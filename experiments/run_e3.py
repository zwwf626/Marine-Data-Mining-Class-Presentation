"""
E3 —— 划区效应（zoning effect）的蒙特卡洛估计：网格之争有没有超过噪声下限？

难点：DGGS 没法像经纬格那样"平移"。但球面上有一个天然的操作：
      **整体旋转点集 ≡ 把同一套网格换一个摆放位置**（球面旋转是保测变换）。

于是可以给任意 DGGS 构造"划区集合"：
      随机旋转 M_r ~ Uniform(SO(3))，r = 1..R
      对每个 r 重新分箱 → 得到同一片海、同一套网格、不同划区的 R 个结果
      R 个结果的离散度 = 划区效应（zoning effect）的非参数置信区间

再用同一批点在**同一旋转**下重抽 R 次泊松样本，得到采样噪声，做方差分解。

判据（本文提出的"噪声下限"判据）：
      若 |theta_A - theta_B| < 2*sd_zoning，则"A 网格 vs B 网格"的差别
      小于"同一网格换个原点"的差别 —— 换网格没有意义。

输出：results/e3_rotation.json
"""
import json
import os
import itertools
import numpy as np
from exp_lib import (VMFField, make_grid, lonlat_to_xyz, xyz_to_lonlat,
                     random_rotation)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

BLOBS = [
    (3, 55, 150, 1.0), (40, 72, 130, 0.7), (150, 42, 180, 1.2), (-80, -10, 120, 0.9),
    (2, 0, 140, 0.6), (62, 15, 160, 0.5), (-20, -45, 220, 0.4), (-175, 57, 150, 0.6),
]

N = 3_000_000
R_ROT = 16
R_NOISE = 4
TARGET_AREA = 5_000.0   # km^2，定义"热点区域"的规模
MIN_COUNT = 3            # 单格至少 3 个点，抑制"一个点占一个小格"的伪高密度

GRIDS = [("H3", 6), ("A5", 10), ("H3", 4), ("A5", 8)]


def hotspot(grid, lon, lat):
    """返回 (热点密度, 选中格子覆盖面积, 选中点索引集合)。"""
    codes = grid.bin(lon, lat)
    uniq, inv, cnt = np.unique(codes, return_inverse=True, return_counts=True)
    A = grid.area(uniq)
    dens = cnt / A
    dens = np.where(cnt >= MIN_COUNT, dens, -np.inf)
    order = np.argsort(-dens)
    cum = np.cumsum(A[order])
    n = int(np.searchsorted(cum, TARGET_AREA) + 1)
    n = max(1, min(n, len(order)))
    sel = order[:n]
    theta = cnt[sel].sum() / A[sel].sum()
    keep = np.isin(inv, sel)
    return float(theta), float(A[sel].sum()), set(np.nonzero(keep)[0].tolist())


def main():
    rng = np.random.default_rng(11)
    field = VMFField(BLOBS, background_share=0.02)
    lon0, lat0 = field.sample(N, rng)
    print(f"sampled {len(lon0):,}")

    rots = [np.eye(3)] + [random_rotation(rng) for _ in range(R_ROT - 1)]
    out = {"n_points": int(N), "n_rotations": R_ROT, "target_area_km2": TARGET_AREA,
           "min_count": MIN_COUNT, "grids": {}}

    ptsets = {g: [] for g in (f"{f} r{r}" for f, r in GRIDS)}
    for fam, res in GRIDS:
        g = make_grid(fam, res)
        thetas, areas, sets = [], [], []
        for M in rots:
            x = lonlat_to_xyz(lon0, lat0) @ M.T  # noqa
            lo, la = xyz_to_lonlat(x)
            th, ar, ps = hotspot(g, lo, la)
            thetas.append(th); areas.append(ar); sets.append(ps)
        ptsets[g.name] = sets
        thetas = np.array(thetas)
        # 组内稳定性：两两 Jaccard
        jac = [len(a & b) / len(a | b)
               for a, b in itertools.combinations(sets, 2)]
        out["grids"][g.name] = dict(
            cell_area=float(g.nominal_area),
            theta_mean=float(thetas.mean()), theta_sd=float(np.std(thetas, ddof=1)),
            theta_cv=float(thetas.std(ddof=1) / thetas.mean()),
            theta_p05=float(np.percentile(thetas, 5)),
            theta_p95=float(np.percentile(thetas, 95)),
            area_mean=float(np.mean(areas)),
            jaccard_mean=float(np.mean(jac)), jaccard_min=float(np.min(jac)),
            thetas=thetas.tolist(),
        )
        print(f"{g.name:10s} A={g.nominal_area:8.2f} theta={thetas.mean():7.2f} "
              f"sd_zoning={thetas.std(ddof=1):5.2f} (CV={thetas.std(ddof=1)/thetas.mean()*100:4.1f}%) "
              f"Jaccard={np.mean(jac):.3f}")

    # ---- 采样噪声（同一旋转，独立泊松重抽）
    noise = {}
    for fam, res in GRIDS:
        g = make_grid(fam, res)
        ts = []
        for _ in range(R_NOISE):
            lo, la = field.sample(N, rng)
            ts.append(hotspot(g, lo, la)[0])
        noise[g.name] = float(np.std(ts, ddof=1))
        out["grids"][g.name]["sd_sampling"] = noise[g.name]

    # ---- 网格对比较
    pairs = [("H3 r6", "A5 r10"), ("H3 r4", "A5 r8")]
    cmp = []
    for a, b in pairs:
        ga, gb = out["grids"][a], out["grids"][b]
        diff = abs(ga["theta_mean"] - gb["theta_mean"])
        pooled = float(np.hypot(ga["theta_sd"], gb["theta_sd"]) / np.sqrt(2))
        cmp.append(dict(
            pair=[a, b],
            area_ratio=ga["cell_area"] / gb["cell_area"],
            theta_a=ga["theta_mean"], theta_b=gb["theta_mean"],
            abs_diff=diff, pooled_sd=pooled,
            diff_over_pooled_sd=diff / pooled,
            within_grid_sd=pooled,
            verdict=("差异小于划区噪声" if diff < pooled else "差异超过划区噪声"),
        ))
        print(f"  {a} vs {b}: |Δ|={diff:.2f}  合并 sd={pooled:.2f}  "
              f"比值={diff/pooled:.2f}  → {cmp[-1]['verdict']}")
    out["pair_comparison"] = cmp

    # ---- 方差分解（以 H3 r6 为例）
    var_z = out["grids"]["H3 r6"]["theta_sd"] ** 2
    var_s = noise["H3 r6"] ** 2
    out["variance_decomposition"] = dict(
        grid="H3 r6", var_zoning=var_z, var_sampling=var_s,
        share_zoning=var_z / (var_z + var_s),
        comment="划区效应与采样噪声的相对大小")
    print(f"  方差分解(H3 r6): 划区占 {var_z/(var_z+var_s)*100:.1f}%")

    with open(os.path.join(OUT, "e3_rotation.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved results/e3_rotation.json")


if __name__ == "__main__":
    main()
