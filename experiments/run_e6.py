"""
E6 —— 合成数据可信度体检：结论对"场"有多敏感？

动机：全部实验用的是解析合成场。若结论只在"8 个高斯团 + 均匀背景"下成立，
      那它就不是关于网格的结论，而是关于这个合成场的结论。

做法：换 6 种形态差异很大的场，重跑关键指标，看哪些结论会翻。

  F1 baseline   8 个中等团 + 2% 背景（正文用的场）
  F2 uniform    纯均匀，无任何结构
  F3 smooth     4 个超大团（σ=1500 km），结构远大于格子
  F4 polar      全部集中在 60–80°N，检验纬度依赖
  F5 sparse-blob 20 个小团（σ=120 km），结构接近格子尺度
  F6 regional   区域型斑块（σ≈6 km ≈ 格子直径），结构就在格子尺度上

指标：
  R2_area  未归一化误差中由面积解释的方差份额（命题 1 的强度）
  cv_zon   同一网格换 8 次摆放位置的热点密度 CV（划区效应）
  jac      选中点集的 Jaccard
  gap      |θ_H3 − θ_A5| / 均值（两族之间的差别）

再做一个采样密度扫描（N = 1e5 … 6.4e6），看 jac 是不是"点太少"造出来的假象。

输出：results/e6_sensitivity.json
"""
import json
import os
import itertools
import numpy as np
from exp_lib import (VMFField, make_grid, lonlat_to_xyz, xyz_to_lonlat,
                     random_rotation, cell_truth, R_EARTH)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

N = 1_500_000
R_ROT = 6
TARGET_AREA = 5_000.0
MIN_COUNT = 3


class FastA5:
    """A5 的面积变异只有 0.41%，敏感性体检里直接用标称面积，省掉逐格多边形积分。"""

    def __init__(self, res):
        self.g = make_grid("A5", res)
        self.name = self.g.name
        self.nominal = self.g.nominal_area

    def bin(self, lon, lat):
        return self.g.bin(lon, lat)

    def area(self, codes):
        return np.full(len(codes), self.nominal)

    def center(self, codes):
        return self.g.center(codes)


def build_fields():
    base = [(3, 55, 600, 1.0), (40, 72, 500, 0.7), (150, 42, 700, 1.2),
            (-80, -10, 400, 0.9), (2, 0, 450, 0.6), (62, 15, 550, 0.5),
            (-20, -45, 800, 0.4), (-175, 57, 600, 0.6)]
    smooth = [(0, 30, 1500, 1.0), (150, -20, 1800, 1.0),
              (-120, 50, 1600, 0.9), (60, -50, 1700, 0.8)]
    polar = [(10, 65, 400, 1.0), (40, 72, 380, 0.9), (-30, 70, 420, 0.8),
             (80, 68, 400, 0.7), (140, 75, 380, 0.6), (-100, 78, 400, 0.7)]
    rng = np.random.default_rng(5)
    small = [(float(rng.uniform(-170, 170)), float(rng.uniform(-60, 70)), 120.0, 1.0)
             for _ in range(20)]
    return {
        "F1_baseline": dict(blobs=base, bg=0.02),
        "F2_uniform": dict(blobs=[], bg=1.0),
        "F3_smooth": dict(blobs=smooth, bg=0.02),
        "F4_polar": dict(blobs=polar, bg=0.02),
        "F5_smallblob": dict(blobs=small, bg=0.05),
    }


def hotspot(grid, lon, lat):
    codes = grid.bin(lon, lat)
    uniq, inv, cnt = np.unique(codes, return_inverse=True, return_counts=True)
    A = grid.area(uniq)
    dens = np.where(cnt >= MIN_COUNT, cnt / A, -np.inf)
    order = np.argsort(-dens)
    cum = np.cumsum(A[order])
    n = max(1, min(int(np.searchsorted(cum, TARGET_AREA) + 1), len(order)))
    sel = order[:n]
    theta = cnt[sel].sum() / A[sel].sum()
    keep = np.isin(inv, sel)
    return float(theta), set(np.nonzero(keep)[0].tolist())


def zoning_stats(grid, lon0, lat0, rots):
    thetas, sets = [], []
    base = lonlat_to_xyz(lon0, lat0)
    for M in rots:
        lo, la = xyz_to_lonlat(base @ M.T)
        th, ps = hotspot(grid, lo, la)
        thetas.append(th); sets.append(ps)
    thetas = np.array(thetas)
    jac = [len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(sets, 2)]
    return dict(theta=float(thetas.mean()),
                cv=float(thetas.std(ddof=1) / thetas.mean()),
                jaccard=float(np.mean(jac)))


def area_r2(grid, lon, lat, n_total, truth_cache=None):
    codes = grid.bin(lon, lat)
    uniq, cnt = np.unique(codes, return_counts=True)
    A = grid.area(uniq)
    t = cell_truth(truth_cache, grid, uniq, n_total)
    m = t >= 5
    if m.sum() < 100:
        return float("nan")
    R = (cnt[m] / np.mean(A)) / (t[m] / A[m])
    x = np.log(A[m] / np.mean(A))
    y = np.log(R)
    if np.std(x) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def regional_patches():
    """区域型高密度斑块：结构尺度 ≈ 格子直径（~6 km）。"""
    rng = np.random.default_rng(31)
    lon_c, lat_c = 15.0, 35.0          # 地中海附近
    half = 900.0                       # km
    n_patch = 60
    px = rng.uniform(-half, half, n_patch)
    py = rng.uniform(-half, half, n_patch)
    sigma = rng.uniform(4.0, 8.0, n_patch)   # 4–8 km，正对 H3 r6 的 6.8 km 直径
    w = rng.random(n_patch) ** 2; w /= w.sum()
    counts = rng.multinomial(400_000, w)
    lons, lats = [], []
    for k in range(n_patch):
        m = counts[k]
        dx = rng.normal(0, sigma[k], m); dy = rng.normal(0, sigma[k], m)
        lons.append(lon_c + np.degrees((px[k] + dx) / (R_EARTH * np.cos(np.radians(lat_c)))))
        lats.append(lat_c + np.degrees((py[k] + dy) / R_EARTH))
    return np.concatenate(lons), np.concatenate(lats)


def main():
    out = {"n_points": N, "n_rotations": R_ROT, "fields": {}}
    rng = np.random.default_rng(2026)
    rots = [random_rotation(rng) for _ in range(R_ROT)]

    h3g = make_grid("H3", 6)
    a5g = FastA5(10)
    llg = make_grid("LatLon", 1.0)

    for fname, cfg in build_fields().items():
        field = VMFField(cfg["blobs"], background_share=cfg["bg"]) if cfg["blobs"] else None
        if field is None:
            # 纯均匀
            z = rng.uniform(-1, 1, N); ph = rng.uniform(0, 2 * np.pi, N)
            r = np.sqrt(1 - z * z)
            lon = np.degrees(np.arctan2(r * np.sin(ph), r * np.cos(ph)))
            lat = np.degrees(np.arcsin(z))
            truth_obj = None
        else:
            lon, lat = field.sample(N, rng)
            truth_obj = field

        rec = {"n_cells_occupied": int(len(np.unique(h3g.bin(lon, lat))))}

        if truth_obj is not None:
            rec["R2_area_H3r6"] = area_r2(h3g, lon, lat, N, truth_obj)
            rec["R2_area_LL1"] = area_r2(llg, lon, lat, N, truth_obj)

        for tag, g in (("H3r6", h3g), ("A5r10", a5g)):
            st = zoning_stats(g, lon, lat, rots)
            rec[f"{tag}_theta"] = st["theta"]
            rec[f"{tag}_cv"] = st["cv"]
            rec[f"{tag}_jaccard"] = st["jaccard"]

        rec["gap_H3_vs_A5"] = abs(rec["H3r6_theta"] - rec["A5r10_theta"]) / \
                              ((rec["H3r6_theta"] + rec["A5r10_theta"]) / 2)
        out["fields"][fname] = rec
        print(f"{fname:14s} R2(H3r6)={rec.get('R2_area_H3r6',float('nan')):.3f} "
              f"cv_H3={rec['H3r6_cv']*100:5.2f}% cv_A5={rec['A5r10_cv']*100:5.2f}% "
              f"jac={rec['H3r6_jaccard']:.3f} gap={rec['gap_H3_vs_A5']*100:5.2f}%")

    # ---------------- 采样密度扫描 ----------------
    print("\n采样密度扫描（H3 r6）")
    field = VMFField(build_fields()["F1_baseline"]["blobs"], background_share=0.02)
    sweep = []
    for n in (100_000, 400_000, 1_600_000, 6_400_000):
        lo, la = field.sample(n, np.random.default_rng(99))
        st = zoning_stats(h3g, lo, la, rots)
        per_cell = n / 2_000_000.0   # 全球平均每格点数（H3 r6 约 2M 格）
        sweep.append(dict(n=n, points_per_cell=per_cell, cv=st["cv"],
                          jaccard=st["jaccard"], theta=st["theta"]))
        print(f"  N={n:>9,}  每格≈{per_cell:6.2f} 点  cv={st['cv']*100:5.2f}%  "
              f"jaccard={st['jaccard']:.3f}")
    out["sampling_sweep"] = sweep

    # ---------------- 区域型斑块：结构就在格子尺度 ----------------
    print("\n区域型斑块（结构 ≈ 格子直径）")
    lo, la = regional_patches()
    reg = {}
    for tag, g in (("H3r6", h3g), ("A5r10", a5g)):
        st = zoning_stats(g, lo, la, rots)
        reg[tag] = st
        print(f"  {tag}: theta={st['theta']:.3f} cv={st['cv']*100:.2f}% jaccard={st['jaccard']:.3f}")
    reg["gap_H3_vs_A5"] = abs(reg["H3r6"]["theta"] - reg["A5r10"]["theta"]) / \
                          ((reg["H3r6"]["theta"] + reg["A5r10"]["theta"]) / 2)
    print(f"  两族差距 {reg['gap_H3_vs_A5']*100:.2f}%")
    out["regional"] = reg

    with open(os.path.join(OUT, "e6_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved results/e6_sensitivity.json")


if __name__ == "__main__":
    main()
