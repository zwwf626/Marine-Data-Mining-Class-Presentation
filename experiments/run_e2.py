"""
E2 —— 裁决 A5 的"Airbnb 排名"论证：缺的那一臂，和真正的机制

A5 官方文档的论证链：
    H3 格子面积不等 → 按【每格房源数】排名 → 排名随面积变化 → "H3 有系统性偏差"
    建议：变量面积网格必须归一化到 km²

本实验要回答三个问题：
  Q1  把"每格"换成"每 km²"，排名会变吗？            → 补上 A5 没做的那一臂
  Q2  如果不变，那 A5 观测到的排名变化从哪来？        → 提出并检验"尺度律"解释
  Q3  等面积网格的"好处"到底是不是等面积带来的？      → 用纯等面积族做对照

方法：
  1. 复现 A5 的选区流程（top-N 格累积到 10 km²，求和 / 10 km²），
     在 120 个合成"城市"上跑 2x2 对照 + 两个尺度对照。
  2. 用 A5 自己（精确等面积、无面积不等）的 r12–r15 四个分辨率拟合尺度律
         theta(A) = C * A^(-beta)
     再用这个只含"格子大小"、不含"面积不等"的模型去预测 H3 r9，
     与实测比较。若吻合，说明 A5 观测到的差异与"面积不等"无关。

输出：results/e2_arms.json, results/e2_scaling.json
"""
import json
import os
import numpy as np
from exp_lib import make_grid, R_EARTH

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

N_CITY = 120
TARGET_AREA = 10.0


def random_city_points(rng, n, lon0, lat0, radius_km=12.0, sigma_km=1.5):
    n_cl = 40
    r = radius_km * np.sqrt(rng.random(n_cl))
    th = rng.random(n_cl) * 2 * np.pi
    cl_lon = lon0 + np.degrees(r * np.cos(th) / (R_EARTH * np.cos(np.radians(lat0))))
    cl_lat = lat0 + np.degrees(r * np.sin(th) / R_EARTH)
    w = rng.random(n_cl) ** 3
    w /= w.sum()
    counts = rng.multinomial(n, w)
    sig = sigma_km * (0.5 + rng.random(n_cl) * 2.0)
    lons, lats = [], []
    for k in range(n_cl):
        m = counts[k]
        if m == 0:
            continue
        d = rng.normal(0, sig[k], (m, 2))
        bad = (np.abs(d) > 3 * sig[k]).any(axis=1)
        while bad.any():
            d[bad] = rng.normal(0, sig[k], (bad.sum(), 2))
            bad = (np.abs(d) > 3 * sig[k]).any(axis=1)
        lons.append(cl_lon[k] + np.degrees(d[:, 0] / (R_EARTH * np.cos(np.radians(cl_lat[k])))))
        lats.append(cl_lat[k] + np.degrees(d[:, 1] / R_EARTH))
    return np.concatenate(lons), np.concatenate(lats)


def select_density(grid, lon, lat, use_area_norm):
    codes = grid.bin(lon, lat)
    uniq, cnt = np.unique(codes, return_counts=True)
    A = grid.area(uniq)
    val = cnt / A if use_area_norm else cnt.astype(float)
    order = np.argsort(-val)
    cum = np.cumsum(A[order])
    n = int(np.searchsorted(cum, TARGET_AREA) + 1)
    n = max(1, min(n, len(order)))
    sel = order[:n]
    return cnt[sel].sum() / A[sel].sum(), n, float(A[sel].mean())


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(len(x)); return r


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def main():
    rng = np.random.default_rng(7)
    cities = []
    for j in range(N_CITY):
        lat0 = float(rng.uniform(-55, 65))
        lon0 = float(rng.uniform(-180, 180))
        target_density = float(np.exp(rng.uniform(np.log(5), np.log(50))))
        n = int(max(400, min(3000, target_density * 12.0)))
        lo, la = random_city_points(rng, n, lon0, lat0)
        cities.append((lo, la))

    arms = {
        "A5_r12_pcell": (make_grid("A5", 12), False),
        "A5_r13_pcell": (make_grid("A5", 13), False),
        "A5_r14_pcell": (make_grid("A5", 14), False),
        "A5_r14_pkm2": (make_grid("A5", 14), True),
        "A5_r15_pcell": (make_grid("A5", 15), False),
        "H3_r8_pcell": (make_grid("H3", 8), False),
        "H3_r9_pcell": (make_grid("H3", 9), False),
        "H3_r9_pkm2": (make_grid("H3", 9), True),
    }

    vals = {k: [] for k in arms}
    ncell = {k: [] for k in arms}
    meanA = {k: [] for k in arms}
    for (lo, la) in cities:
        for k, (g, normed) in arms.items():
            d, nc, ma = select_density(g, lo, la, normed)
            vals[k].append(d); ncell[k].append(nc); meanA[k].append(ma)

    ref = np.array(vals["A5_r15_pcell"])  # 最细的等面积网格做参考
    out = {"n_city": N_CITY, "target_area_km2": TARGET_AREA, "arms": {}}
    for k in arms:
        v = np.array(vals[k])
        top10 = set(np.argsort(-v)[:10]); t10r = set(np.argsort(-ref)[:10])
        out["arms"][k] = dict(
            grid=arms[k][0].name, normalized_by_area=bool(arms[k][1]),
            mean_density=float(v.mean()),
            mean_cells=float(np.mean(ncell[k])),
            mean_cell_area=float(np.mean(meanA[k])),
            spearman_vs_fine=spearman(v, ref),
            top10_overlap=len(top10 & t10r),
            max_rank_shift=float(np.max(np.abs(rankdata(v) - rankdata(ref)))),
        )
        r = out["arms"][k]
        print(f"{k:16s} A={r['mean_cell_area']:7.4f} n={r['mean_cells']:6.1f} "
              f"dens={r['mean_density']:7.3f} rho={r['spearman_vs_fine']:6.3f} "
              f"top10={r['top10_overlap']:2d} shift={r['max_rank_shift']:5.1f}")

    # ---- 尺度律：只用 A5（精确等面积，无面积不等项）拟合 theta = C * A^-beta
    a5 = [(a["mean_cell_area"], a["mean_density"])
          for k, a in out["arms"].items()
          if k.startswith("A5") and k.endswith("pcell")]
    a5.sort()
    la_, ld_ = np.log([p[0] for p in a5]), np.log([p[1] for p in a5])
    beta, logC = np.polyfit(la_, ld_, 1)
    pred = lambda A: float(np.exp(logC + beta * np.log(A)))

    sc = {"fit_beta": float(beta), "fit_C": float(np.exp(logC)),
          "a5_points": [{"area": p[0], "density": p[1], "pred": pred(p[0])} for p in a5],
          "validation": []}
    for k in ["H3_r9_pcell", "H3_r8_pcell", "H3_r9_pkm2"]:
        A = out["arms"][k]["mean_cell_area"]
        obs = out["arms"][k]["mean_density"]
        pr = pred(A)
        sc["validation"].append(dict(arm=k, area=A, observed=obs, predicted=pr,
                                     rel_err=(obs - pr) / pr))
        print(f"  尺度律预测 {k}: A={A:.4f} 观测={obs:.3f} 预测={pr:.3f} "
              f"相对误差={(obs-pr)/pr*100:+.2f}%")

    # ---- 决定性检验：跨城市回归
    # H3 r9 的局部格子面积随纬度变化（0.078~0.127 km^2，1.6 倍），
    # A5 r14 面积恒定。若"面积不等"是机制，则
    #     log(dens_H3 / dens_A5) 应随 log(A_H3 / A_A5) 以斜率 beta 变化。
    dA5 = np.array(vals["A5_r14_pcell"]); dH3 = np.array(vals["H3_r9_pcell"])
    aH3 = np.array(meanA["H3_r9_pcell"]); aA5 = np.array(meanA["A5_r14_pcell"])
    x = np.log(aH3 / aA5); y = np.log(dH3 / dA5)
    sl, ic = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    # 局部插值（只用 r14-r15 两点，紧邻 H3 r9 的尺度）
    a14, d14 = out["arms"]["A5_r14_pcell"]["mean_cell_area"], out["arms"]["A5_r14_pcell"]["mean_density"]
    a15, d15 = out["arms"]["A5_r15_pcell"]["mean_cell_area"], out["arms"]["A5_r15_pcell"]["mean_density"]
    beta_loc = float(np.log(d15 / d14) / np.log(a14 / a15))
    aH9 = out["arms"]["H3_r9_pcell"]["mean_cell_area"]
    pred_local = float(d14 * (a14 / aH9) ** beta_loc)
    obs_h9 = out["arms"]["H3_r9_pcell"]["mean_density"]

    np.savez_compressed(os.path.join(OUT, "e2_percity.npz"),
                        dA5=dA5, dH3=dH3, aH3=aH3, aA5=aA5,
                        dens={k: np.array(v) for k, v in vals.items()},
                        area={k: np.array(v) for k, v in meanA.items()})
    sc["regression_cross_city"] = dict(
        slope=float(sl), intercept=float(ic), r2=r2, n=len(x),
        area_ratio_range=[float(aH3.min() / aA5.max()), float(aH3.max() / aA5.min())],
        comment="若斜率约等于 beta，说明 H3 的'偏差'随格子尺寸变化，即尺度效应，而非面积归一化缺失。")
    sc["local_interp"] = dict(
        beta_local=beta_loc, a_ref=a14, d_ref=d14,
        predicted_at_H3r9=pred_local, observed_H3r9=obs_h9,
        rel_err=(obs_h9 - pred_local) / pred_local)
    sc["gap_decomposition"] = dict(
        observed_gap=float(obs_h9 / out["arms"]["A5_r14_pcell"]["mean_density"]),
        predicted_by_cell_size=float((a14 / aH9) ** beta_loc),
        cell_area_ratio=float(a14 / aH9))
    print(f"  跨城市回归: slope={sl:.3f} R2={r2:.3f}  (beta_local={beta_loc:.3f})")
    print(f"  局部插值预测 H3 r9 = {pred_local:.3f}, 观测 {obs_h9:.3f}, "
          f"相对误差 {(obs_h9-pred_local)/pred_local*100:+.2f}%")
    print(f"  原始差距 = {obs_h9/out['arms']['A5_r14_pcell']['mean_density']:.3f}, "
          f"纯由格子面积预测 = {(a14/aH9)**beta_loc:.3f}")

    h3r9 = make_grid("H3", 9)
    sc["note"] = ("beta 为负说明：格子越小，'top-10km² 密度'越大。"
                  "这使该指标对分辨率敏感，而对'是否除以格面积'不敏感。")

    with open(os.path.join(OUT, "e2_arms.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "e2_scaling.json"), "w") as f:
        json.dump(sc, f, indent=2, ensure_ascii=False)
    print("saved e2_arms.json / e2_scaling.json")


if __name__ == "__main__":
    main()
