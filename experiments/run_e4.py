"""
E4 —— 强度量（intensive quantity）：归一化为什么救不了它，
      以及 HESS 2024 那个 28% 到底是哪一步带来的？

强度量 = 比值型（渔业努力中拖网占比、平均深度、某物种占比）。
它不可加，所以"除以格面积"是空操作 —— 面积归一化对强度量没有任何作用。

本实验三个层次：
  T1  全球平均：经纬网格的"格平均"等价于按 1/cos(phi) 加权 → 高纬被放大。
      等面积网格让"简单平均"自动等于"面积加权平均"。
  T2  分区排序：8 个海域按强度量排序，未加权 vs 面积加权，数逆序对。
  T3  重采样：1°→0.25° 源格搬到目标格上，
        A = 对落入的源格做【无权平均】（HESS 的做法）
        B = 对落入的源格做【面积加权平均】（本文主张）
      比较 NRMSE。这是任何下游模型都突破不了的误差下限。

输出：results/e4_intensive.json
"""
import json
import os
import numpy as np
import h3 as _h3
import a5 as _a5
from exp_lib import make_grid, R_EARTH, LatLonGrid

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

_GL_X, _GL_W = np.polynomial.legendre.leggauss(6)


def f_int(lon, lat):
    """一个'看起来很合理'的强度量场：中纬度高、副热带低，带经向波动。"""
    v = (0.10
         + 0.45 * np.exp(-((lat - 40.0) / 22.0) ** 2)
         + 0.28 * np.exp(-((lat + 38.0) / 18.0) ** 2)
         + 0.08 * np.cos(np.radians(2.0 * lon))
         + 0.04 * np.sin(np.radians(3.0 * lat)) * np.cos(np.radians(lon))
         + 0.075 * np.sin(np.radians(43.0 * lon)) * np.cos(np.radians(31.0 * lat))
         + 0.055 * np.cos(np.radians(29.0 * lon + 40.0)) * np.sin(np.radians(37.0 * lat)))
    return np.clip(v, 0.0, 0.95)


def cell_mean(func, grid, codes):
    """func 在每个格子上的面积加权平均（数值求积）。"""
    A = grid.area(codes)
    if isinstance(grid, LatLonGrid):
        d = grid.d
        i, j = grid._dims(codes)
        lo0 = -180.0 + i * d
        s1 = np.sin(np.radians(-90.0 + j * d))
        s2 = np.sin(np.radians(-90.0 + (j + 1) * d))
        u = 0.5 * (1 + _GL_X); w = 0.5 * _GL_W
        LON = lo0[:, None] + d * u[None, :]
        SIN = s1[:, None] + (s2 - s1)[:, None] * u[None, :]
        LAT = np.degrees(np.arcsin(SIN))
        s = np.zeros(len(codes))
        for a in range(6):
            for b in range(6):
                s += w[a] * w[b] * func(LON[:, a], LAT[:, b])
        return s
    lon_c, lat_c = grid.center(codes)
    r_eq = np.sqrt(A / np.pi)
    pat = [(0.0, 0.0)]
    for ir in (1, 2):
        rr = ir / 3.0 * 0.95
        for k in range(8):
            th = 2 * np.pi * k / 8 + (np.pi / 8 if ir % 2 else 0)
            pat.append((rr * np.cos(th), rr * np.sin(th)))
    coslat = np.cos(np.radians(lat_c))
    s = np.zeros(len(codes))
    for dx, dy in pat:
        s += func(lon_c + np.degrees(dx * r_eq / (R_EARTH * coslat)),
                  lat_c + np.degrees(dy * r_eq / R_EARTH))
    return s / len(pat)


def all_codes(grid):
    """枚举一个网格族的全部格子。"""
    if isinstance(grid, LatLonGrid):
        return np.arange(grid.ni * grid.nj, dtype=np.uint64)
    if grid.family == "H3":
        out = []
        for c in _h3.get_res0_cells():
            out.extend(_h3.cell_to_children(c, grid.res))
        return np.array([int(c, 16) for c in out], dtype=np.uint64)
    codes = list(_a5.get_res0_cells())
    for _ in range(grid.res):
        nxt = []
        for c in codes:
            nxt.extend(_a5.cell_to_children(int(c)))
        codes = nxt
    return np.array(codes, dtype=np.uint64)


def main():
    # 真值：0.1° 细网格面积加权
    gf = LatLonGrid(0.1)
    cf = all_codes(gf)
    vf = cell_mean(f_int, gf, cf)
    Af = gf.area(cf)
    lof, laf = gf.center(cf)
    TRUTH = float((vf * Af).sum() / Af.sum())
    out = {"truth_global_mean": TRUTH}
    print(f"真值（面积加权全球平均）= {TRUTH:.5f}")

    # ---------------- T1 ----------------
    t1 = []
    for fam, res in [("LatLon", 5.0), ("LatLon", 2.0), ("H3", 3), ("H3", 4),
                     ("A5", 7), ("A5", 8)]:
        g = make_grid(fam, res)
        codes = all_codes(g)
        val = cell_mean(f_int, g, codes)
        A = g.area(codes)
        unw = float(val.mean())
        wtd = float((val * A).sum() / A.sum())
        t1.append(dict(grid=g.name, n_cells=int(len(codes)),
                       unweighted=unw, area_weighted=wtd,
                       bias_unweighted=unw - TRUTH, bias_weighted=wtd - TRUTH,
                       rel_bias_unweighted=float((unw - TRUTH) / TRUTH),
                       area_max_over_min=float(A.max() / A.min())))
        print(f"  {g.name:12s} 未加权={unw:.5f} ({t1[-1]['rel_bias_unweighted']*100:+.2f}%)  "
              f"面积加权={wtd:.5f} ({(wtd-TRUTH)/TRUTH*100:+.3f}%)  "
              f"面积比={t1[-1]['area_max_over_min']:.1f}")
    out["T1_global_mean"] = t1

    # ---------------- T2 ----------------
    boxes = [
        ("北海", -4, 60, 48, 62), ("巴伦支海", 20, 60, 68, 80),
        ("西北太平洋", 140, 170, 30, 48), ("秘鲁外海", -85, -70, -20, -5),
        ("几内亚湾", -5, 10, -8, 5), ("阿拉伯海", 55, 70, 8, 22),
        ("南大洋", -60, -20, -65, -50), ("墨西哥湾", -95, -80, 20, 30),
    ]
    g1 = LatLonGrid(1.0)
    c1 = all_codes(g1)
    v1 = cell_mean(f_int, g1, c1)
    A1 = g1.area(c1)
    lo1, la1 = g1.center(c1)

    def est(b, weighted):
        m = (lo1 >= b[1]) & (lo1 <= b[2]) & (la1 >= b[3]) & (la1 <= b[4])
        return float((v1[m] * A1[m]).sum() / A1[m].sum() if weighted else v1[m].mean())

    def tru(b):
        m = (lof >= b[1]) & (lof <= b[2]) & (laf >= b[3]) & (laf <= b[4])
        return float((vf[m] * Af[m]).sum() / Af[m].sum())

    T = np.array([tru(b) for b in boxes])
    U = np.array([est(b, False) for b in boxes])
    W = np.array([est(b, True) for b in boxes])

    def inv(e):
        return int(sum(1 for i in range(len(T)) for j in range(i + 1, len(T))
                       if (T[i] - T[j]) * (e[i] - e[j]) < 0))

    out["T2_regions"] = dict(
        names=[b[0] for b in boxes], truth=T.tolist(),
        unweighted=U.tolist(), weighted=W.tolist(),
        inversions_unweighted=inv(U), inversions_weighted=inv(W),
        n_pairs=len(T) * (len(T) - 1) // 2,
        max_abs_err_unweighted=float(np.max(np.abs(U - T))),
        max_abs_err_weighted=float(np.max(np.abs(W - T))),
    )
    print(f"  T2 排序逆序对：未加权 {inv(U)}/{out['T2_regions']['n_pairs']}，"
          f"面积加权 {inv(W)}/{out['T2_regions']['n_pairs']}")

    # ---------------- T3 ----------------
    src = LatLonGrid(1.0)
    sc = all_codes(src)
    sval = cell_mean(f_int, src, sc)
    sA = src.area(sc)
    slo, sla = src.center(sc)

    t3 = []
    for fam, res in [("H3", 2), ("H3", 3), ("A5", 4), ("A5", 5)]:
        tg = make_grid(fam, res)
        tcode = tg.bin(slo, sla)
        uniq_tc, inv_idx = np.unique(tcode, return_inverse=True)
        truth_t = cell_mean(f_int, tg, uniq_tc)

        cnt = np.bincount(inv_idx, minlength=len(uniq_tc))
        un_w = np.bincount(inv_idx, weights=sval, minlength=len(uniq_tc)) / np.maximum(cnt, 1)
        ar_w = (np.bincount(inv_idx, weights=sval * sA, minlength=len(uniq_tc))
                / np.maximum(np.bincount(inv_idx, weights=sA, minlength=len(uniq_tc)), 1e-12))

        rms = lambda x: float(np.sqrt(np.mean((x - truth_t) ** 2)))
        num = lambda x: rms(x) / float(np.std(truth_t))
        tlo, tla = tg.center(uniq_tc)
        latcorr = lambda e: float(np.corrcoef(np.abs(tla), np.abs(e))[0, 1])
        t3.append(dict(grid=tg.name, n_target=int(len(uniq_tc)),
                       rmse_unweighted=rms(un_w), rmse_weighted=rms(ar_w),
                       nrms_unweighted=num(un_w), nrms_weighted=num(ar_w),
                       improvement=1 - rms(ar_w) / rms(un_w),
                       latcorr_unweighted=latcorr(un_w - truth_t),
                       latcorr_weighted=latcorr(ar_w - truth_t)))
        print(f"  T3 {tg.name:8s} NRMSE 无权={t3[-1]['nrms_unweighted']:.4f} "
              f"面积加权={t3[-1]['nrms_weighted']:.4f}  改善 {t3[-1]['improvement']*100:5.1f}%  "
              f"|误差|~|纬度| {t3[-1]['latcorr_unweighted']:+.2f}→{t3[-1]['latcorr_weighted']:+.2f}")
    out["T3_regrid"] = t3

    with open(os.path.join(OUT, "e4_intensive.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved results/e4_intensive.json")


if __name__ == "__main__":
    main()
