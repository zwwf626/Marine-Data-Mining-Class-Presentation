"""
E5 —— 反转概率地图：网格面积比 alpha 与真值比 beta 的竞争

理论推导（这是全文的核心判决条件）：
  未归一化时，产品报告的是 r_i = n_i / A_mean，而 n_i ≈ A_i * rho_i，
  所以 r_i ∝ A_i * rho_i —— 它排序的是"面积 × 密度"，不是密度。

  对任意两个单元 i, j，令  alpha = A_i / A_j,  beta = rho_i / rho_j,
     真值说"i 更密"  <=>  beta > 1
     产品说"i 更密"  <=>  alpha*beta > 1
  两者不一致（排名反转）<=> beta < 1 < alpha*beta，即
  ================  alpha > 1/beta  ================
  换句话说：**只要两个单元的面积比超过它们真值比的倒数，排名就一定反。**

本脚本在真实（合成）场上把这个条件画成概率地图，检验实测是否符合理论边界。

输出：results/e5_inversion.npz / results/e5_summary.json / cell 数据
"""
import json
import os
import numpy as np
from exp_lib import VMFField, make_grid, aggregate, cell_truth

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

BLOBS = [
    (3, 55, 450, 1.0), (40, 72, 380, 0.7), (150, 42, 520, 1.2), (-80, -10, 320, 0.9),
    (2, 0, 350, 0.6), (62, 15, 420, 0.5), (-20, -45, 600, 0.4), (-175, 57, 450, 0.6),
]
N = 2_000_000
GRIDS = [("H3", 4), ("A5", 8), ("LatLon", 1.0)]
NB = 70
LM = 2.5   # log 轴范围 [-2.5, 2.5]


def main():
    rng = np.random.default_rng(2026)
    field = VMFField(BLOBS, background_share=0.02)
    lon, lat = field.sample(N, rng)

    store = {}
    summary = {"n_points": N, "grids": {}}
    for fam, res in GRIDS:
        g = make_grid(fam, res)
        codes = g.bin(lon, lat)
        uniq, cnt = np.unique(codes, return_counts=True)
        A = g.area(uniq)
        t = cell_truth(field, g, uniq, N)
        store[f"{fam}_{res}"] = dict(cnt=cnt, A=A, t=t)
        summary["grids"][g.name] = dict(
            n_cells=int(len(uniq)), area_mean=float(A.mean()),
            area_min=float(A.min()), area_max=float(A.max()),
            area_max_over_min=float(A.max() / A.min()),
            area_cv=float(A.std() / A.mean()))
        print(f"{g.name}: cells={len(uniq)} A range {A.min():.2f}-{A.max():.2f}")

    np.savez_compressed(os.path.join(OUT, "e5_cells.npz"),
                        **{f"{k}__{p}": v[p] for k, v in store.items() for p in v})

    # ---- 反转率与 alpha、beta 的关系（解析式，可直接画）
    beta_grid = np.exp(np.linspace(-LM, LM, 400))
    alpha_grid = np.exp(np.linspace(-LM, LM, 400))

    # ---- 实测：在 H3 r4 与 LatLon 1° 上随机抽格子对
    inv_maps = {}
    for key in ["H3_4", "LatLon_1.0"]:
        d = store[key]
        A, t, cnt = d["A"], d["t"], d["cnt"].astype(float)
        rho = t / A
        m = (t >= 5) & (cnt > 0)
        A, t, cnt, rho = A[m], t[m], cnt[m], rho[m]
        n = len(A)
        S = 1_500_000
        i = rng.integers(0, n, S)
        j = rng.integers(0, n, S)
        ok = i != j
        i, j = i[ok], j[ok]
        la = np.log(A[i] / A[j])
        lb = np.log(rho[i] / rho[j])
        rep_i = cnt[i] / A[i].mean()
        rep_j = cnt[j] / A[j].mean()
        tru_i = rho[i]; tru_j = rho[j]
        inv = (rep_i > rep_j) != (tru_i > tru_j)
        H, xe, ye = np.histogram2d(lb, la, bins=NB, range=[[-LM, LM], [-LM, LM]])
        Hi, _, _ = np.histogram2d(lb[inv], la[inv], bins=(xe, ye))
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(H > 0, Hi / H, np.nan)
        inv_maps[key] = dict(H=H, Hi=Hi, rate=rate, xe=xe, ye=ye)
        summary[key] = {"pairs": int(len(i)), "inversion_rate": float(inv.mean())}
        print(f"{key}: 随机格子对 {len(i):,} 个，整体反转率 {inv.mean()*100:.2f}%")

    np.savez_compressed(os.path.join(OUT, "e5_inversion.npz"),
                        **{f"{k}__{p}": v[p] for k, v in inv_maps.items() for p in v})
    with open(os.path.join(OUT, "e5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("saved e5_cells.npz / e5_inversion.npz / e5_summary.json")


if __name__ == "__main__":
    main()
