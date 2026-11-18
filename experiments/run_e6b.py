"""
E6b（§7.4 专用）—— R² 到底是网格属性还是场属性？

做法：用纯均匀场（只均匀撒点、无任何结构，误差 100% 来自面积）重算同一个网格，
      与"真实感场"对照。若两者接近，说明 R² 由网格的面积分布决定，与场无关。

输出：results/e6b_r2.json
"""
import json
import os
import numpy as np
from exp_lib import VMFField, make_grid, cell_truth

OUT = os.path.join(os.path.dirname(__file__), "results")
N = 3_000_000
BLOBS = [(3, 55, 600, 1.0), (40, 72, 500, 0.7), (150, 42, 700, 1.2), (-80, -10, 400, 0.9),
         (2, 0, 450, 0.6), (62, 15, 550, 0.5), (-20, -45, 800, 0.4), (-175, 57, 600, 0.6)]


def uniform(n, rng):
    z = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(1 - z * z)
    return np.degrees(np.arctan2(r * np.sin(ph), r * np.cos(ph))), np.degrees(np.arcsin(z))


def main():
    rng = np.random.default_rng(2026)
    field = VMFField(BLOBS, background_share=0.02)
    cases = {"uniform": uniform(N, rng), "baseline": field.sample(N, rng)}

    out = {"n_points": N, "rows": []}
    for cname, (lon, lat) in cases.items():
        for fam, res in [("H3", 4), ("LatLon", 1.0), ("LatLon", 2.0)]:
            g = make_grid(fam, res)
            codes = g.bin(lon, lat)
            uniq, cnt = np.unique(codes, return_counts=True)
            A = g.area(uniq)
            # 均匀场的真值就是常数密度：t_i ∝ A_i
            t = (A / A.sum() * N) if cname == "uniform" else cell_truth(field, g, uniq, N)
            m = t >= 10
            R = (cnt[m] / np.mean(A)) / (t[m] / A[m])
            x = np.log(A[m] / np.mean(A))
            y = np.log(R)
            r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if np.std(x) > 0 else 0.0
            out["rows"].append(dict(field=cname, grid=g.name, r2=r2,
                                    var_logA=float(np.var(x)), n_mask=int(m.sum())))
            print(f"{cname:9s} {g.name:10s} R2={r2:6.3f}  var(logA)={np.var(x):.4f}")

    with open(os.path.join(OUT, "e6b_r2.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved results/e6b_r2.json")


if __name__ == "__main__":
    main()
