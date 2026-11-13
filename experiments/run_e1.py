"""
E1 —— 偏差的解剖：未归一化的"每格"报告到底错在哪？

问题：一个海洋数据产品把"每格多少个船位"直接当密度发布，误差有多大？
      用真实面积归一化之后，误差还剩多少？

做法：von Mises-Fisher 场 + 精确采样 → 三个网格族、各三个分辨率
      对每个网格计算：
        (1) 报告值/真值 的逐格比值 R（未归一化 / 归一化）
        (2) log R 对 log(A/Ā) 的斜率与 R²  ← "偏差=面积比"的身份检验
        (3) 按报告值选出的 top-k 格，其真实密度在全体中的平均分位数
            （完美选择=100%，随机=50%）
输出：results/e1_summary.json
"""
import json
import os
import numpy as np
from exp_lib import (VMFField, make_grid, aggregate, cell_truth,
                     report_ratio, geom_mean, R_EARTH)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

# 8 个现实中真实存在的"渔场"，经纬度是真实位置
BLOBS = [
    (3, 55, 600, 1.0),      # 北海
    (40, 72, 500, 0.7),     # 巴伦支海
    (150, 42, 700, 1.2),    # 西北太平洋
    (-80, -10, 400, 0.9),   # 秘鲁上升流
    (2, 0, 450, 0.6),       # 几内亚湾
    (62, 15, 550, 0.5),     # 阿拉伯海
    (-20, -45, 800, 0.4),   # 南大西洋
    (-175, 57, 600, 0.6),   # 白令海
]

GRIDS = [
    ("H3", 4), ("H3", 5), ("H3", 6),
    ("A5", 8), ("A5", 9), ("A5", 10),
    ("LatLon", 2.0), ("LatLon", 1.0), ("LatLon", 0.1),
]

N = 6_000_000


def pct_rank(x):
    """经验分位数（0-100）。"""
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(len(x)) / max(1, len(x) - 1) * 100.0
    return r


def main():
    rng = np.random.default_rng(20261113)
    field = VMFField(BLOBS, background_share=0.02)
    print(f"field total_mass={field.total_mass:.3e}  sampling N={N:,}")
    lon, lat = field.sample(N, rng)
    print("sampled", len(lon))

    rows = []
    for fam, res in GRIDS:
        g = make_grid(fam, res)
        codes = g.bin(lon, lat)
        uniq, cnt, _ = aggregate(codes)
        A = g.area(uniq)
        t = cell_truth(field, g, uniq, N)

        m = t >= 10.0          # 只分析"期望至少 10 个点"的格子，抑制泊松噪声
        cnt_m, A_m, t_m = cnt[m], A[m], t[m]
        cover = t_m.sum() / N

        Rraw, Rnrm = report_ratio(cnt_m, A_m, t_m, g.nominal_area)
        x = np.log(A_m / g.nominal_area)

        # 身份检验：log R = log(A/Ā)
        # 用 R 而非 log R 时也存在恒等关系；这里对 log R 回归
        lr_raw = np.log(Rraw)
        slope = float(np.polyfit(x, lr_raw, 1)[0])
        r2 = float(np.corrcoef(x, lr_raw)[0, 1] ** 2)

        # top-k 选择
        k = max(200, int(0.01 * m.sum()))
        true_d = t_m / A_m
        pr = pct_rank(true_d)
        sel_raw = np.argsort(-(cnt_m / g.nominal_area))[:k]
        sel_nrm = np.argsort(-(cnt_m / A_m))[:k]
        sel_true = np.argsort(-true_d)[:k]

        rows.append(dict(
            grid=g.name, family=fam, res=res,
            n_cells=int(len(uniq)), n_mask=int(m.sum()),
            nominal_area_km2=float(g.nominal_area),
            area_min=float(A.min()), area_max=float(A.max()),
            area_max_over_min=float(A.max() / A.min()),
            area_cv=float(A.std() / A.mean()),
            coverage=float(cover),
            geomR_raw=geom_mean(Rraw), geomR_nrm=geom_mean(Rnrm),
            medianR_raw=float(np.median(Rraw)), medianR_nrm=float(np.median(Rnrm)),
            slope_logA=slope, r2_logA=r2,
            sd_logR_raw=float(lr_raw.std()),
            sd_logR_nrm=float(np.log(Rnrm).std()),
            pct_selected_raw=float(pr[sel_raw].mean()),
            pct_selected_nrm=float(pr[sel_nrm].mean()),
            pct_selected_oracle=float(pr[sel_true].mean()),
            topk_jaccard_raw=float(len(set(sel_raw) & set(sel_true)) / len(set(sel_raw) | set(sel_true))),
            topk_jaccard_nrm=float(len(set(sel_nrm) & set(sel_true)) / len(set(sel_nrm) | set(sel_true))),
            selmean_area_ratio_raw=float(A_m[sel_raw].mean() / A_m.mean()),
            selmean_area_ratio_nrm=float(A_m[sel_nrm].mean() / A_m.mean()),
        ))
        r = rows[-1]
        print(f"{g.name:14s} cells={r['n_cells']:>9,} A={r['nominal_area_km2']:>10.2f} "
              f"max/min={r['area_max_over_min']:6.2f}  "
              f"slope={slope:6.3f} R2={r2:5.3f}  "
              f"pct_raw={r['pct_selected_raw']:5.1f} pct_nrm={r['pct_selected_nrm']:5.1f} "
              f"cov={cover:5.3f}")

    with open(os.path.join(OUT, "e1_summary.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("saved results/e1_summary.json")


if __name__ == "__main__":
    main()
