"""生成报告全部插图。"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["PingFang SC", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})
C = {"H3": "#1f77b4", "A5": "#d62728", "LatLon": "#2ca02c"}


def load_json(n):
    with open(os.path.join(RES, n)) as f:
        return json.load(f)


# ---------------------------------------------------------------- F1
def fig1():
    z = np.load(os.path.join(RES, "e5_cells.npz"))
    keys = [("H3_4", "H3 r4（六边形）"), ("A5_8", "A5 r8（等面积五边形）"),
            ("LatLon_1.0", "经纬 1°（矩形）")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for k, lab in keys:
        A = z[f"{k}__A"]
        r = A / A.mean()
        fam = k.split("_")[0]
        ax[0].hist(r, bins=np.logspace(-2.1, 2.1, 90), histtype="step", lw=1.8,
                   label=f"{lab}  max/min={A.max()/A.min():.2f}", color=C[fam])
        xs = np.sort(r)
        ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), lw=2,
                   label=f"{lab}", color=C[fam])
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("格子面积 / 平均面积"); ax[0].set_ylabel("格子数")
    ax[0].set_title("(a) 面积分布：A5 是一条竖线，经纬格横跨两个数量级")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("格子面积 / 平均面积"); ax[1].set_ylabel("累积比例")
    ax[1].set_title("(b) 累积分布")
    ax[1].scatter([1], [0.5], color="k", zorder=5, s=18, marker="+")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f1_area_distribution.png")); plt.close(fig)


# ---------------------------------------------------------------- F2
def fig2():
    z = np.load(os.path.join(RES, "e5_cells.npz"))
    keys = [("H3_4", "H3 r4"), ("A5_8", "A5 r8"), ("LatLon_1.0", "经纬 1°")]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for a, (k, lab) in zip(ax, keys):
        A = z[f"{k}__A"]; t = z[f"{k}__t"]; c = z[f"{k}__cnt"].astype(float)
        m = (t >= 100) & (c > 20)
        A, t, c = A[m], t[m], c[m]
        x = np.log(A / A.mean())
        R = (c / c.mean() * c.mean() / (t / A))  # 未归一化报告值 / 真值 = (c/Ā)/(t/A)
        R = (c / A.mean()) / (t / A)
        y = np.log(R)
        hb = a.hexbin(x, y, gridsize=55, bins="log", cmap="viridis", mincnt=1,
                      extent=[-1.2, 1.2, -1.2, 1.2])
        xx = np.linspace(-1.2, 1.2, 10)
        a.plot(xx, xx, "r--", lw=1.5, label="恒等式 y = x")
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        a.set_title(f"{lab}\n$R^2$={r2:.3f}", fontsize=10)
        a.set_xlabel(r"$\log(A_i/\bar A)$")
        a.set_xlim(-1.2, 1.2); a.set_ylim(-1.2, 1.2)
        a.legend(fontsize=8, loc="upper left")
    ax[0].set_ylabel("log（报告密度 / 真密度）")
    fig.suptitle("未归一化报告：相对误差逐格落在 y=x 上（散点宽度＝泊松噪声）；A5 因面积恒定而无此误差", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f2_bias_identity.png")); plt.close(fig)


# ---------------------------------------------------------------- F3
def fig3():
    d = np.load(os.path.join(RES, "e5_inversion.npz"))
    keys = [("H3_4", "H3 r4（面积比 ≤ 2.38）"), ("LatLon_1.0", "经纬 1°（面积比 ≤ 114）")]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for a, (k, lab) in zip(ax, keys):
        rate = d[f"{k}__rate"].T
        xe, ye = d[f"{k}__xe"], d[f"{k}__ye"]
        im = a.pcolormesh(xe, ye, rate, cmap="RdYlBu_r", vmin=0, vmax=1, shading="auto")
        g = np.linspace(-2.5, 2.5, 50)
        a.plot(g, -g, "k--", lw=2, label=r"理论边界 $\alpha = 1/\beta$")
        a.set_xlabel(r"$\log\beta=\log(\rho_i/\rho_j)$   ← 真值更密 | 真值更疏 →")
        a.set_title(lab, fontsize=10)
        a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5)
        a.legend(fontsize=8, loc="lower left")
        a.grid(False)
    ax[0].set_ylabel(r"$\log\alpha=\log(A_i/A_j)$")
    fig.colorbar(im, ax=ax, label="排名反转的实际频率", fraction=0.025)
    fig.suptitle("反转概率地图：实测反转区与理论边界 α=1/β 吻合", y=1.0)
    fig.savefig(os.path.join(FIG, "f3_inversion_map.png"), bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- F4
def fig4():
    arms = load_json("e2_arms.json")["arms"]
    sc = load_json("e2_scaling.json")
    pc = np.load(os.path.join(RES, "e2_percity.npz"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    # (a) 尺度律
    a5 = sorted([(v["mean_cell_area"], v["mean_density"], k) for k, v in arms.items()
                 if k.startswith("A5") and k.endswith("pcell")])
    ax[0].plot([p[0] for p in a5], [p[1] for p in a5], "o-", color=C["A5"], lw=2,
               label="A5（精确等面积，可当地板）")
    for k in ["H3_r8_pcell", "H3_r9_pcell"]:
        ax[0].scatter([arms[k]["mean_cell_area"]], [arms[k]["mean_density"]],
                      s=90, marker="s", color=C["H3"], zorder=5,
                      label="H3（变面积）" if k == "H3_r8_pcell" else None)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("格子面积 (km²)"); ax[0].set_ylabel("报告的热点密度（点/km²）")
    ax[0].set_title("(a) 同一套流程，只改格子大小\n「top-10km² 密度」随格子变细而系统性升高")
    ax[0].legend(fontsize=8)
    # (b) 跨城市回归
    x = np.log(pc["aH3"] / pc["aA5"]); y = np.log(pc["dH3"] / pc["dA5"])
    ax[1].scatter(x, y, s=14, alpha=0.6, color=C["H3"])
    sl, ic = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax[1].plot(xx, sl * xx + ic, "r-", lw=2,
               label=f"拟合斜率 {sl:.2f}（$R^2$={sc['regression_cross_city']['r2']:.2f}）")
    ax[1].axhline(0, color="k", lw=0.8, ls=":")
    ax[1].set_xlabel(r"$\log(A_{\rm H3}/A_{\rm A5})$　格子面积比")
    ax[1].set_ylabel(r"$\log(\theta_{\rm H3}/\theta_{\rm A5})$")
    ax[1].set_title("(b) 若「面积不等」是机制，斜率应为正；\n实测为负＝纯尺度效应（格子越小读数越高）")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f4_a5_verdict.png")); plt.close(fig)


# ---------------------------------------------------------------- F5
def fig5():
    d = load_json("e3_rotation.json")
    names = list(d["grids"].keys())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    data = [d["grids"][n]["thetas"] for n in names]
    bp = ax[0].boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for p, n in zip(bp["boxes"], names):
        p.set_facecolor(C[n.split()[0]]); p.set_alpha(0.35)
    for i, n in enumerate(names, 1):
        ax[0].scatter(np.full(len(data[i - 1]), i) + np.random.default_rng(i).normal(0, 0.05, len(data[i - 1])),
                      data[i - 1], s=14, color=C[n.split()[0]], zorder=4)
    ax[0].set_xticklabels([n.replace(" ", "\n") for n in names])
    ax[0].set_ylabel("热点密度 θ（点/km²）")
    ax[0].set_title("(a) 同一片海、同一套网格，只把网格整体旋转 16 次")
    for i, n in enumerate(names, 1):
        g = d["grids"][n]
        ax[0].annotate(f"CV={g['theta_cv']*100:.1f}%", (i, g["theta_mean"]),
                       textcoords="offset points", xytext=(0, 26),
                       ha="center", fontsize=8)
    # (b) Jaccard 稳定性
    jac = [d["grids"][n]["jaccard_mean"] for n in names]
    ax[1].bar(range(len(names)), jac, color=[C[n.split()[0]] for n in names], alpha=0.75)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels([n.replace(" ", "\n") for n in names])
    ax[1].set_ylabel("两次旋转选中同一批点的 Jaccard")
    ax[1].set_title("(b) 「最热的 5,000 km²」本身就不稳定\n两个网格一样不稳定")
    ax[1].axhline(0.5, ls=":", color="k")
    for i, v in enumerate(jac):
        ax[1].annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 4),
                       ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f5_zoning_noise.png")); plt.close(fig)


# ---------------------------------------------------------------- F6
def fig6():
    d = load_json("e4_intensive.json")
    t1 = d["T1_global_mean"]
    names = [t["grid"] for t in t1]
    un = [t["rel_bias_unweighted"] * 100 for t in t1]
    T = d["truth_global_mean"]
    wt = [(t["bias_weighted"] / T) * 100 for t in t1]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, un, w, label="格平均（未加权）", color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, wt, w, label="面积加权平均", color="#1f77b4", alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("全球平均的相对偏差 (%)")
    ax.set_title("强度量的全球平均：经纬格未加权偏差 −9.3%，等面积网格 ≈ 0")
    for i, v in enumerate(un):
        ax.annotate(f"{v:+.2f}", (i - w / 2, v), textcoords="offset points",
                    xytext=(0, -12 if v < 0 else 4), ha="center", fontsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f6_intensive_bias.png")); plt.close(fig)


# ---------------------------------------------------------------- F7
def fig7():
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x_hi = 2.15
    X, Y = np.meshgrid(np.linspace(0, x_hi, 600), np.linspace(-1.3, 1.3, 600))
    wedge = ((Y < 0) & (Y > -X))
    ax.contourf(X, Y, wedge.astype(float), levels=[-0.5, 0.5, 1.5],
                colors=["#eef7ee", "#fdecea"], alpha=1.0)
    ax.plot(np.linspace(0, 1.3, 50), -np.linspace(0, 1.3, 50), "k--", lw=1.8)
    ax.axhline(0, color="k", lw=1.0)

    ax.text(0.75, -0.42, "排名反转区\n（大格子明明更稀，却排得更前）", fontsize=10, ha="center",
            bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.text(0.95, -1.20, "排名正确\n（密度差距 > 面积比）", fontsize=10, ha="center",
            bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.text(1.55, 0.45, "排名正确\n（大格子确实更密）", fontsize=10, ha="center",
            bbox=dict(fc="white", ec="none", alpha=0.8))

    marks = [
        (1.04, "A5 任意两格", C["A5"]),
        (2.38, "H3 任意两格", C["H3"]),
        (2.00, "经纬1° 赤道/60°", "#2ca02c"),
        (3.86, "赤道/75°", "#2ca02c"),
        (11.5, "赤道/85°", "#2ca02c"),
    ]
    for a, lab, col in marks:
        x = np.log10(a)
        if x <= x_hi:
            ax.axvline(x, color=col, ls=":", lw=1.3, alpha=0.9)
            ax.annotate(lab, (x + 0.012, 1.27), rotation=90, fontsize=8, color=col,
                        ha="left", va="top")

    ax.set_xlim(0, x_hi); ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel(r"$\log_{10}\alpha$　两个区域被比较时的面积比")
    ax.set_ylabel(r"$\log_{10}\gamma$　大格子区域的密度 / 小格子区域的密度")
    ax.set_title("判决相图：什么时候「网格」才会改变结论")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f7_phase.png")); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("figures ->", os.listdir(FIG))
