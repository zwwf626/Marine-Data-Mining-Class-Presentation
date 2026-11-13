"""
exp_lib.py —— 网格 / 场 / 采样 / 度量 的公共工具

设计要点
---------
1. 三种网格族统一接口：bin() 把 (lon,lat) 映射到整数 code，area() 给出真实球面面积。
   - H3   : 真实面积逐格计算（h3.cell_area）
   - A5   : 真实面积用球面多边形余量公式逐格计算（不是用标称值！）
   - LatLon: 球面矩形精确面积 R^2 * dlon * (sin(phi2)-sin(phi1))
2. 真值场用 von Mises-Fisher 核叠加，概率密度精确可解析积分，采样也是精确的
   （不是"小区域近似"），所以"真值"不依赖于任何参考网格。
3. 每个格子的真值质量用数值求积（经纬格用 4x4 Gauss-Legendre，六边形/五边形用
   切平面圆盘点阵），误差远小于我们要观测的 1%-100% 量级效应。
"""

import numpy as np
import h3
import a5

R_EARTH = 6371.0088  # km


# --------------------------------------------------------------------------
# 球面几何
# --------------------------------------------------------------------------
def lonlat_to_xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo),
                     np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


def xyz_to_lonlat(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lon, lat


def random_rotation(rng):
    """均匀随机的 SO(3) 旋转矩阵（QR 法 + 符号修正）。"""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def spherical_polygon_area(lonlat):
    """球面多边形面积（km^2），lonlat 为 [(lon,lat),...]，要求顶点顺序环绕。"""
    n = len(lonlat)
    s = 0.0
    for i in range(n):
        lo1, la1 = lonlat[i]
        lo2, la2 = lonlat[(i + 1) % n]
        s += np.radians(lo2 - lo1) * (2 + np.sin(np.radians(la1)) + np.sin(np.radians(la2)))
    return abs(s) * R_EARTH * R_EARTH / 2.0


# --------------------------------------------------------------------------
# 网格族
# --------------------------------------------------------------------------
class _Base:
    family = "base"

    def __init__(self, res):
        self.res = res
        self.name = f"{self.family} r{res}"
        self._area_cache = {}

    # 统一：返回 uint64 code 数组
    def bin(self, lon, lat):
        raise NotImplementedError

    # 统一：返回 code 对应的真实面积（km^2）
    def area(self, codes):
        raise NotImplementedError

    # 统一：返回 code 中心 (lon, lat)
    def center(self, codes):
        raise NotImplementedError

    # 统一：标称面积（km^2）—— 用于"假设等面积"这个错误做法
    @property
    def nominal_area(self):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name} A={self.nominal_area:.3f}km2>"


class H3Grid(_Base):
    family = "H3"

    def bin(self, lon, lat):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        return np.array([int(h3.latlng_to_cell(float(a), float(b), self.res), 16)
                         for a, b in zip(lat, lon)], dtype=np.uint64)

    def area(self, codes):
        out = np.empty(len(codes), dtype=float)
        for i, c in enumerate(codes):
            k = int(c)
            v = self._area_cache.get(k)
            if v is None:
                v = h3.cell_area(h3.int_to_str(k), unit="km^2")
                self._area_cache[k] = v
            out[i] = v
        return out

    def center(self, codes):
        la = np.empty(len(codes)); lo = np.empty(len(codes))
        for i, c in enumerate(codes):
            y, x = h3.cell_to_latlng(h3.int_to_str(int(c)))
            la[i] = y; lo[i] = x
        return lo, la

    @property
    def nominal_area(self):
        return h3.average_hexagon_area(self.res, unit="km^2")


class A5Grid(_Base):
    family = "A5"

    def bin(self, lon, lat):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        return np.array([a5.lonlat_to_cell((float(b), float(a)), self.res)
                         for a, b in zip(lat, lon)], dtype=np.uint64)

    def area(self, codes):
        out = np.empty(len(codes), dtype=float)
        for i, c in enumerate(codes):
            k = int(c)
            v = self._area_cache.get(k)
            if v is None:
                v = spherical_polygon_area(a5.cell_to_boundary(k))
                self._area_cache[k] = v
            out[i] = v
        return out

    def center(self, codes):
        la = np.empty(len(codes)); lo = np.empty(len(codes))
        for i, c in enumerate(codes):
            x, y = a5.cell_to_lonlat(int(c))
            lo[i] = x; la[i] = y
        return lo, la

    @property
    def nominal_area(self):
        return a5.cell_area(self.res) / 1e6


class LatLonGrid(_Base):
    family = "LatLon"

    def __init__(self, res_deg):
        self.d = float(res_deg)
        self.res = res_deg
        self.name = f"LatLon {res_deg:g}deg"
        self.ni = int(round(360.0 / self.d))
        self.nj = int(round(180.0 / self.d))
        self._area_cache = {}

    def bin(self, lon, lat):
        lon = np.asarray(lon, dtype=float); lat = np.asarray(lat, dtype=float)
        i = np.clip(((lon + 180.0) / self.d).astype(np.int64), 0, self.ni - 1)
        j = np.clip(((lat + 90.0) / self.d).astype(np.int64), 0, self.nj - 1)
        return (i * self.nj + j).astype(np.uint64)

    def _dims(self, codes):
        codes = np.asarray(codes, dtype=np.int64)
        return codes // self.nj, codes % self.nj

    def area(self, codes):
        i, j = self._dims(codes)
        p1 = np.radians(-90.0 + j * self.d)
        p2 = np.radians(-90.0 + (j + 1) * self.d)
        return R_EARTH ** 2 * np.radians(self.d) * (np.sin(p2) - np.sin(p1))

    def center(self, codes):
        i, j = self._dims(codes)
        return (-180.0 + (i + 0.5) * self.d, -90.0 + (j + 0.5) * self.d)

    @property
    def nominal_area(self):
        return 4 * np.pi * R_EARTH ** 2 / (self.ni * self.nj)


def make_grid(family, res):
    if family == "H3":
        return H3Grid(res)
    if family == "A5":
        return A5Grid(res)
    if family == "LatLon":
        return LatLonGrid(res)
    raise ValueError(family)


# --------------------------------------------------------------------------
# 真值场：vMF 核叠加
# --------------------------------------------------------------------------
class VMFField:
    """
    lambda(x) = bg + sum_k amp_k * exp(kappa_k * (u_k . x - 1))

    exp(kappa (u.x - 1)) 在球面上的积分 = 2*pi*(1-exp(-2 kappa))/kappa  [立体角]
    因此核 k 的总质量(km^2) = R^2 * 2*pi*(1-exp(-2k))/k * amp
    当 k 很大时 ≈ 2*pi*sigma^2*amp，与平面高斯一致。
    """

    def __init__(self, blobs, background_share=0.02):
        # blobs: list of (lon, lat, sigma_km, amp)
        self.u = np.array([lonlat_to_xyz(b[0], b[1]) for b in blobs])
        self.sigma = np.array([b[2] for b in blobs], dtype=float)
        self.kappa = (R_EARTH / self.sigma) ** 2
        self.amp = np.array([b[3] for b in blobs], dtype=float)
        self.sigma_min = float(self.sigma.min())

        kern_mass = R_EARTH ** 2 * 2 * np.pi * (1 - np.exp(-2 * self.kappa)) / self.kappa
        blob_mass = float((kern_mass * self.amp).sum())
        # 让背景质量占 background_share
        self.bg = background_share / (1 - background_share) * blob_mass / (4 * np.pi * R_EARTH ** 2)
        self._kern_mass = kern_mass

    def __call__(self, lon, lat):
        x = lonlat_to_xyz(lon, lat)
        out = np.full(x.shape[:-1], self.bg, dtype=float)
        cosang = x @ self.u.T  # (...,K)
        for k in range(len(self.u)):
            out = out + self.amp[k] * np.exp(self.kappa[k] * (cosang[..., k] - 1))
        return out

    @property
    def total_mass(self):
        return self.bg * 4 * np.pi * R_EARTH ** 2 + float((self._kern_mass * self.amp).sum())

    def sample(self, n, rng):
        """精确采样：混合分布（背景均匀 + 各核 vMF）。返回 (lon, lat)。"""
        w = np.concatenate([[self.bg * 4 * np.pi * R_EARTH ** 2],
                            self._kern_mass * self.amp])
        w = w / w.sum()
        counts = rng.multinomial(n, w)
        lons, lats = [], []
        # 背景
        nb = counts[0]
        if nb:
            z = rng.uniform(-1, 1, nb)
            ph = rng.uniform(0, 2 * np.pi, nb)
            r = np.sqrt(np.maximum(0, 1 - z * z))
            lons.append(np.degrees(np.arctan2(r * np.sin(ph), r * np.cos(ph))))
            lats.append(np.degrees(np.arcsin(z)))
        for k in range(len(self.u)):
            nk = counts[k + 1]
            if nk == 0:
                continue
            pts = _vmf_sample(self.u[k], self.kappa[k], nk, rng)
            lo, la = xyz_to_lonlat(pts)
            lons.append(lo); lats.append(la)
        return np.concatenate(lons), np.concatenate(lats)


def _vmf_sample(u, kappa, n, rng):
    """3D von Mises-Fisher 精确采样（Wood 1994）。"""
    eta = rng.random(n)
    w = 1.0 + np.log(eta + (1 - eta) * np.exp(-2 * kappa)) / kappa
    w = np.clip(w, -1.0, 1.0)
    phi = rng.random(n) * 2 * np.pi
    s = np.sqrt(np.maximum(0.0, 1 - w * w))
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return (w[:, None] * u[None, :]
            + s[:, None] * (np.cos(phi)[:, None] * e1[None, :]
                            + np.sin(phi)[:, None] * e2[None, :]))


# --------------------------------------------------------------------------
# 单格真值质量：数值求积
# --------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)


def _disk_pattern(n_ring=2, m=6):
    """单位圆内的点阵（含圆心），等权。"""
    pts = [(0.0, 0.0)]
    for ir in range(1, n_ring + 1):
        rr = ir / (n_ring + 1) * 0.95
        for j in range(m):
            th = 2 * np.pi * j / m + (0.5 * np.pi / m if ir % 2 else 0)
            pts.append((rr * np.cos(th), rr * np.sin(th)))
    return np.array(pts)


def cell_truth(field, grid, codes, n_total):
    """
    返回每个格子的真值"点数"（= ∫λdA / ∫λdA_total * n_total）。
    经纬格用 4x4 Gauss-Legendre（几乎精确）；多边形格用切平面圆盘点阵。
    """
    scale = n_total / field.total_mass
    A = grid.area(codes)
    lon_c, lat_c = grid.center(codes)

    if isinstance(grid, LatLonGrid):
        d = grid.d
        i, j = grid._dims(codes)
        # 在 (lon, sin(phi)) 上做张量积 Gauss-Legendre
        lo0 = -180.0 + i * d
        s1 = np.sin(np.radians(-90.0 + j * d))
        s2 = np.sin(np.radians(-90.0 + (j + 1) * d))
        u = 0.5 * (1 + _GL_X); w = 0.5 * _GL_W
        LON = lo0[:, None] + d * u[None, :]            # (nc,4)
        SIN = s1[:, None] + (s2 - s1)[:, None] * u[None, :]
        LAT = np.degrees(np.arcsin(SIN))
        mass = np.zeros(len(codes))
        for a in range(4):
            for b in range(4):
                mass += w[a] * w[b] * field(LON[:, a], LAT[:, b])
        return mass * A * scale

    # 多边形：同心点阵
    r_eq = np.sqrt(A / np.pi)  # 等面积圆半径
    ratio = float(np.max(r_eq / field.sigma_min))
    if ratio < 0.05:
        pat = np.array([[0.0, 0.0]])
    elif ratio < 0.15:
        pat = _disk_pattern(1, 6)
    else:
        pat = _disk_pattern(3, 8)

    coslat = np.cos(np.radians(lat_c))
    mass = np.zeros(len(codes))
    for (dx, dy) in pat:
        lo = lon_c + np.degrees(dx * r_eq / (R_EARTH * coslat))
        la = lat_c + np.degrees(dy * r_eq / R_EARTH)
        mass += field(lo, la)
    return mass / len(pat) * A * scale


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------
def aggregate(codes, n_cells_hint=None):
    """把逐点 code 压成 (uniq_codes, counts)。"""
    uniq, inv, cnt = np.unique(codes, return_inverse=True, return_counts=True)
    return uniq, cnt, inv


def per_cell_density(grid, codes, n_total):
    """返回 (uniq, count, area, 真实密度, 未归一化报告值)。"""
    uniq, cnt, inv = aggregate(codes)
    A = grid.area(uniq)
    return uniq, cnt, A


# --------------------------------------------------------------------------
# 度量
# --------------------------------------------------------------------------
def safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))


def report_ratio(counts, area, truth_mass, nominal_area):
    """
    返回"报告值 / 真值"的逐格比值 R（无量纲）。

      - 未归一化（把每格当成等面积 Ā）:  R = (n/Ā) / (t/A) ≈ A/Ā
      - 归一化（用真实面积）:            R = (n/A) / (t/A) ≈ 1

    R>1 表示该格密度被高估。这个量对泊松噪声稳健（大数定律），
    而且可以直接读出"高估了多少倍"。
    """
    true_d = truth_mass / area
    rep_raw = counts / nominal_area
    rep_nrm = counts / area
    return rep_raw / true_d, rep_nrm / true_d


def geom_mean(x):
    return float(np.exp(np.mean(np.log(np.maximum(x, 1e-12)))))


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def topk_jaccard(score_a, score_b, k):
    ia = set(np.argsort(-score_a)[:k])
    ib = set(np.argsort(-score_b)[:k])
    return len(ia & ib) / len(ia | ib)
