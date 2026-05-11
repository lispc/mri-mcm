"""
Watson 分布 + Stick 信号模型。

对应 MCM 中的细胞内圆柱隔室（小胶质细胞突起）。
引用：NODDI (Zhang et al. 2012), NeuroImage, Eq. 2

单位约定（与所有隔室模块统一）：
    b           : s/mm²
    d_parallel  : mm²/s
"""

import numpy as np
from scipy.special import hyp1f1, erf
from utils import uniform_sphere_grid

# ---------------------------------------------------------------------------
# Cached sphere grids
# ---------------------------------------------------------------------------
_GRID_CACHE = {}


def _get_sphere_grid(n_theta: int = 60, n_phi: int = 60):
    """Return cached (directions, weights) for given resolution."""
    key = (n_theta, n_phi)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = uniform_sphere_grid(n_theta, n_phi)
    return _GRID_CACHE[key]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def watson_normalization(k: float) -> float:
    """Watson 分布的归一化常数 M(1/2, 3/2, k)。"""
    return hyp1f1(0.5, 1.5, k)


def watson_pdf(cos_theta: np.ndarray, k: float) -> np.ndarray:
    """Watson 分布的概率密度值（已归一化）。"""
    norm = watson_normalization(k)
    return np.exp(k * cos_theta**2) / (4 * np.pi * norm)


def stick_signal(cos_theta: np.ndarray, b: float, d_parallel: float) -> np.ndarray:
    """单个 stick 的扩散信号：S = exp(-b * d_parallel * cos²(theta))"""
    return np.exp(-b * d_parallel * cos_theta**2)


def watson_stick_isotropic_exact(b: float, d_parallel: float) -> float:
    """k=0 时 Watson-Stick 信号的解析解。"""
    arg = np.sqrt(b * d_parallel)
    if arg < 1e-10:
        return 1.0 - b * d_parallel / 3.0
    return 0.5 * np.sqrt(np.pi) / arg * erf(arg)


def watson_stick_signal(
    q: np.ndarray,
    mu: np.ndarray,
    b: float,
    k: float,
    d_parallel: float = 1.0e-3,
    n_theta: int = 60,
    n_phi: int = 60,
) -> float:
    """
    计算 Watson-分散 stick 隔室的扩散加权信号。

    S_IC(q, b, k) = ∫_{S²} f_Watson(n; mu, k) * exp(-b * d_parallel * (q·n)²) dn

    Parameters
    ----------
    q : (3,) array
        梯度方向向量（会被归一化）
    mu : (3,) array
        平均取向向量（会被归一化）
    b : float
        b 值 (s/mm²)
    k : float
        Watson 集中参数
    d_parallel : float
        沿 stick 方向的扩散率 (mm²/s)，默认 1.0×10⁻³
    n_theta, n_phi : int
        积分网格分辨率

    Returns
    -------
    float
        信号值 S_IC
    """
    q = np.asarray(q, dtype=float)
    q = q / (np.linalg.norm(q) + 1e-15)
    mu = np.asarray(mu, dtype=float)
    mu = mu / (np.linalg.norm(mu) + 1e-15)

    if b == 0.0:
        return 1.0

    directions, weights = _get_sphere_grid(n_theta, n_phi)
    cos_theta_mu_n = directions @ mu
    cos_theta_q_n = directions @ q
    watson_weights = np.exp(k * cos_theta_mu_n**2)
    signals = np.exp(-b * d_parallel * cos_theta_q_n**2)
    norm = watson_normalization(k)
    return float(np.sum(watson_weights * signals * weights) / norm)


def watson_stick_signal_batch(
    q_directions: np.ndarray,
    mu: np.ndarray,
    b: float,
    k: float,
    d_parallel: float = 1.0e-3,
    n_theta: int = 60,
    n_phi: int = 60,
) -> np.ndarray:
    """
    批量计算多个梯度方向上的 Watson-Stick 信号（矩阵运算优化版）。

    Parameters
    ----------
    q_directions : (N, 3) array
        N 个梯度方向向量（会被归一化）
    mu : (3,) array
        平均取向向量（会被归一化）
    b : float
        b 值
    k : float
        Watson 集中参数
    d_parallel : float
        扩散率 (mm²/s)
    n_theta, n_phi : int
        积分网格分辨率

    Returns
    -------
    (N,) array
        各方向的信号值
    """
    q_directions = np.asarray(q_directions, dtype=float)
    norms = np.linalg.norm(q_directions, axis=1, keepdims=True)
    q_norm = q_directions / (norms + 1e-15)
    mu = np.asarray(mu, dtype=float)
    mu = mu / (np.linalg.norm(mu) + 1e-15)

    if b == 0.0:
        return np.ones(len(q_norm), dtype=float)

    directions, weights = _get_sphere_grid(n_theta, n_phi)
    # (M, 3) @ (3, N) -> (M, N)
    cos_theta_q_n = directions @ q_norm.T
    cos_theta_mu_n = directions @ mu
    watson_weights = np.exp(k * cos_theta_mu_n**2)[:, None]
    signals = np.exp(-b * d_parallel * cos_theta_q_n**2)
    norm = watson_normalization(k)
    # sum over M samples, divide by norm
    return np.sum(watson_weights * signals * weights[:, None], axis=0) / norm
