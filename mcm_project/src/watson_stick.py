"""
Watson 分布 + Stick 信号模型。

对应 MCM 中的细胞内圆柱隔室（小胶质细胞突起）。
引用：NODDI (Zhang et al. 2012), NeuroImage, Eq. 2
"""

import numpy as np
from scipy.special import hyp1f1, erf
from utils import uniform_sphere_grid


def watson_normalization(k: float) -> float:
    """
    Watson 分布的归一化常数 M(1/2, 3/2, k)。
    
    f(n) = M(1/2, 3/2, k)^{-1} * exp(k * (mu·n)^2)
    
    Parameters
    ----------
    k : float
        Watson 集中参数，k >= 0
        
    Returns
    -------
    float
        归一化常数 M(1/2, 3/2, k)
    """
    return hyp1f1(0.5, 1.5, k)


def watson_pdf(cos_theta: np.ndarray, k: float) -> np.ndarray:
    """
    Watson 分布的概率密度值（已归一化）。
    
    Parameters
    ----------
    cos_theta : np.ndarray
        mu·n 的点积值（余弦），范围 [-1, 1]
    k : float
        Watson 集中参数
        
    Returns
    -------
    np.ndarray
        概率密度值，在球面上积分 = 1
    """
    norm = watson_normalization(k)
    return np.exp(k * cos_theta**2) / (4 * np.pi * norm)


def stick_signal(cos_theta: np.ndarray, b: float, d_parallel: float) -> np.ndarray:
    """
    单个 stick（零半径圆柱）的扩散信号。
    
    S_stick = exp(-b * d_parallel * cos²(theta))
    
    其中 theta 是梯度方向 q 与 stick 方向 n 的夹角。
    
    Parameters
    ----------
    cos_theta : np.ndarray
        q·n 的点积值
    b : float
        b 值 (s/mm²)
    d_parallel : float
        沿 stick 方向的扩散率 (mm²/s)
        
    Returns
    -------
    np.ndarray
        信号值
    """
    return np.exp(-b * d_parallel * cos_theta**2)


def watson_stick_isotropic_exact(b: float, d_parallel: float) -> float:
    """
    k=0（完全各向同性）时 Watson-Stick 信号的解析解。
    
    S = (1/2) * sqrt(pi / (b*d_parallel)) * erf(sqrt(b*d_parallel))
    
    用于数值验证。
    """
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
    q : np.ndarray, shape (3,)
        单位梯度方向向量
    mu : np.ndarray, shape (3,)
        平均取向单位向量（圆柱隔室主方向）
    b : float
        b 值 (s/mm²)
    k : float
        Watson 集中参数
    d_parallel : float
        沿 stick 方向的扩散率 (mm²/s)，默认 1.0e-3
    n_theta, n_phi : int
        球面积分的网格分辨率
        
    Returns
    -------
    float
        信号值 S_IC
    """
    # 防御性归一化
    q = np.asarray(q, dtype=float)
    q = q / (np.linalg.norm(q) + 1e-15)
    mu = np.asarray(mu, dtype=float)
    mu = mu / (np.linalg.norm(mu) + 1e-15)

    if b == 0.0:
        return 1.0

    # 获取球面采样点和权重
    directions, weights = uniform_sphere_grid(n_theta, n_phi)
    
    # 计算 cos(theta) = mu · n
    cos_theta_mu_n = directions @ mu  # (N_samples,)
    
    # 计算 cos(theta) = q · n
    cos_theta_q_n = directions @ q    # (N_samples,)
    
    # Watson 分布权重
    watson_weights = np.exp(k * cos_theta_mu_n**2)
    
    # Stick 信号
    signals = np.exp(-b * d_parallel * cos_theta_q_n**2)
    
    # 加权积分（权重已包含 sin(theta) dtheta dphi / 4pi）
    # 但这里 Watson 分布未归一化，需要额外除以归一化常数
    norm = watson_normalization(k)
    
    # 积分 = sum(f(n) * S(n) * dOmega)
    # f(n) = exp(k*(mu·n)²) / (4*pi * norm)
    # dOmega = sin(theta) dtheta dphi
    # 所以积分 = sum(exp(k*(mu·n)²) * S(n) * sin(theta) dtheta dphi) / (4*pi * norm)
    # weights = sin(theta) dtheta dphi / (4*pi)
    # 所以积分 = sum(watson_weights * signals * weights) / norm
    
    result = np.sum(watson_weights * signals * weights) / norm
    
    return float(result)


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
    批量计算多个梯度方向上的 Watson-Stick 信号。
    
    Parameters
    ----------
    q_directions : np.ndarray, shape (N, 3)
        N 个单位梯度方向向量
    mu : np.ndarray, shape (3,)
        平均取向单位向量
    b : float
        b 值
    k : float
        Watson 集中参数
    d_parallel : float
        扩散率
    n_theta, n_phi : int
        积分网格分辨率
        
    Returns
    -------
    np.ndarray, shape (N,)
        各方向的信号值
    """
    # 预计算球面采样（避免重复计算）
    directions, weights = uniform_sphere_grid(n_theta, n_phi)
    cos_theta_mu_n = directions @ mu
    watson_weights = np.exp(k * cos_theta_mu_n**2)
    norm = watson_normalization(k)
    
    results = []
    for q in q_directions:
        cos_theta_q_n = directions @ q
        signals = np.exp(-b * d_parallel * cos_theta_q_n**2)
        result = np.sum(watson_weights * signals * weights) / norm
        results.append(result)
    
    return np.array(results)
