"""
通用工具函数：球面采样、数值积分辅助等。
"""

import numpy as np


def fibonacci_sphere_sampling(n_samples: int) -> np.ndarray:
    """
    斐波那契球面均匀采样，返回 N×3 的单位方向向量。
    
    参考：https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """
    indices = np.arange(0, n_samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_samples)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    directions = np.stack([x, y, z], axis=-1)
    return directions


def uniform_sphere_grid(n_theta: int = 60, n_phi: int = 60) -> tuple:
    """
    经纬度网格采样，返回 (directions, weights)。
    
    directions: (n_theta*n_phi, 3) 单位向量
    weights: (n_theta*n_phi,) 积分权重（已归一化，sum(weights)=1）
    """
    # 使用中点法则避免端点问题
    theta = np.linspace(0, np.pi, n_theta + 2)[1:-1]  # 去掉端点
    phi = np.linspace(0, 2 * np.pi, n_phi + 1)[:-1]   # 周期边界，去掉最后一个点
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    directions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    
    # 中点法则权重
    dtheta = np.pi / (n_theta + 1)
    dphi = 2 * np.pi / n_phi
    
    # sin(theta) 是球面面积元的一部分
    weights = np.sin(theta_grid).ravel() * dtheta * dphi / (4 * np.pi)
    
    # 归一化确保 sum(weights) = 1
    weights = weights / np.sum(weights)
    
    return directions, weights
