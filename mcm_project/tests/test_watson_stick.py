"""
Watson-Stick 模块的多角度验证测试。

验证策略：
1. Watson 分布归一化（球面积分 = 1）
2. k=0 时与解析解对比
3. k→∞ 时退化为单方向 stick（放宽精度要求）
4. 信号范围 [0, 1]
5. 参数单调性（物理直觉验证）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from watson_stick import (
    watson_normalization,
    watson_pdf,
    stick_signal,
    watson_stick_signal,
    watson_stick_isotropic_exact,
    watson_stick_signal_batch,
)
from utils import uniform_sphere_grid


class TestWatsonNormalization:
    """验证 Watson 分布的数学性质。"""
    
    def test_normalization_positive(self):
        """归一化常数必须为正。"""
        for k in [0, 1, 5, 10, 30]:
            assert watson_normalization(k) > 0
    
    def test_pdf_integrates_to_one(self):
        """Watson PDF 在球面上积分 = 1。"""
        directions, weights = uniform_sphere_grid(120, 120)
        # weights 已归一化
        assert np.isclose(np.sum(weights), 1.0, atol=1e-10), \
            f"Weights don't sum to 1: {np.sum(weights)}"
        
        for k in [0, 1, 5, 10, 30]:
            cos_vals = directions[:, 2]  # 假设 mu = z-axis
            pdf_vals = watson_pdf(cos_vals, k)
            integral = np.sum(pdf_vals * weights * 4 * np.pi)
            assert np.isclose(integral, 1.0, atol=1e-2), \
                f"k={k}: PDF integral = {integral}, expected 1.0"

    def test_k0_is_uniform(self):
        """k=0 时 Watson 分布退化为均匀分布。"""
        directions, weights = uniform_sphere_grid(60, 60)
        cos_vals = directions[:, 2]
        pdf_vals = watson_pdf(cos_vals, 0.0)
        expected = 1.0 / (4 * np.pi)
        assert np.allclose(pdf_vals, expected, atol=1e-10), \
            "k=0 PDF not uniform"


class TestStickSignal:
    """验证 Stick 信号的基本性质。"""
    
    def test_parallel_direction_no_attenuation_when_dzero(self):
        """q 与 n 平行且 d_parallel=0 时，信号应为 1。"""
        cos_theta = 1.0
        b = 2000.0
        d_parallel = 0.0
        s = stick_signal(np.array([cos_theta]), b, d_parallel)
        assert np.isclose(s[0], 1.0)
    
    def test_perpendicular_direction_no_attenuation(self):
        """q 与 n 垂直时，信号应为 1（stick 垂直方向完全受限）。"""
        cos_theta = 0.0
        b = 4000.0
        d_parallel = 1.0e-3
        s = stick_signal(np.array([cos_theta]), b, d_parallel)
        assert np.isclose(s[0], 1.0)
    
    def test_parallel_direction_attenuation(self):
        """q 与 n 平行时，信号应有最大衰减。"""
        cos_theta = 1.0
        b = 2000.0
        d_parallel = 1.0e-3
        s = stick_signal(np.array([cos_theta]), b, d_parallel)
        expected = np.exp(-b * d_parallel)
        assert np.isclose(s[0], expected)
    
    def test_signal_range(self):
        """Stick 信号必须在 [0, 1] 范围内。"""
        cos_thetas = np.linspace(-1, 1, 100)
        b = 4000.0
        d_parallel = 1.0e-3
        signals = stick_signal(cos_thetas, b, d_parallel)
        assert np.all(signals >= 0) and np.all(signals <= 1)


class TestWatsonStickIntegration:
    """验证 Watson-Stick 信号积分的正确性。"""
    
    def test_k0_vs_analytical(self):
        """
        k=0 时与解析解对比。
        
        解析解：S = 0.5 * sqrt(pi/(b*d)) * erf(sqrt(b*d))
        """
        q = np.array([1.0, 0.0, 0.0])
        mu = np.array([0.0, 0.0, 1.0])
        d_parallel = 1.0e-3
        
        for b in [500, 1000, 2000, 4000]:
            numerical = watson_stick_signal(q, mu, b, 0.0, d_parallel, n_theta=120, n_phi=120)
            analytical = watson_stick_isotropic_exact(b, d_parallel)
            rel_err = abs(numerical - analytical) / analytical
            assert rel_err < 0.03, \
                f"b={b}: numerical={numerical:.6f}, analytical={analytical:.6f}, rel_err={rel_err:.4f}"

    def test_k_large_degeneration(self):
        """
        k→∞ 时退化为单方向 stick。
        
        当 k 很大时，Watson 分布集中在 mu 附近，信号应接近 exp(-b*d*(q·mu)²)。
        注意：大 k 时数值积分精度有限，放宽 tolerance。
        """
        q = np.array([1.0, 0.0, 0.0])
        mu = np.array([1.0, 0.0, 0.0])
        b = 2000.0
        d_parallel = 1.0e-3
        
        expected = np.exp(-b * d_parallel * (q @ mu)**2)
        
        # 使用高密度网格
        for k in [50, 100]:
            numerical = watson_stick_signal(q, mu, b, k, d_parallel, n_theta=300, n_phi=300)
            rel_err = abs(numerical - expected) / expected
            # 大 k 时放宽精度要求（数值积分极限）
            assert rel_err < 0.15, \
                f"k={k}: numerical={numerical:.6f}, expected={expected:.6f}, rel_err={rel_err:.4f}"

    def test_signal_always_in_range(self):
        """Watson-Stick 信号必须在 [0, 1] 范围内。"""
        mu = np.array([0.0, 0.0, 1.0])
        d_parallel = 1.0e-3
        
        np.random.seed(42)
        for _ in range(20):
            q = np.random.randn(3)
            q /= np.linalg.norm(q)
            for b in [1000, 2000, 4000]:
                for k in [0, 1, 5, 10]:
                    s = watson_stick_signal(q, mu, b, k, d_parallel, n_theta=60, n_phi=60)
                    assert 0 <= s <= 1, f"Signal out of range: s={s} for b={b}, k={k}"

    def test_k_effect_when_q_parallel_mu(self):
        """
        当 q // mu 时，k 增加应使信号减小。
        
        物理直觉：k 越大 → stick 越集中在 mu 方向 → 更多 stick 与 q 平行 →
        沿 q 方向的扩散衰减越大 → 信号越小。
        """
        q = np.array([0.0, 0.0, 1.0])
        mu = np.array([0.0, 0.0, 1.0])
        b = 2000.0
        d_parallel = 1.0e-3
        
        signals = []
        for k in [0, 1, 5, 10, 20]:
            s = watson_stick_signal(q, mu, b, k, d_parallel, n_theta=120, n_phi=120)
            signals.append(s)
        
        # k 增加时信号应单调不增
        for i in range(len(signals) - 1):
            assert signals[i+1] <= signals[i] + 0.01, \
                f"Signal not decreasing with k (q//mu): {signals}"
    
    def test_k_effect_when_q_perpendicular_mu(self):
        """
        当 q ⟂ mu 时，k 增加应使信号增加。
        
        物理直觉：k 越大 → stick 越集中在 mu 方向 → 更多 stick 垂直于 q →
        沿 q 方向的扩散衰减越小 → 信号越大。
        """
        q = np.array([1.0, 0.0, 0.0])
        mu = np.array([0.0, 0.0, 1.0])
        b = 2000.0
        d_parallel = 1.0e-3
        
        signals = []
        for k in [0, 1, 5, 10, 20]:
            s = watson_stick_signal(q, mu, b, k, d_parallel, n_theta=120, n_phi=120)
            signals.append(s)
        
        # k 增加时信号应单调不减
        for i in range(len(signals) - 1):
            assert signals[i+1] >= signals[i] - 0.01, \
                f"Signal not increasing with k (q⟂mu): {signals}"

    def test_batch_consistency(self):
        """批量计算与逐个计算结果一致。"""
        np.random.seed(42)
        q_dirs = np.random.randn(10, 3)
        q_dirs /= np.linalg.norm(q_dirs, axis=1, keepdims=True)
        mu = np.array([0.0, 0.0, 1.0])
        
        batch_results = watson_stick_signal_batch(q_dirs, mu, 2000.0, 5.0, n_theta=60, n_phi=60)
        
        individual_results = []
        for q in q_dirs:
            s = watson_stick_signal(q, mu, 2000.0, 5.0, n_theta=60, n_phi=60)
            individual_results.append(s)
        
        assert np.allclose(batch_results, individual_results, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
