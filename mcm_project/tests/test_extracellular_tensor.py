"""Tests for extracellular_tensor.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from extracellular_tensor import extracellular_tensor_signal


class TestValidation:
    def test_negative_b_raises(self):
        with pytest.raises(ValueError):
            extracellular_tensor_signal(np.array([1, 0, 0]), np.array([1, 0, 0]), -1.0, 1.0, 0.5)

    def test_negative_d_parallel_raises(self):
        with pytest.raises(ValueError):
            extracellular_tensor_signal(np.array([1, 0, 0]), np.array([1, 0, 0]), 1000.0, -1.0, 0.5)

    def test_negative_d_perp_raises(self):
        with pytest.raises(ValueError):
            extracellular_tensor_signal(np.array([1, 0, 0]), np.array([1, 0, 0]), 1000.0, 1.0, -0.5)

    def test_wrong_q_shape_raises(self):
        with pytest.raises(ValueError):
            extracellular_tensor_signal(np.array([1, 0]), np.array([1, 0, 0]), 1000.0, 1.0, 0.5)

    def test_wrong_mu_shape_raises(self):
        with pytest.raises(ValueError):
            extracellular_tensor_signal(np.array([1, 0, 0]), np.array([1, 0]), 1000.0, 1.0, 0.5)


class TestSpecialDirections:
    def test_parallel_direction(self):
        """q ∥ μ → attenuation governed by d_parallel."""
        q = np.array([1.0, 0.0, 0.0])
        mu = np.array([1.0, 0.0, 0.0])
        b, d_para, d_perp = 2000.0, 1.2e-3, 0.6e-3
        s = extracellular_tensor_signal(q, mu, b, d_para, d_perp)
        assert s == pytest.approx(np.exp(-b * d_para))

    def test_perpendicular_direction(self):
        """q ⟂ μ → attenuation governed by d_perp."""
        q = np.array([0.0, 1.0, 0.0])
        mu = np.array([1.0, 0.0, 0.0])
        b, d_para, d_perp = 2000.0, 1.2e-3, 0.6e-3
        s = extracellular_tensor_signal(q, mu, b, d_para, d_perp)
        assert s == pytest.approx(np.exp(-b * d_perp))

    def test_45_degree_direction(self):
        """q at 45° to μ → ADC = (d_parallel + d_perp) / 2."""
        q = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        mu = np.array([1.0, 0.0, 0.0])
        b, d_para, d_perp = 2000.0, 1.2e-3, 0.6e-3
        s = extracellular_tensor_signal(q, mu, b, d_para, d_perp)
        expected_adc = 0.5 * d_para + 0.5 * d_perp
        assert s == pytest.approx(np.exp(-b * expected_adc))


class TestIsotropicLimit:
    def test_d_parallel_equals_d_perp(self):
        """d_parallel = d_perp → isotropic Gaussian diffusion."""
        q = np.array([0.5, 0.5, np.sqrt(0.5)])
        mu = np.array([1.0, 0.0, 0.0])
        b, D = 2000.0, 1.0e-3
        s = extracellular_tensor_signal(q, mu, b, D, D)
        assert s == pytest.approx(np.exp(-b * D))

    def test_isotropic_independent_of_direction(self):
        """Isotropic tensor should give same signal for any q."""
        mu = np.array([1.0, 0.0, 0.0])
        b, D = 2000.0, 1.0e-3
        directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        ]
        signals = [extracellular_tensor_signal(q, mu, b, D, D) for q in directions]
        for s in signals:
            assert s == pytest.approx(signals[0], abs=1e-12)


class TestEdgeCases:
    def test_b_zero(self):
        s = extracellular_tensor_signal(np.array([1, 0, 0]), np.array([1, 0, 0]), 0.0, 1.0, 0.5)
        assert s == pytest.approx(1.0)

    def test_signal_range(self):
        """Signal must always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            q = rng.normal(size=3)
            mu = rng.normal(size=3)
            b = rng.uniform(0, 4000)
            d_para = rng.uniform(0.5e-3, 2.0e-3)
            d_perp = rng.uniform(0.1e-3, 1.0e-3)
            s = extracellular_tensor_signal(q, mu, b, d_para, d_perp)
            assert 0.0 <= s <= 1.0

    def test_non_unit_vectors_normalised(self):
        """Function should normalise non-unit input vectors."""
        q = np.array([2.0, 0.0, 0.0])
        mu = np.array([3.0, 0.0, 0.0])
        s = extracellular_tensor_signal(q, mu, 2000.0, 1.2e-3, 0.6e-3)
        assert s == pytest.approx(np.exp(-2000.0 * 1.2e-3))
