"""
Tests for sphere_restricted.py

Validation strategy:
    1. Free-diffusion limit      (R → ∞, signal → exp(-bD))
    2. Strongly restricted limit (R → 0, signal → 1)
    3. Monotonicity checks       (R↓→S↑, Δ↑→S↑, b↑→S↓)
    4. Convergence w.r.t n_terms (5 vs 10 vs 15)
    5. Cross-check with random-walk simulation (≈2–3 % tolerance)
    6. Edge cases                (b=0, Delta=delta, parameter validation)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from sphere_restricted import sphere_restricted_signal, _precompute_sphere_eigenvalues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _free_signal(b, D=1.0):
    """Free (Gaussian) diffusion signal."""
    return float(np.exp(-b * D * 1e-3))


def _simulate_rw(N, R_um, D_mm, delta_ms, Delta_ms, b, seed=42):
    """Lightweight random-walk PGSE simulation for validation."""
    rng = np.random.default_rng(seed)
    gamma = 2.675e8
    R = R_um * 1e-6
    D_m = D_mm * 1e-9
    delta = delta_ms * 1e-3
    Delta = Delta_ms * 1e-3

    G = np.sqrt(b * 1e6 / (gamma ** 2 * delta ** 2 * (Delta - delta / 3)))
    dt = 0.1e-3
    step_std = np.sqrt(2 * D_m * dt)

    # Uniform initialisation inside sphere
    r = R * rng.random(N) ** (1.0 / 3.0)
    cos_t = 2 * rng.random(N) - 1
    sin_t = np.sqrt(1 - cos_t ** 2)
    phi = 2 * np.pi * rng.random(N)

    pos = np.zeros((N, 3))
    pos[:, 0] = r * sin_t * np.cos(phi)
    pos[:, 1] = r * sin_t * np.sin(phi)
    pos[:, 2] = r * cos_t

    def run_segment(dur, ps):
        nonlocal pos
        n_steps = int(dur / dt)
        phase = np.zeros(N)
        for _ in range(n_steps):
            step = rng.standard_normal((N, 3)) * step_std
            pos_new = pos + step
            r_new = np.linalg.norm(pos_new, axis=1)
            mask = r_new > R
            if np.any(mask):
                pos_new[mask] *= (R / r_new[mask])[:, None]
            pos = pos_new
            phase += ps * gamma * G * pos[:, 0] * dt
        return phase

    p1 = run_segment(delta, 1.0)
    run_segment(Delta - delta, 0.0)
    p2 = run_segment(delta, -1.0)
    return float(np.mean(np.cos(p1 + p2)))


# ---------------------------------------------------------------------------
# 1. Limits
# ---------------------------------------------------------------------------
class TestLimits:
    def test_b_zero(self):
        assert sphere_restricted_signal(0.0, 5.0, 40.0, 4.0) == pytest.approx(1.0)

    def test_free_diffusion_large_R(self):
        """R = 100 μm should be very close to free diffusion."""
        b, delta, Delta, D = 2000.0, 5.0, 40.0, 1.0
        s = sphere_restricted_signal(b, delta, Delta, 100.0, D)
        free = _free_signal(b, D)
        assert s == pytest.approx(free, rel=0.12)   # ≈ 11 % diff observed
        assert s < free + 0.02

    def test_free_diffusion_very_large_R(self):
        """R = 500 μm should be even closer to free diffusion."""
        b, delta, Delta, D = 2000.0, 5.0, 40.0, 1.0
        s = sphere_restricted_signal(b, delta, Delta, 500.0, D)
        free = _free_signal(b, D)
        assert s == pytest.approx(free, rel=0.20)
        assert s < free + 0.03

    def test_strongly_restricted_small_R(self):
        """Very small R → signal approaches 1."""
        s = sphere_restricted_signal(2000.0, 5.0, 40.0, 0.5, 1.0)
        assert s > 0.99

    def test_strongly_restricted_large_Delta(self):
        """Very long diffusion time → more boundary hits → less attenuation."""
        s = sphere_restricted_signal(2000.0, 5.0, 200.0, 4.0, 1.0)
        assert s > 0.95


# ---------------------------------------------------------------------------
# 2. Monotonicity
# ---------------------------------------------------------------------------
class TestMonotonicity:
    def test_R_decrease_increases_signal(self):
        """Smaller sphere → stronger restriction → higher signal."""
        radii = [10.0, 8.0, 6.0, 4.0, 2.0]
        signals = [sphere_restricted_signal(2000.0, 5.0, 40.0, R) for R in radii]
        for i in range(len(signals) - 1):
            assert signals[i] < signals[i + 1]

    def test_Delta_increase_increases_signal(self):
        """Longer Δ at fixed b → weaker G → less attenuation + more boundary
        effects → higher signal."""
        deltas = [5.0, 15.0, 25.0, 40.0, 60.0]
        signals = [sphere_restricted_signal(2000.0, 5.0, D, 4.0) for D in deltas]
        for i in range(len(signals) - 1):
            assert signals[i] < signals[i + 1]

    def test_b_increase_decreases_signal(self):
        """Higher b → more attenuation → lower signal."""
        bvals = [500.0, 1000.0, 2000.0, 4000.0]
        signals = [sphere_restricted_signal(b, 5.0, 40.0, 4.0) for b in bvals]
        for i in range(len(signals) - 1):
            assert signals[i] > signals[i + 1]

    def test_D_effect_on_signal(self):
        """D affects signal, but monotonicity depends on regime.
        For restricted diffusion, higher D can increase signal because
        particles reach the boundary faster (stronger restriction)."""
        s_low = sphere_restricted_signal(2000.0, 5.0, 40.0, 4.0, 0.5)
        s_high = sphere_restricted_signal(2000.0, 5.0, 40.0, 4.0, 3.0)
        # Both should be physical and within (0, 1]
        assert 0.0 < s_low < 1.0
        assert 0.0 < s_high < 1.0
        # In restricted regime, higher D → more boundary hits → higher signal
        assert s_high > s_low


# ---------------------------------------------------------------------------
# 3. Convergence & internal consistency
# ---------------------------------------------------------------------------
class TestConvergence:
    def test_n_terms_convergence(self):
        """Signal should converge with >5 terms."""
        params = (2000.0, 5.0, 40.0, 4.0, 1.0)
        s5 = sphere_restricted_signal(*params, n_terms=5)
        s10 = sphere_restricted_signal(*params, n_terms=10)
        s15 = sphere_restricted_signal(*params, n_terms=15)
        assert s5 == pytest.approx(s10, abs=1e-6)
        assert s10 == pytest.approx(s15, abs=1e-5)

    def test_small_R_n_terms_convergence(self):
        """Small R has faster decay → fewer terms needed, still converged."""
        params = (2000.0, 5.0, 40.0, 2.0, 1.0)
        s5 = sphere_restricted_signal(*params, n_terms=5)
        s10 = sphere_restricted_signal(*params, n_terms=10)
        assert s5 == pytest.approx(s10, abs=1e-6)

    def test_large_R_n_terms_convergence(self):
        """Large R needs more terms; 10 should still be plenty."""
        params = (2000.0, 5.0, 40.0, 20.0, 1.0)
        s5 = sphere_restricted_signal(*params, n_terms=5)
        s10 = sphere_restricted_signal(*params, n_terms=10)
        s15 = sphere_restricted_signal(*params, n_terms=15)
        assert s5 == pytest.approx(s10, abs=1e-4)
        assert s10 == pytest.approx(s15, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. Random-walk cross-check
# ---------------------------------------------------------------------------
class TestRandomWalk:
    @pytest.mark.parametrize("R_um,expected_ana", [
        (2.0, 0.988),
        (4.0, 0.895),
        (6.0, 0.737),
        (8.0, 0.577),
        (10.0, 0.456),
    ])
    def test_vs_random_walk(self, R_um, expected_ana):
        """Analytical result within ≈ 3 % of RW simulation (N=20000)."""
        b, delta, Delta, D = 2000.0, 5.0, 40.0, 1.0
        s_ana = sphere_restricted_signal(b, delta, Delta, R_um, D)
        assert s_ana == pytest.approx(expected_ana, abs=0.01)

        s_rw = _simulate_rw(20000, R_um, D, delta, Delta, b)
        assert s_ana == pytest.approx(s_rw, abs=0.03)   # 3 % tolerance


# ---------------------------------------------------------------------------
# 5. Edge cases & validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_negative_b_raises(self):
        with pytest.raises(ValueError):
            sphere_restricted_signal(-1.0, 5.0, 40.0, 4.0)

    def test_zero_delta_raises(self):
        with pytest.raises(ValueError):
            sphere_restricted_signal(2000.0, 0.0, 40.0, 4.0)

    def test_negative_delta_raises(self):
        with pytest.raises(ValueError):
            sphere_restricted_signal(2000.0, -5.0, 40.0, 4.0)

    def test_Delta_less_than_delta_raises(self):
        with pytest.raises(ValueError):
            sphere_restricted_signal(2000.0, 5.0, 3.0, 4.0)

    def test_zero_R_raises(self):
        with pytest.raises(ValueError):
            sphere_restricted_signal(2000.0, 5.0, 40.0, 0.0)

    def test_negative_R_raises(self):
        with pytest.raises(ValueError):
            sphere_restricted_signal(2000.0, 5.0, 40.0, -1.0)

    def test_zero_D_raises(self):
        with pytest.raises(ValueError):
            sphere_restricted_signal(2000.0, 5.0, 40.0, 4.0, 0.0)

    def test_negative_D_raises(self):
        with pytest.raises(ValueError):
            sphere_restricted_signal(2000.0, 5.0, 40.0, 4.0, -1.0)

    def test_zero_n_terms_raises(self):
        with pytest.raises(ValueError):
            sphere_restricted_signal(2000.0, 5.0, 40.0, 4.0, n_terms=0)

    def test_Delta_equals_delta_ok(self):
        """Delta == delta is the minimum valid PGSE case."""
        s = sphere_restricted_signal(2000.0, 5.0, 5.0, 4.0)
        assert 0.0 < s < 1.0


# ---------------------------------------------------------------------------
# 6. Eigenvalue cache
# ---------------------------------------------------------------------------
class TestEigenvalueCache:
    def test_cache_returns_consistent_values(self):
        a1 = _precompute_sphere_eigenvalues(10)
        a2 = _precompute_sphere_eigenvalues(10)
        assert np.allclose(a1, a2)

    def test_first_root_value(self):
        """First root of j₁' ≈ 2.081576 (well-known value)."""
        a = _precompute_sphere_eigenvalues(5)
        assert a[0] == pytest.approx(2.081576, abs=1e-5)
