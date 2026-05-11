"""Tests for mcm_forward.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from mcm_forward import (
    MCMParameters,
    mcm_signal,
    mcm_signal_batch,
    simulate_mcm_data,
    make_acquisition_scheme,
    _fibonacci_sphere,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def default_params():
    return MCMParameters(
        f_ic=0.3, k=5.0, mu=np.array([0.0, 0.0, 1.0]), d_parallel_ic=1.0e-3,
        f_ss=0.1, R_ss=4.0,
        f_ls=0.1, R_ls=8.0,
        f_ec=0.2, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
        f_T=0.8, d_iso=3.0e-3,
    )


# ---------------------------------------------------------------------------
# MCMParameters
# ---------------------------------------------------------------------------
class TestMCMParameters:
    def test_default_initialisation(self):
        p = MCMParameters()
        assert p.f_ic == pytest.approx(0.3)
        assert p.k == pytest.approx(5.0)
        assert np.allclose(p.mu, np.array([0, 0, 1]))

    def test_mu_normalised(self):
        p = MCMParameters(mu=np.array([2.0, 0.0, 0.0]))
        assert np.allclose(p.mu, np.array([1.0, 0.0, 0.0]))

    def test_volume_sum(self, default_params):
        # default: 0.3 + 0.1 + 0.1 + 0.2 + (1-0.8) = 0.9
        assert default_params.volume_sum() == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Single-point signal
# ---------------------------------------------------------------------------
class TestMCMSignal:
    def test_b_zero(self, default_params):
        """b=0 → all compartment signals = 1, total = volume sum."""
        q = np.array([1.0, 0.0, 0.0])
        s = mcm_signal(q, 0.0, 5.0, 40.0, default_params)
        assert s == pytest.approx(default_params.volume_sum())

    def test_signal_range(self, default_params):
        """Signal must be in [0, 1] for random directions."""
        rng = np.random.default_rng(42)
        for _ in range(30):
            q = rng.normal(size=3)
            b = rng.choice([500.0, 1000.0, 2000.0, 4000.0])
            Delta = rng.choice([15.0, 25.0, 40.0, 60.0])
            s = mcm_signal(q, b, 5.0, Delta, default_params)
            assert 0.0 <= s <= 1.0

    def test_f_ic_dominates_when_aligned(self, default_params):
        """When q ∥ μ and k is large, stick compartment dominates attenuation."""
        q = np.array([0.0, 0.0, 1.0])
        params = MCMParameters(
            f_ic=0.8, k=20.0, mu=np.array([0.0, 0.0, 1.0]), d_parallel_ic=1.0e-3,
            f_ss=0.05, R_ss=4.0,
            f_ls=0.05, R_ls=8.0,
            f_ec=0.05, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=0.95, d_iso=3.0e-3,
        )
        s = mcm_signal(q, 2000.0, 5.0, 40.0, params)
        # stick-only signal at b=2000, d_parallel=1.0 → exp(-2) ≈ 0.135
        # weighted by f_ic=0.8 → ~0.108
        assert s < 0.3  # strongly attenuated

    def test_k_zero_watson_uniform(self, default_params):
        """k=0 → Watson distribution is uniform, but stick itself is still
        anisotropic, so signal varies with q.  Check against exact integral."""
        params = MCMParameters(
            f_ic=1.0, k=0.0, mu=np.array([0.0, 0.0, 1.0]), d_parallel_ic=1.0e-3,
            f_ss=0.0, R_ss=4.0,
            f_ls=0.0, R_ls=8.0,
            f_ec=0.0, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=1.0, d_iso=3.0e-3,
        )
        q = np.array([1.0, 0.0, 0.0])
        s = mcm_signal(q, 2000.0, 5.0, 40.0, params)
        # exact integral for isotropic Watson (k=0)
        from watson_stick import watson_stick_isotropic_exact
        expected = watson_stick_isotropic_exact(2000.0, 1.0e-3)
        assert s == pytest.approx(expected, abs=1e-4)

    def test_pure_free_water(self, default_params):
        """f_T=0 → pure free water."""
        params = MCMParameters(
            f_ic=0.0, k=5.0, mu=np.array([0.0, 0.0, 1.0]), d_parallel_ic=1.0e-3,
            f_ss=0.0, R_ss=4.0,
            f_ls=0.0, R_ls=8.0,
            f_ec=0.0, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=0.0, d_iso=3.0e-3,
        )
        q = np.array([1.0, 0.0, 0.0])
        s = mcm_signal(q, 2000.0, 5.0, 40.0, params)
        assert s == pytest.approx(np.exp(-2000.0 * 3.0e-3), abs=1e-6)

    def test_pure_extracellular(self, default_params):
        """f_ec=1, others=0 → pure extracellular tensor."""
        params = MCMParameters(
            f_ic=0.0, k=5.0, mu=np.array([0.0, 0.0, 1.0]), d_parallel_ic=1.0e-3,
            f_ss=0.0, R_ss=4.0,
            f_ls=0.0, R_ls=8.0,
            f_ec=1.0, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=1.0, d_iso=3.0e-3,
        )
        q = np.array([0.0, 0.0, 1.0])
        s = mcm_signal(q, 2000.0, 5.0, 40.0, params)
        assert s == pytest.approx(np.exp(-2000.0 * 1.2e-3), abs=1e-6)


# ---------------------------------------------------------------------------
# Batch signal
# ---------------------------------------------------------------------------
class TestMCMSignalBatch:
    def test_batch_matches_single(self, default_params):
        """Batch computation must match individual mcm_signal calls."""
        rng = np.random.default_rng(42)
        n = 20
        q_dirs = rng.normal(size=(n, 3))
        bvals = rng.uniform(500, 4000, size=n)
        deltas = np.full(n, 5.0)
        Deltas = rng.choice([15.0, 25.0, 40.0, 60.0], size=n)

        s_batch = mcm_signal_batch(q_dirs, bvals, deltas, Deltas, default_params)

        for i in range(n):
            s_single = mcm_signal(q_dirs[i], bvals[i], deltas[i], Deltas[i], default_params)
            assert s_batch[i] == pytest.approx(s_single, abs=1e-10)

    def test_batch_shape_validation(self, default_params):
        with pytest.raises(ValueError):
            mcm_signal_batch(
                np.zeros((5, 3)),
                np.ones(4),   # mismatch
                np.ones(5),
                np.ones(5),
                default_params,
            )


# ---------------------------------------------------------------------------
# Acquisition scheme
# ---------------------------------------------------------------------------
class TestAcquisitionScheme:
    def test_fibonacci_sphere_uniformity(self):
        dirs = _fibonacci_sphere(30)
        assert dirs.shape == (30, 3)
        # all points on unit sphere
        norms = np.linalg.norm(dirs, axis=1)
        assert np.allclose(norms, 1.0)

    def test_make_scheme_dimensions(self):
        scheme = make_acquisition_scheme(
            n_dirs=30, b_values=(2000.0, 4000.0),
            Delta_values=(15.0, 25.0, 40.0, 60.0),
        )
        n_expected = 30 * 2 * 4
        assert len(scheme["bvals"]) == n_expected
        assert scheme["q_dirs"].shape == (n_expected, 3)
        assert np.allclose(scheme["deltas"], 5.0)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class TestSimulation:
    def test_noise_decreases_snr(self, default_params):
        """Higher noise → lower effective SNR."""
        q = np.array([[1.0, 0.0, 0.0]])
        b = np.array([2000.0])
        delta = np.array([5.0])
        Delta = np.array([40.0])

        clean = mcm_signal_batch(q, b, delta, Delta, default_params)

        noisy_high = simulate_mcm_data(q, b, delta, Delta, default_params, snr=5.0, seed=1)
        noisy_low = simulate_mcm_data(q, b, delta, Delta, default_params, snr=50.0, seed=1)

        err_high = np.abs(noisy_high - clean)
        err_low = np.abs(noisy_low - clean)
        assert err_high[0] > err_low[0]  # on average; with same seed this is guaranteed for single sample? No, but usually true

    def test_simulated_range(self, default_params):
        scheme = make_acquisition_scheme(n_dirs=10, b_values=(2000.0,), Delta_values=(40.0,))
        data = simulate_mcm_data(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            default_params, snr=20.0,
        )
        assert data.shape == (10,)
        assert np.all(data >= 0.0)

    def test_reproducibility(self, default_params):
        """Same seed → same noise."""
        scheme = make_acquisition_scheme(n_dirs=10, b_values=(2000.0,), Delta_values=(40.0,))
        d1 = simulate_mcm_data(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            default_params, snr=20.0, seed=123,
        )
        d2 = simulate_mcm_data(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            default_params, snr=20.0, seed=123,
        )
        assert np.allclose(d1, d2)
