"""
End-to-end validation: simulate → fit → recover parameters.

Tests the full pipeline on moderately-sized acquisition schemes to keep
runtime reasonable while still exercising all model components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from mcm_forward import MCMParameters, make_acquisition_scheme, simulate_mcm_data
from mcm_fit import fit_mcm, fit_mcm_multi_start


class TestEndToEndRecovery:
    """Fit to simulated data and check parameter recovery."""

    @pytest.fixture(scope="class")
    def medium_scheme(self):
        """Medium scheme: 15 dirs × 2 b × 2 Delta = 60 points."""
        return make_acquisition_scheme(
            n_dirs=15,
            b_values=(1000.0, 2000.0),
            Delta_values=(25.0, 40.0),
        )

    def test_recover_noiseless_full_model(self, medium_scheme):
        """Noiseless data should give near-perfect recovery."""
        true = MCMParameters(
            f_ic=0.3, k=5.0, mu=np.array([0.2, 0.3, np.sqrt(0.87)]),
            d_parallel_ic=1.0,
            f_ss=0.1, R_ss=4.0,
            f_ls=0.1, R_ls=8.0,
            f_ec=0.2, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=0.8, d_iso=3.0,
        )
        data = simulate_mcm_data(
            medium_scheme["q_dirs"], medium_scheme["bvals"],
            medium_scheme["deltas"], medium_scheme["Deltas"],
            true, snr=1e6, seed=42,
        )
        result = fit_mcm_multi_start(
            medium_scheme["q_dirs"], medium_scheme["bvals"],
            medium_scheme["deltas"], medium_scheme["Deltas"],
            data, n_starts=2, max_nfev=2000, seed=42,
        )
        fitted = result["params"]

        assert result["R2"] > 0.95
        assert fitted.f_ic == pytest.approx(true.f_ic, abs=0.10)
        assert fitted.k == pytest.approx(true.k, rel=0.3)
        # Sphere radii are notoriously hard to decouple from volume fractions
        assert fitted.R_ss == pytest.approx(true.R_ss, abs=2.0)
        assert fitted.R_ls == pytest.approx(true.R_ls, abs=2.5)
        assert fitted.f_T == pytest.approx(true.f_T, abs=0.15)

    def test_recover_low_snr_full_model(self, medium_scheme):
        """SNR=100: parameters should still be approximately recovered."""
        true = MCMParameters(
            f_ic=0.25, k=8.0, mu=np.array([0.0, 0.0, 1.0]),
            d_parallel_ic=1.0,
            f_ss=0.15, R_ss=3.5,
            f_ls=0.05, R_ls=7.0,
            f_ec=0.25, d_parallel_ec=1.0e-3, d_perp_ec=0.5e-3,
            f_T=0.75, d_iso=3.0,
        )
        data = simulate_mcm_data(
            medium_scheme["q_dirs"], medium_scheme["bvals"],
            medium_scheme["deltas"], medium_scheme["Deltas"],
            true, snr=100.0, seed=42,
        )
        result = fit_mcm_multi_start(
            medium_scheme["q_dirs"], medium_scheme["bvals"],
            medium_scheme["deltas"], medium_scheme["Deltas"],
            data, n_starts=3, max_nfev=2000, seed=42,
        )
        fitted = result["params"]

        assert result["R2"] > 0.90
        # Volume fractions should be in the right ballpark
        assert fitted.f_ic == pytest.approx(true.f_ic, abs=0.10)
        assert fitted.f_ss == pytest.approx(true.f_ss, abs=0.10)
        assert fitted.f_ls == pytest.approx(true.f_ls, abs=0.10)
        assert fitted.f_ec == pytest.approx(true.f_ec, abs=0.10)
        assert fitted.f_T == pytest.approx(true.f_T, abs=0.10)

    def test_mu_direction_recovery(self, medium_scheme):
        """The principal direction mu should be recovered."""
        true_mu = np.array([0.3, 0.4, np.sqrt(0.75)])
        true = MCMParameters(
            f_ic=0.4, k=10.0, mu=true_mu,
            d_parallel_ic=1.0,
            f_ss=0.1, R_ss=4.0,
            f_ls=0.1, R_ls=8.0,
            f_ec=0.2, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=0.8, d_iso=3.0,
        )
        data = simulate_mcm_data(
            medium_scheme["q_dirs"], medium_scheme["bvals"],
            medium_scheme["deltas"], medium_scheme["Deltas"],
            true, snr=1e6, seed=42,
        )
        result = fit_mcm(
            medium_scheme["q_dirs"], medium_scheme["bvals"],
            medium_scheme["deltas"], medium_scheme["Deltas"],
            data, max_nfev=2000,
        )
        fitted = result["params"]
        # Angular error in degrees
        cos_angle = np.clip(np.dot(true_mu, fitted.mu), -1.0, 1.0)
        angle_err = np.degrees(np.arccos(abs(cos_angle)))
        assert angle_err < 15.0  # within 15 degrees


class TestPipelineIntegrity:
    """Sanity checks on the full pipeline."""

    def test_volume_constraint_enforced_after_fit(self):
        scheme = make_acquisition_scheme(n_dirs=10, b_values=(2000.0,), Delta_values=(40.0,))
        true = MCMParameters(
            f_ic=0.3, k=5.0, mu=np.array([0, 0, 1]),
            f_ss=0.1, R_ss=4.0,
            f_ls=0.1, R_ls=8.0,
            f_ec=0.2, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=0.8,
        )
        data = simulate_mcm_data(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            true, snr=50.0, seed=1,
        )
        result = fit_mcm(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            data, max_nfev=500,
        )
        # After normalisation, volume fractions must sum to ~1
        vsum = result["params"].volume_sum()
        assert vsum == pytest.approx(1.0, abs=1e-6)

    def test_predicted_matches_observed_for_noiseless(self):
        """On noiseless data, predicted should closely match observed."""
        from mcm_forward import mcm_signal_batch

        scheme = make_acquisition_scheme(n_dirs=10, b_values=(2000.0,), Delta_values=(40.0,))
        true = MCMParameters(
            f_ic=0.3, k=5.0, mu=np.array([0, 0, 1]),
            f_ss=0.1, R_ss=4.0,
            f_ls=0.1, R_ls=8.0,
            f_ec=0.2, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=0.8,
        )
        data = simulate_mcm_data(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            true, snr=1e6, seed=1,
        )
        result = fit_mcm(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            data, max_nfev=500,
        )
        predicted = mcm_signal_batch(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            result["params"],
        )
        rmse = np.sqrt(np.mean((predicted - data) ** 2))
        assert rmse < 0.01
