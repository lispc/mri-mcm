"""Tests for mcm_fit.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from mcm_forward import MCMParameters, make_acquisition_scheme, simulate_mcm_data
from mcm_fit import (
    params_to_vec,
    vec_to_params,
    normalise_volumes,
    fit_mcm,
    fit_mcm_multi_start,
    _DEFAULT_INITIAL,
    _LOWER_BOUNDS,
    _UPPER_BOUNDS,
)


# ---------------------------------------------------------------------------
# Vector conversion
# ---------------------------------------------------------------------------
class TestVectorConversion:
    def test_round_trip(self):
        p = MCMParameters(
            f_ic=0.25, k=10.0, mu=np.array([0.5, 0.5, np.sqrt(0.5)]),
            d_parallel_ic=1.0e-3,
            f_ss=0.15, R_ss=3.5,
            f_ls=0.05, R_ls=7.0,
            f_ec=0.3, d_parallel_ec=1.5e-3, d_perp_ec=0.8e-3,
            f_T=0.75, d_iso=3.0e-3,
        )
        vec = params_to_vec(p)
        p2 = vec_to_params(vec)
        assert p.f_ic == pytest.approx(p2.f_ic)
        assert p.k == pytest.approx(p2.k)
        assert np.allclose(p.mu, p2.mu)
        assert p.f_ss == pytest.approx(p2.f_ss)
        assert p.R_ss == pytest.approx(p2.R_ss)
        assert p.f_ls == pytest.approx(p2.f_ls)
        assert p.R_ls == pytest.approx(p2.R_ls)
        assert p.f_ec == pytest.approx(p2.f_ec)
        assert p.d_parallel_ec == pytest.approx(p2.d_parallel_ec)
        assert p.d_perp_ec == pytest.approx(p2.d_perp_ec)
        assert p.f_T == pytest.approx(p2.f_T)

    def test_default_initial_in_bounds(self):
        assert np.all(_DEFAULT_INITIAL >= _LOWER_BOUNDS)
        assert np.all(_DEFAULT_INITIAL <= _UPPER_BOUNDS)


# ---------------------------------------------------------------------------
# Volume normalisation
# ---------------------------------------------------------------------------
class TestVolumeNormalisation:
    def test_normalises_to_one(self):
        p = MCMParameters(
            f_ic=0.3, f_ss=0.1, f_ls=0.1, f_ec=0.2, f_T=0.8,
        )
        p_norm = normalise_volumes(p)
        assert p_norm.volume_sum() == pytest.approx(1.0)

    def test_already_normalised_unchanged(self):
        p = MCMParameters(
            f_ic=0.2, f_ss=0.2, f_ls=0.2, f_ec=0.2, f_T=0.8,
        )
        # volume_sum = 0.2+0.2+0.2+0.2+0.2 = 1.0
        p_norm = normalise_volumes(p)
        assert p_norm.f_ic == pytest.approx(0.2)
        assert p_norm.f_ss == pytest.approx(0.2)

    def test_preserves_ratios(self):
        p = MCMParameters(
            f_ic=0.6, f_ss=0.3, f_ls=0.0, f_ec=0.0, f_T=1.0,
        )
        # volume_sum = 0.6+0.3+0+0+0 = 0.9
        p_norm = normalise_volumes(p)
        assert p_norm.f_ic / p_norm.f_ss == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Fitting — parameter recovery on noiseless data
# ---------------------------------------------------------------------------
class TestFittingNoiseless:
    @pytest.fixture
    def small_scheme(self):
        """A small scheme for fast tests."""
        return make_acquisition_scheme(
            n_dirs=10,
            b_values=(1000.0, 2000.0),
            Delta_values=(25.0, 40.0),
        )

    def test_recover_pure_free_water(self, small_scheme):
        """Fit to pure free-water data should recover f_T≈0."""
        true_params = MCMParameters(
            f_ic=0.0, k=5.0, mu=np.array([0, 0, 1]), d_parallel_ic=1.0e-3,
            f_ss=0.0, R_ss=4.0,
            f_ls=0.0, R_ls=8.0,
            f_ec=0.0, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=0.0, d_iso=3.0e-3,
        )
        data = simulate_mcm_data(
            small_scheme["q_dirs"], small_scheme["bvals"],
            small_scheme["deltas"], small_scheme["Deltas"],
            true_params, snr=1e6, seed=1,
        )
        result = fit_mcm(
            small_scheme["q_dirs"], small_scheme["bvals"],
            small_scheme["deltas"], small_scheme["Deltas"],
            data, max_nfev=500,
        )
        assert result["success"]
        assert result["R2"] > 0.99
        # f_T should be close to 0
        assert result["params"].f_T < 0.1

    def test_recover_pure_extracellular(self, small_scheme):
        """Fit to pure EC data should recover EC parameters."""
        true_params = MCMParameters(
            f_ic=0.0, k=5.0, mu=np.array([0, 0, 1]), d_parallel_ic=1.0e-3,
            f_ss=0.0, R_ss=4.0,
            f_ls=0.0, R_ls=8.0,
            f_ec=1.0, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=1.0, d_iso=3.0e-3,
        )
        data = simulate_mcm_data(
            small_scheme["q_dirs"], small_scheme["bvals"],
            small_scheme["deltas"], small_scheme["Deltas"],
            true_params, snr=1e6, seed=1,
        )
        result = fit_mcm(
            small_scheme["q_dirs"], small_scheme["bvals"],
            small_scheme["deltas"], small_scheme["Deltas"],
            data, max_nfev=500,
        )
        assert result["success"]
        assert result["R2"] > 0.99
        assert result["params"].f_ec > 0.9

    def test_recover_full_model_noisy(self, small_scheme):
        """Fit to full MCM data with moderate SNR."""
        true_params = MCMParameters(
            f_ic=0.3, k=5.0, mu=np.array([0, 0, 1]), d_parallel_ic=1.0e-3,
            f_ss=0.1, R_ss=4.0,
            f_ls=0.1, R_ls=8.0,
            f_ec=0.2, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=0.8, d_iso=3.0e-3,
        )
        data = simulate_mcm_data(
            small_scheme["q_dirs"], small_scheme["bvals"],
            small_scheme["deltas"], small_scheme["Deltas"],
            true_params, snr=50.0, seed=1,
        )
        result = fit_mcm(
            small_scheme["q_dirs"], small_scheme["bvals"],
            small_scheme["deltas"], small_scheme["Deltas"],
            data, max_nfev=1000,
        )
        assert result["success"]
        assert result["R2"] > 0.90  # 12 params, 40 pts, SNR=50 → moderate recovery


# ---------------------------------------------------------------------------
# Multi-start
# ---------------------------------------------------------------------------
class TestMultiStart:
    def test_multi_start_better_than_single(self):
        scheme = make_acquisition_scheme(
            n_dirs=10, b_values=(2000.0,), Delta_values=(40.0,),
        )
        true_params = MCMParameters(
            f_ic=0.3, k=5.0, mu=np.array([0, 0, 1]), d_parallel_ic=1.0e-3,
            f_ss=0.1, R_ss=4.0,
            f_ls=0.1, R_ls=8.0,
            f_ec=0.2, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
            f_T=0.8, d_iso=3.0e-3,
        )
        data = simulate_mcm_data(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            true_params, snr=50.0, seed=1,
        )

        result_single = fit_mcm(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            data, max_nfev=500,
        )

        result_multi = fit_mcm_multi_start(
            scheme["q_dirs"], scheme["bvals"],
            scheme["deltas"], scheme["Deltas"],
            data, n_starts=3, max_nfev=500, seed=1,
        )

        assert result_multi["cost"] <= result_single["cost"] + 1e-6
