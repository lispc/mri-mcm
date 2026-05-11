"""
MCM Parameter Fitting — Non-linear least-squares inversion

Fits the 5-compartment MCM model to DWI data using
scipy.optimize.least_squares with box constraints.

Parameter vector (12 free parameters):
    [f_ic, k, theta, phi,
     f_ss, R_ss,
     f_ls, R_ls,
     f_ec, d_parallel_ec, d_perp_ec,
     f_T]

Fixed:
    d_parallel_ic = 1.0 × 10⁻³ mm²/s
    d_iso         = 3.0 × 10⁻³ mm²/s (CSF)

Volume constraint (handled by normalisation inside the objective):
    f_ic + f_ss + f_ls + f_ec + (1 - f_T) = 1
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import least_squares

from mcm_forward import (
    MCMParameters,
    mcm_signal_batch,
    make_acquisition_scheme,
)


# ---------------------------------------------------------------------------
# Parameter bounds (physical ranges)
# ---------------------------------------------------------------------------
_PARAM_NAMES = [
    "f_ic", "k", "theta", "phi",
    "f_ss", "R_ss",
    "f_ls", "R_ls",
    "f_ec", "d_parallel_ec", "d_perp_ec",
    "f_T",
]

_LOWER_BOUNDS = np.array([
    0.0,    # f_ic
    0.0,    # k
    0.0,    # theta
    0.0,    # phi
    0.0,    # f_ss
    2.0,    # R_ss (μm)
    0.0,    # f_ls
    6.0,    # R_ls (μm)
    0.0,    # f_ec
    0.5e-3, # d_parallel_ec (mm²/s)
    0.1e-3, # d_perp_ec (mm²/s)
    0.0,    # f_T
])

_UPPER_BOUNDS = np.array([
    1.0,    # f_ic
    30.0,   # k
    np.pi,  # theta
    2 * np.pi,  # phi
    1.0,    # f_ss
    6.0,    # R_ss
    1.0,    # f_ls
    12.0,   # R_ls
    1.0,    # f_ec
    2.0e-3, # d_parallel_ec
    1.0e-3, # d_perp_ec
    1.0,    # f_T
])

_DEFAULT_INITIAL = np.array([
    0.3,     # f_ic
    5.0,     # k
    0.5 * np.pi,   # theta
    0.0,     # phi
    0.1,     # f_ss
    4.0,     # R_ss
    0.1,     # f_ls
    8.0,     # R_ls
    0.2,     # f_ec
    1.2e-3,  # d_parallel_ec (mm²/s)
    0.6e-3,  # d_perp_ec (mm²/s)
    0.8,     # f_T
])


# ---------------------------------------------------------------------------
# Vector <-> MCMParameters conversions
# ---------------------------------------------------------------------------
def params_to_vec(params: MCMParameters) -> np.ndarray:
    """Pack MCMParameters into a 12-element vector."""
    theta = np.arccos(np.clip(params.mu[2], -1.0, 1.0))
    phi = np.arctan2(params.mu[1], params.mu[0])
    return np.array([
        params.f_ic,
        params.k,
        theta,
        phi,
        params.f_ss,
        params.R_ss,
        params.f_ls,
        params.R_ls,
        params.f_ec,
        params.d_parallel_ec,
        params.d_perp_ec,
        params.f_T,
    ])


def vec_to_params(vec: np.ndarray) -> MCMParameters:
    """Unpack a 12-element vector into MCMParameters."""
    theta = np.clip(vec[2], 0.0, np.pi)
    phi = vec[3]
    mu = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    return MCMParameters(
        f_ic=vec[0],
        k=vec[1],
        mu=mu,
        d_parallel_ic=1.0,
        f_ss=vec[4],
        R_ss=vec[5],
        f_ls=vec[6],
        R_ls=vec[7],
        f_ec=vec[8],
        d_parallel_ec=vec[9],
        d_perp_ec=vec[10],
        f_T=vec[11],
    )


# ---------------------------------------------------------------------------
# Volume-fraction normalisation
# ---------------------------------------------------------------------------
def normalise_volumes(params: MCMParameters) -> MCMParameters:
    """
    Scale all volume fractions so that they sum to exactly 1.
    Returns a new MCMParameters object with normalised fractions.
    """
    vsum = params.volume_sum()
    if vsum <= 0:
        return params

    scale = 1.0 / vsum
    return MCMParameters(
        f_ic=params.f_ic * scale,
        k=params.k,
        mu=params.mu.copy(),
        d_parallel_ic=params.d_parallel_ic,
        f_ss=params.f_ss * scale,
        R_ss=params.R_ss,
        f_ls=params.f_ls * scale,
        R_ls=params.R_ls,
        f_ec=params.f_ec * scale,
        d_parallel_ec=params.d_parallel_ec,
        d_perp_ec=params.d_perp_ec,
        f_T=1.0 - (1.0 - params.f_T) * scale,
    )


# ---------------------------------------------------------------------------
# Objective / residual function
# ---------------------------------------------------------------------------
def make_residuals(
    q_dirs: np.ndarray,
    bvals: np.ndarray,
    deltas: np.ndarray,
    Deltas: np.ndarray,
    observed: np.ndarray,
    sigma: Optional[np.ndarray] = None,
):
    """
    Factory that returns a residual function for scipy.optimize.least_squares.

    Parameters
    ----------
    observed : np.ndarray, shape (N,)
        Observed (noisy) signals.
    sigma : np.ndarray or float, optional
        Noise standard deviation. If None, unit weighting is used.
    """
    if sigma is None:
        sigma = 1.0
    sigma = np.asarray(sigma)
    if sigma.ndim == 0:
        sigma = np.full_like(observed, float(sigma))

    def residuals(vec: np.ndarray) -> np.ndarray:
        params = vec_to_params(vec)
        params = normalise_volumes(params)
        predicted = mcm_signal_batch(q_dirs, bvals, deltas, Deltas, params)
        return (predicted - observed) / sigma

    return residuals


# ---------------------------------------------------------------------------
# Single fit
# ---------------------------------------------------------------------------
def fit_mcm(
    q_dirs: np.ndarray,
    bvals: np.ndarray,
    deltas: np.ndarray,
    Deltas: np.ndarray,
    observed: np.ndarray,
    initial: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    method: str = "trf",
    max_nfev: int = 2000,
    ftol: float = 1e-8,
) -> dict:
    """
    Fit the MCM model to observed DWI data.

    Parameters
    ----------
    q_dirs, bvals, deltas, Deltas : arrays
        Acquisition scheme.
    observed : np.ndarray, shape (N,)
        Observed signal values.
    initial : np.ndarray, optional
        Initial parameter guess (12-element vector).  If None, uses
        the physiologically-motivated default.
    sigma : np.ndarray or float, optional
        Noise standard deviation for weighted least squares.
    method : str
        Optimization algorithm passed to scipy.optimize.least_squares.
        Default "trf" (trust-region reflective) handles bounds well.
    max_nfev : int
        Maximum function evaluations.
    ftol : float
        Cost function tolerance.

    Returns
    -------
    dict with keys:
        'params'      : fitted MCMParameters (volume-normalised)
        'x'           : fitted parameter vector
        'cost'        : final sum of squared residuals
        'nfev'        : number of function evaluations
        'success'     : optimizer success flag
        'message'     : optimizer message
        'R2'          : coefficient of determination
        'n_params'    : number of free parameters (12)
        'n_data'      : number of data points
    """
    if initial is None:
        x0 = _DEFAULT_INITIAL.copy()
    else:
        x0 = np.asarray(initial, dtype=float).copy()
        if len(x0) != 12:
            raise ValueError(f"initial must be a 12-element vector, got {len(x0)}")

    # Clip initial guess to bounds (defensive)
    x0 = np.clip(x0, _LOWER_BOUNDS, _UPPER_BOUNDS)

    residual_fn = make_residuals(q_dirs, bvals, deltas, Deltas, observed, sigma)

    result = least_squares(
        residual_fn,
        x0,
        bounds=(_LOWER_BOUNDS, _UPPER_BOUNDS),
        method=method,
        max_nfev=max_nfev,
        ftol=ftol,
    )

    x_fit = result.x
    params_fit = normalise_volumes(vec_to_params(x_fit))

    # R²
    predicted = mcm_signal_batch(q_dirs, bvals, deltas, Deltas, params_fit)
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "params": params_fit,
        "x": x_fit,
        "cost": result.cost,
        "nfev": result.nfev,
        "success": result.success,
        "message": result.message,
        "R2": R2,
        "n_params": 12,
        "n_data": len(observed),
    }


# ---------------------------------------------------------------------------
# Multi-start fitting
# ---------------------------------------------------------------------------
def fit_mcm_multi_start(
    q_dirs: np.ndarray,
    bvals: np.ndarray,
    deltas: np.ndarray,
    Deltas: np.ndarray,
    observed: np.ndarray,
    n_starts: int = 3,
    sigma: Optional[np.ndarray] = None,
    seed: int = 42,
    **kwargs,
) -> dict:
    """
    Fit MCM from multiple random initialisations and return the best result.

    Parameters
    ----------
    n_starts : int
        Number of random restarts.  Default 3.
    seed : int
        Random seed for reproducibility.
    **kwargs
        Passed to fit_mcm (method, max_nfev, etc.).

    Returns
    -------
    dict
        Same format as fit_mcm, with additional key 'n_starts'.
    """
    rng = np.random.default_rng(seed)

    best_result = None
    best_cost = np.inf

    for i in range(n_starts):
        if i == 0:
            # First start from the physiologically-motivated default
            x0 = _DEFAULT_INITIAL.copy()
        else:
            # Random initialisation within bounds
            x0 = rng.uniform(_LOWER_BOUNDS, _UPPER_BOUNDS)

        try:
            result = fit_mcm(
                q_dirs, bvals, deltas, Deltas, observed,
                initial=x0, sigma=sigma, **kwargs,
            )
        except Exception:
            continue

        if result["cost"] < best_cost:
            best_cost = result["cost"]
            best_result = result

    if best_result is None:
        raise RuntimeError("All optimisation attempts failed")

    best_result["n_starts"] = n_starts
    return best_result
