"""
MCM Forward Model — Multi-Compartment Microstructure Model

Combines five tissue compartments into a single DWI signal:

    S = f_IC·S_IC + f_SS·S_SS + f_LS·S_LS + f_EC·S_EC + (1-f_T)·S_FW

Compartments:
    S_IC  — Intracellular stick + Watson distribution (microglia processes)
    S_SS  — Small-sphere restricted diffusion (microglia soma)
    S_LS  — Large-sphere restricted diffusion (astrocyte soma)
    S_EC  — Extracellular anisotropic tensor
    S_FW  — Free water (isotropic)

Volume constraint (enforced externally during fitting):
    f_IC + f_SS + f_LS + f_EC + (1-f_T) = 1

Reference:
    Canals et al. (2022) J Neurosci — MCM 5-compartment model for glia.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np

from watson_stick import watson_stick_signal, watson_stick_signal_batch
from sphere_restricted import sphere_restricted_signal
from extracellular_tensor import extracellular_tensor_signal, extracellular_tensor_signal_batch
from free_water import free_water_signal


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------
@dataclass
class MCMParameters:
    """Container for all MCM model parameters.

    Fields are stored in *physical* units (μm, ms, mm²/s, etc.).
    The forward model handles all SI conversions internally.
    """
    # --- Intracellular (Watson + Stick) ---
    f_ic: float = 0.3           # volume fraction [0, 1]
    k: float = 5.0              # Watson concentration [0, ∞)
    mu: np.ndarray = None       # principal direction (3,) unit vector
    d_parallel_ic: float = 1.0e-3  # mm²/s

    # --- Small sphere (microglia soma) ---
    f_ss: float = 0.1           # volume fraction [0, 1]
    R_ss: float = 4.0           # μm

    # --- Large sphere (astrocyte soma) ---
    f_ls: float = 0.1           # volume fraction [0, 1]
    R_ls: float = 8.0           # μm

    # --- Extracellular tensor ---
    f_ec: float = 0.2           # volume fraction [0, 1]
    d_parallel_ec: float = 1.2e-3  # mm²/s
    d_perp_ec: float = 0.6e-3      # mm²/s

    # --- Free water ---
    f_T: float = 0.8            # total tissue water fraction [0, 1]
    d_iso: float = 3.0e-3       # mm²/s (CSF at 37 °C)

    def __post_init__(self):
        if self.mu is None:
            self.mu = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            self.mu = np.asarray(self.mu, dtype=float)
            self.mu = self.mu / (np.linalg.norm(self.mu) + 1e-15)

    def volume_sum(self) -> float:
        """Sum of all compartment volume fractions."""
        return self.f_ic + self.f_ss + self.f_ls + self.f_ec + (1.0 - self.f_T)


# ---------------------------------------------------------------------------
# Single-point signal
# ---------------------------------------------------------------------------
def mcm_signal(
    q: np.ndarray,
    b: float,
    delta: float,
    Delta: float,
    params: MCMParameters,
) -> float:
    """
    Compute the MCM signal for a single acquisition point.

    Parameters
    ----------
    q : np.ndarray, shape (3,)
        Unit gradient direction.
    b : float
        b-value in s/mm².
    delta : float
        Gradient pulse duration δ in milliseconds.
    Delta : float
        Diffusion time Δ in milliseconds.
    params : MCMParameters
        Model parameters.

    Returns
    -------
    float
        Predicted normalised signal S/S₀.
    """
    # --- Intracellular (Watson + Stick) ---
    s_ic = watson_stick_signal(
        q=q,
        mu=params.mu,
        b=b,
        k=params.k,
        d_parallel=params.d_parallel_ic,
    )

    # --- Small sphere ---
    s_ss = sphere_restricted_signal(
        b=b,
        delta=delta,
        Delta=Delta,
        R=params.R_ss,
        D=1.0,  # intracellular D fixed
    )

    # --- Large sphere ---
    s_ls = sphere_restricted_signal(
        b=b,
        delta=delta,
        Delta=Delta,
        R=params.R_ls,
        D=1.0,
    )

    # --- Extracellular tensor ---
    s_ec = extracellular_tensor_signal(
        q=q,
        mu=params.mu,
        b=b,
        d_parallel=params.d_parallel_ec,
        d_perp=params.d_perp_ec,
    )

    # --- Free water ---
    s_fw = free_water_signal(b=b, d_iso=params.d_iso)

    # --- weighted sum ---
    total = (
        params.f_ic * s_ic
        + params.f_ss * s_ss
        + params.f_ls * s_ls
        + params.f_ec * s_ec
        + (1.0 - params.f_T) * s_fw
    )
    return float(np.clip(total, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Batch signal
# ---------------------------------------------------------------------------
def mcm_signal_batch(
    q_dirs: np.ndarray,
    bvals: np.ndarray,
    deltas: np.ndarray,
    Deltas: np.ndarray,
    params: MCMParameters,
) -> np.ndarray:
    """
    Compute MCM signals for a batch of acquisition points.

    Parameters
    ----------
    q_dirs : np.ndarray, shape (N, 3)
        Gradient direction vectors (will be normalised).
    bvals : np.ndarray, shape (N,)
        b-values in s/mm².
    deltas : np.ndarray, shape (N,)
        Gradient pulse durations δ in milliseconds.
    Deltas : np.ndarray, shape (N,)
        Diffusion times Δ in milliseconds.
    params : MCMParameters
        Model parameters.

    Returns
    -------
    np.ndarray, shape (N,)
        Predicted signals S/S₀.
    """
    q_dirs = np.asarray(q_dirs, dtype=float)
    bvals = np.asarray(bvals, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    Deltas = np.asarray(Deltas, dtype=float)

    n = len(bvals)
    if q_dirs.shape != (n, 3):
        raise ValueError(f"q_dirs shape {q_dirs.shape} incompatible with N={n}")
    if len(deltas) != n or len(Deltas) != n:
        raise ValueError("All input arrays must have the same length")

    # Normalise directions
    q_norm = q_dirs / (np.linalg.norm(q_dirs, axis=1, keepdims=True) + 1e-15)

    # Pre-compute isotropic compartment signals (spheres + free water)
    s_ss_all = np.array([
        sphere_restricted_signal(bvals[i], deltas[i], Deltas[i], params.R_ss)
        for i in range(n)
    ])
    s_ls_all = np.array([
        sphere_restricted_signal(bvals[i], deltas[i], Deltas[i], params.R_ls)
        for i in range(n)
    ])
    s_fw_all = np.array([
        free_water_signal(bvals[i], params.d_iso)
        for i in range(n)
    ])

    # Group by unique (b, delta, Delta) to batch anisotropic compartments
    # For now, we batch Watson-Stick and EC per unique b (most common case)
    unique_b = np.unique(bvals)
    s_ic_all = np.zeros(n, dtype=float)
    s_ec_all = np.zeros(n, dtype=float)

    for b_u in unique_b:
        mask = bvals == b_u
        q_batch = q_norm[mask]
        # Batch Watson-Stick for all directions at this b
        s_ic_batch = watson_stick_signal_batch(
            q_batch, params.mu, b_u, params.k, params.d_parallel_ic,
        )
        s_ic_all[mask] = s_ic_batch
        # Batch EC tensor for all directions at this b
        s_ec_batch = extracellular_tensor_signal_batch(
            q_batch, params.mu, b_u, params.d_parallel_ec, params.d_perp_ec,
        )
        s_ec_all[mask] = s_ec_batch

    signals = (
        params.f_ic * s_ic_all
        + params.f_ss * s_ss_all
        + params.f_ls * s_ls_all
        + params.f_ec * s_ec_all
        + (1.0 - params.f_T) * s_fw_all
    )
    return np.clip(signals, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Acquisition scheme helpers
# ---------------------------------------------------------------------------
def make_acquisition_scheme(
    n_dirs: int = 30,
    b_values: tuple = (2000.0, 4000.0),
    Delta_values: tuple = (15.0, 25.0, 40.0, 60.0),
    delta_ms: float = 5.0,
    seed: int = 42,
) -> dict:
    """
    Generate a standard DWI acquisition scheme.

    Returns a dict with keys:
        'q_dirs'  : (N, 3)  gradient directions
        'bvals'   : (N,)    b-values
        'deltas'  : (N,)    δ values
        'Deltas'  : (N,)    Δ values

    Total number of points = n_dirs × len(b_values) × len(Delta_values).
    """
    rng = np.random.default_rng(seed)

    # Uniform directions on sphere (Fibonacci sphere)
    directions = _fibonacci_sphere(n_dirs)

    # Cartesian product: dirs × b × Delta
    q_list, b_list, d_list, D_list = [], [], [], []
    for b in b_values:
        for Delta in Delta_values:
            for q in directions:
                q_list.append(q)
                b_list.append(b)
                d_list.append(delta_ms)
                D_list.append(Delta)

    return {
        "q_dirs": np.array(q_list),
        "bvals": np.array(b_list),
        "deltas": np.array(d_list),
        "Deltas": np.array(D_list),
    }


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return n approximately uniform points on the unit sphere."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + np.sqrt(5)) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------
def simulate_mcm_data(
    q_dirs: np.ndarray,
    bvals: np.ndarray,
    deltas: np.ndarray,
    Deltas: np.ndarray,
    params: MCMParameters,
    snr: float = 20.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate noise-corrupted MCM signals with Rician noise.

    The noise model assumes the data are magnitude images.  For SNR defined
    as S₀ / σ (signal-to-noise ratio on the b=0 image):

        noisy = sqrt( (S + n_real)² + n_imag² )

    where n_real, n_imag ~ N(0, σ²) and σ = 1 / SNR (since S₀ = 1).

    Parameters
    ----------
    q_dirs, bvals, deltas, Deltas : arrays
        Acquisition scheme (same as mcm_signal_batch).
    params : MCMParameters
        Ground-truth model parameters.
    snr : float
        Signal-to-noise ratio on the b=0 image.  Default 20.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Noisy signal array with the same shape as the input scheme.
    """
    rng = np.random.default_rng(seed)
    signals = mcm_signal_batch(q_dirs, bvals, deltas, Deltas, params)

    sigma = 1.0 / snr  # noise std on S₀ = 1
    noise_real = rng.normal(0.0, sigma, size=signals.shape)
    noise_imag = rng.normal(0.0, sigma, size=signals.shape)

    noisy = np.sqrt((signals + noise_real) ** 2 + noise_imag ** 2)
    return noisy
