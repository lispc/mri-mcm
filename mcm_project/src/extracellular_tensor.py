"""
Extracellular Tensor Compartment for MCM

Implements the DWI signal for anisotropic Gaussian diffusion in the
extracellular space, modelled as a radially symmetric diffusion tensor
aligned with the principal direction μ.

Physics:
    The extracellular space is approximated as a cylindrically symmetric
    (prolate) diffusion tensor with eigenvalues d_∥ and d_⊥:

        S_EC(q, b) = exp( -b · [ d_∥ (q·μ)² + d_⊥ (1 - (q·μ)²) ] )

    where q is the unit gradient direction and μ is the unit principal
    direction of the tensor.

    Special cases:
    - q ∥ μ  →  S = exp(-b · d_∥)     (max attenuation)
    - q ⟂ μ  →  S = exp(-b · d_⊥)     (min attenuation)
    - d_∥ = d_⊥  →  isotropic Gaussian, S = exp(-b · D)

Reference:
    - Canals et al. (2022) J Neurosci — MCM 5-compartment model for glia.

Units (input interface):
    b           : s/mm²
    d_parallel  : mm²/s  (diffusivity along μ)
    d_perp      : mm²/s  (diffusivity perpendicular to μ)
"""

import numpy as np


def extracellular_tensor_signal(
    q: np.ndarray,
    mu: np.ndarray,
    b: float,
    d_parallel: float,
    d_perp: float,
) -> float:
    """
    Compute the DWI signal for the extracellular tensor compartment.

    Parameters
    ----------
    q : np.ndarray, shape (3,)
        Unit gradient direction vector.
    mu : np.ndarray, shape (3,)
        Unit principal direction of the diffusion tensor.
    b : float
        b-value in s/mm².
    d_parallel : float
        Diffusivity parallel to μ in mm²/s.
    d_perp : float
        Diffusivity perpendicular to μ in mm²/s.

    Returns
    -------
    float
        Normalised signal S/S₀ in the range (0, 1].

    Raises
    ------
    ValueError
        If parameters are non-physical.
    """
    # --- validation ---------------------------------------------------------
    if b < 0:
        raise ValueError("b must be non-negative")
    if d_parallel < 0 or d_perp < 0:
        raise ValueError("diffusivities must be non-negative")

    q = np.asarray(q, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if q.shape != (3,) or mu.shape != (3,):
        raise ValueError("q and mu must be 3-element vectors")

    # normalise (defensive, in case caller passes non-unit vectors)
    q_norm = q / (np.linalg.norm(q) + 1e-15)
    mu_norm = mu / (np.linalg.norm(mu) + 1e-15)

    if b == 0.0:
        return 1.0

    # --- signal -------------------------------------------------------------
    cos_sq = float(np.dot(q_norm, mu_norm) ** 2)
    adc = d_parallel * cos_sq + d_perp * (1.0 - cos_sq)
    return float(np.exp(-b * adc))
