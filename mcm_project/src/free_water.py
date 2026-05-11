"""
Free Water (Isotropic) Compartment for MCM

Implements the DWI signal for unrestricted isotropic Gaussian diffusion,
representing cerebrospinal fluid (CSF) or other free-water compartments.

Physics:
    S_FW(b) = exp(-b · d_iso)

    where d_iso is the isotropic free-water diffusivity.

    Typical values:
    - d_iso = 3.0 × 10⁻³ mm²/s  at 37 °C (CSF)
    - d_iso = 2.0 × 10⁻³ mm²/s  at room temperature

Reference:
    - Canals et al. (2022) J Neurosci — MCM 5-compartment model for glia.

Units (input interface):
    b     : s/mm²
    d_iso : mm²/s  (default 3.0 × 10⁻³)
"""

import numpy as np


def free_water_signal(
    b: float,
    d_iso: float = 3.0e-3,
) -> float:
    """
    Compute the DWI signal for the free-water compartment.

    Parameters
    ----------
    b : float
        b-value in s/mm².
    d_iso : float, optional
        Isotropic diffusivity in mm²/s. Default is 3.0×10⁻³ mm²/s
        (typical for CSF at body temperature).

    Returns
    -------
    float
        Normalised signal S/S₀ in the range (0, 1].

    Raises
    ------
    ValueError
        If parameters are non-physical.
    """
    if b < 0:
        raise ValueError("b must be non-negative")
    if d_iso < 0:
        raise ValueError("d_iso must be non-negative")

    if b == 0.0:
        return 1.0

    return float(np.exp(-b * d_iso))
