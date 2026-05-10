"""
Sphere Restricted Diffusion Compartment for MCM

Implements the PGSE (pulsed gradient spin echo) signal attenuation for
water molecules diffusing within a reflecting spherical boundary.

Physics:
    The model is based on the eigenfunction expansion of the diffusion
    propagator in a sphere with reflecting boundary conditions.
    The signal is computed using the displacement correlation function
    approach, which avoids the numerical instabilities of direct Neuman
    series formulas found in the literature.

    Key equations (1D projection along gradient direction):
        ln(S/S_0) = γ² G² Σ_n B_n · (J₁ₙ - J₃ₙ)

        B_n     = 2R² / [α_n² (α_n² - 2)]
        λ_n     = α_n² D / R²

        J₁ₙ - J₃ₙ = [2(1 - λ_nδ - e^{-λ_nδ})
                     + e^{-λ_n(Δ-δ)} - 2e^{-λ_nΔ} + e^{-λ_n(Δ+δ)}] / λ_n²

    where α_n are the roots of j₁'(α) = 0 (spherical Bessel function
    derivative), ordered by increasing value.

    In the free-diffusion limit (R → ∞):
        Σ B_n λ_n = D  →  S/S_0 → exp(-bD)

    In the strongly restricted limit (R → 0 or Δ → ∞):
        S/S_0 → 1  (no net displacement along gradient)

References:
    - Neuman CH (1974) Spin echo of spins diffusing in a bounded medium.
      J Chem Phys 60:4508-4511.
    - Murday JS, Cotts RM (1968) Self-diffusion coefficient of liquid
      lithium. J Chem Phys 48:4938-4945.
    - Grebenkov DS (2007) NMR survey of reflected Brownian motion.
      Rev Mod Phys 79:1077-1137.

Units (input interface):
    b       : s/mm²
    delta   : ms  (gradient pulse duration)
    Delta   : ms  (diffusion time)
    R       : μm  (sphere radius)
    D       : μm²/ms  (diffusion coefficient)
"""

import numpy as np
from numpy.polynomial import polynomial as P

# ---------------------------------------------------------------------------
# Cached eigenvalues
# ---------------------------------------------------------------------------
_ALPHA_R_CACHE = None


def _j1_prime(y: float) -> float:
    """Derivative of spherical Bessel function j₁(y)."""
    if abs(y) < 1e-10:
        return 1.0
    return (y * y - 2.0) * np.sin(y) / (y ** 3) + 2.0 * np.cos(y) / (y * y)


def _precompute_sphere_eigenvalues(n_max: int = 20) -> np.ndarray:
    """Return first n_max roots of j₁'(α) = 0."""
    global _ALPHA_R_CACHE
    if _ALPHA_R_CACHE is not None and len(_ALPHA_R_CACHE) >= n_max:
        return _ALPHA_R_CACHE[:n_max]

    from scipy.optimize import brentq

    roots = []
    # Search on a dense grid up to (n_max + 2) * π
    y_max = (n_max + 2) * np.pi
    search = np.linspace(0.1, y_max, int(y_max * 100))

    for i in range(len(search) - 1):
        a, b = search[i], search[i + 1]
        fa, fb = _j1_prime(a), _j1_prime(b)
        if fa == 0.0 or fb == 0.0 or fa * fb < 0.0:
            try:
                root = brentq(_j1_prime, a, b)
                if root > 0.1 and all(abs(root - r) > 0.01 for r in roots):
                    roots.append(root)
                    if len(roots) >= n_max:
                        break
            except ValueError:
                pass

    _ALPHA_R_CACHE = np.array(sorted(roots))
    return _ALPHA_R_CACHE[:n_max]


# ---------------------------------------------------------------------------
# Core signal function
# ---------------------------------------------------------------------------
def sphere_restricted_signal(
    b: float,
    delta: float,
    Delta: float,
    R: float,
    D: float = 1.0,
    n_terms: int = 10,
) -> float:
    """
    Compute the PGSE DWI signal for restricted diffusion in a sphere.

    Parameters
    ----------
    b : float
        b-value in s/mm².
    delta : float
        Gradient pulse duration δ in milliseconds.
    Delta : float
        Diffusion time Δ in milliseconds (time between gradient onsets).
    R : float
        Sphere radius in micrometers (μm).
    D : float, optional
        Diffusion coefficient in μm²/ms. Default is 1.0 (typical for
        intracellular water at 37 °C).
    n_terms : int, optional
        Number of eigenmodes to sum. Default 10 is sufficient for
        all practical parameter ranges in MCM (R = 2–12 μm,
        Δ = 5–60 ms).  Convergence is typically reached with n_terms=5.

    Returns
    -------
    float
        Normalised signal S/S₀ in the range (0, 1].

    Raises
    ------
    ValueError
        If any parameter is non-physical (negative or zero where
        inadmissible).
    """
    # --- validation ---------------------------------------------------------
    if b < 0:
        raise ValueError("b must be non-negative")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if Delta <= 0:
        raise ValueError("Delta must be positive")
    if Delta < delta:
        raise ValueError("Delta must be >= delta (PGSE sequence constraint)")
    if R <= 0:
        raise ValueError("R must be positive")
    if D <= 0:
        raise ValueError("D must be positive")
    if n_terms < 1:
        raise ValueError("n_terms must be at least 1")

    # trivial case
    if b == 0.0:
        return 1.0

    # --- unit conversion ----------------------------------------------------
    gamma = 2.675e8          # rad s⁻¹ T⁻¹
    R_m = R * 1e-6           # m
    D_m = D * 1e-9           # m² s⁻¹
    delta_s = delta * 1e-3   # s
    Delta_s = Delta * 1e-3   # s

    # gradient strength from b-value (b in s/m² after conversion)
    b_si = b * 1e6
    denom = gamma ** 2 * delta_s ** 2 * (Delta_s - delta_s / 3.0)
    G = np.sqrt(b_si / denom)

    # --- eigenmodes ---------------------------------------------------------
    alpha_R = _precompute_sphere_eigenvalues(n_max=n_terms)

    total = 0.0
    for i in range(min(n_terms, len(alpha_R))):
        ar = alpha_R[i]
        ar2 = ar * ar

        # B_n = 2 R² / [α_n² (α_n² - 2)]
        B = 2.0 * R_m * R_m / (ar2 * (ar2 - 2.0))

        # λ_n = α_n² D / R²
        lam = ar2 * D_m / (R_m * R_m)

        lam_d = lam * delta_s
        lam_D = lam * Delta_s
        lam_Dmd = lam * (Delta_s - delta_s)
        lam_Dpd = lam * (Delta_s + delta_s)

        # Numerator of (J₁ₙ - J₃ₙ) – written in an overflow-safe form
        if lam_d < 1e-4:
            # Series expansion for tiny λδ to avoid catastrophic cancellation
            # 1 - x - e^{-x} = -x²/2 + x³/6 - x⁴/24 + x⁵/120 - ...
            x = lam_d
            poly = x * x * (-0.5 + x / 6.0 - x * x / 24.0 + x ** 3 / 120.0)
            term = 2.0 * poly
            # e^{-y} ≈ 1 - y + y²/2 - y³/6 for the cross terms
            term += (
                (1.0 - lam_Dmd + lam_Dmd * lam_Dmd / 2.0 - lam_Dmd ** 3 / 6.0)
                - 2.0 * (1.0 - lam_D + lam_D * lam_D / 2.0 - lam_D ** 3 / 6.0)
                + (1.0 - lam_Dpd + lam_Dpd * lam_Dpd / 2.0 - lam_Dpd ** 3 / 6.0)
            )
        else:
            term = (
                2.0 * (1.0 - lam_d - np.exp(-lam_d))
                + np.exp(-lam_Dmd)
                - 2.0 * np.exp(-lam_D)
                + np.exp(-lam_Dpd)
            )

        total += B * term / (lam * lam)

    lnS = gamma ** 2 * G ** 2 * total
    return float(np.exp(lnS))
