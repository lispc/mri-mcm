# MCM: Multi-Compartment Microstructure Model for DWI

A Python implementation of a **5-compartment microstructure model** for diffusion-weighted MRI (DWI), designed to distinguish **microglia** from **astrocytes** via their distinct cell body sizes and process orientations.

> This project implements the modeling framework described in *Canals et al. (2022)* and addresses the notorious numerical instability of sphere-restricted diffusion formulas through a displacement-correlation-function approach.

---

## Overview

The model decomposes the DWI signal into five biologically interpretable compartments:

| Compartment | Symbol | Biophysical Meaning | Key Parameters |
|-------------|--------|---------------------|----------------|
| **Intracellular Processes** | IC | Microglial cell processes (sticks with Watson-distributed orientations) | `f_ic`, `k`, `μ` |
| **Small Spheres** | SS | Microglial cell bodies | `f_ss`, `R_ss` |
| **Large Spheres** | LS | Astrocytic cell bodies | `f_ls`, `R_ls` |
| **Extracellular Space** | EC | Diffusion between cells (anisotropic tensor) | `f_ec`, `d_∥`, `d_⊥` |
| **Free Water** | FW | CSF / unbound water | `1-f_T` (fixed `d_iso=3.0×10⁻³`) |

$$
S = f_{IC} \cdot S_{IC} + f_{SS} \cdot S_{SS} + f_{LS} \cdot S_{LS} + f_{EC} \cdot S_{EC} + (1-f_T) \cdot S_{FW}
$$

**Free parameters per voxel:** 12 (volume fractions, orientation dispersion, sphere radii, EC diffusivities)

---

## Key Features

### 🔬 Biologically-Driven Design
- **Watson-distributed sticks** model microglial process orientations with concentration parameter `k`
- **Two sphere sizes** (2–6 μm and 6–12 μm) capture the distinct soma sizes of microglia and astrocytes
- **Multi-b, multi-Δ acquisition** leverages 240 measurements per voxel (30 directions × 2 b-values × 4 diffusion times)

### 🛡️ Numerically Stable Sphere-Restricted Diffusion
After extensive testing, we found that **direct implementations of classic literature formulas** (Neuman 1974, Murday-Cotts, Stepišnik) suffer from catastrophic numerical issues:

| Formula | Problem | Symptom |
|---------|---------|---------|
| Neuman direct series | `cosh(350)` overflow | `inf` at typical parameters |
| Murday-Cotts | Coefficient mismatch | Signal ≈ 1 regardless of parameters |
| Stepišnik | Exponential cancellation | `NaN` in intermediate terms |

**Our solution** uses the **displacement correlation function** (Grebenkov 2007 framework), deriving the signal from analytically integrated eigenfunction expansions. Verified against Monte Carlo random-walk simulations (N=20,000, error < 3%).

> 📖 See [`docs/sphere_restricted_derivation.md`](docs/sphere_restricted_derivation.md) for the full derivation and disaster log.

### ⚡ Performance-Optimized
- **Sphere grid caching** (`@lru_cache`) eliminates repeated 60×60 grid generation
- **Batch evaluation** groups signals by unique `b`-value for vectorized computation
- **3.3× speedup** over naive implementation:

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| `mcm_signal_batch(60 pts)` | 8.2 ms | 2.5 ms | **3.3×** |
| `fit_mcm(500 evals)` | 3.5 s | 1.18 s | **3.0×** |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd mcm_project

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:** NumPy ≥ 2.0, SciPy ≥ 1.10, Matplotlib ≥ 3.8

---

## Quick Start

### Forward Model — Simulate DWI Signal

```python
from src.mcm_forward import MCMParameters, make_acquisition_scheme, simulate_mcm_data

# Define microstructural parameters
params = MCMParameters(
    f_ic=0.30,          # Microglial processes: 30%
    k=8.0,              # Moderately aligned
    mu=(1.0, 0.0, 0.0), # Along x-axis
    f_ss=0.10, R_ss=4.0,   # Microglial soma: 10%, 4 μm
    f_ls=0.10, R_ls=8.0,   # Astrocytic soma: 10%, 8 μm
    f_ec=0.20,              # Extracellular: 20%
    d_parallel_ec=1.2e-3,   # EC parallel diffusivity (mm²/s)
    d_perp_ec=0.6e-3,       # EC perpendicular diffusivity (mm²/s)
    f_T=0.80,               # Tissue water fraction
)

# Create acquisition scheme: 30 dirs × 2 b-values × 4 Δ-times
scheme = make_acquisition_scheme(n_dirs=30, b_values=(2000, 4000))

# Simulate signal (240 measurements)
signal = simulate_mcm_data(params, **scheme, delta=5.0, Delta_list=[15, 25, 40, 60])
print(signal.shape)  # (240,)
```

### Parameter Fitting — Recover Microstructure

```python
from src.mcm_fit import fit_mcm_multi_start

# Fit to observed data with multiple random starts
result = fit_mcm_multi_start(
    **scheme,
    delta=5.0,
    Delta_list=[15, 25, 40, 60],
    observed=signal,        # Your normalized DWI data
    n_starts=3,             # 3 random initializations
    max_nfev=2000,          # Max function evaluations
    seed=42,
)

fitted = result["params"]
print(f"f_ic = {fitted.f_ic:.3f}, k = {fitted.k:.2f}, R_ss = {fitted.R_ss:.1f} μm")
```

---

## Project Structure

```
mcm_project/
├── src/                          # Core implementation
│   ├── mcm_forward.py            # 5-compartment forward model + simulator
│   ├── mcm_fit.py                # Nonlinear least-squares fitting
│   ├── watson_stick.py           # Watson-distributed stick compartment
│   ├── sphere_restricted.py      # Sphere-restricted diffusion (PGSE)
│   ├── extracellular_tensor.py   # Anisotropic EC tensor compartment
│   ├── free_water.py             # Free water (isotropic) compartment
│   └── utils.py                  # Sphere sampling utilities (cached)
│
├── tests/                        # Test suite (94 tests, all passing)
│   ├── test_end_to_end.py        # Full parameter recovery validation
│   ├── test_mcm_forward.py       # Forward model correctness
│   ├── test_mcm_fit.py           # Fitting convergence
│   ├── test_sphere_restricted.py # Sphere diffusion vs random walk
│   ├── test_watson_stick.py      # Watson distribution limits
│   ├── test_extracellular_tensor.py
│   └── test_free_water.py
│
├── docs/                         # Documentation
│   ├── design.md                 # Mathematical definitions & parameter table
│   ├── TECHNICAL_REPORT.md       # Parameter recovery audit, code quality, speed
│   ├── sphere_restricted_derivation.md  # Derivation & formula disaster log
│   ├── fitting_scale_and_data.md # Data scale, voxel counts, compute estimates
│   └── mcm_visual_summary.png    # Model schematic
│
├── notebooks/                    # (Jupyter notebooks — TBD)
├── scripts/
│   └── visualize_mcm.py          # Generate model schematic figure
├── requirements.txt
└── README.md                     # This file
```

---

## Running Tests

```bash
# Run full test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

All 94 tests pass, including:
- **End-to-end parameter recovery**: Ground-truth parameters recovered within tolerances (f_ic ±0.05, R ±1 μm, k within factor 2)
- **Sphere diffusion validation**: Analytic formula matches random-walk simulation (< 3% error)
- **Boundary condition tests**: Free-diffusion limit ($R \to \infty$), strongly-restricted limit ($R \to 0$)
- **Watson distribution limits**: $k \to 0$ (isotropic) and $k \to \infty$ (single stick)

---

## Documentation Index

| Document | What You'll Find |
|----------|-----------------|
| [`docs/design.md`](docs/design.md) | Full mathematical definitions, parameter ranges, acquisition scheme details |
| [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) | Code quality audit, parameter recovery results, fitting speed benchmarks, known limitations |
| [`docs/sphere_restricted_derivation.md`](docs/sphere_restricted_derivation.md) | **Why literature formulas fail** and how the displacement-correlation-function method solves it |
| [`docs/fitting_scale_and_data.md`](docs/fitting_scale_and_data.md) | Data scale (240 pts/voxel), brain volume estimates (~100k voxels), compute time (1.2 s/voxel), output tensor dimensions |

---

## Parameter Summary

| Parameter | Symbol | Range | Default | Biology |
|-----------|--------|-------|---------|---------|
| IC volume fraction | `f_ic` | [0, 1] | 0.30 | Microglial process density |
| Watson concentration | `k` | [0, 30] | 5.0 | Process alignment (0=isotropic, ∞=perfect) |
| Small sphere fraction | `f_ss` | [0, 1] | 0.10 | Microglial soma density |
| Small sphere radius | `R_ss` | [2, 6] μm | 4.0 μm | Microglial soma size |
| Large sphere fraction | `f_ls` | [0, 1] | 0.10 | Astrocytic soma density |
| Large sphere radius | `R_ls` | [6, 12] μm | 8.0 μm | Astrocytic soma size |
| EC volume fraction | `f_ec` | [0, 1] | 0.20 | Extracellular space |
| EC parallel diffusivity | `d_parallel_ec` | [0.5, 2.0]×10⁻³ mm²/s | 1.2×10⁻³ | Along-main-axis diffusion |
| EC perpendicular diffusivity | `d_perp_ec` | [0.1, 1.0]×10⁻³ mm²/s | 0.6×10⁻³ | Across-fiber diffusion |
| Tissue water fraction | `f_T` | [0, 1] | 0.80 | Total bound water |

---

## Performance & Scale

| Scale | Voxels | Single-Core | 16-Core Parallel |
|-------|--------|-------------|------------------|
| ROI debugging | 1,000 | 20 min | 1.5 min |
| Half-brain | 10,000 | 3.3 h | 12 min |
| Full brain | 100,000 | 33 h | 2 h |

> Per-voxel fitting time: **~1.2 s** (optimized Python). Embarrassingly parallel per voxel.

See [`docs/fitting_scale_and_data.md`](docs/fitting_scale_and_data.md) for full details.

---

## Citation

If you use this code, please cite:

> Canals, S., et al. (2022). *Mapping microglia and astrocyte activation in vivo using diffusion MRI*. Nature Communications.

The sphere-restricted diffusion implementation follows the displacement-correlation-function framework described in:

> Grebenkov, D.S. (2007). *NMR survey of reflected Brownian motion*. Reviews of Modern Physics, 79(3), 1077.

---

## License

MIT License — see `LICENSE` file for details.
