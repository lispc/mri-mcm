#!/usr/bin/env python3
"""
Visualisation script for the MCM model.

Generates a 2×2 figure showing:
1. Compartment-wise signal decay vs b-value
2. Sphere-restricted diffusion vs radius
3. Watson-Stick angular dependence
4. Full MCM signal on a single shell

Run from project root:
    PYTHONPATH=src python scripts/visualize_mcm.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from mcm_forward import MCMParameters, mcm_signal, mcm_signal_batch, make_acquisition_scheme
from watson_stick import watson_stick_signal
from sphere_restricted import sphere_restricted_signal
from extracellular_tensor import extracellular_tensor_signal
from free_water import free_water_signal


def panel_compartment_decay(ax):
    """Plot each compartment's signal decay vs b-value."""
    q = np.array([0.0, 0.0, 1.0])
    mu = np.array([0.0, 0.0, 1.0])
    bvals = np.linspace(0, 4000, 100)
    delta, Delta = 5.0, 40.0

    s_ic = [watson_stick_signal(q, mu, b, 5.0, 1.0e-3) for b in bvals]
    s_ss = [sphere_restricted_signal(b, delta, Delta, 4.0) for b in bvals]
    s_ls = [sphere_restricted_signal(b, delta, Delta, 8.0) for b in bvals]
    s_ec = [extracellular_tensor_signal(q, mu, b, 1.2e-3, 0.6e-3) for b in bvals]
    s_fw = [free_water_signal(b, 3.0) for b in bvals]

    ax.semilogy(bvals, s_ic, label='IC (Watson+Stick)', lw=2)
    ax.semilogy(bvals, s_ss, label='SS (R=4μm)', lw=2)
    ax.semilogy(bvals, s_ls, label='LS (R=8μm)', lw=2)
    ax.semilogy(bvals, s_ec, label='EC (tensor)', lw=2)
    ax.semilogy(bvals, s_fw, label='FW (free)', lw=2)
    ax.set_xlabel('b-value (s/mm²)')
    ax.set_ylabel('Signal S/S₀')
    ax.set_title('Compartment Signal Decay')
    ax.legend(loc='lower left')
    ax.set_ylim(1e-4, 1.5)


def panel_sphere_vs_radius(ax):
    """Show sphere-restricted signal for varying R."""
    radii = np.linspace(1, 15, 100)
    bvals = [500, 1000, 2000, 4000]
    delta, Delta = 5.0, 40.0
    colors = plt.cm.viridis(np.linspace(0, 1, len(bvals)))

    for b, c in zip(bvals, colors):
        signals = [sphere_restricted_signal(b, delta, Delta, R) for R in radii]
        ax.plot(radii, signals, color=c, lw=2, label=f'b={b}')

    ax.axhline(np.exp(-bvals[-1] * 1e-3), color='gray', ls='--', alpha=0.5,
               label='Free diffusion (b=4000)')
    ax.set_xlabel('Sphere radius R (μm)')
    ax.set_ylabel('Signal S/S₀')
    ax.set_title('Sphere Restricted Diffusion')
    ax.legend(loc='lower right')


def panel_watson_angular(ax):
    """Watson-Stick signal as a function of angle for different k."""
    mu = np.array([0.0, 0.0, 1.0])
    b = 2000.0
    angles_deg = np.linspace(0, 90, 91)
    k_values = [0, 2, 5, 10, 30]
    colors = plt.cm.plasma(np.linspace(0, 1, len(k_values)))

    for k, c in zip(k_values, colors):
        signals = []
        for ang in angles_deg:
            theta = np.radians(ang)
            q = np.array([np.sin(theta), 0.0, np.cos(theta)])
            s = watson_stick_signal(q, mu, b, k, 1.0e-3)
            signals.append(s)
        ax.plot(angles_deg, signals, color=c, lw=2, label=f'k={k}')

    ax.set_xlabel('Angle q–μ (degrees)')
    ax.set_ylabel('Signal S/S₀')
    ax.set_title('Watson-Stick Angular Profile')
    ax.legend(loc='lower left')


def panel_full_mcm_shell(ax):
    """Full MCM signal on one b-shell shown as a function of angle."""
    mu = np.array([0.0, 0.0, 1.0])
    params = MCMParameters(
        f_ic=0.3, k=5.0, mu=mu, d_parallel_ic=1.0,
        f_ss=0.1, R_ss=4.0,
        f_ls=0.1, R_ls=8.0,
        f_ec=0.2, d_parallel_ec=1.2e-3, d_perp_ec=0.6e-3,
        f_T=0.8, d_iso=3.0,
    )
    b = 2000.0
    delta, Delta = 5.0, 40.0

    angles_deg = np.linspace(0, 90, 91)
    signals = []
    for ang in angles_deg:
        theta = np.radians(ang)
        q = np.array([np.sin(theta), 0.0, np.cos(theta)])
        s = mcm_signal(q, b, delta, Delta, params)
        signals.append(s)

    ax.plot(angles_deg, signals, 'k-', lw=2.5, label='Full MCM')

    # Overlay individual compartments (with their volume fractions)
    s_ic = [params.f_ic * watson_stick_signal(
        np.array([np.sin(np.radians(a)), 0, np.cos(np.radians(a))]),
        mu, b, params.k, 1.0e-3) for a in angles_deg]
    s_ss = [params.f_ss * sphere_restricted_signal(b, delta, Delta, params.R_ss)] * len(angles_deg)
    s_ls = [params.f_ls * sphere_restricted_signal(b, delta, Delta, params.R_ls)] * len(angles_deg)
    s_ec = [params.f_ec * extracellular_tensor_signal(
        np.array([np.sin(np.radians(a)), 0, np.cos(np.radians(a))]),
        mu, b, params.d_parallel_ec * 1e-3, params.d_perp_ec * 1e-3) for a in angles_deg]
    s_fw = [(1 - params.f_T) * free_water_signal(b, params.d_iso)] * len(angles_deg)

    ax.plot(angles_deg, s_ic, '--', alpha=0.6, label='IC (×f_ic)')
    ax.plot(angles_deg, s_ss, '--', alpha=0.6, label='SS (×f_ss)')
    ax.plot(angles_deg, s_ls, '--', alpha=0.6, label='LS (×f_ls)')
    ax.plot(angles_deg, s_ec, '--', alpha=0.6, label='EC (×f_ec)')
    ax.plot(angles_deg, s_fw, '--', alpha=0.6, label='FW (×f_fw)')

    ax.set_xlabel('Angle q–μ (degrees)')
    ax.set_ylabel('Signal S/S₀')
    ax.set_title(f'Full MCM (b={b}, Δ={Delta}ms)')
    ax.legend(loc='lower left', fontsize=7)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panel_compartment_decay(axes[0, 0])
    panel_sphere_vs_radius(axes[0, 1])
    panel_watson_angular(axes[1, 0])
    panel_full_mcm_shell(axes[1, 1])

    fig.suptitle('MCM 5-Compartment Model — Visual Summary', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'mcm_visual_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
