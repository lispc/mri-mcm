"""Tests for free_water.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from free_water import free_water_signal


class TestValidation:
    def test_negative_b_raises(self):
        with pytest.raises(ValueError):
            free_water_signal(-1.0)

    def test_negative_d_iso_raises(self):
        with pytest.raises(ValueError):
            free_water_signal(1000.0, -1.0)


class TestSignal:
    def test_default_d_iso(self):
        """Default d_iso = 3.0 × 10⁻³ mm²/s."""
        b = 2000.0
        s = free_water_signal(b)
        assert s == pytest.approx(np.exp(-b * 3.0e-3))

    def test_custom_d_iso(self):
        b = 1000.0
        d_iso = 2.0
        s = free_water_signal(b, d_iso)
        assert s == pytest.approx(np.exp(-b * 2.0e-3))

    def test_b_zero(self):
        assert free_water_signal(0.0) == pytest.approx(1.0)

    def test_high_b_attenuation(self):
        """At b=4000, free-water signal should be very small."""
        s = free_water_signal(4000.0)
        assert s < 1e-4

    def test_signal_range(self):
        """Signal must always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            b = rng.uniform(0, 5000)
            d_iso = rng.uniform(1.0, 4.0)
            s = free_water_signal(b, d_iso)
            assert 0.0 <= s <= 1.0

    def test_b_increase_decreases_signal(self):
        """Higher b → more attenuation → lower signal."""
        bvals = [500.0, 1000.0, 2000.0, 4000.0]
        signals = [free_water_signal(b) for b in bvals]
        for i in range(len(signals) - 1):
            assert signals[i] > signals[i + 1]

    def test_d_iso_increase_decreases_signal(self):
        """Higher d_iso → more attenuation → lower signal."""
        disos = [1.0, 2.0, 3.0, 4.0]
        signals = [free_water_signal(2000.0, d) for d in disos]
        for i in range(len(signals) - 1):
            assert signals[i] > signals[i + 1]
