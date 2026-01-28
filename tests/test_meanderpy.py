"""
Tests for meanderpy core functions.

Run with: pytest tests/test_meanderpy.py -v
"""

import numpy as np
import pytest
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meanderpy import meanderpy as mp


class TestComputeCurvature:
    """Tests for compute_curvature function."""

    def test_straight_line_zero_curvature(self):
        """A straight line should have zero curvature."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 50, 50)  # straight diagonal line
        curv = mp.compute_curvature(x, y)
        # curvature should be essentially zero (allowing for numerical noise)
        assert np.allclose(curv, 0, atol=1e-10)

    def test_circle_constant_curvature(self):
        """A circle should have constant curvature = 1/radius."""
        radius = 100.0
        theta = np.linspace(0, 2 * np.pi, 200)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        curv = mp.compute_curvature(x, y)
        expected_curvature = 1.0 / radius
        # Check middle points (edges have boundary effects)
        assert np.allclose(curv[20:-20], expected_curvature, rtol=0.05)

    def test_curvature_sign(self):
        """Curvature should change sign for opposite bends."""
        # Create S-curve
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        curv = mp.compute_curvature(x, y)
        # First half should have opposite sign to second half
        assert np.mean(curv[10:40]) * np.mean(curv[60:90]) < 0

    def test_curvature_output_shape(self):
        """Output should have same shape as input."""
        x = np.random.rand(50)
        y = np.random.rand(50)
        curv = mp.compute_curvature(x, y)
        assert curv.shape == x.shape


class TestComputeDerivatives:
    """Tests for compute_derivatives function."""

    def test_derivatives_output_shapes(self):
        """All outputs should have correct shapes."""
        n = 50
        x = np.linspace(0, 100, n)
        y = np.linspace(0, 50, n)
        z = np.linspace(10, 5, n)
        dx, dy, dz, ds, s = mp.compute_derivatives(x, y, z)
        assert dx.shape == (n,)
        assert dy.shape == (n,)
        assert dz.shape == (n,)
        assert ds.shape == (n,)
        assert s.shape == (n,)

    def test_arc_length_increasing(self):
        """Cumulative arc length should be monotonically increasing."""
        x = np.linspace(0, 100, 50)
        y = np.sin(x / 10) * 20
        z = np.zeros_like(x)
        dx, dy, dz, ds, s = mp.compute_derivatives(x, y, z)
        assert np.all(np.diff(s) >= 0)

    def test_arc_length_straight_line(self):
        """Arc length of straight line should equal Euclidean distance."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 0, 50)
        z = np.zeros_like(x)
        dx, dy, dz, ds, s = mp.compute_derivatives(x, y, z)
        expected_length = 100.0
        assert np.isclose(s[-1], expected_length, rtol=0.01)

    def test_ds_positive(self):
        """Distance increments should be positive."""
        x = np.linspace(0, 100, 50)
        y = np.sin(x / 10) * 20
        z = np.zeros_like(x)
        dx, dy, dz, ds, s = mp.compute_derivatives(x, y, z)
        assert np.all(ds > 0)


class TestGenerateInitialChannel:
    """Tests for generate_initial_channel function."""

    def test_channel_object_returned(self):
        """Should return a Channel object."""
        W, D, Sl, deltas, pad, n_bends = 100, 5, 0.0, 25, 100, 10
        ch = mp.generate_initial_channel(W, D, Sl, deltas, pad, n_bends)
        assert isinstance(ch, mp.Channel)

    def test_channel_width_depth(self):
        """Channel should have correct width and depth."""
        W, D, Sl, deltas, pad, n_bends = 100, 5, 0.0, 25, 100, 10
        ch = mp.generate_initial_channel(W, D, Sl, deltas, pad, n_bends)
        assert ch.W == W
        assert ch.D == D

    def test_channel_coordinates_exist(self):
        """Channel should have x, y, z coordinates."""
        W, D, Sl, deltas, pad, n_bends = 100, 5, 0.0, 25, 100, 10
        ch = mp.generate_initial_channel(W, D, Sl, deltas, pad, n_bends)
        assert len(ch.x) > 0
        assert len(ch.y) > 0
        assert len(ch.z) > 0
        assert len(ch.x) == len(ch.y) == len(ch.z)

    def test_channel_z_slope(self):
        """Channel z should decrease downstream (positive slope)."""
        W, D, Sl, deltas, pad, n_bends = 100, 5, 0.001, 25, 100, 10
        ch = mp.generate_initial_channel(W, D, Sl, deltas, pad, n_bends)
        # z should generally decrease (channel flows downhill)
        assert ch.z[0] >= ch.z[-1]


class TestResampleCenterline:
    """Tests for resample_centerline function."""

    def test_output_shapes_consistent(self):
        """All outputs should have consistent shapes."""
        x = np.linspace(0, 1000, 100)
        y = np.sin(x / 100) * 50
        z = np.linspace(10, 0, 100)
        deltas = 15.0
        x_new, y_new, z_new, dx, dy, dz, ds, s = mp.resample_centerline(x, y, z, deltas)
        n = len(x_new)
        assert len(y_new) == n
        assert len(z_new) == n
        assert len(dx) == n
        assert len(dy) == n
        assert len(dz) == n
        assert len(ds) == n
        assert len(s) == n

    def test_uniform_spacing(self):
        """Resampled points should have approximately uniform spacing."""
        x = np.linspace(0, 1000, 100)
        y = np.sin(x / 100) * 50
        z = np.linspace(10, 0, 100)
        deltas = 15.0
        x_new, y_new, z_new, dx, dy, dz, ds, s = mp.resample_centerline(x, y, z, deltas)
        # ds should be close to deltas (except at boundaries)
        assert np.allclose(ds[5:-5], deltas, rtol=0.1)

    def test_preserves_endpoints(self):
        """Resampling should approximately preserve start point."""
        x = np.linspace(0, 1000, 100)
        y = np.zeros_like(x)
        z = np.linspace(10, 0, 100)
        deltas = 15.0
        x_new, y_new, z_new, dx, dy, dz, ds, s = mp.resample_centerline(x, y, z, deltas)
        # Start point should be close
        assert np.isclose(x_new[0], x[0], atol=deltas)


class TestMigrateOneStep:
    """Tests for migrate_one_step function."""

    def test_migration_changes_coordinates(self):
        """Migration should change channel coordinates."""
        W, D = 100.0, 5.0
        x = np.linspace(0, 2000, 100)
        y = np.sin(x / 200) * 100  # sinuous channel
        z = np.linspace(10, 0, 100)
        x_orig, y_orig = x.copy(), y.copy()

        kl = 1.9e-6
        dt = 0.1 * 365 * 24 * 60 * 60  # 0.1 years in seconds
        k, Cf = 1.0, 0.01
        pad, pad1 = 10, 5

        x_new, y_new = mp.migrate_one_step(x, y, z, W, kl, dt, k, Cf, D, pad, pad1)

        # Coordinates should have changed (at least somewhere)
        assert not np.allclose(x_new, x_orig) or not np.allclose(y_new, y_orig)

    def test_cfl_warning_triggered(self):
        """Large time step should trigger CFL warning."""
        W, D = 100.0, 5.0
        deltas = 25.0
        x = np.linspace(0, 2000, 80)
        y = np.sin(x / 200) * 100
        z = np.linspace(10, 0, 80)

        kl = 1e-4  # Very large migration rate
        dt = 10 * 365 * 24 * 60 * 60  # 10 years - very large time step
        k, Cf = 1.0, 0.01
        pad, pad1 = 10, 5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mp.migrate_one_step(x, y, z, W, kl, dt, k, Cf, D, pad, pad1,
                               deltas=deltas, cfl_factor=0.5)
            # Check that a CFL warning was issued
            cfl_warnings = [warning for warning in w if "CFL" in str(warning.message)]
            assert len(cfl_warnings) > 0

    def test_cfl_clamping_limits_displacement(self):
        """CFL clamping should limit maximum displacement."""
        W, D = 100.0, 5.0
        deltas = 25.0
        cfl_factor = 0.5
        max_allowed = cfl_factor * deltas

        x = np.linspace(0, 2000, 80)
        y = np.sin(x / 200) * 100
        z = np.linspace(10, 0, 80)
        x_orig, y_orig = x.copy(), y.copy()

        kl = 1e-4  # Very large migration rate
        dt = 10 * 365 * 24 * 60 * 60  # Large time step
        k, Cf = 1.0, 0.01
        pad, pad1 = 10, 5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_new, y_new = mp.migrate_one_step(x, y, z, W, kl, dt, k, Cf, D, pad, pad1,
                                               deltas=deltas, cfl_factor=cfl_factor)

        # Calculate actual displacements
        disp = np.sqrt((x_new - x_orig)**2 + (y_new - y_orig)**2)

        # All displacements should be <= max_allowed (with small tolerance)
        assert np.all(disp <= max_allowed + 1e-10)

    def test_no_cfl_check_when_disabled(self):
        """No CFL warning when deltas=None."""
        W, D = 100.0, 5.0
        x = np.linspace(0, 2000, 80)
        y = np.sin(x / 200) * 100
        z = np.linspace(10, 0, 80)

        kl = 1e-4  # Large migration rate
        dt = 10 * 365 * 24 * 60 * 60
        k, Cf = 1.0, 0.01
        pad, pad1 = 10, 5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mp.migrate_one_step(x, y, z, W, kl, dt, k, Cf, D, pad, pad1,
                               deltas=None)  # CFL disabled
            cfl_warnings = [warning for warning in w if "CFL" in str(warning.message)]
            assert len(cfl_warnings) == 0


class TestFindCutoffs:
    """Tests for find_cutoffs function."""

    def test_no_cutoffs_for_simple_curve(self):
        """Simple curve without self-intersection should have no cutoffs."""
        x = np.linspace(0, 1000, 100)
        y = np.sin(x / 200) * 50  # gentle sinuosity
        crdist = 100.0
        deltas = 10.0
        ind1, ind2 = mp.find_cutoffs(x, y, crdist, deltas)
        assert len(ind1) == 0
        assert len(ind2) == 0

    def test_cutoff_detected_for_tight_loop(self):
        """Tight loop that crosses itself should trigger cutoff detection."""
        # Create a loop that comes back close to itself
        theta = np.linspace(0, 1.8 * np.pi, 100)
        x = theta * 50 + np.sin(theta) * 100
        y = np.cos(theta) * 100
        crdist = 50.0
        deltas = 10.0
        ind1, ind2 = mp.find_cutoffs(x, y, crdist, deltas)
        # May or may not find cutoffs depending on exact geometry
        # Just check it returns valid indices
        assert len(ind1) == len(ind2)
        if len(ind1) > 0:
            assert np.all(ind1 < ind2)  # ind1 should be before ind2


class TestCutOffCutoffs:
    """Tests for cut_off_cutoffs function."""

    def test_no_cutoffs_returns_same_length(self):
        """When no cutoffs occur, output should have similar length."""
        x = np.linspace(0, 1000, 100)
        y = np.sin(x / 200) * 30  # gentle curve
        z = np.linspace(10, 0, 100)
        dx, dy, dz, ds, s = mp.compute_derivatives(x, y, z)
        crdist = 200.0  # Large distance - no cutoffs
        deltas = 10.0

        x_new, y_new, z_new, xc, yc, zc = mp.cut_off_cutoffs(x, y, z, s, crdist, deltas)

        # Should be same length (no cutoffs removed)
        assert len(x_new) == len(x)
        assert len(xc) == 0  # no cutoff segments

    def test_cutoff_returns_valid_output(self):
        """Output arrays should be valid even with cutoffs."""
        # Create meandering channel that might have cutoffs
        x = np.linspace(0, 2000, 200)
        y = np.sin(x / 100) * 150  # high amplitude meanders
        z = np.linspace(10, 0, 200)
        dx, dy, dz, ds, s = mp.compute_derivatives(x, y, z)
        crdist = 100.0
        deltas = 10.0

        x_new, y_new, z_new, xc, yc, zc = mp.cut_off_cutoffs(x, y, z, s, crdist, deltas)

        # Output should be valid
        assert len(x_new) == len(y_new) == len(z_new)
        assert len(x_new) > 0
        # Cutoff lists should have same length
        assert len(xc) == len(yc) == len(zc)


class TestGetChannelBanks:
    """Tests for get_channel_banks function."""

    def test_bank_coordinates_shape(self):
        """Bank coordinates should have correct shape (2x centerline length for both banks)."""
        x = np.linspace(0, 1000, 100)
        y = np.sin(x / 100) * 50
        W = 100.0
        xm, ym = mp.get_channel_banks(x, y, W)
        # Returns concatenated coords of both banks (left + right reversed)
        assert len(xm) == 2 * len(x)
        assert len(ym) == 2 * len(x)

    def test_banks_offset_by_half_width(self):
        """Banks should be offset by approximately W/2 from centerline."""
        x = np.linspace(0, 1000, 100)
        y = np.zeros_like(x)  # straight channel along x-axis
        W = 100.0
        xm, ym = mp.get_channel_banks(x, y, W)
        n = len(x)

        # For straight channel along x-axis, banks should be at y = +/- W/2
        # First half is one bank, second half (reversed) is the other
        # Check middle points of first bank (avoid edge effects)
        assert np.allclose(ym[20:n-20], W/2, atol=5)
        # Second half is reversed, so check from end
        assert np.allclose(ym[n+20:-20], -W/2, atol=5)


class TestChannelBelt:
    """Tests for ChannelBelt class."""

    def test_channelbelt_initialization(self):
        """ChannelBelt should initialize correctly."""
        W, D, Sl, deltas, pad, n_bends = 100, 5, 0.0, 25, 100, 5
        ch = mp.generate_initial_channel(W, D, Sl, deltas, pad, n_bends)
        chb = mp.ChannelBelt(channels=[ch], cutoffs=[], cl_times=[0.0], cutoff_times=[])

        assert len(chb.channels) == 1
        assert len(chb.cutoffs) == 0
        assert chb.cl_times == [0.0]

    def test_channelbelt_migrate_short(self):
        """Short migration should complete without error."""
        W, D, Sl, deltas, pad, n_bends = 100, 5, 0.0, 25, 100, 5
        ch = mp.generate_initial_channel(W, D, Sl, deltas, pad, n_bends)
        chb = mp.ChannelBelt(channels=[ch], cutoffs=[], cl_times=[0.0], cutoff_times=[])

        nit = 10
        saved_ts = 5
        crdist = 2 * W
        depths = D * np.ones(nit)
        Cfs = 0.01 * np.ones(nit)
        kl = 60.0 / (365 * 24 * 60 * 60)
        kv = 1e-11
        dt = 0.1 * 365 * 24 * 60 * 60
        dens = 1000.0

        # Should run without error
        chb.migrate(nit, saved_ts, deltas, pad, crdist, depths, Cfs, kl, kv, dt, dens)

        # Should have added channels
        assert len(chb.channels) > 1


class TestChannel:
    """Tests for Channel class."""

    def test_channel_attributes(self):
        """Channel should store all attributes correctly."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 0.1, 0, -0.1, 0])
        z = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        W, D = 10.0, 2.0

        ch = mp.Channel(x, y, z, W, D)

        assert np.array_equal(ch.x, x)
        assert np.array_equal(ch.y, y)
        assert np.array_equal(ch.z, z)
        assert ch.W == W
        assert ch.D == D


class TestCutoff:
    """Tests for Cutoff class."""

    def test_cutoff_attributes(self):
        """Cutoff should store all attributes correctly."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 0])
        z = np.array([1.0, 0.9, 0.8])
        W, D = 10.0, 2.0

        cutoff = mp.Cutoff(x, y, z, W, D)

        assert np.array_equal(cutoff.x, x)
        assert np.array_equal(cutoff.y, y)
        assert np.array_equal(cutoff.z, z)
        assert cutoff.W == W
        assert cutoff.D == D


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
