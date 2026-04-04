"""Tests for time alignment: snap_to_nwp_basis, basis/forecast mapping, horizon separation.

Covers:
- snap_to_nwp_basis with avoid_exact=True/False
- KST→UTC conversion and back
- Horizon-0 leakage prevention (avoid_exact=True)
- Horizon>0 uses exact boundary (avoid_exact=False)
- Basis_time/forecast_time mapping correctness per horizon
- Edge: timestamp exactly on NWP release boundary
- Edge: timestamp before first release of the day
"""

import numpy as np
import pandas as pd
import pytest

from src.data.training_data_builder.time_alignment import (
    convert_timezone,
    create_nwp_basis_mapping,
    create_nwp_file_mapping,
    parse_frequency,
    snap_to_nwp_basis,
)


# ---------------------------------------------------------------------------
# parse_frequency
# ---------------------------------------------------------------------------


class TestParseFrequency:

    def test_basic(self):
        assert parse_frequency("6h") == (6, "h")
        assert parse_frequency("1h") == (1, "h")
        assert parse_frequency("3hr") == (3, "h")
        assert parse_frequency("12hours") == (12, "h")

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Only hour units"):
            parse_frequency("30m")

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid frequency"):
            parse_frequency("abc")


# ---------------------------------------------------------------------------
# convert_timezone
# ---------------------------------------------------------------------------


class TestConvertTimezone:

    def test_kst_to_utc(self):
        """KST → UTC: subtract 9 hours."""
        kst = pd.DatetimeIndex(["2024-01-15 09:00", "2024-01-15 15:00"])
        utc = convert_timezone(kst, "KST", "UTC")
        expected = pd.DatetimeIndex(["2024-01-15 00:00", "2024-01-15 06:00"])
        pd.testing.assert_index_equal(utc, expected)

    def test_utc_to_kst(self):
        """UTC → KST: add 9 hours."""
        utc = pd.DatetimeIndex(["2024-01-15 00:00"])
        kst = convert_timezone(utc, "UTC", "KST")
        expected = pd.DatetimeIndex(["2024-01-15 09:00"])
        pd.testing.assert_index_equal(kst, expected)

    def test_roundtrip(self):
        """KST→UTC→KST = identity."""
        original = pd.DatetimeIndex(["2024-01-15 12:00"])
        roundtrip = convert_timezone(
            convert_timezone(original, "KST", "UTC"), "UTC", "KST"
        )
        pd.testing.assert_index_equal(roundtrip, original)


# ---------------------------------------------------------------------------
# snap_to_nwp_basis
# ---------------------------------------------------------------------------


class TestSnapToNWPBasis:
    """Test NWP basis_time snapping with 6h release cycle."""

    def test_between_releases(self):
        """Timestamp between releases snaps to previous release."""
        utc = pd.DatetimeIndex(["2024-01-15 08:30"])
        snapped = snap_to_nwp_basis(utc, "6h")
        expected = pd.DatetimeIndex(["2024-01-15 06:00"])
        pd.testing.assert_index_equal(snapped, expected)

    def test_exact_release_no_avoid(self):
        """Exact release time, avoid_exact=False → stays on same release."""
        utc = pd.DatetimeIndex(["2024-01-15 12:00"])
        snapped = snap_to_nwp_basis(utc, "6h", avoid_exact=False)
        expected = pd.DatetimeIndex(["2024-01-15 12:00"])
        pd.testing.assert_index_equal(snapped, expected)

    def test_exact_release_with_avoid(self):
        """Exact release time, avoid_exact=True → snaps to PREVIOUS release."""
        utc = pd.DatetimeIndex(["2024-01-15 12:00"])
        snapped = snap_to_nwp_basis(utc, "6h", avoid_exact=True)
        expected = pd.DatetimeIndex(["2024-01-15 06:00"])
        pd.testing.assert_index_equal(snapped, expected)

    def test_midnight_avoid_exact(self):
        """Midnight with avoid_exact → snaps to previous day 18:00."""
        utc = pd.DatetimeIndex(["2024-01-15 00:00"])
        snapped = snap_to_nwp_basis(utc, "6h", avoid_exact=True)
        expected = pd.DatetimeIndex(["2024-01-14 18:00"])
        pd.testing.assert_index_equal(snapped, expected)

    def test_3h_frequency(self):
        """3-hour release cycle."""
        utc = pd.DatetimeIndex([
            "2024-01-15 04:00",  # between 3:00 and 6:00
            "2024-01-15 06:00",  # exact release
        ])
        snapped = snap_to_nwp_basis(utc, "3h", avoid_exact=False)
        expected = pd.DatetimeIndex([
            "2024-01-15 03:00",
            "2024-01-15 06:00",
        ])
        pd.testing.assert_index_equal(snapped, expected)

    def test_minutes_seconds_zeroed(self):
        """Sub-hour components are zeroed out."""
        utc = pd.DatetimeIndex(["2024-01-15 08:45:30"])
        snapped = snap_to_nwp_basis(utc, "6h")
        expected = pd.DatetimeIndex(["2024-01-15 06:00"])
        pd.testing.assert_index_equal(snapped, expected)

    def test_multiple_timestamps(self):
        """Vector operation on multiple timestamps."""
        utc = pd.DatetimeIndex([
            "2024-01-15 01:00",  # → 00:00
            "2024-01-15 07:00",  # → 06:00
            "2024-01-15 13:00",  # → 12:00
            "2024-01-15 19:00",  # → 18:00
        ])
        snapped = snap_to_nwp_basis(utc, "6h", avoid_exact=False)
        expected = pd.DatetimeIndex([
            "2024-01-15 00:00",
            "2024-01-15 06:00",
            "2024-01-15 12:00",
            "2024-01-15 18:00",
        ])
        pd.testing.assert_index_equal(snapped, expected)


# ---------------------------------------------------------------------------
# Horizon-0 leakage prevention
# ---------------------------------------------------------------------------


class TestHorizon0Leakage:
    """Horizon=0 must use avoid_exact=True to prevent observation leakage.

    If SCADA time exactly aligns with an NWP release, using that release's
    data at horizon=0 would mean the forecast was "issued" at the same time
    as the observation — i.e. the model could see the current observation.
    """

    def test_file_mapping_horizon0_avoids_exact(self):
        """create_nwp_file_mapping with horizon=0 uses avoid_exact=True."""
        # SCADA KST 09:00 → UTC 00:00 (exactly on 6h boundary)
        kst = pd.DatetimeIndex(["2024-01-15 09:00"])
        mapping = create_nwp_file_mapping(kst, "6h", "KST", "UTC", forecasting_horizon=0)

        # Should snap to PREVIOUS release (18:00 on Jan 14), not 00:00
        assert "2024-01-14_18" in mapping
        assert "2024-01-15_00" not in mapping

    def test_file_mapping_horizon1_uses_exact(self):
        """create_nwp_file_mapping with horizon>0 uses avoid_exact=False."""
        kst = pd.DatetimeIndex(["2024-01-15 09:00"])
        mapping = create_nwp_file_mapping(kst, "6h", "KST", "UTC", forecasting_horizon=1)

        # horizon>0: avoid_exact=False → snaps to 00:00 (current release)
        assert "2024-01-15_00" in mapping


# ---------------------------------------------------------------------------
# Basis mapping for continuous format
# ---------------------------------------------------------------------------


class TestBasisMapping:
    """create_nwp_basis_mapping always uses avoid_exact=True."""

    def test_always_avoids_exact(self):
        """Even when SCADA aligns with NWP release, uses previous cycle."""
        kst = pd.DatetimeIndex(["2024-01-15 09:00"])  # → UTC 00:00 (exact)
        mapping = create_nwp_basis_mapping(kst, "6h", "KST", "UTC")

        assert "2024-01-14_18" in mapping
        assert "2024-01-15_00" not in mapping

    def test_forecast_time_equals_scada_utc(self):
        """Forecast time = SCADA time converted to UTC (no horizon shift)."""
        kst = pd.DatetimeIndex(["2024-01-15 15:00"])  # → UTC 06:00
        mapping = create_nwp_basis_mapping(kst, "6h", "KST", "UTC")

        # UTC 06:00 is exact release → avoid → snaps to 00:00
        assert "2024-01-15_00" in mapping
        assert "2024-01-15 06:00:00" in mapping["2024-01-15_00"]


# ---------------------------------------------------------------------------
# Per-horizon file mapping: forecast_time = scada_utc + horizon
# ---------------------------------------------------------------------------


class TestPerHorizonMapping:
    """create_nwp_file_mapping with various horizons."""

    def test_horizon_shifts_forecast_time(self):
        """Forecast time = SCADA UTC + horizon hours."""
        kst = pd.DatetimeIndex(["2024-01-15 12:00"])  # → UTC 03:00
        mapping = create_nwp_file_mapping(kst, "6h", "KST", "UTC", forecasting_horizon=3)

        # basis: UTC 03:00 → snap to 00:00 (avoid_exact=False, 3%6=3≠0)
        assert "2024-01-15_00" in mapping
        # forecast_time: 03:00 + 3h = 06:00
        assert "2024-01-15 06:00:00" in mapping["2024-01-15_00"]

    def test_horizon_crosses_day_boundary(self):
        """Forecast time can cross into next day."""
        kst = pd.DatetimeIndex(["2024-01-16 06:00"])  # → UTC Jan 15 21:00
        mapping = create_nwp_file_mapping(kst, "6h", "KST", "UTC", forecasting_horizon=6)

        # basis: UTC 21:00 → snap to 18:00
        assert "2024-01-15_18" in mapping
        # forecast: 21:00 + 6h = 03:00 next day
        assert "2024-01-16 03:00:00" in mapping["2024-01-15_18"]

    def test_multiple_scada_same_basis(self):
        """Multiple SCADA times mapping to same basis produce multiple forecast times."""
        kst = pd.DatetimeIndex([
            "2024-01-15 10:00",  # UTC 01:00 → basis 00:00
            "2024-01-15 11:00",  # UTC 02:00 → basis 00:00
            "2024-01-15 12:00",  # UTC 03:00 → basis 00:00
        ])
        mapping = create_nwp_file_mapping(kst, "6h", "KST", "UTC", forecasting_horizon=1)

        assert "2024-01-15_00" in mapping
        fc_times = mapping["2024-01-15_00"]
        # forecast = UTC + 1h → 02:00, 03:00, 04:00
        assert "2024-01-15 02:00:00" in fc_times
        assert "2024-01-15 03:00:00" in fc_times
        assert "2024-01-15 04:00:00" in fc_times
