"""Tests for NWPDataStore: horizon extraction, missing fill, future-fill warnings.

Covers:
- get_for_horizon: correct basis_time/forecast_time lookup
- Horizon-0 avoid_exact vs horizon>0 exact boundary
- _fill_missing_basis_times: nearest-previous fill, future-fill warning
- _interpolate_if_needed: 3h→1h interpolation
- No future data leaking into current predictions
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.data.training_data_builder.nwp_store import NWPDataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_store(
    basis_hours: list[int],
    forecast_offsets: list[int],
    nwp_tz: str = "UTC",
    scada_tz: str = "KST",
    col_name: str = "ecmwf_forecast_temperature",
    base_date: str = "2024-01-15",
) -> NWPDataStore:
    """Build a NWPDataStore from simple basis_hour + forecast_offset specs.

    Each (basis_hour, forecast_offset) pair creates one row. Values are
    set to basis_hour * 100 + forecast_offset for easy verification.
    """
    rows = []
    for bh in basis_hours:
        bt = pd.Timestamp(f"{base_date} {bh:02d}:00")
        for fo in forecast_offsets:
            ft = bt + pd.Timedelta(hours=fo)
            val = bh * 100 + fo
            rows.append({"basis_time": bt, "forecast_time": ft, col_name: val, "is_valid": True})

    df = pd.DataFrame(rows).set_index(["basis_time", "forecast_time"])
    return NWPDataStore(df, nwp_name="ecmwf", nwp_tz=nwp_tz, scada_tz=scada_tz)


# ---------------------------------------------------------------------------
# get_for_horizon
# ---------------------------------------------------------------------------


class TestGetForHorizon:
    """Test horizon extraction from NWPDataStore."""

    def test_horizon0_avoids_exact(self):
        """Horizon=0 with avoid_exact: SCADA exactly on release → uses previous cycle."""
        # basis 00:00 and 06:00, each with offsets 0..11h
        store = _build_store([0, 6], list(range(12)))

        # SCADA KST 15:00 → UTC 06:00 (exact 6h boundary)
        kst = pd.DatetimeIndex(["2024-01-15 15:00"])

        result = store.get_for_horizon(0, kst, "6h")

        # avoid_exact=True → basis=00:00, forecast=06:00 → value = 0*100 + 6 = 6
        assert result["ecmwf_forecast_temperature"].iloc[0] == 6

    def test_horizon1_uses_exact(self):
        """Horizon=1 with avoid_exact=False: SCADA on release → uses current cycle."""
        store = _build_store([0, 6], list(range(12)))

        # SCADA KST 15:00 → UTC 06:00 (exact boundary)
        kst = pd.DatetimeIndex(["2024-01-15 15:00"])

        result = store.get_for_horizon(1, kst, "6h")

        # avoid_exact=False → basis=06:00, forecast=06:00+1=07:00 → value = 6*100 + 1 = 601
        assert result["ecmwf_forecast_temperature"].iloc[0] == 601

    def test_between_releases(self):
        """SCADA between releases → snaps to previous release."""
        store = _build_store([0, 6], list(range(12)))

        # SCADA KST 13:00 → UTC 04:00 (between 00 and 06)
        kst = pd.DatetimeIndex(["2024-01-15 13:00"])

        result = store.get_for_horizon(2, kst, "6h")

        # basis=00:00 (04%6=4, snap back 4h), forecast=04+2=06:00 → value = 0*100 + 6 = 6
        assert result["ecmwf_forecast_temperature"].iloc[0] == 6

    def test_multiple_horizons_same_scada(self):
        """Different horizons from same SCADA time → different forecast_times."""
        store = _build_store([0], list(range(24)))

        kst = pd.DatetimeIndex(["2024-01-15 12:00"])  # UTC 03:00

        for h in [0, 1, 3, 6]:
            result = store.get_for_horizon(h, kst, "6h")
            # basis depends on avoid_exact
            if h == 0:
                # avoid → previous release: since 03%6=3, snap to 00:00 anyway
                # but avoid_exact, 3!=0 so no difference: basis=00:00
                expected = 0 * 100 + (3 + h)  # 3
            else:
                expected = 0 * 100 + (3 + h)  # 3+h
            assert result["ecmwf_forecast_temperature"].iloc[0] == expected

    def test_result_indexed_by_scada_kst(self):
        """Output index matches original SCADA KST timestamps."""
        store = _build_store([0], list(range(24)))

        kst = pd.DatetimeIndex([
            "2024-01-15 10:00",
            "2024-01-15 11:00",
            "2024-01-15 12:00",
        ])

        result = store.get_for_horizon(1, kst, "6h")
        pd.testing.assert_index_equal(result.index, kst)


# ---------------------------------------------------------------------------
# _fill_missing_basis_times
# ---------------------------------------------------------------------------


class TestFillMissingBasisTimes:

    def test_fill_from_previous(self):
        """Missing basis_time filled from nearest previous."""
        # Only basis 00:00 exists; 06:00 is "missing"
        store = _build_store([0], list(range(12)))

        filled = NWPDataStore._fill_missing_basis_times(
            store._df, ["2024-01-15_06"], "test_nwp",
        )

        # Check that 06:00 basis_time now exists
        bt_06 = pd.Timestamp("2024-01-15 06:00")
        assert bt_06 in filled.index.get_level_values("basis_time")

        # Filled rows should have is_valid=False
        filled_rows = filled.loc[bt_06]
        assert (filled_rows["is_valid"] == False).all()

        # Forecast times should be shifted +6h from source (00:00)
        original_fts = store._df.loc[pd.Timestamp("2024-01-15 00:00")].index
        expected_fts = original_fts + pd.Timedelta(hours=6)
        pd.testing.assert_index_equal(filled_rows.index, expected_fts)

    def test_future_fill_warning(self, caplog):
        """When only future data available, fills with warning."""
        # Only basis 12:00 exists; 06:00 is "missing" → must use future
        store = _build_store([12], list(range(12)))

        with caplog.at_level(logging.WARNING):
            filled = NWPDataStore._fill_missing_basis_times(
                store._df, ["2024-01-15_06"], "test_nwp",
            )

        assert "FUTURE cycle" in caplog.text

        bt_06 = pd.Timestamp("2024-01-15 06:00")
        assert bt_06 in filled.index.get_level_values("basis_time")

    def test_unfillable(self, caplog):
        """No data at all → error logged, basis_time not in output."""
        df = pd.DataFrame(
            columns=["is_valid"],
            index=pd.MultiIndex.from_tuples([], names=["basis_time", "forecast_time"]),
        )

        with caplog.at_level(logging.ERROR):
            result = NWPDataStore._fill_missing_basis_times(
                df, ["2024-01-15_06"], "test_nwp",
            )

        assert "could not be filled" in caplog.text

    def test_no_missing_returns_unchanged(self):
        """Empty missing list → DataFrame unchanged."""
        store = _build_store([0, 6], [0, 1, 2])
        original_len = len(store._df)

        result = NWPDataStore._fill_missing_basis_times(store._df, [], "test")
        assert len(result) == original_len


# ---------------------------------------------------------------------------
# _interpolate_if_needed
# ---------------------------------------------------------------------------


class TestInterpolateIfNeeded:

    def test_3h_to_1h(self):
        """3h forecast data interpolated to 1h resolution."""
        bt = pd.Timestamp("2024-01-15 00:00")
        ft_index = pd.DatetimeIndex([
            "2024-01-15 00:00",
            "2024-01-15 03:00",
            "2024-01-15 06:00",
        ])
        df = pd.DataFrame({
            "temperature": [10.0, 13.0, 16.0],
        }, index=pd.MultiIndex.from_arrays(
            [[bt, bt, bt], ft_index],
            names=["basis_time", "forecast_time"],
        ))

        result = NWPDataStore._interpolate_if_needed(df, "3h", "1h")

        # Should have 7 rows: 00, 01, 02, 03, 04, 05, 06
        bt_group = result.loc[bt]
        assert len(bt_group) == 7

        # Linear interpolation: temp at 01:00 = 11.0, at 02:00 = 12.0
        assert bt_group.loc[pd.Timestamp("2024-01-15 01:00"), "temperature"] == pytest.approx(11.0)
        assert bt_group.loc[pd.Timestamp("2024-01-15 02:00"), "temperature"] == pytest.approx(12.0)

    def test_already_fine_resolution(self):
        """1h data with 1h target → returned unchanged."""
        bt = pd.Timestamp("2024-01-15 00:00")
        ft_index = pd.DatetimeIndex([
            "2024-01-15 00:00",
            "2024-01-15 01:00",
        ])
        df = pd.DataFrame({
            "temperature": [10.0, 11.0],
        }, index=pd.MultiIndex.from_arrays(
            [[bt, bt], ft_index],
            names=["basis_time", "forecast_time"],
        ))

        result = NWPDataStore._interpolate_if_needed(df, "1h", "1h")
        assert len(result) == 2

    def test_is_valid_forward_filled(self):
        """is_valid is forward-filled, not interpolated."""
        bt = pd.Timestamp("2024-01-15 00:00")
        ft_index = pd.DatetimeIndex([
            "2024-01-15 00:00",
            "2024-01-15 03:00",
        ])
        df = pd.DataFrame({
            "temperature": [10.0, 13.0],
            "is_valid": [True, False],
        }, index=pd.MultiIndex.from_arrays(
            [[bt, bt], ft_index],
            names=["basis_time", "forecast_time"],
        ))

        result = NWPDataStore._interpolate_if_needed(df, "3h", "1h")
        bt_group = result.loc[bt]

        # 00:00 → True, 01:00-03:00 → True (ffill from 00:00)...
        # actually 03:00 is False, so ffill: 00→True, 01→True, 02→True, 03→False
        assert bt_group.loc[pd.Timestamp("2024-01-15 00:00"), "is_valid"] == True
        assert bt_group.loc[pd.Timestamp("2024-01-15 01:00"), "is_valid"] == True
        assert bt_group.loc[pd.Timestamp("2024-01-15 03:00"), "is_valid"] == False


# ---------------------------------------------------------------------------
# No future data leakage
# ---------------------------------------------------------------------------


class TestNoFutureLeakage:
    """Ensure NWP data from future basis_times never leaks into predictions."""

    def test_horizon0_never_uses_current_release(self):
        """At horizon=0, even if current release exists, previous is used."""
        # basis 00:00 and 06:00 both exist
        store = _build_store([0, 6], list(range(12)))

        # SCADA at exact 06:00 UTC (KST 15:00)
        kst = pd.DatetimeIndex(["2024-01-15 15:00"])

        result = store.get_for_horizon(0, kst, "6h")

        # Must use basis 00:00 (previous), NOT 06:00 (current)
        # forecast_time = UTC 06:00, basis=00:00 → value = 0*100+6 = 6
        assert result["ecmwf_forecast_temperature"].iloc[0] == 6

    def test_horizon_gt0_at_boundary_uses_current(self):
        """At horizon>0, exact boundary uses current release (no leakage)."""
        store = _build_store([0, 6], list(range(12)))

        kst = pd.DatetimeIndex(["2024-01-15 15:00"])  # UTC 06:00

        result = store.get_for_horizon(3, kst, "6h")

        # avoid_exact=False → basis=06:00, forecast=06+3=09:00 → value = 6*100+3 = 603
        assert result["ecmwf_forecast_temperature"].iloc[0] == 603
