"""Tests for _recompute_wind_derived() and aggregate_to_farm().

Covers:
- ECMWF-style wspd/wdir recomputation from U/V components
- KMA-style wspd/wdir recomputation from U-component/V-component
- Mixed ECMWF + KMA sources in single DataFrame
- Calm wind threshold (wspd < 0.01 m/s → wdir = 0)
- Missing U/V components → warning, scalar average preserved
- Case-insensitive column matching
- aggregate_to_farm with wind recomputation end-to-end
- Opposing wind vectors: scalar average vs vector average correctness
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.data.training_data_builder.pipeline import (
    _recompute_wind_derived,
    aggregate_to_farm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index(n: int = 5) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h")


# ---------------------------------------------------------------------------
# ECMWF-style recomputation
# ---------------------------------------------------------------------------


class TestRecomputeECMWF:
    """ECMWF-style: {prefix}_forecast_wspd{height} ↔ u{height}/v{height}."""

    def test_basic_recompute(self):
        """wspd and wdir are recomputed from U/V at given height."""
        idx = _make_index()
        u = np.array([3.0, 0.0, -3.0, 0.0, 1.0])
        v = np.array([4.0, 5.0, 4.0, -5.0, 0.0])

        df = pd.DataFrame({
            "ecmwf_forecast_u100": u,
            "ecmwf_forecast_v100": v,
            "ecmwf_forecast_wspd100": np.full(5, 999.0),  # placeholder
            "ecmwf_forecast_wdir100": np.full(5, 999.0),  # placeholder
        }, index=idx)

        result = _recompute_wind_derived(df)

        expected_wspd = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(result["ecmwf_forecast_wspd100"], expected_wspd)

        expected_wdir = (180 + np.degrees(np.arctan2(u, v))) % 360
        np.testing.assert_allclose(result["ecmwf_forecast_wdir100"], expected_wdir)

    def test_multiple_heights(self):
        """Handles multiple height levels independently."""
        idx = _make_index(3)
        df = pd.DataFrame({
            "ecmwf_forecast_u10": [3.0, 0.0, -3.0],
            "ecmwf_forecast_v10": [4.0, 5.0, 4.0],
            "ecmwf_forecast_wspd10": [0.0, 0.0, 0.0],
            "ecmwf_forecast_wdir10": [0.0, 0.0, 0.0],
            "ecmwf_forecast_u100": [6.0, 0.0, -6.0],
            "ecmwf_forecast_v100": [8.0, 10.0, 8.0],
            "ecmwf_forecast_wspd100": [0.0, 0.0, 0.0],
            "ecmwf_forecast_wdir100": [0.0, 0.0, 0.0],
        }, index=idx)

        result = _recompute_wind_derived(df)

        np.testing.assert_allclose(result["ecmwf_forecast_wspd10"], [5.0, 5.0, 5.0])
        np.testing.assert_allclose(result["ecmwf_forecast_wspd100"], [10.0, 10.0, 10.0])

    def test_wspd_only_no_wdir(self):
        """When wdir column is absent, only wspd is recomputed."""
        idx = _make_index(2)
        df = pd.DataFrame({
            "ecmwf_forecast_u100": [3.0, 0.0],
            "ecmwf_forecast_v100": [4.0, 5.0],
            "ecmwf_forecast_wspd100": [0.0, 0.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        np.testing.assert_allclose(result["ecmwf_forecast_wspd100"], [5.0, 5.0])
        assert "ecmwf_forecast_wdir100" not in result.columns


# ---------------------------------------------------------------------------
# KMA-style recomputation
# ---------------------------------------------------------------------------


class TestRecomputeKMA:
    """KMA-style: {prefix}_forecast_wspd ↔ U-component/V-component."""

    def test_basic_recompute(self):
        """wspd/wdir recomputed from U-component/V-component."""
        idx = _make_index(3)
        u = np.array([3.0, 0.0, -3.0])
        v = np.array([4.0, 5.0, 4.0])

        df = pd.DataFrame({
            "kma_forecast_U-component": u,
            "kma_forecast_V-component": v,
            "kma_forecast_wspd": np.full(3, 999.0),
            "kma_forecast_wdir": np.full(3, 999.0),
        }, index=idx)

        result = _recompute_wind_derived(df)

        expected_wspd = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(result["kma_forecast_wspd"], expected_wspd)

        expected_wdir = (180 + np.degrees(np.arctan2(u, v))) % 360
        np.testing.assert_allclose(result["kma_forecast_wdir"], expected_wdir)

    def test_kma_wspd_only(self):
        """When wdir is absent, only wspd is recomputed."""
        idx = _make_index(2)
        df = pd.DataFrame({
            "kma_forecast_U-component": [3.0, -3.0],
            "kma_forecast_V-component": [4.0, 4.0],
            "kma_forecast_wspd": [0.0, 0.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        np.testing.assert_allclose(result["kma_forecast_wspd"], [5.0, 5.0])


# ---------------------------------------------------------------------------
# Calm wind threshold
# ---------------------------------------------------------------------------


class TestCalmWind:
    """Wind direction set to 0 when wspd < CALM_THRESHOLD (0.01 m/s)."""

    def test_zero_wind(self):
        """Both U and V are zero → wspd=0, wdir=0."""
        idx = _make_index(1)
        df = pd.DataFrame({
            "ecmwf_forecast_u100": [0.0],
            "ecmwf_forecast_v100": [0.0],
            "ecmwf_forecast_wspd100": [999.0],
            "ecmwf_forecast_wdir100": [999.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        assert result["ecmwf_forecast_wspd100"].iloc[0] == pytest.approx(0.0)
        assert result["ecmwf_forecast_wdir100"].iloc[0] == 0.0

    def test_near_zero_wind(self):
        """wspd just below threshold → wdir forced to 0."""
        idx = _make_index(1)
        tiny = 0.005  # < 0.01 threshold
        df = pd.DataFrame({
            "ecmwf_forecast_u100": [tiny],
            "ecmwf_forecast_v100": [0.0],
            "ecmwf_forecast_wspd100": [999.0],
            "ecmwf_forecast_wdir100": [999.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        assert result["ecmwf_forecast_wspd100"].iloc[0] == pytest.approx(tiny)
        assert result["ecmwf_forecast_wdir100"].iloc[0] == 0.0

    def test_above_threshold(self):
        """wspd above threshold → wdir computed normally."""
        idx = _make_index(1)
        df = pd.DataFrame({
            "ecmwf_forecast_u100": [0.1],
            "ecmwf_forecast_v100": [0.0],
            "ecmwf_forecast_wspd100": [999.0],
            "ecmwf_forecast_wdir100": [999.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        assert result["ecmwf_forecast_wspd100"].iloc[0] == pytest.approx(0.1)
        assert result["ecmwf_forecast_wdir100"].iloc[0] != 0.0


# ---------------------------------------------------------------------------
# Mixed ECMWF + KMA
# ---------------------------------------------------------------------------


class TestMixedSources:
    """DataFrame with both ECMWF and KMA columns."""

    def test_both_recomputed(self):
        """Both naming conventions handled in the same DataFrame."""
        idx = _make_index(2)
        df = pd.DataFrame({
            # ECMWF
            "ecmwf_forecast_u100": [3.0, 0.0],
            "ecmwf_forecast_v100": [4.0, 5.0],
            "ecmwf_forecast_wspd100": [0.0, 0.0],
            # KMA
            "kma_forecast_U-component": [6.0, 0.0],
            "kma_forecast_V-component": [8.0, 10.0],
            "kma_forecast_wspd": [0.0, 0.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        np.testing.assert_allclose(result["ecmwf_forecast_wspd100"], [5.0, 5.0])
        np.testing.assert_allclose(result["kma_forecast_wspd"], [10.0, 10.0])


# ---------------------------------------------------------------------------
# Missing U/V → warning, scalar average preserved
# ---------------------------------------------------------------------------


class TestMissingUV:
    """wspd/wdir without matching U/V → warning logged, values unchanged."""

    def test_warns_no_uv(self, caplog):
        """Warning emitted for unmatched wspd columns."""
        idx = _make_index(2)
        df = pd.DataFrame({
            "src_forecast_wspd": [5.0, 10.0],  # no matching U/V
            "src_forecast_wdir": [90.0, 180.0],
        }, index=idx)

        with caplog.at_level(logging.WARNING):
            result = _recompute_wind_derived(df)

        assert "could not be recomputed" in caplog.text
        # Values unchanged
        np.testing.assert_array_equal(result["src_forecast_wspd"], [5.0, 10.0])
        np.testing.assert_array_equal(result["src_forecast_wdir"], [90.0, 180.0])

    def test_no_warning_when_no_wspd(self, caplog):
        """No wind columns at all → no warning."""
        idx = _make_index(2)
        df = pd.DataFrame({
            "temperature": [15.0, 16.0],
            "pressure": [1013.0, 1012.0],
        }, index=idx)

        with caplog.at_level(logging.WARNING):
            result = _recompute_wind_derived(df)

        assert "could not be recomputed" not in caplog.text


# ---------------------------------------------------------------------------
# Case-insensitive matching
# ---------------------------------------------------------------------------


class TestCaseInsensitive:
    """Column lookup is case-insensitive for U/V."""

    def test_mixed_case_uv(self):
        """Uppercase/lowercase U/V columns still matched."""
        idx = _make_index(2)
        df = pd.DataFrame({
            "ECMWF_forecast_U100": [3.0, 0.0],
            "ecmwf_forecast_V100": [4.0, 5.0],
            "ecmwf_forecast_wspd100": [0.0, 0.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        np.testing.assert_allclose(result["ecmwf_forecast_wspd100"], [5.0, 5.0])


# ---------------------------------------------------------------------------
# aggregate_to_farm end-to-end with wind recomputation
# ---------------------------------------------------------------------------


class TestAggregateFarmWind:
    """Verify vector-correct wind recomputation after farm-level averaging."""

    def test_opposing_winds_vector_average(self):
        """Opposing U vectors cancel out; scalar wspd average would be wrong."""
        idx = _make_index(1)

        # Turbine A: U=5, V=0 → wspd=5, wdir=270°
        turbine_a = pd.DataFrame({
            "ecmwf_forecast_u100": [5.0],
            "ecmwf_forecast_v100": [0.0],
            "ecmwf_forecast_wspd100": [5.0],
            "ecmwf_forecast_wdir100": [270.0],
            "is_valid": [True],
        }, index=idx)

        # Turbine B: U=-5, V=0 → wspd=5, wdir=90°
        turbine_b = pd.DataFrame({
            "ecmwf_forecast_u100": [-5.0],
            "ecmwf_forecast_v100": [0.0],
            "ecmwf_forecast_wspd100": [5.0],
            "ecmwf_forecast_wdir100": [90.0],
            "is_valid": [True],
        }, index=idx)

        farm = aggregate_to_farm({"A": turbine_a, "B": turbine_b})

        # Vector average: U_avg=0, V_avg=0 → wspd=0 (not 5!)
        assert farm["ecmwf_forecast_wspd100"].iloc[0] == pytest.approx(0.0, abs=1e-10)
        # Calm wind → wdir=0
        assert farm["ecmwf_forecast_wdir100"].iloc[0] == 0.0

    def test_same_direction_winds(self):
        """Same-direction winds: scalar and vector averages agree."""
        idx = _make_index(1)

        turbine_a = pd.DataFrame({
            "ecmwf_forecast_u100": [3.0],
            "ecmwf_forecast_v100": [4.0],
            "ecmwf_forecast_wspd100": [5.0],
            "ecmwf_forecast_wdir100": [0.0],
            "is_valid": [True],
        }, index=idx)

        turbine_b = pd.DataFrame({
            "ecmwf_forecast_u100": [3.0],
            "ecmwf_forecast_v100": [4.0],
            "ecmwf_forecast_wspd100": [5.0],
            "ecmwf_forecast_wdir100": [0.0],
            "is_valid": [True],
        }, index=idx)

        farm = aggregate_to_farm({"A": turbine_a, "B": turbine_b})

        # Same direction: U_avg=3, V_avg=4 → wspd=5
        assert farm["ecmwf_forecast_wspd100"].iloc[0] == pytest.approx(5.0)

    def test_is_valid_and_across_turbines(self):
        """is_valid is AND-ed across turbines."""
        idx = _make_index(3)

        turbine_a = pd.DataFrame({
            "ecmwf_forecast_u100": [1.0, 2.0, 3.0],
            "ecmwf_forecast_v100": [1.0, 2.0, 3.0],
            "ecmwf_forecast_wspd100": [0.0, 0.0, 0.0],
            "is_valid": [True, True, False],
        }, index=idx)

        turbine_b = pd.DataFrame({
            "ecmwf_forecast_u100": [1.0, 2.0, 3.0],
            "ecmwf_forecast_v100": [1.0, 2.0, 3.0],
            "ecmwf_forecast_wspd100": [0.0, 0.0, 0.0],
            "is_valid": [True, False, True],
        }, index=idx)

        farm = aggregate_to_farm({"A": turbine_a, "B": turbine_b})
        expected_valid = [True, False, False]
        np.testing.assert_array_equal(farm["is_valid"], expected_valid)

    def test_object_dtype_columns(self):
        """Columns with object dtype after averaging don't break np.sqrt.

        When np.stack produces an object array (e.g. boolean/object
        columns mixed before separation), the resulting DataFrame has
        object-dtype columns. np.sqrt fails on Python float objects
        unless explicitly cast.
        """
        idx = _make_index(2)

        turbine_a = pd.DataFrame({
            "ecmwf_forecast_u100": pd.array([3.0, 0.0], dtype=object),
            "ecmwf_forecast_v100": pd.array([4.0, 5.0], dtype=object),
            "ecmwf_forecast_wspd100": pd.array([0.0, 0.0], dtype=object),
            "ecmwf_forecast_wdir100": pd.array([0.0, 0.0], dtype=object),
            "is_valid": [True, True],
        }, index=idx)

        turbine_b = pd.DataFrame({
            "ecmwf_forecast_u100": pd.array([3.0, 0.0], dtype=object),
            "ecmwf_forecast_v100": pd.array([4.0, 5.0], dtype=object),
            "ecmwf_forecast_wspd100": pd.array([0.0, 0.0], dtype=object),
            "ecmwf_forecast_wdir100": pd.array([0.0, 0.0], dtype=object),
            "is_valid": [True, True],
        }, index=idx)

        farm = aggregate_to_farm({"A": turbine_a, "B": turbine_b})
        np.testing.assert_allclose(
            farm["ecmwf_forecast_wspd100"], [5.0, 5.0],
        )

    def test_empty_turbine_dict_raises(self):
        """Empty dict raises ValueError."""
        with pytest.raises(ValueError, match="No turbine"):
            aggregate_to_farm({})

    def test_no_common_index_raises(self):
        """Disjoint indices raises ValueError."""
        idx_a = pd.date_range("2024-01-01", periods=3, freq="h")
        idx_b = pd.date_range("2024-06-01", periods=3, freq="h")

        a = pd.DataFrame({"val": [1, 2, 3]}, index=idx_a)
        b = pd.DataFrame({"val": [4, 5, 6]}, index=idx_b)

        with pytest.raises(ValueError, match="No common time index"):
            aggregate_to_farm({"A": a, "B": b})

    def test_partial_overlap_drops_timestamps(self, caplog):
        """Partial overlap → warning + dropped timestamps."""
        idx_a = pd.date_range("2024-01-01 00:00", periods=5, freq="h")
        idx_b = pd.date_range("2024-01-01 02:00", periods=5, freq="h")

        a = pd.DataFrame({"val": np.arange(5, dtype=float)}, index=idx_a)
        b = pd.DataFrame({"val": np.arange(5, dtype=float)}, index=idx_b)

        with caplog.at_level(logging.WARNING):
            farm = aggregate_to_farm({"A": a, "B": b})

        assert len(farm) == 3  # intersection of [00..04] and [02..06]
        assert "dropped" in caplog.text


# ---------------------------------------------------------------------------
# Wind direction edge cases
# ---------------------------------------------------------------------------


class TestWindDirectionEdgeCases:
    """Meteorological wind direction convention edge cases."""

    def test_north_wind(self):
        """Wind from north: U=0, V<0 → wdir=360° (or 0°)."""
        idx = _make_index(1)
        df = pd.DataFrame({
            "ecmwf_forecast_u100": [0.0],
            "ecmwf_forecast_v100": [-5.0],
            "ecmwf_forecast_wspd100": [0.0],
            "ecmwf_forecast_wdir100": [0.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        # atan2(0, -5) = π → (180 + 180) % 360 = 0 or 360
        wdir = result["ecmwf_forecast_wdir100"].iloc[0]
        assert wdir == pytest.approx(0.0) or wdir == pytest.approx(360.0)

    def test_south_wind(self):
        """Wind from south: U=0, V>0 → wdir=180°."""
        idx = _make_index(1)
        df = pd.DataFrame({
            "ecmwf_forecast_u100": [0.0],
            "ecmwf_forecast_v100": [5.0],
            "ecmwf_forecast_wspd100": [0.0],
            "ecmwf_forecast_wdir100": [0.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        assert result["ecmwf_forecast_wdir100"].iloc[0] == pytest.approx(180.0)

    def test_east_wind(self):
        """Wind from east: U<0, V=0 → wdir=90°."""
        idx = _make_index(1)
        df = pd.DataFrame({
            "ecmwf_forecast_u100": [-5.0],
            "ecmwf_forecast_v100": [0.0],
            "ecmwf_forecast_wspd100": [0.0],
            "ecmwf_forecast_wdir100": [0.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        assert result["ecmwf_forecast_wdir100"].iloc[0] == pytest.approx(90.0)

    def test_west_wind(self):
        """Wind from west: U>0, V=0 → wdir=270°."""
        idx = _make_index(1)
        df = pd.DataFrame({
            "ecmwf_forecast_u100": [5.0],
            "ecmwf_forecast_v100": [0.0],
            "ecmwf_forecast_wspd100": [0.0],
            "ecmwf_forecast_wdir100": [0.0],
        }, index=idx)

        result = _recompute_wind_derived(df)
        assert result["ecmwf_forecast_wdir100"].iloc[0] == pytest.approx(270.0)
