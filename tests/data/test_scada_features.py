"""Tests for SCADA feature engineering: validation, lagged features, target preparation.

Covers:
- _validate_hourly_index: gaps detected, single-row passthrough
- create_lagged_features: correct shift, is_valid AND propagation
- prepare_target: horizon shifting, basis_time/forecast_time separation
- Edge: horizon > data length, horizon=0
"""

import numpy as np
import pandas as pd
import pytest

from src.data.training_data_builder.scada import (
    _validate_hourly_index,
    clip_positive_columns,
    create_lagged_features,
    prepare_target,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hourly_index(n: int = 10, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h")


def _make_scada(n: int = 10, with_valid: bool = True) -> pd.DataFrame:
    """Simple SCADA-like DataFrame with hourly index."""
    idx = _hourly_index(n)
    df = pd.DataFrame({
        "power": np.arange(n, dtype=float) * 10,
        "wind_speed": np.arange(n, dtype=float) + 5,
    }, index=idx)
    if with_valid:
        df["is_valid"] = True
    return df


# ---------------------------------------------------------------------------
# _validate_hourly_index
# ---------------------------------------------------------------------------


class TestValidateHourlyIndex:

    def test_uniform_passes(self):
        """Uniform 1h spacing → no error."""
        _validate_hourly_index(_hourly_index(10), "test")

    def test_single_row_passes(self):
        """Single row → no validation possible, passes."""
        _validate_hourly_index(_hourly_index(1), "test")

    def test_empty_passes(self):
        """Empty index → passes."""
        _validate_hourly_index(pd.DatetimeIndex([]), "test")

    def test_gap_raises(self):
        """2-hour gap detected → ValueError."""
        idx = pd.DatetimeIndex([
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 03:00",  # gap: missing 02:00
        ])
        with pytest.raises(ValueError, match="not uniformly 1-hour"):
            _validate_hourly_index(idx, "test_context")

    def test_sub_hourly_raises(self):
        """30-min spacing → detected as irregular."""
        idx = pd.DatetimeIndex([
            "2024-01-01 00:00",
            "2024-01-01 00:30",
            "2024-01-01 01:00",
        ])
        with pytest.raises(ValueError, match="not uniformly 1-hour"):
            _validate_hourly_index(idx, "test_context")

    def test_context_in_message(self):
        """Context string appears in error message."""
        idx = pd.DatetimeIndex([
            "2024-01-01 00:00",
            "2024-01-01 05:00",
        ])
        with pytest.raises(ValueError, match="my_function"):
            _validate_hourly_index(idx, "my_function")


# ---------------------------------------------------------------------------
# create_lagged_features
# ---------------------------------------------------------------------------


class TestCreateLaggedFeatures:

    def test_lag0_columns(self):
        """Lag-0 column names follow convention."""
        scada = _make_scada(10)
        result = create_lagged_features(scada, max_lag=0)

        assert "basis_time_lag0_power" in result.columns
        assert "basis_time_lag0_wind_speed" in result.columns

    def test_lag_shift_values(self):
        """Lag-k row at time t contains the value from time t-k."""
        scada = _make_scada(10)
        result = create_lagged_features(scada, max_lag=2)

        # At time index=5 (value=50): lag1 should have value from index=4 (40)
        idx_5 = scada.index[5]
        assert result.loc[idx_5, "basis_time_lag0_power"] == 50.0
        assert result.loc[idx_5, "basis_time_lag1_power"] == 40.0
        assert result.loc[idx_5, "basis_time_lag2_power"] == 30.0

    def test_nan_rows_dropped(self):
        """First max_lag rows dropped (NaN from shifting)."""
        scada = _make_scada(10)
        result = create_lagged_features(scada, max_lag=3)
        assert len(result) == 10 - 3  # first 3 rows dropped

    def test_is_valid_and_propagation(self):
        """is_valid = AND of all lag validities."""
        scada = _make_scada(6)
        # Mark row at index=2 as invalid
        scada.loc[scada.index[2], "is_valid"] = False

        result = create_lagged_features(scada, max_lag=2)

        # Row at index=2 (lag0 invalid) → invalid
        assert result.loc[scada.index[2], "is_valid"] == False
        # Row at index=3 (lag1 = index=2 which is invalid) → invalid
        assert result.loc[scada.index[3], "is_valid"] == False
        # Row at index=4 (lag2 = index=2 which is invalid) → invalid
        assert result.loc[scada.index[4], "is_valid"] == False
        # Row at index=5 (lag2 = index=3, lag1 = index=4, lag0 = index=5, all valid)
        assert result.loc[scada.index[5], "is_valid"] == True

    def test_no_is_valid_column(self):
        """When is_valid absent, output also has no is_valid."""
        scada = _make_scada(10, with_valid=False)
        result = create_lagged_features(scada, max_lag=1)
        assert "is_valid" not in result.columns

    def test_gap_in_index_raises(self):
        """Non-uniform index → ValueError."""
        idx = pd.DatetimeIndex([
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 03:00",  # gap
        ])
        scada = pd.DataFrame({"power": [1, 2, 3]}, index=idx)
        with pytest.raises(ValueError, match="not uniformly"):
            create_lagged_features(scada, max_lag=1)


# ---------------------------------------------------------------------------
# prepare_target
# ---------------------------------------------------------------------------


class TestPrepareTarget:

    def test_horizon0_no_shift(self):
        """Horizon=0: values at same time, column renamed."""
        scada = _make_scada(5)
        result = prepare_target(scada, horizon=0)

        assert "forecast_time_power" in result.columns
        # Value at index=2 → same as original
        np.testing.assert_equal(
            result["forecast_time_power"].values,
            scada["power"].values,
        )

    def test_horizon_shift(self):
        """Horizon=k: row at time t contains value from t+k."""
        scada = _make_scada(10)
        result = prepare_target(scada, horizon=3)

        # Row at original index=0 should contain value from index=3
        first_idx = result.index[0]
        original_idx_0 = scada.index[0]
        assert first_idx == original_idx_0
        assert result.loc[first_idx, "forecast_time_power"] == scada["power"].iloc[3]

    def test_horizon_drops_tail(self):
        """Last `horizon` rows dropped (NaN from backward shift)."""
        scada = _make_scada(10)
        result = prepare_target(scada, horizon=3)
        assert len(result) == 10 - 3

    def test_target_is_valid_shifted(self):
        """target_is_valid reflects validity at the forecast time, not basis."""
        scada = _make_scada(6)
        scada.loc[scada.index[4], "is_valid"] = False  # invalid at t=4

        result = prepare_target(scada, horizon=2)

        # Row at t=2: target is at t+2=4 (invalid)
        assert result.loc[scada.index[2], "target_is_valid"] == False
        # Row at t=0: target is at t+2=2 (valid)
        assert result.loc[scada.index[0], "target_is_valid"] == True

    def test_horizon_exceeds_length(self):
        """Horizon >= data length → all rows dropped."""
        scada = _make_scada(3)
        result = prepare_target(scada, horizon=3)
        assert len(result) == 0

    def test_column_prefix(self):
        """All data columns prefixed with forecast_time_."""
        scada = _make_scada(5)
        result = prepare_target(scada, horizon=1)
        data_cols = [c for c in result.columns if c != "target_is_valid"]
        assert all(c.startswith("forecast_time_") for c in data_cols)

    def test_gap_in_index_raises(self):
        """Non-uniform index → ValueError."""
        idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 05:00"])
        scada = pd.DataFrame({"power": [1, 2], "is_valid": [True, True]}, index=idx)
        with pytest.raises(ValueError, match="not uniformly"):
            prepare_target(scada, horizon=1)


# ---------------------------------------------------------------------------
# clip_positive_columns
# ---------------------------------------------------------------------------


class TestClipPositive:

    def test_clips_matching_suffix(self):
        """Columns matching suffix are clipped to >= 0."""
        df = pd.DataFrame({
            "forecast_time_KPX_pwr": [-5.0, 0.0, 10.0],
            "basis_time_lag0_KPX_pwr": [-3.0, 5.0, 8.0],
            "temperature": [-1.0, 0.0, 1.0],
        })
        result = clip_positive_columns(df, ["KPX_pwr"])

        np.testing.assert_array_equal(result["forecast_time_KPX_pwr"], [0.0, 0.0, 10.0])
        np.testing.assert_array_equal(result["basis_time_lag0_KPX_pwr"], [0.0, 5.0, 8.0])
        # temperature not matched → unchanged
        np.testing.assert_array_equal(result["temperature"], [-1.0, 0.0, 1.0])

    def test_no_match_returns_unchanged(self):
        """No matching columns → DataFrame unchanged."""
        df = pd.DataFrame({"a": [-1.0, 2.0]})
        result = clip_positive_columns(df, ["nonexistent"])
        pd.testing.assert_frame_equal(result, df)
