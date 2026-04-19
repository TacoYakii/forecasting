"""Tests for JejuNMAPEEvaluator."""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import ParametricForecastResult
from src.utils.nMAPE import JejuNMAPEEvaluator, to_nmape_frames


CAPACITY = 100.0


def _make_observed_csv(path, start, periods, value=50.0, is_valid=True):
    idx = pd.date_range(start, periods=periods, freq="h")
    df = pd.DataFrame(
        {
            "forecast_time": idx,
            "forecast_time_observed_KPX_pwr": np.full(periods, value),
            "is_valid": np.full(periods, is_valid, dtype=bool),
        }
    )
    df.to_csv(path, index=False)


def _make_result(loc_value=50.0):
    """24 hourly basis times on 2024-01-01, horizons 1..40."""
    basis = pd.date_range("2024-01-01 00:00", periods=24, freq="h")
    H = 40
    loc = np.full((24, H), loc_value)
    scale = np.full((24, H), 1.0)
    return ParametricForecastResult(
        dist_name="normal",
        params={"loc": loc, "scale": scale},
        basis_index=basis,
    )


class TestJejuDayAhead:
    def test_full_pipeline(self, tmp_path):
        """Day-ahead uses basis 10:00, horizons 14-37."""
        result = _make_result(loc_value=50.0)
        frames = to_nmape_frames(result)

        obs_file = tmp_path / "obs.csv"
        _make_observed_csv(obs_file, "2024-01-01", 96, value=50.0)

        out_dir = tmp_path / "out"
        evaluator = JejuNMAPEEvaluator(
            capacity=CAPACITY,
            observed_file=obs_file,
            output_dir=out_dir,
            forecast_frames=frames,
        )
        evaluator.run(day_ahead=True, real_time=False)

        for fn in [
            "day_ahead_hourly.csv",
            "day_ahead_daily_nMAPE.csv",
            "day_ahead_monthly_summary.csv",
        ]:
            assert (out_dir / fn).exists(), f"{fn} not created"

        da_hourly = pd.read_csv(out_dir / "day_ahead_hourly.csv")
        np.testing.assert_allclose(da_hourly["nMAPE"], 0.0, atol=1e-9)

    def test_basis_hour_filtering(self, tmp_path):
        """Only basis_time with hour==10 should appear in day-ahead."""
        result = _make_result(loc_value=50.0)
        frames = to_nmape_frames(result)

        obs_file = tmp_path / "obs.csv"
        _make_observed_csv(obs_file, "2024-01-01", 96, value=50.0)

        out_dir = tmp_path / "out"
        evaluator = JejuNMAPEEvaluator(
            capacity=CAPACITY,
            observed_file=obs_file,
            output_dir=out_dir,
            forecast_frames=frames,
        )
        evaluator.run(day_ahead=True, real_time=False)

        da_hourly = pd.read_csv(out_dir / "day_ahead_hourly.csv")
        basis_hours = pd.to_datetime(da_hourly["basis_time"]).dt.hour.unique()
        assert list(basis_hours) == [10]

    def test_da_no_capacity_filter(self, tmp_path):
        """DA should NOT filter by capacity — all rows kept."""
        result = _make_result(loc_value=50.0)
        frames = to_nmape_frames(result)

        obs_file = tmp_path / "obs.csv"
        # value=5.0 < 10% of 100, but Jeju DA should keep it
        _make_observed_csv(obs_file, "2024-01-01", 96, value=5.0)

        out_dir = tmp_path / "out"
        evaluator = JejuNMAPEEvaluator(
            capacity=CAPACITY,
            observed_file=obs_file,
            output_dir=out_dir,
            forecast_frames=frames,
        )
        evaluator.run(day_ahead=True, real_time=False)

        da_hourly = pd.read_csv(out_dir / "day_ahead_hourly.csv")
        assert len(da_hourly) > 0
        # nMAPE = |50 - 5| / 100 * 100 = 45.0
        np.testing.assert_allclose(da_hourly["nMAPE"], 45.0, atol=1e-9)


class TestJejuRealTime:
    def test_rt_no_filter(self, tmp_path):
        """RT should NOT filter by capacity — all rows kept."""
        result = _make_result(loc_value=50.0)
        frames = to_nmape_frames(result)

        obs_file = tmp_path / "obs.csv"
        # value=5.0 < 10% capacity, but RT should keep it
        _make_observed_csv(obs_file, "2024-01-01", 96, value=5.0)

        out_dir = tmp_path / "out"
        evaluator = JejuNMAPEEvaluator(
            capacity=CAPACITY,
            observed_file=obs_file,
            output_dir=out_dir,
            forecast_frames=frames,
        )
        evaluator.run(day_ahead=False, real_time=True)

        rt_hourly = pd.read_csv(out_dir / "real_time_hourly.csv")
        assert len(rt_hourly) > 0
        # nMAPE = |50 - 5| / 100 * 100 = 45.0
        np.testing.assert_allclose(rt_hourly["nMAPE"], 45.0, atol=1e-9)

    def test_rt_ignores_is_valid(self, tmp_path):
        """RT should work even when is_valid=False."""
        result = _make_result(loc_value=50.0)
        frames = to_nmape_frames(result)

        obs_file = tmp_path / "obs.csv"
        _make_observed_csv(
            obs_file, "2024-01-01", 96, value=50.0, is_valid=False
        )

        out_dir = tmp_path / "out"
        evaluator = JejuNMAPEEvaluator(
            capacity=CAPACITY,
            observed_file=obs_file,
            output_dir=out_dir,
            forecast_frames=frames,
        )
        evaluator.run(day_ahead=False, real_time=True)

        rt_hourly = pd.read_csv(out_dir / "real_time_hourly.csv")
        assert len(rt_hourly) > 0
