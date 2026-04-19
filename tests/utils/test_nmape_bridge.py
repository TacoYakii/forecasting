"""Tests for the ForecastResult -> NMAPEEvaluator bridge."""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import (
    GridForecastResult,
    ParametricForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
)
from src.utils.nMAPE import NMAPEEvaluator, to_nmape_frames


# ---------------------------------------------------------------------------
# Adapter unit tests
# ---------------------------------------------------------------------------


N, H = 10, 3


@pytest.fixture
def basis_index():
    return pd.date_range("2024-01-01 00:00", periods=N, freq="h")


def _check_frames(frames, basis, horizons, expected_mu=None):
    assert set(frames.keys()) == set(horizons)
    step = pd.Timedelta(hours=1)
    for h in horizons:
        df = frames[h]
        assert list(df.columns) == ["basis_time", "forecast_time", "mu"]
        assert len(df) == len(basis)
        assert (df["forecast_time"] == basis + h * step).all()
        assert (df["basis_time"] == basis).all()
        if expected_mu is not None:
            np.testing.assert_allclose(df["mu"].to_numpy(), expected_mu)


class TestAdapter:
    def test_parametric_normal(self, basis_index):
        loc = np.full((N, H), 5.0)
        scale = np.full((N, H), 1.0)
        result = ParametricForecastResult(
            dist_name="normal",
            params={"loc": loc, "scale": scale},
            basis_index=basis_index,
        )
        frames = to_nmape_frames(result)
        _check_frames(
            frames, basis_index, [1, 2, 3], expected_mu=np.full(N, 5.0)
        )

    def test_quantile(self, basis_index):
        # Symmetric quantiles around 10.0 -> mean ≈ 10.0
        q_data = {
            0.1: np.full((N, H), 8.0),
            0.5: np.full((N, H), 10.0),
            0.9: np.full((N, H), 12.0),
        }
        result = QuantileForecastResult(
            quantiles_data=q_data, basis_index=basis_index
        )
        frames = to_nmape_frames(result)
        assert set(frames.keys()) == {1, 2, 3}
        # QuantileDistribution.mean() integrates the piecewise-linear CDF —
        # check it's in a sensible range, not exact equality.
        for h in (1, 2, 3):
            mu = frames[h]["mu"].to_numpy()
            assert mu.shape == (N,)
            assert np.all((mu >= 8.0) & (mu <= 12.0))

    def test_sample(self, basis_index):
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=3.0, scale=0.5, size=(N, 200, H))
        result = SampleForecastResult(samples=samples, basis_index=basis_index)
        frames = to_nmape_frames(result)
        assert set(frames.keys()) == {1, 2, 3}
        for h in (1, 2, 3):
            mu = frames[h]["mu"].to_numpy()
            np.testing.assert_allclose(
                mu, samples[:, :, h - 1].mean(axis=1)
            )

    def test_grid(self, basis_index):
        G = 11
        grid = np.linspace(0.0, 10.0, G)
        # Point mass at bin index 5 (value = 5.0) for all (n, h)
        prob = np.zeros((N, G, H))
        prob[:, 5, :] = 1.0
        result = GridForecastResult(
            grid=grid, prob=prob, basis_index=basis_index
        )
        frames = to_nmape_frames(result)
        _check_frames(
            frames, basis_index, [1, 2, 3], expected_mu=np.full(N, 5.0)
        )

    def test_horizons_subset(self, basis_index):
        loc = np.full((N, H), 1.0)
        scale = np.full((N, H), 1.0)
        result = ParametricForecastResult(
            dist_name="normal",
            params={"loc": loc, "scale": scale},
            basis_index=basis_index,
        )
        frames = to_nmape_frames(result, horizons=[2])
        assert set(frames.keys()) == {2}
        assert len(frames[2]) == N

    def test_custom_step(self, basis_index):
        loc = np.full((N, 2), 0.0)
        scale = np.full((N, 2), 1.0)
        result = ParametricForecastResult(
            dist_name="normal",
            params={"loc": loc, "scale": scale},
            basis_index=basis_index,
        )
        frames = to_nmape_frames(result, step=pd.Timedelta(minutes=30))
        assert (
            frames[1]["forecast_time"] == basis_index + pd.Timedelta(minutes=30)
        ).all()
        assert (
            frames[2]["forecast_time"] == basis_index + pd.Timedelta(hours=1)
        ).all()

    def test_non_datetime_basis_raises(self):
        loc = np.zeros((N, 1))
        scale = np.ones((N, 1))
        result = ParametricForecastResult(
            dist_name="normal",
            params={"loc": loc, "scale": scale},
            basis_index=pd.RangeIndex(N),
        )
        with pytest.raises(TypeError, match="DatetimeIndex"):
            to_nmape_frames(result)

    def test_out_of_range_horizon_raises(self, basis_index):
        loc = np.zeros((N, H))
        scale = np.ones((N, H))
        result = ParametricForecastResult(
            dist_name="normal",
            params={"loc": loc, "scale": scale},
            basis_index=basis_index,
        )
        with pytest.raises(ValueError, match="out of range"):
            to_nmape_frames(result, horizons=[99])


# ---------------------------------------------------------------------------
# NMAPEEvaluator integration tests
# ---------------------------------------------------------------------------


CAPACITY = 100.0


def _make_observed_csv(path, start, periods, value=50.0):
    idx = pd.date_range(start, periods=periods, freq="h")
    df = pd.DataFrame(
        {
            "forecast_time": idx,
            "forecast_time_observed_KPX_pwr": np.full(periods, value),
            "is_valid": np.ones(periods, dtype=bool),
        }
    )
    df.to_csv(path, index=False)


def _make_result_for_kpx(loc_value=50.0):
    """24 hourly basis times on 2024-01-01, horizons 1..40."""
    basis = pd.date_range("2024-01-01 00:00", periods=24, freq="h")
    H_full = 40
    loc = np.full((24, H_full), loc_value)
    scale = np.full((24, H_full), 1.0)
    return ParametricForecastResult(
        dist_name="normal",
        params={"loc": loc, "scale": scale},
        basis_index=basis,
    )


class TestEvaluatorWithFrames:
    def test_xor_validation_both_none(self, tmp_path):
        obs_file = tmp_path / "obs.csv"
        _make_observed_csv(obs_file, "2024-01-01", 24)
        with pytest.raises(ValueError, match="exactly one"):
            NMAPEEvaluator(
                capacity=CAPACITY,
                observed_file=obs_file,
                output_dir=tmp_path / "out",
            )

    def test_xor_validation_both_given(self, tmp_path):
        obs_file = tmp_path / "obs.csv"
        _make_observed_csv(obs_file, "2024-01-01", 24)
        with pytest.raises(ValueError, match="exactly one"):
            NMAPEEvaluator(
                capacity=CAPACITY,
                observed_file=obs_file,
                output_dir=tmp_path / "out",
                forecast_files={1: "a.csv"},
                forecast_frames={1: pd.DataFrame()},
            )

    def test_missing_columns_in_frame(self, tmp_path):
        obs_file = tmp_path / "obs.csv"
        _make_observed_csv(obs_file, "2024-01-01", 24)
        bad_frame = pd.DataFrame({"basis_time": [], "mu": []})
        evaluator = NMAPEEvaluator(
            capacity=CAPACITY,
            observed_file=obs_file,
            output_dir=tmp_path / "out",
            forecast_frames={1: bad_frame},
        )
        with pytest.raises(KeyError, match="forecast_time"):
            evaluator._load_data()

    def test_full_pipeline(self, tmp_path):
        result = _make_result_for_kpx(loc_value=50.0)
        frames = to_nmape_frames(result)

        obs_file = tmp_path / "obs.csv"
        # Need coverage from basis[0] + 1h to basis[-1] + 40h.
        # basis[-1] = 2024-01-01 23:00 + 40h = 2024-01-03 15:00.
        # Cover generously: 2024-01-01 00:00 + 96h.
        _make_observed_csv(obs_file, "2024-01-01", 96, value=50.0)

        out_dir = tmp_path / "out"
        evaluator = NMAPEEvaluator(
            capacity=CAPACITY,
            observed_file=obs_file,
            output_dir=out_dir,
            forecast_frames=frames,
        )
        evaluator.run(day_ahead=True, real_time=True)

        for fn in [
            "day_ahead_hourly.csv",
            "day_ahead_daily_nMAPE.csv",
            "day_ahead_monthly_summary.csv",
            "real_time_hourly.csv",
            "real_time_daily_nMAPE.csv",
            "real_time_monthly_summary.csv",
        ]:
            assert (out_dir / fn).exists(), f"{fn} not created"

        # mu == observed (both 50.0) so nMAPE should be 0.
        da_hourly = pd.read_csv(out_dir / "day_ahead_hourly.csv")
        np.testing.assert_allclose(da_hourly["nMAPE_DA"], 0.0, atol=1e-9)

        rt_hourly = pd.read_csv(out_dir / "real_time_hourly.csv")
        np.testing.assert_allclose(rt_hourly["nMAPE"], 0.0, atol=1e-9)

    def test_nonzero_nmape(self, tmp_path):
        # forecast mu = 50, observed = 60 -> |50-60|/100*100 = 10.0
        result = _make_result_for_kpx(loc_value=50.0)
        frames = to_nmape_frames(result)
        obs_file = tmp_path / "obs.csv"
        _make_observed_csv(obs_file, "2024-01-01", 96, value=60.0)

        out_dir = tmp_path / "out"
        evaluator = NMAPEEvaluator(
            capacity=CAPACITY,
            observed_file=obs_file,
            output_dir=out_dir,
            forecast_frames=frames,
        )
        evaluator.run(day_ahead=True, real_time=True)

        da_hourly = pd.read_csv(out_dir / "day_ahead_hourly.csv")
        np.testing.assert_allclose(da_hourly["nMAPE_DA"], 10.0, atol=1e-9)

        rt_hourly = pd.read_csv(out_dir / "real_time_hourly.csv")
        np.testing.assert_allclose(rt_hourly["nMAPE"], 10.0, atol=1e-9)
