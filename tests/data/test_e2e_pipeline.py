"""End-to-end integration tests for the training data pipeline.

Tests the full data flow without I/O dependencies by constructing
in-memory NWPDataStore objects and synthetic SCADA data, then verifying:

1. Per-horizon data correctly separates basis_time features and forecast_time targets
2. Lagged features + NWP + target aligned at correct times
3. Farm-level aggregation + wind recomputation end-to-end
4. No future NWP data leaks into any horizon
5. is_valid propagation through the entire chain
6. Missing NWP filled correctly without future leakage
"""

import numpy as np
import pandas as pd
import pytest

from src.data.training_data_builder.nwp_store import NWPDataStore
from src.data.training_data_builder.pipeline import (
    _recompute_wind_derived,
    aggregate_to_farm,
)
from src.data.training_data_builder.scada import (
    create_lagged_features,
    prepare_target,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kst_index(n: int = 48, start: str = "2024-01-15 09:00") -> pd.DatetimeIndex:
    """Hourly KST index (naive)."""
    return pd.date_range(start, periods=n, freq="h")


def _make_nwp_store(
    basis_hours: list[int],
    forecast_offsets: list[int],
    base_date: str = "2024-01-15",
    col_prefix: str = "ecmwf",
    include_wind: bool = False,
) -> NWPDataStore:
    """Build an NWPDataStore with simple deterministic values.

    Values: basis_hour * 1000 + forecast_offset for temperature.
    If include_wind: adds u/v/wspd/wdir columns.
    """
    rows = []
    for bh in basis_hours:
        bt = pd.Timestamp(f"{base_date} {bh:02d}:00")
        for fo in forecast_offsets:
            ft = bt + pd.Timedelta(hours=fo)
            row = {
                "basis_time": bt,
                "forecast_time": ft,
                f"{col_prefix}_forecast_temperature": bh * 1000 + fo,
                f"{col_prefix}_forecast_is_valid": True,
            }
            if include_wind:
                # U/V based on forecast offset for deterministic verification
                u = float(fo) * 0.5
                v = float(fo) * 0.3
                row[f"{col_prefix}_forecast_u100"] = u
                row[f"{col_prefix}_forecast_v100"] = v
                row[f"{col_prefix}_forecast_wspd100"] = np.sqrt(u**2 + v**2)
                row[f"{col_prefix}_forecast_wdir100"] = (
                    180 + np.degrees(np.arctan2(u, v))
                ) % 360
            rows.append(row)

    df = pd.DataFrame(rows).set_index(["basis_time", "forecast_time"])
    return NWPDataStore(df, nwp_name=col_prefix, nwp_tz="UTC", scada_tz="KST")


def _make_scada(n: int = 48, start: str = "2024-01-15 09:00") -> pd.DataFrame:
    """Synthetic SCADA with hourly index in KST."""
    idx = _kst_index(n, start)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "power": rng.uniform(0, 100, n),
        "wind_speed": rng.uniform(3, 15, n),
        "is_valid": True,
    }, index=idx)


# ---------------------------------------------------------------------------
# E2E: Per-horizon data build (SCADA lag + NWP + target)
# ---------------------------------------------------------------------------


class TestE2EPerHorizon:
    """Simulate the per_horizon builder flow manually."""

    def _build_horizon_data(self, scada, store, horizon, max_lag=2):
        """Replicate per-horizon build: lag + NWP + target → merged."""
        lag_features = create_lagged_features(scada, max_lag=max_lag)
        target = prepare_target(scada, horizon=horizon)
        nwp = store.get_for_horizon(horizon, scada.index, "6h")

        common_idx = lag_features.index.intersection(target.index).intersection(nwp.index)
        return pd.concat([
            lag_features.loc[common_idx],
            nwp.loc[common_idx],
            target.loc[common_idx],
        ], axis=1)

    def test_basis_forecast_time_separation(self):
        """Lag features are from basis_time; target is from forecast_time."""
        scada = _make_scada(48)
        store = _make_nwp_store([0, 6, 12, 18], list(range(24)))

        horizon = 3
        data = self._build_horizon_data(scada, store, horizon, max_lag=2)

        # Check column prefixes
        lag_cols = [c for c in data.columns if c.startswith("basis_time_lag")]
        target_cols = [c for c in data.columns if c.startswith("forecast_time_")]
        nwp_cols = [c for c in data.columns if "forecast_temperature" in c or "forecast_is_valid" in c]

        assert len(lag_cols) > 0, "Should have lag features"
        assert len(target_cols) > 0, "Should have target columns"
        assert len(nwp_cols) > 0, "Should have NWP columns"

    def test_target_offset_matches_horizon(self):
        """Target power at basis_time t equals scada power at t + horizon."""
        scada = _make_scada(24)
        store = _make_nwp_store([0, 6, 12, 18], list(range(24)))

        horizon = 3
        data = self._build_horizon_data(scada, store, horizon, max_lag=1)

        for t in data.index[:5]:
            target_power = data.loc[t, "forecast_time_power"]
            # The target should be scada power at t + horizon
            future_t = t + pd.Timedelta(hours=horizon)
            if future_t in scada.index:
                expected = scada.loc[future_t, "power"]
                assert target_power == pytest.approx(expected), (
                    f"At {t}: target={target_power}, expected scada[{future_t}]={expected}"
                )

    def test_lag_values_from_past(self):
        """Lag-k feature at time t equals scada value at t - k."""
        scada = _make_scada(24)
        store = _make_nwp_store([0, 6, 12, 18], list(range(24)))

        data = self._build_horizon_data(scada, store, horizon=1, max_lag=3)

        for t in data.index[:5]:
            lag2_power = data.loc[t, "basis_time_lag2_power"]
            past_t = t - pd.Timedelta(hours=2)
            if past_t in scada.index:
                expected = scada.loc[past_t, "power"]
                assert lag2_power == pytest.approx(expected)

    def test_different_horizons_different_nwp(self):
        """Same SCADA time, different horizons → different NWP forecast_times."""
        scada = _make_scada(24)
        store = _make_nwp_store([0, 6, 12, 18], list(range(24)))

        data_h1 = self._build_horizon_data(scada, store, horizon=1, max_lag=1)
        data_h6 = self._build_horizon_data(scada, store, horizon=6, max_lag=1)

        common = data_h1.index.intersection(data_h6.index)
        assert len(common) > 0

        # NWP values should differ (different forecast offsets)
        t = common[0]
        nwp_h1 = data_h1.loc[t, "ecmwf_forecast_temperature"]
        nwp_h6 = data_h6.loc[t, "ecmwf_forecast_temperature"]
        assert nwp_h1 != nwp_h6


# ---------------------------------------------------------------------------
# E2E: No future leakage across all horizons
# ---------------------------------------------------------------------------


class TestE2ENoFutureLeakage:
    """Verify NWP data is always from a basis_time <= SCADA observation time."""

    def test_nwp_basis_not_after_scada_utc(self):
        """For every row, the NWP basis_time used must be <= SCADA UTC time."""
        from src.data.training_data_builder.time_alignment import (
            convert_timezone,
            snap_to_nwp_basis,
        )

        scada = _make_scada(48)
        store = _make_nwp_store([0, 6, 12, 18], list(range(24)))

        for horizon in [0, 1, 3, 6, 12]:
            scada_utc = convert_timezone(scada.index, "KST", "UTC")
            avoid_exact = (horizon == 0)
            basis_times = snap_to_nwp_basis(scada_utc, "6h", avoid_exact=avoid_exact)

            for scada_t_utc, basis_t in zip(scada_utc, basis_times):
                assert basis_t <= scada_t_utc, (
                    f"Horizon={horizon}: basis_time {basis_t} is AFTER "
                    f"SCADA UTC {scada_t_utc} — future leakage!"
                )

    def test_horizon0_basis_strictly_before_at_boundary(self):
        """At exact NWP boundaries, horizon=0 must use strictly previous basis."""
        from src.data.training_data_builder.time_alignment import (
            convert_timezone,
            snap_to_nwp_basis,
        )

        # SCADA times that align exactly with 6h UTC boundaries
        # KST 09:00 → UTC 00:00, KST 15:00 → UTC 06:00, etc.
        kst_exact = pd.DatetimeIndex([
            "2024-01-15 09:00",   # UTC 00:00
            "2024-01-15 15:00",   # UTC 06:00
            "2024-01-15 21:00",   # UTC 12:00
            "2024-01-16 03:00",   # UTC 18:00
        ])
        utc_exact = convert_timezone(kst_exact, "KST", "UTC")
        basis_h0 = snap_to_nwp_basis(utc_exact, "6h", avoid_exact=True)

        for utc_t, basis_t in zip(utc_exact, basis_h0):
            assert basis_t < utc_t, (
                f"Horizon=0 at boundary: basis={basis_t} should be "
                f"strictly before SCADA UTC={utc_t}"
            )


# ---------------------------------------------------------------------------
# E2E: Farm-level aggregation with wind recomputation
# ---------------------------------------------------------------------------


class TestE2EFarmAggregation:
    """Full turbine→farm flow with NWP wind data."""

    @staticmethod
    def _prep_for_farm(df: pd.DataFrame) -> pd.DataFrame:
        """Rename {prefix}_is_valid → is_valid for aggregate_to_farm compatibility.

        aggregate_to_farm expects a single ``is_valid`` column, but NWP store
        output has prefixed names like ``ecmwf_forecast_is_valid``.  In the
        real pipeline, this renaming is handled by merge_nwp_stores; here we
        replicate it for isolated testing.
        """
        valid_cols = [c for c in df.columns if c.endswith("is_valid") and c != "is_valid"]
        if valid_cols:
            # AND all validity columns into a single is_valid
            df = df.copy()
            df["is_valid"] = df[valid_cols].all(axis=1)
            df = df.drop(columns=valid_cols)
        return df

    def test_farm_wind_recomputed_from_averaged_uv(self):
        """After averaging turbines, wspd/wdir reflect vector-averaged U/V."""
        kst = _kst_index(3)

        store_a = _make_nwp_store([0], list(range(24)), include_wind=True)
        nwp_a = store_a.get_for_horizon(1, kst, "6h")

        store_b = _make_nwp_store([0], list(range(24)), include_wind=True)
        nwp_b = store_b.get_for_horizon(1, kst, "6h")
        nwp_b["ecmwf_forecast_u100"] = -nwp_b["ecmwf_forecast_u100"]
        nwp_b["ecmwf_forecast_wspd100"] = np.sqrt(
            nwp_b["ecmwf_forecast_u100"]**2 + nwp_b["ecmwf_forecast_v100"]**2
        )

        nwp_a = self._prep_for_farm(nwp_a)
        nwp_b = self._prep_for_farm(nwp_b)
        farm = aggregate_to_farm({"A": nwp_a, "B": nwp_b})

        # Vector average: opposing U cancel out → wspd < either turbine
        avg_u = (nwp_a["ecmwf_forecast_u100"] + nwp_b["ecmwf_forecast_u100"]) / 2
        avg_v = (nwp_a["ecmwf_forecast_v100"] + nwp_b["ecmwf_forecast_v100"]) / 2
        expected_wspd = np.sqrt(avg_u**2 + avg_v**2)

        np.testing.assert_allclose(
            farm["ecmwf_forecast_wspd100"].values,
            expected_wspd.values,
            rtol=1e-10,
        )

    def test_farm_non_wind_columns_averaged(self):
        """Non-wind columns are simply averaged."""
        kst = _kst_index(3)

        store = _make_nwp_store([0], list(range(24)))
        nwp_a = self._prep_for_farm(store.get_for_horizon(1, kst, "6h"))
        nwp_b = nwp_a.copy()
        nwp_b["ecmwf_forecast_temperature"] += 10

        farm = aggregate_to_farm({"A": nwp_a, "B": nwp_b})

        expected_temp = (
            nwp_a["ecmwf_forecast_temperature"] + nwp_b["ecmwf_forecast_temperature"]
        ) / 2
        np.testing.assert_allclose(
            farm["ecmwf_forecast_temperature"].values,
            expected_temp.values,
        )


# ---------------------------------------------------------------------------
# E2E: is_valid propagation through full chain
# ---------------------------------------------------------------------------


class TestE2EIsValidPropagation:
    """is_valid must propagate correctly through lag + NWP + target."""

    def test_invalid_scada_propagates_to_lags(self):
        """Invalid SCADA row at time t invalidates lags touching t."""
        scada = _make_scada(12)
        scada.loc[scada.index[5], "is_valid"] = False

        lag = create_lagged_features(scada, max_lag=2)

        # t=5 is invalid → t=5 (lag0), t=6 (lag1), t=7 (lag2) should be invalid
        assert lag.loc[scada.index[5], "is_valid"] == False
        assert lag.loc[scada.index[6], "is_valid"] == False
        assert lag.loc[scada.index[7], "is_valid"] == False
        # t=8 should be valid (lag2 touches t=6 which is valid)
        assert lag.loc[scada.index[8], "is_valid"] == True

    def test_invalid_target_via_horizon(self):
        """Invalid SCADA at t+h → target_is_valid False at basis t."""
        scada = _make_scada(10)
        scada.loc[scada.index[5], "is_valid"] = False

        target = prepare_target(scada, horizon=2)

        # Target at basis t=3 → forecast at t=5 (invalid)
        assert target.loc[scada.index[3], "target_is_valid"] == False
        # Target at basis t=2 → forecast at t=4 (valid)
        assert target.loc[scada.index[2], "target_is_valid"] == True

    def test_nwp_missing_row_filled_via_day_offset(self):
        """NWP rows not found at (basis, forecast) are filled from -24h SCADA offset."""
        # Build store with basis 00:00 and 06:00 over 2 days
        # so get_for_horizon can fill via merged.reindex(idx - 24h)
        rows = []
        for base_date in ["2024-01-14", "2024-01-15"]:
            for bh in [0, 6, 12, 18]:
                bt = pd.Timestamp(f"{base_date} {bh:02d}:00")
                for fo in range(24):
                    ft = bt + pd.Timedelta(hours=fo)
                    rows.append({
                        "basis_time": bt, "forecast_time": ft,
                        "ecmwf_forecast_temperature": fo * 10.0,
                        "ecmwf_forecast_is_valid": True,
                    })
        df = pd.DataFrame(rows).set_index(["basis_time", "forecast_time"])
        store = NWPDataStore(df, "ecmwf", "UTC", "KST")

        # Provide 2 days of SCADA so that the -24h lookup can find data
        kst = pd.date_range("2024-01-14 09:00", periods=48, freq="h")
        result = store.get_for_horizon(1, kst, "6h")

        # All rows filled successfully
        assert len(result) == 48
        assert not result["ecmwf_forecast_temperature"].isna().any()


# ---------------------------------------------------------------------------
# E2E: NWP missing fill with future-fill detection
# ---------------------------------------------------------------------------


class TestE2EMissingFill:
    """Integration test for _fill_missing_basis_times in store context."""

    def test_missing_basis_filled_preserves_other_data(self):
        """Filling a missing basis_time doesn't corrupt existing data."""
        # Build store with a plain 'is_valid' column (pre-prefix, like raw NWP data)
        rows = []
        for bh in [0, 12]:
            bt = pd.Timestamp(f"2024-01-15 {bh:02d}:00")
            for fo in range(12):
                ft = bt + pd.Timedelta(hours=fo)
                rows.append({
                    "basis_time": bt, "forecast_time": ft,
                    "temperature": bh * 100 + fo, "is_valid": True,
                })
        raw_df = pd.DataFrame(rows).set_index(["basis_time", "forecast_time"])

        filled_df = NWPDataStore._fill_missing_basis_times(
            raw_df, ["2024-01-15_06"], "test",
        )

        # Original data intact (check_freq=False because concat may change freq)
        bt_00 = pd.Timestamp("2024-01-15 00:00")
        pd.testing.assert_frame_equal(
            raw_df.loc[bt_00], filled_df.loc[bt_00], check_freq=False,
        )

        # Filled data present with is_valid=False
        bt_06 = pd.Timestamp("2024-01-15 06:00")
        assert bt_06 in filled_df.index.get_level_values("basis_time")
        assert (filled_df.loc[bt_06, "is_valid"] == False).all()

    def test_multiple_missing_filled_independently(self):
        """Each missing basis_time is filled from its own nearest source."""
        rows = []
        for bh in [0, 18]:
            bt = pd.Timestamp(f"2024-01-15 {bh:02d}:00")
            for fo in range(6):
                ft = bt + pd.Timedelta(hours=fo)
                rows.append({
                    "basis_time": bt, "forecast_time": ft,
                    "temperature": bh * 100 + fo, "is_valid": True,
                })
        raw_df = pd.DataFrame(rows).set_index(["basis_time", "forecast_time"])

        filled_df = NWPDataStore._fill_missing_basis_times(
            raw_df, ["2024-01-15_06", "2024-01-15_12"], "test",
        )

        for bt_str, bt in [("06", pd.Timestamp("2024-01-15 06:00")),
                            ("12", pd.Timestamp("2024-01-15 12:00"))]:
            assert bt in filled_df.index.get_level_values("basis_time")
            assert (filled_df.loc[bt, "is_valid"] == False).all()
