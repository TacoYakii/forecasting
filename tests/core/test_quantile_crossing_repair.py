"""Tests for quantile crossing repair via isotonic regression (PAVA)."""

import numpy as np
import pandas as pd
import pytest

from src.core.forecast_results import (
    QuantileForecastResult,
    _repair_quantile_crossings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quantiles_data(levels, values_3d):
    """Build quantiles_data dict from levels and (N, Q, H) array."""
    return {lv: values_3d[:, i, :] for i, lv in enumerate(levels)}


def _is_monotone(quantiles_data):
    """Check that quantile values are non-decreasing across levels."""
    levels = sorted(quantiles_data.keys())
    stacked = np.stack([quantiles_data[lv] for lv in levels], axis=1)
    return np.all(np.diff(stacked, axis=1) >= 0)


# ---------------------------------------------------------------------------
# Tests for _repair_quantile_crossings
# ---------------------------------------------------------------------------

class TestRepairQuantileCrossings:
    """Tests for the standalone repair function."""

    def test_already_monotone_unchanged(self):
        """Already monotone input is returned exactly unchanged."""
        levels = [0.1, 0.5, 0.9]
        # (N=3, Q=3, H=2), strictly increasing across Q
        vals = np.array([[[1, 2], [3, 4], [5, 6]],
                         [[0, 1], [2, 3], [4, 5]],
                         [[10, 20], [30, 40], [50, 60]]], dtype=float)
        data = _make_quantiles_data(levels, vals)
        repaired = _repair_quantile_crossings(data)

        for lv in levels:
            np.testing.assert_array_equal(repaired[lv], data[lv])

    def test_simple_crossing_two_levels(self):
        """Two-level crossing: both values become their mean."""
        data = {0.1: np.array([[5.0]]), 0.9: np.array([[3.0]])}
        repaired = _repair_quantile_crossings(data)

        assert repaired[0.1][0, 0] == pytest.approx(4.0)
        assert repaired[0.9][0, 0] == pytest.approx(4.0)

    def test_multi_level_crossing(self):
        """Multi-level crossing produces non-decreasing output."""
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        vals = np.array([[[5.0], [3.0], [4.0], [2.0], [6.0]]])  # (1,5,1)
        data = _make_quantiles_data(levels, vals)
        repaired = _repair_quantile_crossings(data)

        assert _is_monotone(repaired)

    def test_row_independence(self):
        """Repair is applied per-row: monotone rows stay unchanged."""
        levels = [0.1, 0.5, 0.9]
        vals = np.array([
            [[1.0], [5.0], [3.0]],   # crossed
            [[1.0], [2.0], [3.0]],   # monotone
        ])
        data = _make_quantiles_data(levels, vals)
        repaired = _repair_quantile_crossings(data)

        assert _is_monotone(repaired)
        # Monotone row preserved exactly
        np.testing.assert_array_equal(
            repaired[0.1][1], data[0.1][1]
        )
        np.testing.assert_array_equal(
            repaired[0.5][1], data[0.5][1]
        )
        np.testing.assert_array_equal(
            repaired[0.9][1], data[0.9][1]
        )

    def test_horizon_independence(self):
        """Each horizon column is repaired independently."""
        levels = [0.1, 0.5, 0.9]
        vals = np.array([
            [[1.0, 5.0], [2.0, 3.0], [3.0, 4.0]],  # h=0 ok, h=1 crossed
        ])
        data = _make_quantiles_data(levels, vals)
        repaired = _repair_quantile_crossings(data)

        assert _is_monotone(repaired)
        # h=0 column preserved exactly
        for lv in levels:
            assert repaired[lv][0, 0] == data[lv][0, 0]

    def test_1d_arrays_rejected(self):
        """1-D arrays are not handled by repair (constructor normalizes to 2-D)."""
        # _repair_quantile_crossings expects 2-D arrays;
        # 1-D normalization happens in QuantileForecastResult.__init__
        data = {0.1: np.array([[5.0], [1.0]]), 0.9: np.array([[3.0], [2.0]])}
        repaired = _repair_quantile_crossings(data)

        assert repaired[0.1].ndim == 2
        assert repaired[0.9][0, 0] >= repaired[0.1][0, 0]
        # Already monotone row preserved
        assert repaired[0.1][1, 0] == 1.0
        assert repaired[0.9][1, 0] == 2.0

    def test_single_level_passthrough(self):
        """Single quantile level: trivially monotone, no change."""
        data = {0.5: np.array([[7.0, 8.0]])}
        repaired = _repair_quantile_crossings(data)

        np.testing.assert_array_equal(repaired[0.5], data[0.5])

    def test_all_equal_values(self):
        """All-equal values are non-decreasing, preserved exactly."""
        levels = [0.1, 0.5, 0.9]
        vals = np.array([[[3.0], [3.0], [3.0]]])
        data = _make_quantiles_data(levels, vals)
        repaired = _repair_quantile_crossings(data)

        for lv in levels:
            np.testing.assert_array_equal(repaired[lv], data[lv])

    def test_nan_cells_skipped(self):
        """Cells containing NaN are passed through without error."""
        levels = [0.1, 0.5, 0.9]
        vals = np.array([
            [[np.nan], [1.0], [2.0]],   # NaN row — skipped
            [[5.0], [3.0], [4.0]],       # crossed — repaired
        ])
        data = _make_quantiles_data(levels, vals)
        repaired = _repair_quantile_crossings(data)

        # NaN preserved
        assert np.isnan(repaired[0.1][0, 0])
        # Crossed row repaired
        repaired_vec = [repaired[lv][1, 0] for lv in levels]
        assert all(b >= a for a, b in zip(repaired_vec, repaired_vec[1:]))


# ---------------------------------------------------------------------------
# Integration: QuantileForecastResult constructor
# ---------------------------------------------------------------------------

class TestQuantileForecastResultCrossingRepair:
    """Verify repair is applied at construction time."""

    @pytest.fixture
    def basis_index(self):
        return pd.date_range("2024-01-01", periods=3, freq="h")

    def test_constructor_enforces_monotonicity(self, basis_index):
        """Crossed data becomes monotone after construction."""
        N, H = len(basis_index), 2
        result = QuantileForecastResult(
            quantiles_data={
                0.1: np.full((N, H), 5.0),
                0.5: np.full((N, H), 3.0),
                0.9: np.full((N, H), 4.0),
            },
            basis_index=basis_index,
        )

        assert _is_monotone(result.quantiles_data)

    def test_to_distribution_monotone(self, basis_index):
        """ppf from repaired result is non-decreasing."""
        N, H = len(basis_index), 2
        result = QuantileForecastResult(
            quantiles_data={
                0.1: np.full((N, H), 5.0),
                0.9: np.full((N, H), 3.0),
            },
            basis_index=basis_index,
        )
        dist = result.to_distribution(h=1)
        tau = np.array([0.1, 0.9])
        ppf_vals = dist.ppf(tau)  # (N, 2)
        assert np.all(ppf_vals[:, 1] >= ppf_vals[:, 0])

    def test_reindex_preserves_monotonicity(self, basis_index):
        """Reindexed result remains monotone (no double-repair issue)."""
        N, H = len(basis_index), 1
        result = QuantileForecastResult(
            quantiles_data={
                0.1: np.array([[5.0], [1.0], [5.0]]),
                0.9: np.array([[3.0], [2.0], [3.0]]),
            },
            basis_index=basis_index,
        )
        subset = basis_index[:2]
        reindexed = result.reindex(subset)

        assert _is_monotone(reindexed.quantiles_data)

    def test_1d_input_normalizes_and_works(self, basis_index):
        """1-D arrays are normalized to 2-D; to_distribution and reindex work."""
        N = len(basis_index)
        result = QuantileForecastResult(
            quantiles_data={
                0.1: np.full(N, 5.0),   # 1-D
                0.9: np.full(N, 3.0),   # 1-D, crossed
            },
            basis_index=basis_index,
        )
        # Stored as 2-D
        assert result.quantiles_data[0.1].ndim == 2
        assert result.horizon == 1
        # Monotone after repair
        assert _is_monotone(result.quantiles_data)
        # to_distribution works
        dist = result.to_distribution(h=1)
        assert dist.ppf(np.array([0.9]))[0, 0] >= dist.ppf(np.array([0.1]))[0, 0]
        # reindex works
        reindexed = result.reindex(basis_index[:2])
        assert len(reindexed) == 2
        assert _is_monotone(reindexed.quantiles_data)
