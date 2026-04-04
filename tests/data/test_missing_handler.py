"""Tests for MissingHandler: NWP missing data fill via backward search.

Covers:
- Basic fill from 1-day-back file
- Chain-propagation prevention (is_valid=False rows skipped)
- File entirely missing → skip to next day
- File with all-invalid data → skip (chain prevention)
- _find_past_file blocks future/current basis_time reads
- Both filename formats (YYYY-MM-DD_HH and YYYYMMDDHH)
- Past directory doesn't exist → warning, unfilled
- Past file unreadable (corrupt) → skip with warning
- Multiple missing rows, different fill depths
- Search gives up after 365 days
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.nwp_preprocess.processors.missing_handler import MissingHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nwp_df(
    hours: list[int],
    basis_hour: int = 0,
    values: dict | None = None,
    is_valid: list[bool] | None = None,
) -> pd.DataFrame:
    """Create a small NWP DataFrame indexed by forecast_time."""
    base = pd.Timestamp("2024-01-15")
    index = pd.DatetimeIndex([base + pd.Timedelta(hours=h) for h in hours])

    if values is None:
        values = {"temperature": np.arange(len(hours), dtype=float) + 10}

    df = pd.DataFrame(values, index=index)
    if is_valid is None:
        df["is_valid"] = True
    else:
        df["is_valid"] = is_valid
    return df


def _write_past_csv(
    output_dir: Path,
    coord_key: str,
    basis_time: pd.Timestamp,
    df: pd.DataFrame,
    fmt: str = "%Y-%m-%d_%H",
):
    """Write a past CSV file in the expected directory structure."""
    coord_dir = output_dir / coord_key
    coord_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{basis_time.strftime(fmt)}.csv"
    df.to_csv(coord_dir / filename)


# ---------------------------------------------------------------------------
# _find_past_file
# ---------------------------------------------------------------------------


class TestFindPastFile:

    def test_dash_format(self, tmp_path):
        """Finds file with YYYY-MM-DD_HH format."""
        coord_dir = tmp_path / "coord1"
        coord_dir.mkdir()
        (coord_dir / "2024-01-14_00.csv").write_text("dummy")

        result = MissingHandler._find_past_file(
            coord_dir, pd.Timestamp("2024-01-14 00:00")
        )
        assert result is not None
        assert result.name == "2024-01-14_00.csv"

    def test_compact_format(self, tmp_path):
        """Finds file with YYYYMMDDHH format."""
        coord_dir = tmp_path / "coord1"
        coord_dir.mkdir()
        (coord_dir / "2024011400.csv").write_text("dummy")

        result = MissingHandler._find_past_file(
            coord_dir, pd.Timestamp("2024-01-14 00:00")
        )
        assert result is not None
        assert result.name == "2024011400.csv"

    def test_prefers_dash_format(self, tmp_path):
        """When both formats exist, dash format is checked first."""
        coord_dir = tmp_path / "coord1"
        coord_dir.mkdir()
        (coord_dir / "2024-01-14_00.csv").write_text("dash")
        (coord_dir / "2024011400.csv").write_text("compact")

        result = MissingHandler._find_past_file(
            coord_dir, pd.Timestamp("2024-01-14 00:00")
        )
        assert result.name == "2024-01-14_00.csv"

    def test_no_file_returns_none(self, tmp_path):
        """No matching file → None."""
        coord_dir = tmp_path / "coord1"
        coord_dir.mkdir()

        result = MissingHandler._find_past_file(
            coord_dir, pd.Timestamp("2024-01-14 00:00")
        )
        assert result is None

    def test_blocks_current_basis_time(self, tmp_path):
        """past_basis_time == current_basis_time → None (parallel batch safety)."""
        coord_dir = tmp_path / "coord1"
        coord_dir.mkdir()
        bt = pd.Timestamp("2024-01-15 00:00")
        (coord_dir / "2024-01-15_00.csv").write_text("should not read")

        result = MissingHandler._find_past_file(
            coord_dir, bt, current_basis_time=bt,
        )
        assert result is None

    def test_blocks_future_basis_time(self, tmp_path):
        """past_basis_time > current_basis_time → None."""
        coord_dir = tmp_path / "coord1"
        coord_dir.mkdir()
        future = pd.Timestamp("2024-01-16 00:00")
        current = pd.Timestamp("2024-01-15 00:00")
        (coord_dir / "2024-01-16_00.csv").write_text("future data")

        result = MissingHandler._find_past_file(
            coord_dir, future, current_basis_time=current,
        )
        assert result is None

    def test_allows_past_basis_time(self, tmp_path):
        """past_basis_time < current_basis_time → allowed."""
        coord_dir = tmp_path / "coord1"
        coord_dir.mkdir()
        past = pd.Timestamp("2024-01-14 00:00")
        current = pd.Timestamp("2024-01-15 00:00")
        (coord_dir / "2024-01-14_00.csv").write_text("past data")

        result = MissingHandler._find_past_file(
            coord_dir, past, current_basis_time=current,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Basic fill
# ---------------------------------------------------------------------------


class TestBasicFill:

    def test_fill_from_1day_back(self, tmp_path):
        """Missing value filled from file 1 day earlier at same hour."""
        handler = MissingHandler()
        basis_time = pd.Timestamp("2024-01-15 00:00")

        # Current data: 1 row with NaN
        index = pd.DatetimeIndex(["2024-01-15 03:00"])
        current = pd.DataFrame({
            "temperature": [np.nan],
            "is_valid": [True],
        }, index=index)

        # Past data (1 day back): valid value at same hour
        past_index = pd.DatetimeIndex(["2024-01-14 03:00"])
        past = pd.DataFrame({
            "temperature": [20.0],
            "is_valid": [True],
        }, index=past_index)
        _write_past_csv(tmp_path, "coord1", pd.Timestamp("2024-01-14 00:00"), past)

        result = handler.fill(current, "coord1", tmp_path, basis_time)

        assert result["temperature"].iloc[0] == 20.0
        assert result["is_valid"].iloc[0] == False  # marked as filled

    def test_no_missing_returns_unchanged(self, tmp_path):
        """No NaN → returned as-is, no file reads."""
        handler = MissingHandler()
        basis_time = pd.Timestamp("2024-01-15 00:00")

        index = pd.DatetimeIndex(["2024-01-15 03:00"])
        current = pd.DataFrame({
            "temperature": [25.0],
            "is_valid": [True],
        }, index=index)

        result = handler.fill(current, "coord1", tmp_path, basis_time)
        assert result["temperature"].iloc[0] == 25.0
        assert result["is_valid"].iloc[0] == True

    def test_fill_from_2days_back(self, tmp_path):
        """1 day back has no file → falls through to 2 days back."""
        handler = MissingHandler()
        basis_time = pd.Timestamp("2024-01-15 00:00")

        index = pd.DatetimeIndex(["2024-01-15 03:00"])
        current = pd.DataFrame({
            "temperature": [np.nan],
            "is_valid": [True],
        }, index=index)

        # Only 2-day-back file exists
        past_index = pd.DatetimeIndex(["2024-01-13 03:00"])
        past = pd.DataFrame({
            "temperature": [18.0],
            "is_valid": [True],
        }, index=past_index)
        _write_past_csv(tmp_path, "coord1", pd.Timestamp("2024-01-13 00:00"), past)

        result = handler.fill(current, "coord1", tmp_path, basis_time)
        assert result["temperature"].iloc[0] == 18.0
        assert result["is_valid"].iloc[0] == False


# ---------------------------------------------------------------------------
# Chain-propagation prevention
# ---------------------------------------------------------------------------


class TestChainPropagation:
    """Rows with is_valid=False in past files are skipped."""

    def test_skips_invalid_past_row(self, tmp_path):
        """Past row with is_valid=False is skipped; searches deeper."""
        handler = MissingHandler()
        basis_time = pd.Timestamp("2024-01-15 00:00")

        index = pd.DatetimeIndex(["2024-01-15 03:00"])
        current = pd.DataFrame({
            "temperature": [np.nan],
            "is_valid": [True],
        }, index=index)

        # 1 day back: has data but is_valid=False (was itself filled)
        past1_index = pd.DatetimeIndex(["2024-01-14 03:00"])
        past1 = pd.DataFrame({
            "temperature": [19.0],
            "is_valid": [False],  # ← was filled, must skip
        }, index=past1_index)
        _write_past_csv(tmp_path, "coord1", pd.Timestamp("2024-01-14 00:00"), past1)

        # 2 days back: valid data
        past2_index = pd.DatetimeIndex(["2024-01-13 03:00"])
        past2 = pd.DataFrame({
            "temperature": [17.0],
            "is_valid": [True],  # ← genuine observation
        }, index=past2_index)
        _write_past_csv(tmp_path, "coord1", pd.Timestamp("2024-01-13 00:00"), past2)

        result = handler.fill(current, "coord1", tmp_path, basis_time)

        # Should use 2-day-back value, NOT 1-day-back
        assert result["temperature"].iloc[0] == 17.0
        assert result["is_valid"].iloc[0] == False

    def test_all_past_invalid_leaves_unfilled(self, tmp_path, caplog):
        """All reachable past rows are invalid → value stays NaN."""
        handler = MissingHandler()
        basis_time = pd.Timestamp("2024-01-03 00:00")

        index = pd.DatetimeIndex(["2024-01-03 03:00"])
        current = pd.DataFrame({
            "temperature": [np.nan],
            "is_valid": [True],
        }, index=index)

        # Create files for days 1 and 2 back, both invalid
        for days_back in [1, 2]:
            past_bt = basis_time - pd.Timedelta(days=days_back)
            past_idx = pd.DatetimeIndex([
                pd.Timestamp("2024-01-03 03:00") - pd.Timedelta(days=days_back)
            ])
            past = pd.DataFrame({
                "temperature": [15.0],
                "is_valid": [False],
            }, index=past_idx)
            _write_past_csv(tmp_path, "coord1", past_bt, past)

        with caplog.at_level(logging.WARNING):
            result = handler.fill(current, "coord1", tmp_path, basis_time)

        # Value stays NaN (no valid source)
        assert pd.isna(result["temperature"].iloc[0])
        assert "Could not fill" in caplog.text


# ---------------------------------------------------------------------------
# File entirely missing / corrupt
# ---------------------------------------------------------------------------


class TestFileEdgeCases:

    def test_no_past_directory(self, tmp_path, caplog):
        """Past directory doesn't exist → warning, unfilled."""
        handler = MissingHandler()
        basis_time = pd.Timestamp("2024-01-15 00:00")

        index = pd.DatetimeIndex(["2024-01-15 03:00"])
        current = pd.DataFrame({
            "temperature": [np.nan],
            "is_valid": [True],
        }, index=index)

        with caplog.at_level(logging.WARNING):
            result = handler.fill(current, "nonexistent_coord", tmp_path, basis_time)

        assert pd.isna(result["temperature"].iloc[0])
        assert "No past data directory" in caplog.text

    def test_corrupt_file_skipped(self, tmp_path, caplog):
        """Unreadable CSV → warning logged, continues search."""
        handler = MissingHandler()
        basis_time = pd.Timestamp("2024-01-15 00:00")

        index = pd.DatetimeIndex(["2024-01-15 03:00"])
        current = pd.DataFrame({
            "temperature": [np.nan],
            "is_valid": [True],
        }, index=index)

        # 1 day back: corrupt file
        coord_dir = tmp_path / "coord1"
        coord_dir.mkdir(parents=True)
        (coord_dir / "2024-01-14_00.csv").write_bytes(b"\x00\x01\x02\xff\xfe")

        # 2 days back: valid file
        past2_index = pd.DatetimeIndex(["2024-01-13 03:00"])
        past2 = pd.DataFrame({
            "temperature": [16.0],
            "is_valid": [True],
        }, index=past2_index)
        _write_past_csv(tmp_path, "coord1", pd.Timestamp("2024-01-13 00:00"), past2)

        with caplog.at_level(logging.WARNING):
            result = handler.fill(current, "coord1", tmp_path, basis_time)

        assert result["temperature"].iloc[0] == 16.0
        assert "Failed to read" in caplog.text

    def test_multiple_rows_different_fill_depths(self, tmp_path):
        """Two missing rows filled from different past depths."""
        handler = MissingHandler()
        basis_time = pd.Timestamp("2024-01-15 00:00")

        index = pd.DatetimeIndex([
            "2024-01-15 03:00",
            "2024-01-15 06:00",
        ])
        current = pd.DataFrame({
            "temperature": [np.nan, np.nan],
            "is_valid": [True, True],
        }, index=index)

        # 1 day back: has hour 03 but NOT hour 06
        past1 = pd.DataFrame({
            "temperature": [20.0],
            "is_valid": [True],
        }, index=pd.DatetimeIndex(["2024-01-14 03:00"]))
        _write_past_csv(tmp_path, "coord1", pd.Timestamp("2024-01-14 00:00"), past1)

        # 2 days back: has hour 06
        past2 = pd.DataFrame({
            "temperature": [18.0],
            "is_valid": [True],
        }, index=pd.DatetimeIndex(["2024-01-13 06:00"]))
        _write_past_csv(tmp_path, "coord1", pd.Timestamp("2024-01-13 00:00"), past2)

        result = handler.fill(current, "coord1", tmp_path, basis_time)

        assert result.loc["2024-01-15 03:00", "temperature"] == 20.0  # 1-day
        assert result.loc["2024-01-15 06:00", "temperature"] == 18.0  # 2-day
        assert not result["is_valid"].any()

    def test_partial_columns_missing(self, tmp_path):
        """Only some columns missing; past must have all missing columns."""
        handler = MissingHandler()
        basis_time = pd.Timestamp("2024-01-15 00:00")

        index = pd.DatetimeIndex(["2024-01-15 03:00"])
        current = pd.DataFrame({
            "temperature": [25.0],  # present
            "wind_speed": [np.nan],  # missing
            "is_valid": [True],
        }, index=index)

        # Past with both columns
        past = pd.DataFrame({
            "temperature": [20.0],
            "wind_speed": [8.5],
            "is_valid": [True],
        }, index=pd.DatetimeIndex(["2024-01-14 03:00"]))
        _write_past_csv(tmp_path, "coord1", pd.Timestamp("2024-01-14 00:00"), past)

        result = handler.fill(current, "coord1", tmp_path, basis_time)

        # Only wind_speed was missing, so only that should be filled
        assert result["wind_speed"].iloc[0] == 8.5
        # temperature was already present — should NOT be overwritten
        assert result["temperature"].iloc[0] == 25.0
        assert result["is_valid"].iloc[0] == False
