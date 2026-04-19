"""Jeju-specific nMAPE evaluator.

Jeju rules differ from mainland KPX:
- No validity filtering (neither ``is_valid`` nor 10%-capacity threshold).
- Day-ahead: single 10:00 bidding window (next day 00:00-23:00).
- Real-time: inherited from base, no filtering.
"""

import os
from pathlib import Path
from typing import Union

import pandas as pd

from .evaluator import NMAPEEvaluator


class JejuNMAPEEvaluator(NMAPEEvaluator):
    """nMAPE evaluator with Jeju-island rules.

    Differences from the base ``NMAPEEvaluator``:

    Validity:
        No validity filtering is applied.  The ``is_valid`` column and
        the 10%-capacity threshold used in the mainland KPX evaluator
        are both skipped.  All periods are evaluated as-is.

    Day-Ahead market:
        A single bidding window at 10:00 the previous day targets
        the next day 00:00-23:00 (horizons 14-37).

    Example:
        >>> evaluator = JejuNMAPEEvaluator(
        ...     capacity=100.0,
        ...     observed_file="obs.csv",
        ...     output_dir="results/",
        ...     forecast_frames=frames,
        ... )
        >>> evaluator.run(day_ahead=True, real_time=False)
    """

    # Jeju day-ahead: basis 10:00, next day 00:00-23:00
    _DA_BASIS_HOUR = 10
    _DA_HORIZON_MIN = 14  # 10:00 + 14h = 00:00 next day
    _DA_HORIZON_MAX = 37  # 10:00 + 37h = 23:00 next day

    def _read_observed(self, fp: Union[str, Path]) -> pd.DataFrame:
        """Read observed data ignoring is_valid column."""
        fp = str(fp)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Observed file not found: {fp}")

        time_col = self.cols["observed_time_col"]
        pwr_col = self.cols["observed_mu_col"]

        df = pd.read_csv(fp)
        if time_col not in df.columns:
            raise KeyError(
                f"Time column '{time_col}' is missing in observed file {fp}."
            )
        if pwr_col not in df.columns:
            raise KeyError(
                f"Observed power column '{pwr_col}' is missing in {fp}."
            )

        df[time_col] = pd.to_datetime(df[time_col])
        # No is_valid check; strict_valid=True so inherited RT uses all rows
        df["strict_valid"] = True

        return df

    def evaluate_day_ahead(self):
        """Evaluate day-ahead nMAPE with Jeju rules.

        Single 10:00 bidding window, no validity filtering.
        """
        print("Evaluating Day-Ahead Market (Jeju)...")
        b_time_col = self.cols["basis_time_col"]
        f_time_col = self.cols["forecast_time_col"]

        if self.merged_df is None or self.merged_df.empty:
            raise ValueError("Merged dataframe is empty.")

        # Basis hour + horizon filter
        mask = (
            (self.merged_df[b_time_col].dt.hour == self._DA_BASIS_HOUR)
            & self.merged_df["horizon"].between(
                self._DA_HORIZON_MIN, self._DA_HORIZON_MAX
            )
        )
        day_ahead = self.merged_df[mask].copy()

        if day_ahead.empty:
            raise ValueError(
                f"No day-ahead forecasts found "
                f"(basis hour={self._DA_BASIS_HOUR}, "
                f"horizons {self._DA_HORIZON_MIN}-{self._DA_HORIZON_MAX})."
            )

        day_ahead = day_ahead.set_index(f_time_col)
        day_ahead.to_csv(os.path.join(self.output_dir, "day_ahead_hourly.csv"))

        # Daily nMAPE
        daily_nmape = day_ahead.groupby(
            pd.DatetimeIndex(day_ahead.index).date
        )["nMAPE"].mean()
        daily_nmape.index.name = "date"
        daily_nmape.to_csv(
            os.path.join(self.output_dir, "day_ahead_daily_nMAPE.csv"),
            header=["nMAPE"],
        )

        # Monthly summary
        monthly_nmape = day_ahead.groupby(
            pd.DatetimeIndex(day_ahead.index).to_period("M")
        )["nMAPE"].mean()
        monthly_summary = pd.DataFrame(
            {"monthly_dayahead_nMAPE": monthly_nmape}
        )
        monthly_summary.index.name = "month"
        monthly_summary.to_csv(
            os.path.join(self.output_dir, "day_ahead_monthly_summary.csv")
        )
        print("Day-Ahead Evaluation (Jeju) Complete.")
