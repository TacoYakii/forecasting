import pandas as pd
import numpy as np
import os
import concurrent.futures
from typing import Dict, Optional, Union
from pathlib import Path


class NMAPEEvaluator:
    """
    Evaluator for Wind Power Forecasting Performance based on Korean Power Exchange (KPX) rules.
    It calculates nMAPE for Day-Ahead (09:00, 16:00 bids -> next day 00:00~23:00) and Real-Time (2-hour ahead) markets.
    Observed data is provided as a single file; forecast data is provided per horizon.

    [Outputs Generated in output_dir]
    Day-Ahead Market (전일시장):
        - day_ahead_hourly.csv: Hourly nMAPE and validity for 1st(09:00) and 2nd(16:00) bidding.
        - day_ahead_daily_nMAPE.csv: Daily average nMAPE (computed only for valid periods).
        - day_ahead_monthly_summary.csv: Monthly average nMAPE and valid_ratio (proportion of valid periods).
    Real-Time Market (실시간시장):
        - real_time_hourly.csv: Hourly nMAPE and validity for 2-hour ahead forecasts.
        - real_time_daily_nMAPE.csv: Daily average nMAPE (computed only for valid periods).
        - real_time_monthly_summary.csv: Monthly average nMAPE and valid_ratio.
    """
    def __init__(
        self,
        capacity: float,
        forecast_files: Dict[int, Union[str, Path]],
        observed_file: Union[str, Path],
        output_dir: Union[str, Path],
        column_mapping: Optional[Dict[str, str]] = None
        ):
        """
        Args:
            capacity: The capacity of the wind farm.
            forecast_files: Dictionary mapping forecast horizon (int) to the CSV file path.
                            Example: {1: "model_nm/horizon_1.csv", ...}
            observed_file: Path to the single observed data CSV file.
                            Example: "scada/farm_level.csv"
            output_dir: Directory where evaluation results (CSVs) will be saved.
            column_mapping: Dictionary to override default column names.
        """
        self.capacity = capacity
        self.forecast_files = forecast_files
        self.observed_file = observed_file
        self.output_dir = output_dir

        # column mapping setting
        self.cols = {
            "forecast_time_col": "forecast_time",
            "basis_time_col": "basis_time",
            "observed_time_col": "forecast_time",
            "forecast_mu_col": "mu",
            "observed_mu_col": "forecast_time_observed_KPX_pwr",
            "is_valid_col": "is_valid"
        }
        
        if column_mapping is not None:
            self.cols.update(column_mapping)
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.forecast_df: Optional[pd.DataFrame] = None
        self.observed_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None

    def _read_single_forecast(self, horizon: int, fp: str):
        if not os.path.exists(fp):
            print(f"[WARNING] Forecast file not found: {fp}")
            return None
            
        req_cols = [self.cols["basis_time_col"], self.cols["forecast_time_col"], self.cols["forecast_mu_col"]]
        
        df = pd.read_csv(fp, usecols=lambda c: c in req_cols)
        missing_cols = [c for c in req_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Required columns missing in forecast file {fp}. Missing: {missing_cols}")
            
        df[self.cols["basis_time_col"]] = pd.to_datetime(df[self.cols["basis_time_col"]])
        df[self.cols["forecast_time_col"]] = pd.to_datetime(df[self.cols["forecast_time_col"]])
        df['horizon'] = horizon
        return df

    def _load_data(self):
        print("Loading forecast data...")
        all_forecasts = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._read_single_forecast, horizon, fp) for horizon, fp in self.forecast_files.items()]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    all_forecasts.append(res)
            
        if not all_forecasts:
            raise ValueError("No valid forecast data could be loaded. Please check the 'forecast_files' dictionary paths and file contents.")
            
        self.forecast_df = pd.concat(all_forecasts, ignore_index=True)
        
        print("Loading observed data...")
        self.observed_df = self._read_observed(self.observed_file)

    def _read_observed(self, fp: Union[str, Path]):
        fp = str(fp)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Observed file not found: {fp}")

        time_col = self.cols["observed_time_col"]
        pwr_col = self.cols["observed_mu_col"]
        val_col = self.cols["is_valid_col"]
        obs_req_cols = [time_col, pwr_col, val_col]

        df = pd.read_csv(fp, usecols=lambda c: c in obs_req_cols)

        if time_col not in df.columns:
            raise KeyError(f"Time column '{time_col}' is missing in observed file {fp}.")
        if pwr_col not in df.columns:
            raise KeyError(f"Observed power column '{pwr_col}' is missing in {fp}.")

        df[time_col] = pd.to_datetime(df[time_col])

        # check validity (usage rate >= 10% and is_valid == True)
        pwr = pd.to_numeric(df[pwr_col], errors='coerce').fillna(0)
        is_val = df.get(val_col, pd.Series(False, index=df.index)).astype(bool)
        df['strict_valid'] = is_val & (pwr >= float(self.capacity) * 0.1)

        return df

    def _merge_data(self):
        print("Merging forecast and observed data...")

        if self.forecast_df is None or self.observed_df is None:
            raise ValueError("Forecast or observed data is missing. Please ensure _load_data() is called successfully.")

        f_time_col = self.cols["forecast_time_col"]
        o_time_col = self.cols["observed_time_col"]

        obs = self.observed_df.rename(columns={o_time_col: f_time_col}) if o_time_col != f_time_col else self.observed_df

        self.merged_df = pd.merge(
            self.forecast_df,
            obs,
            on=f_time_col,
            how='inner'
        )

        if self.merged_df.empty:
            raise ValueError("Merged dataframe is empty! Make sure forecast_time and observed_time match.")

        self.merged_df['nMAPE'] = np.abs(self.merged_df[self.cols["forecast_mu_col"]] - self.merged_df[self.cols["observed_mu_col"]]) / self.capacity * 100

    def evaluate_day_ahead(self):
        print("Evaluating Day-Ahead Market...")
        b_time_col = self.cols["basis_time_col"]
        f_time_col = self.cols["forecast_time_col"]
        
        if self.merged_df is None or self.merged_df.empty:
            raise ValueError("Merged dataframe is empty! Make sure basis and forecast times match exactly.")
            
        mask_09 = (self.merged_df[b_time_col].dt.hour == 9) & (self.merged_df['horizon'].between(15, 38)) # 1st bidding: 09:00 -> 15-hour ahead to 38-hour ahead -> 00:00 ~ 23:00 (next day)
        df_09 = self.merged_df[mask_09].copy()
        
        mask_16 = (self.merged_df[b_time_col].dt.hour == 16) & (self.merged_df['horizon'].between(8, 31)) # 2nd bidding: 16:00 -> 8-hour ahead to 31-hour ahead -> 00:00 ~ 23:00 (next day)
        df_16 = self.merged_df[mask_16].copy()
        
        res_09 = df_09.set_index(f_time_col)[['nMAPE', 'strict_valid']]
        res_16 = df_16.set_index(f_time_col)[['nMAPE', 'strict_valid']]
        
        day_ahead = res_09.join(res_16, lsuffix='_09', rsuffix='_16', how='inner')
        
        if day_ahead.empty:
            raise ValueError("No matching 09:00 and 16:00 bids for Day-Ahead evaluation.")
            
        day_ahead['strict_valid'] = day_ahead['strict_valid_09'] & day_ahead['strict_valid_16']
        day_ahead['nMAPE_DA'] = (day_ahead['nMAPE_09'] + day_ahead['nMAPE_16']) / 2
        
        day_ahead.to_csv(os.path.join(self.output_dir, "day_ahead_hourly.csv"))
        
        valid_bids = day_ahead[day_ahead['strict_valid']] # Use only valid bids
        
        daily_nMAPE = valid_bids.groupby(pd.DatetimeIndex(valid_bids.index).date)['nMAPE_DA'].mean()
        daily_nMAPE.index.name = "date"
        daily_nMAPE.to_csv(os.path.join(self.output_dir, "day_ahead_daily_nMAPE.csv"), header=["nMAPE"])
        
        monthly_groups = day_ahead.groupby(pd.DatetimeIndex(day_ahead.index).to_period('M'))
        valid_monthly_groups = valid_bids.groupby(pd.DatetimeIndex(valid_bids.index).to_period('M'))
        
        monthly_nMAPE = valid_monthly_groups['nMAPE_DA'].mean()
        valid_ratio = (valid_monthly_groups.size() / monthly_groups.size()).fillna(0)
        
        monthly_summary = pd.DataFrame({
            'monthly_dayahead_nMAPE': monthly_nMAPE,
            'valid_ratio': valid_ratio
        })
        monthly_summary.index.name = "month"
        monthly_summary.to_csv(os.path.join(self.output_dir, "day_ahead_monthly_summary.csv"))
        print("Day-Ahead Evaluation Complete.")

    def evaluate_real_time(self):
        print("Evaluating Real-Time Market...")
        f_time_col = self.cols["forecast_time_col"]
        
        if self.merged_df is None or self.merged_df.empty:
            raise ValueError("Merged dataframe is empty! Make sure basis and forecast times match exactly.")
            
        df_rt = self.merged_df[self.merged_df['horizon'] == 2].copy() # Real-time: exactly 2 hours ahead forecast
        
        if df_rt.empty:
            raise ValueError("No real-time (horizon=2) forecasts found.")
            
        df_rt = df_rt.set_index(f_time_col)
        df_rt.to_csv(os.path.join(self.output_dir, "real_time_hourly.csv"))
        
        valid_rt = df_rt[df_rt['strict_valid']] # Use only valid bids
        
        daily_nMAPE = valid_rt.groupby(pd.DatetimeIndex(valid_rt.index).date)['nMAPE'].mean()
        daily_nMAPE.index.name = "date"
        daily_nMAPE.to_csv(os.path.join(self.output_dir, "real_time_daily_nMAPE.csv"), header=["nMAPE"])
        
        monthly_groups = df_rt.groupby(pd.DatetimeIndex(df_rt.index).to_period('M'))
        valid_monthly_groups = valid_rt.groupby(pd.DatetimeIndex(valid_rt.index).to_period('M'))
        
        monthly_nMAPE = valid_monthly_groups['nMAPE'].mean()
        valid_ratio = (valid_monthly_groups.size() / monthly_groups.size()).fillna(0)
        
        monthly_summary = pd.DataFrame({
            'monthly_realtime_nMAPE': monthly_nMAPE,
            'valid_ratio': valid_ratio
        })
        monthly_summary.index.name = "month"
        monthly_summary.to_csv(os.path.join(self.output_dir, "real_time_monthly_summary.csv"))
        print("Real-Time Evaluation Complete.")

    def run(self, day_ahead: bool = True, real_time: bool = True):
        """Runs the whole preprocessing and evaluation pipeline.

        Args:
            day_ahead: Whether to evaluate Day-Ahead market. Default True.
            real_time: Whether to evaluate Real-Time market. Default True.
        """
        if not day_ahead and not real_time:
            raise ValueError("At least one of 'day_ahead' or 'real_time' must be True.")

        self._load_data()
        self._merge_data()
        if day_ahead:
            self.evaluate_day_ahead()
        if real_time:
            self.evaluate_real_time()
        print(f"Evaluation finished successfully. Check '{self.output_dir}' for results.")
