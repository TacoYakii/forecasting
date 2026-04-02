"""
Runner classes for orchestrating forecasting workflows.

This module provides orchestrator classes that manage model evaluation
externally — models only handle fit/forecast, runners handle everything else.

- RollingRunner:     Time-series rolling evaluation (Statistical, Deep, Foundation)
- PerHorizonRunner:  Per-horizon independent ML models
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
    Self,
)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .base_model import BaseModel, BaseForecaster
from .forecast_distribution import DISTRIBUTION_REGISTRY
from .forecast_results import ParametricForecastResult

if TYPE_CHECKING:
    from .forecast_results import QuantileForecastResult, SampleForecastResult


# ======================================================================
# Protocols — define how RollingRunner interacts with models
# ======================================================================

@runtime_checkable
class StatefulPredictor(Protocol):
    """
    Models that forecast from mutable internal state (ARIMA family).

    At each rolling step:
      1. forecast(horizon) → ParametricForecastResult
      2. update_state(y_actual) to advance one step
    """

    is_fitted_: bool

    def forecast(
        self,
        horizon: int,
        x_future: Optional[np.ndarray] = None,
    ) -> ParametricForecastResult: ...

    def update_state(
        self,
        y_new: float,
        x_new: Optional[np.ndarray] = None,
    ) -> None: ...


@runtime_checkable
class ContextPredictor(Protocol):
    """
    Models that forecast from a context window (deep learning, foundation).

    At each rolling step the runner builds a context window and calls
    predict_from_context() — no internal state mutation.
    """

    is_fitted_: bool

    def predict_from_context(
        self,
        context_y: np.ndarray,
        horizon: int,
        **kwargs,
    ): ...


# ======================================================================
# RollingRunner — time-series rolling evaluation
# ======================================================================

class RollingRunner:
    """
    Rolling multi-step-ahead forecast orchestrator.

    Dispatches between two rolling strategies based on the model type:

      StatefulPredictor (ARIMA family):
        1. forecast(horizon) from current state
        2. Record output
        3. update_state(y_actual) to advance one step

      ContextPredictor (Deep learning / Foundation):
        1. Build context window (all data up to current time)
        2. predict_from_context(context, horizon)
        3. Record output, slide window forward

    Args:
        model:           A fitted model (StatefulPredictor or ContextPredictor).
        dataset:         Full DataFrame (train + test), sorted by time index.
        y_col:           Target column name.
        forecast_period: (start, end) tuple defining the evaluation period.
        exog_cols:       Exogenous column names for Statistical/Foundation models.
        futr_cols:       Future-known exogenous columns for Deep models (NWP etc.).
        hist_cols:       Historical-only exogenous columns for Deep models (SCADA obs etc.).

    Example:
        >>> model = ArimaGarchForecaster(dataset=train_df, ...).fit()
        >>> runner = RollingRunner(
        ...     model, dataset=full_df, y_col="power",
        ...     forecast_period=("2023-07-01", "2023-12-31"),
        ... )
        >>> result = runner.run(horizon=24)
    """

    def __init__(
        self,
        model: Union[StatefulPredictor, ContextPredictor],
        dataset: pd.DataFrame,
        y_col: str,
        forecast_period: Tuple,
        exog_cols: Optional[List[str]] = None,
        futr_cols: Optional[List[str]] = None,
        hist_cols: Optional[List[str]] = None,
    ):
        if not model.is_fitted_:
            raise RuntimeError("Model must be fitted before rolling forecast.")

        self._model = model
        self._dataset = dataset.sort_index()
        self._y_col = y_col
        self._exog_cols = exog_cols or []
        self._futr_cols = futr_cols or []
        self._hist_cols = hist_cols or []
        self._forecast_period = forecast_period

        self._forecast_data = self._dataset.loc[
            forecast_period[0] : forecast_period[1]
        ]
        if len(self._forecast_data) == 0:
            raise ValueError(
                f"Forecast period {forecast_period} yields no data."
            )

    def run(
        self,
        horizon: int,
        method: str = "forecast",
        method_kwargs: Optional[dict] = None,
        dist_name: str = "normal",
        show_progress: bool = True,
    ):
        """
        Execute rolling evaluation over the forecast period.

        Args:
            horizon:       Steps ahead at each basis time.
            method:        For StatefulPredictor: model method name
                           (e.g. "forecast", "simulate_paths").
                           Ignored for ContextPredictor.
            method_kwargs: Extra kwargs passed to the model method.
            dist_name:     Distribution name for the output.
            show_progress: If True, display a tqdm progress bar.

        Returns:
            ParametricForecastResult, SampleForecastResult, or
            QuantileForecastResult depending on the model's return type.
        """
        if isinstance(self._model, StatefulPredictor):
            return self._run_stateful(
                horizon, method, method_kwargs, dist_name, show_progress,
            )
        elif isinstance(self._model, ContextPredictor):
            return self._run_context(
                horizon, method_kwargs, dist_name, show_progress,
            )
        else:
            raise TypeError(
                f"Model {type(self._model).__name__} does not satisfy "
                f"StatefulPredictor or ContextPredictor protocol."
            )

    # ------------------------------------------------------------------
    # StatefulPredictor (ARIMA family)
    # ------------------------------------------------------------------

    def _run_stateful(self, horizon, method, method_kwargs, dist_name, show_progress):
        assert isinstance(self._model, StatefulPredictor)
        model = self._model

        predict_fn = getattr(model, method, None)
        if predict_fn is None or not callable(predict_fn):
            raise AttributeError(
                f"Model {type(model).__name__} has no callable "
                f"method '{method}'"
            )

        kwargs = dict(method_kwargs) if method_kwargs else {}
        base_seed = kwargs.pop("seed", None)

        forecast_times = self._forecast_data.index
        y_actual = self._forecast_data[self._y_col].to_numpy()
        n_steps = len(forecast_times)

        x_aligned = (
            self._forecast_data[self._exog_cols].to_numpy()
            if self._exog_cols
            else np.empty((n_steps, 0))
        )
        n_exog = x_aligned.shape[1] if x_aligned.ndim == 2 else 0

        results = []
        iterator = tqdm(
            range(n_steps), desc=f"Rolling {method}", disable=not show_progress,
        )
        for t in iterator:
            x_future = self._slice_x_future(x_aligned, t, horizon, n_exog)

            call_kwargs = {**kwargs, "horizon": horizon, "x_future": x_future}
            if base_seed is not None:
                call_kwargs["seed"] = base_seed + t

            output = predict_fn(**call_kwargs)
            results.append(output)

            x_t = x_aligned[t] if n_exog > 0 else None
            model.update_state(y_actual[t], x_t)

        basis_index = pd.Index(forecast_times[:n_steps])
        return self._collect_results(results, basis_index, horizon, dist_name)

    # ------------------------------------------------------------------
    # ContextPredictor (Deep learning / Foundation)
    # ------------------------------------------------------------------

    def _run_context(self, horizon, method_kwargs, dist_name, show_progress):
        assert isinstance(self._model, ContextPredictor)
        model = self._model

        # Suppress PyTorch Lightning progress bars during rolling predict
        pl_logger = logging.getLogger("pytorch_lightning")
        prev_level = pl_logger.level
        pl_logger.setLevel(logging.ERROR)

        kwargs = dict(method_kwargs) if method_kwargs else {}
        forecast_times = self._forecast_data.index
        n_steps = len(forecast_times)

        # Determine exog column sets:
        # - Deep models use futr_cols/hist_cols (split)
        # - Foundation/other models use exog_cols (unified)
        has_futr = len(self._futr_cols) > 0
        has_hist = len(self._hist_cols) > 0
        has_exog = len(self._exog_cols) > 0

        idx = self._dataset.index
        freq = pd.infer_freq(idx) or "h" if isinstance(idx, pd.DatetimeIndex) else "h"

        results = []
        iterator = tqdm(
            range(n_steps), desc="Rolling predict", disable=not show_progress,
        )
        for t in iterator:
            current_time = forecast_times[t]

            context_data = self._dataset.loc[:current_time].iloc[:-1]
            if len(context_data) == 0:
                context_data = self._dataset.loc[:current_time]

            context_y = context_data[self._y_col].to_numpy()
            context_index = context_data.index

            # Build context_X and future_X based on futr/hist split or exog_cols
            context_X = None
            future_X = None
            future_index = None

            if has_futr or has_hist:
                # Deep model path: futr_cols + hist_cols
                all_context_cols = self._futr_cols + self._hist_cols
                if all_context_cols:
                    context_X = context_data[all_context_cols].to_numpy()

                if has_futr:
                    futr_start = t + 1
                    futr_end = min(t + horizon, n_steps)
                    futr_data = self._forecast_data.iloc[futr_start:futr_end]

                    if len(futr_data) < horizon:
                        pad_n = horizon - len(futr_data)
                        last_row = (
                            futr_data.iloc[[-1]]
                            if len(futr_data) > 0
                            else context_data.iloc[[-1]]
                        )
                        pad_df = pd.concat([last_row] * pad_n, ignore_index=True)
                        last_time = (
                            futr_data.index[-1]
                            if len(futr_data) > 0
                            else current_time
                        )
                        pad_df.index = pd.date_range(
                            last_time, periods=pad_n + 1, freq=freq,
                        )[1:]
                        futr_data = pd.concat([futr_data, pad_df])

                    # Only futr_cols go into future_X (no hist leakage)
                    future_X = futr_data[self._futr_cols].to_numpy()[:horizon]
                    future_index = futr_data.index[:horizon]

            elif has_exog:
                # Foundation/other model path: exog_cols unified (context only)
                context_X = context_data[self._exog_cols].to_numpy()

            output = model.predict_from_context(
                context_y=context_y,
                horizon=horizon,
                context_index=context_index,
                context_X=context_X,
                future_X=future_X,
                future_index=future_index,
                **kwargs,
            )
            results.append(output)

        pl_logger.setLevel(prev_level)

        basis_index = pd.Index(forecast_times)
        return self._collect_results(results, basis_index, horizon, dist_name)

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_results(results, basis_index, horizon, dist_name):
        """
        Aggregate per-step outputs into a single result object.

        All models return ForecastResult objects directly. This method
        concatenates them along axis=0 (the basis/time dimension).

        Dispatch by return type:
          - ParametricForecastResult → concatenate params along axis=0
          - SampleForecastResult     → concatenate samples along axis=0
          - QuantileForecastResult   → concatenate quantile dicts along axis=0
        """
        from src.core.forecast_results import (
            ParametricForecastResult,
            QuantileForecastResult,
            SampleForecastResult,
        )

        first = results[0]

        if isinstance(first, ParametricForecastResult):
            param_keys = list(first.params.keys())
            stacked_params = {}
            for key in param_keys:
                stacked_params[key] = np.concatenate(
                    [r.params[key] for r in results], axis=0
                )  # (N, H)
            return ParametricForecastResult(
                dist_name=first.dist_name,
                params=stacked_params,
                basis_index=basis_index,
                model_name=first.model_name,
            )

        if isinstance(first, SampleForecastResult):
            samples_all = np.concatenate(
                [r.samples for r in results], axis=0,
            )
            return SampleForecastResult(
                samples=samples_all,
                basis_index=basis_index,
                model_name=first.model_name,
            )

        if isinstance(first, QuantileForecastResult):
            q_levels = first.quantile_levels
            quantiles_data = {}
            for q in q_levels:
                quantiles_data[q] = np.concatenate(
                    [r.quantiles_data[q] for r in results], axis=0,
                )
            return QuantileForecastResult(
                quantiles_data=quantiles_data,
                basis_index=basis_index,
                model_name=first.model_name,
            )

        raise TypeError(
            f"Unsupported return type from model method: {type(first).__name__}."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_x_future(x_aligned, t, horizon, n_exog):
        """Extract exogenous features for the next ``horizon`` steps."""
        if n_exog == 0:
            return None
        end_idx = min(t + horizon, len(x_aligned))
        x_slice = x_aligned[t:end_idx]
        if len(x_slice) < horizon:
            return np.vstack([
                x_slice,
                np.zeros((horizon - len(x_slice), n_exog)),
            ])
        return x_slice


# ======================================================================
# PerHorizonRunner — per-horizon independent ML models
# ======================================================================

class PerHorizonRunner(BaseModel):
    """
    Wrapper that trains independent models per forecast horizon and produces
    a unified ParametricForecastResult.

    Given a directory of per-horizon CSV files (horizon_1.csv, ..., horizon_H.csv),
    this runner instantiates one Forecaster per horizon via MODEL_REGISTRY,
    trains all of them, and assembles (mu, sigma) into a single
    ParametricForecastResult of shape (N_basis, H).

    Args:
        data_dir: Directory containing horizon_*.csv files.
        model_name: Registry key ("xgboost", "ngboost", "catboost", "lr", "gbm", "pgbm").
        y_col: Target column name in each CSV.
        exog_cols: Feature columns. None = use all except y_col.
        training_period: (start, end) for training split, shared across all horizons.
        forecast_period: (start, end) for forecast split, shared across all horizons.
        hyperparameter: Passed to every per-horizon model.
        horizons: Explicit horizon list (1-indexed). None = auto-discover from data_dir.
        dist_name: Distribution name for the output ParametricForecastResult.
        n_jobs: Number of parallel model trainings (1=sequential, -1=all cores).
        enable_logging: Enable file logging.
        save_dir: Custom directory for saving.
        verbose: Enable console logging.

    Example:
        >>> runner = PerHorizonRunner(
        ...     data_dir="data/training_dataset/sinan/w100002/",
        ...     model_name="ngboost",
        ...     y_col="forecast_time_observed_KPX_pwr",
        ...     training_period=("2020-01-01", "2022-12-31"),
        ...     forecast_period=("2023-01-01", "2023-06-30"),
        ... )
        >>> runner.fit()
        >>> result = runner.forecast()  # ParametricForecastResult(N, H)
        >>> result.to_distribution(6)   # 6-step-ahead ParametricDistribution
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        model_name: str,
        y_col: str,
        exog_cols: Optional[List[str]] = None,
        training_period: Optional[Tuple] = None,
        forecast_period: Optional[Tuple] = None,
        hyperparameter: Optional[Dict] = None,
        horizons: Optional[List[int]] = None,
        dist_name: str = "normal",
        n_jobs: int = 1,
        enable_logging: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.y_col = y_col
        self.exog_cols = exog_cols
        self.training_period = training_period
        self.forecast_period = forecast_period
        self.model_hyperparameter = dict(hyperparameter) if hyperparameter else {}
        self.dist_name = dist_name
        self.n_jobs = n_jobs
        self.verbose = verbose

        if dist_name not in DISTRIBUTION_REGISTRY:
            raise ValueError(
                f"Distribution '{dist_name}' not supported. "
                f"Available: {list(DISTRIBUTION_REGISTRY.keys())}"
            )

        if horizons is not None:
            self._horizons = sorted(horizons)
        else:
            self._horizons = self._discover_horizons()

        if not self._horizons:
            raise ValueError(f"No horizon CSV files found in {self.data_dir}")

        self._models: Dict[int, BaseForecaster] = {}
        self._datasets: Dict[int, pd.DataFrame] = {}

        super().__init__(
            hyperparameter=hyperparameter,
            enable_logging=enable_logging,
            save_dir=save_dir,
            verbose=verbose,
        )

    def __repr__(self) -> str:
        n_fitted = sum(1 for m in self._models.values() if m.is_fitted_)
        return (
            f"PerHorizonRunner(model='{self.model_name}', "
            f"horizons={len(self._horizons)}, fitted={n_fitted})"
        )

    @property
    def horizons(self) -> List[int]:
        """Sorted list of forecast horizons (1-indexed)."""
        return list(self._horizons)

    # ------------------------------------------------------------------
    # Data discovery & loading
    # ------------------------------------------------------------------

    def _discover_horizons(self) -> List[int]:
        """Glob data_dir for horizon_*.csv and extract horizon integers."""
        pattern = re.compile(r"^horizon_(\d+)\.csv$")
        found = []
        for f in self.data_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                h = int(m.group(1))
                if h > 0:
                    found.append(h)
        return sorted(found)

    def _load_dataset(self, h: int) -> pd.DataFrame:
        """Load a single horizon CSV with basis_time as DatetimeIndex."""
        path = self.data_dir / f"horizon_{h}.csv"
        df = pd.read_csv(path, index_col="basis_time", parse_dates=True)
        df.index = pd.DatetimeIndex(df.index)
        return df

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _fit_single_horizon(self, h: int) -> Tuple[int, BaseForecaster, pd.DataFrame]:
        """Train a single horizon model. Returns (h, fitted_model, full_df)."""
        from src.core.registry import MODEL_REGISTRY

        df = self._load_dataset(h)
        model_cls = MODEL_REGISTRY.get(self.model_name)

        train_df = df.loc[self.training_period[0]:self.training_period[1]]

        model = model_cls(
            dataset=train_df,
            y_col=self.y_col,
            exog_cols=self.exog_cols,
            hyperparameter=dict(self.model_hyperparameter),
            enable_logging=False,
            save_dir=str(self.base_dir / f"horizon_{h}"),
            verbose=False,
        )
        model.fit()

        if self.enable_logging:
            self.logger.info(f"Horizon {h}/{self._horizons[-1]} trained.")

        return h, model, df

    def fit(self) -> Self:
        """
        Train all per-horizon models.

        Returns:
            Self for method chaining.
        """
        if self.enable_logging:
            self.logger.info(
                f"Training {len(self._horizons)} {self.model_name} models "
                f"(n_jobs={self.n_jobs})..."
            )

        if self.n_jobs == 1:
            for h in self._horizons:
                h_key, model, df = self._fit_single_horizon(h)
                self._models[h_key] = model
                self._datasets[h_key] = df
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_single_horizon)(h) for h in self._horizons
            )
            for h_key, model, df in results:
                self._models[h_key] = model
                self._datasets[h_key] = df

        self.is_fitted_ = True

        if self.enable_logging:
            self.logger.info("All horizon models trained.")

        return self

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def _extract_result(self, model: BaseForecaster, h: int) -> Tuple[ParametricForecastResult, pd.Index]:
        """
        Extract ParametricForecastResult and forecast_index from a fitted per-horizon model.
        """
        df = self._datasets[h]
        forecast_data = df.loc[self.forecast_period[0]:self.forecast_period[1]]
        forecast_X = forecast_data[model.exog_cols].to_numpy()
        forecast_index = forecast_data.index

        result = model.forecast(forecast_X, forecast_index)
        return result, forecast_index

    def forecast(self) -> ParametricForecastResult:
        """
        Generate unified multi-horizon forecast.

        Returns:
            ParametricForecastResult with shape (N_common, len(horizons)).
        """
        if not self.is_fitted_:
            raise RuntimeError("Runner not fitted. Call fit() first.")

        horizon_results: Dict[int, Tuple[ParametricForecastResult, pd.Index]] = {}
        for h in self._horizons:
            model = self._models[h]
            fp, forecast_index = self._extract_result(model, h)
            horizon_results[h] = (fp, forecast_index)

        common_idx = horizon_results[self._horizons[0]][1]
        for h in self._horizons[1:]:
            common_idx = common_idx.intersection(horizon_results[h][1])

        if len(common_idx) == 0:
            raise ValueError("No common basis_time indices across horizons.")

        max_len = max(len(v[1]) for v in horizon_results.values())
        if len(common_idx) < max_len * 0.9 and self.enable_logging:
            self.logger.warning(
                f"Common index ({len(common_idx)}) is significantly smaller "
                f"than max horizon index ({max_len}). "
                f"Some horizons may have missing data."
            )

        # Collect param keys from the first result
        first_fp = horizon_results[self._horizons[0]][0]
        param_keys = list(first_fp.params.keys())
        H = len(self._horizons)
        N = len(common_idx)

        # Build (N, H) matrices for each param key
        params_matrices: Dict[str, np.ndarray] = {
            key: np.empty((N, H), dtype=float) for key in param_keys
        }

        for col_idx, h in enumerate(self._horizons):
            fp, basis_idx = horizon_results[h]
            for key in param_keys:
                vals = fp.params[key]
                series = pd.Series(np.ravel(vals), index=basis_idx)
                params_matrices[key][:, col_idx] = series.loc[common_idx].values

        # Get model_name from any horizon's result
        first_result = next(iter(horizon_results.values()))[0]
        return ParametricForecastResult(
            dist_name=self.dist_name,
            params=params_matrices,
            basis_index=common_idx,
            model_name=first_result.model_name,
        )

    def forecast_horizon(self, h: int) -> ParametricForecastResult:
        """
        Get the ParametricForecastResult for a single horizon.

        Args:
            h: Forecast horizon (1-indexed, must be in self.horizons).

        Returns:
            ParametricForecastResult with shape (T, 1) for horizon h.
        """
        if h not in self._models:
            raise ValueError(
                f"Horizon {h} not available. Available: {self._horizons}"
            )
        model = self._models[h]
        df = self._datasets[h]
        forecast_data = df.loc[self.forecast_period[0]:self.forecast_period[1]]
        forecast_X = forecast_data[model.exog_cols].to_numpy()
        forecast_index = forecast_data.index
        return model.forecast(forecast_X, forecast_index)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save_model_specific(self, model_path: Path) -> Path:
        """Save all per-horizon models."""
        for h, model in self._models.items():
            model.save_model()
        return self.base_dir

    def _load_model_specific(self, model_path: Path) -> None:
        """Load all per-horizon models from their saved directories."""
        from src.core.registry import MODEL_REGISTRY

        for h in self._horizons:
            df = self._load_dataset(h)
            model_cls = MODEL_REGISTRY.get(self.model_name)

            train_df = df.loc[self.training_period[0]:self.training_period[1]]

            model = model_cls(
                dataset=train_df,
                y_col=self.y_col,
                exog_cols=self.exog_cols,
                hyperparameter=dict(self.model_hyperparameter),
                enable_logging=False,
                save_dir=str(self.base_dir / f"horizon_{h}"),
                verbose=False,
            )

            horizon_dir = self.base_dir / f"horizon_{h}"
            model_files = list(horizon_dir.glob(f"{model_cls.__name__}_model*"))
            if not model_files:
                raise FileNotFoundError(
                    f"No saved model found in {horizon_dir}"
                )
            model.load_model(model_files[0])
            self._models[h] = model
            self._datasets[h] = df
