"""
TSMixerx forecaster: NeuralForecast TSMixerx wrapper.

TSMixerx is an all-MLP architecture based on the MLP-Mixer paradigm from
vision, adapted to time series. It alternates two mixing blocks:
time-mixing (per-feature MLP across time steps) and feature-mixing
(per-timestep MLP across features). No attention, no recurrence, no
convolution — only MLPs.

The "x" suffix indicates exogenous variable support. TSMixerx handles
both future-known (futr_exog) and historical (hist_exog) covariates
by concatenating them into the feature dimension.

Although TSMixerx is MULTIVARIATE=True in NeuralForecast, we use it as
a univariate model by passing n_series=1 (one target series per farm).

Reference:
    Chen et al., "TSMixer: An All-MLP Architecture for Time Series
    Forecasting", TMLR 2023.
"""

from neuralforecast.models import TSMixerx

from src.core.base_deep_model import BaseDeepModel
from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register_model(name="tsmixerx")
class TSMixerxForecaster(BaseDeepModel):
    """
    TSMixerx (MLP-Mixer for time series) probabilistic forecaster.

    Wraps NeuralForecast TSMixerx with the unified BaseDeepModel interface.
    Produces QuantileForecastResult output by default (MQLoss).

    Note:
        TSMixerx is MULTIVARIATE=True but we use n_series=1 for univariate
        (single target series) to stay consistent with the existing
        fit-runner infrastructure.

    Loss selection (via hyperparameter dict):
        loss_type (str): "quantile" (default), "distribution", "implicit_quantile"
        distribution (str): For loss_type="distribution". Default: "StudentT"

    Model-specific hyperparameters:
        n_block (int): Number of mixing blocks. Default: 2
        ff_dim (int): Feed-forward MLP hidden size. Default: 64
        dropout (float): Dropout probability. Default: 0.0
        revin (bool): Reversible instance normalization. Default: True

    Example:
        >>> model = TSMixerxForecaster(
        ...     hyperparameter={
        ...         "input_size": 144,
        ...         "prediction_length": 48,
        ...         "loss_type": "quantile",
        ...     }
        ... )
        >>> model.fit(dataset=df, y_col="power",
        ...           futr_cols=["nwp_wspd"], hist_cols=["obs_wspd"])
        >>> result = model.forecast()
        >>> result.to_distribution(6).ppf(0.9)
    """

    SUPPORTS_HIST_EXOG = True

    def _create_model(self) -> TSMixerx:
        hp = dict(self._model_hp)

        # Build exogenous feature lists (futr/hist split)
        futr_exog = self.futr_cols or None
        hist_exog = self.hist_cols or None

        loss, valid_loss = self._create_loss(
            default_loss_type="quantile",
            default_distribution="StudentT",
        )

        optional = {}
        if self._scaler_type is not None:
            optional["scaler_type"] = self._scaler_type

        return TSMixerx(
            h=self._prediction_length,
            input_size=self._input_size,
            n_series=1,  # univariate usage: single target series
            loss=loss,
            valid_loss=valid_loss,
            max_steps=self._max_steps,
            batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            early_stop_patience_steps=self._early_stop,
            futr_exog_list=futr_exog,
            hist_exog_list=hist_exog,
            **optional,
            # Model architecture params
            n_block=hp.pop("n_block", 2),
            ff_dim=hp.pop("ff_dim", 64),
            dropout=hp.pop("dropout", 0.0),
            revin=hp.pop("revin", True),
            accelerator=self._accelerator,
            enable_progress_bar=self._enable_progress_bar,
            **hp,
        )
