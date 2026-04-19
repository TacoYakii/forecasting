"""
TimesNet forecaster: NeuralForecast TimesNet wrapper.

TimesNet converts 1D time series into 2D tensors via FFT-detected
periodicity, then applies 2D Inception CNN blocks to capture both
intra-period (within a cycle) and inter-period (across cycles) patterns.
This yields a unique architecture that explicitly encodes multi-
periodicity in the data.

Exogenous variable support:
    - futr_exog: Supported. Future covariates are concatenated into the
      input tensor before 2D reshaping.
    - hist_exog: NOT supported. TimesNet's 1D→2D reshape + CNN pipeline
      does not handle historical-only variables (no mechanism to fill
      their future values). Use TFT or NHITS if hist_exog is needed.

Reference:
    Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General
    Time Series Analysis", ICLR 2023.
"""

from neuralforecast.models import TimesNet

from src.core.base_deep_model import BaseDeepModel
from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register_model(name="timesnet")
class TimesNetForecaster(BaseDeepModel):
    """
    TimesNet (2D CNN on FFT-decomposed series) probabilistic forecaster.

    Wraps NeuralForecast TimesNet with the unified BaseDeepModel interface.
    Produces QuantileForecastResult output by default (MQLoss).

    Loss selection (via hyperparameter dict):
        loss_type (str): "quantile" (default), "distribution", "implicit_quantile"
        distribution (str): For loss_type="distribution". Default: "StudentT"

    Model-specific hyperparameters:
        hidden_size (int): Embedding dimension. Default: 64
        conv_hidden_size (int): 2D conv hidden channels. Default: 64
        top_k (int): Number of top-k periods from FFT. Default: 5
        num_kernels (int): Inception kernel variety. Default: 6
        encoder_layers (int): Number of TimesBlock layers. Default: 2
        dropout (float): Dropout probability. Default: 0.1

    Example:
        >>> model = TimesNetForecaster(
        ...     hyperparameter={
        ...         "input_size": 144,
        ...         "prediction_length": 48,
        ...         "loss_type": "quantile",
        ...     }
        ... )
        >>> model.fit(dataset=df, y_col="power", futr_cols=["nwp_wspd"])
        >>> result = model.forecast()
        >>> result.to_distribution(6).ppf(0.9)
    """

    SUPPORTS_HIST_EXOG = False

    def _create_model(self) -> TimesNet:
        hp = dict(self._model_hp)

        # Build exogenous feature lists
        # hist_exog always None — TimesNet doesn't support it (EXOGENOUS_HIST=False)
        # fit() already strips hist_cols with a warning if provided
        futr_exog = self.futr_cols or None

        loss, valid_loss = self._create_loss(
            default_loss_type="quantile",
            default_distribution="StudentT",
        )

        optional = {}
        if self._scaler_type is not None:
            optional["scaler_type"] = self._scaler_type

        return TimesNet(
            h=self._prediction_length,
            input_size=self._input_size,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=self._max_steps,
            batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            early_stop_patience_steps=self._early_stop,
            futr_exog_list=futr_exog,
            hist_exog_list=None,
            **optional,
            # Model architecture params
            hidden_size=hp.pop("hidden_size", 64),
            conv_hidden_size=hp.pop("conv_hidden_size", 64),
            top_k=hp.pop("top_k", 5),
            num_kernels=hp.pop("num_kernels", 6),
            encoder_layers=hp.pop("encoder_layers", 2),
            dropout=hp.pop("dropout", 0.1),
            accelerator=self._accelerator,
            enable_progress_bar=self._enable_progress_bar,
            **hp,
        )
