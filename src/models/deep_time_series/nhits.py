"""
NHITS forecaster: NeuralForecast Neural Hierarchical Interpolation wrapper.

NHITS is a direct multi-horizon forecaster that decomposes the forecast into
hierarchical stacks operating at different frequency bands via multi-rate
input pooling and hierarchical interpolation. Each stack models a distinct
time scale (e.g., low-frequency synoptic/diurnal trends, mid-frequency ramp
events, high-frequency residuals), and stacks are combined residually.

Because NHITS is non-recurrent (RECURRENT=False), it does not suffer from
the error-compounding issue that affects autoregressive models like DeepAR
at long horizons.

Default: MQLoss (quantile regression). Configurable via loss_type hyperparameter.

Reference:
    Challu et al., "NHITS: Neural Hierarchical Interpolation for Time Series
    Forecasting", AAAI, 2023.
"""

from neuralforecast.models import NHITS

from src.core.base_deep_model import BaseDeepModel
from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register_model(name="nhits")
class NHITSForecaster(BaseDeepModel):
    """
    Neural Hierarchical Interpolation (NHITS) probabilistic forecaster.

    Wraps NeuralForecast NHITS with the unified BaseDeepModel interface.
    Produces QuantileForecastResult output by default (MQLoss).

    Exogenous variable support:
        - futr_exog: Supported.
        - hist_exog: Supported. NHITS concatenates historical exogenous features
          into the input tensor of its MLP stacks.

    Loss selection (via hyperparameter dict):
        loss_type (str): "quantile" (default), "distribution", "implicit_quantile"
        distribution (str): For loss_type="distribution". Default: "StudentT"

    Model-specific hyperparameters (wind power defaults at h=48, 1h resolution):
        n_blocks (list[int]): Blocks per stack. Default: [1, 1, 1]
        mlp_units (list[list[int]]): MLP sizes per stack. Default: [[512, 512]] * 3
        n_pool_kernel_size (list[int]): Input pooling kernels. Default: [8, 4, 1]
        n_freq_downsample (list[int]): Stack downsampling ratios. Default: [24, 8, 1]
            Interprets stacks as daily(24h) / ramp(8h) / HF-residual decomposition.
        pooling_mode (str): "MaxPool1d" or "AvgPool1d". Default: "AvgPool1d"
            Avg pooling is better matched to wind power smoothing characteristics.
        interpolation_mode (str): "linear", "nearest", or "cubic". Default: "linear"
        dropout_prob_theta (float): Basis dropout probability. Default: 0.0
        activation (str): MLP activation. Default: "ReLU"

    Example:
        >>> model = NHITSForecaster(
        ...     hyperparameter={
        ...         "input_size": 168,
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

    def _create_model(self) -> NHITS:
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

        return NHITS(
            h=self._prediction_length,
            input_size=self._input_size,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=self._max_steps,
            batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            early_stop_patience_steps=self._early_stop,
            futr_exog_list=futr_exog,
            hist_exog_list=hist_exog,
            **optional,
            # Multi-scale decomposition (wind power domain defaults)
            n_blocks=hp.pop("n_blocks", [1, 1, 1]),
            mlp_units=hp.pop("mlp_units", [[512, 512]] * 3),
            n_pool_kernel_size=hp.pop("n_pool_kernel_size", [8, 4, 1]),
            n_freq_downsample=hp.pop("n_freq_downsample", [24, 8, 1]),
            pooling_mode=hp.pop("pooling_mode", "AvgPool1d"),
            interpolation_mode=hp.pop("interpolation_mode", "linear"),
            dropout_prob_theta=hp.pop("dropout_prob_theta", 0.0),
            activation=hp.pop("activation", "ReLU"),
            accelerator=self._accelerator,
            enable_progress_bar=self._enable_progress_bar,
            **hp,
        )
