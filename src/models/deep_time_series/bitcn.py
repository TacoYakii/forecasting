"""
BiTCN forecaster: NeuralForecast BiTCN wrapper.

BiTCN is a Bidirectional Temporal Convolutional Network that uses dilated
causal convolutions to capture long-range temporal dependencies without
recurrence or attention. Inspired by WaveNet, it applies parallel
convolutional branches to historical context and future covariates,
merging them via residual connections for direct multi-horizon output.

Exogenous variable support:
    - futr_exog: Supported via a dedicated future convolutional branch.
    - hist_exog: Supported via historical convolutional branch.

Reference:
    Oord et al., "WaveNet: A Generative Model for Raw Audio" (2016);
    NeuralForecast BiTCN implementation (Bai et al., TCN 2018 extension).
"""

from neuralforecast.models import BiTCN

from src.core.base_deep_model import BaseDeepModel
from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register_model(name="bitcn")
class BiTCNForecaster(BaseDeepModel):
    """
    Bidirectional Temporal Convolutional Network probabilistic forecaster.

    Wraps NeuralForecast BiTCN with the unified BaseDeepModel interface.
    Produces QuantileForecastResult output by default (MQLoss).

    Loss selection (via hyperparameter dict):
        loss_type (str): "quantile" (default), "distribution", "implicit_quantile"
        distribution (str): For loss_type="distribution". Default: "StudentT"

    Model-specific hyperparameters:
        hidden_size (int): TCN hidden channel dimension. Default: 16
        dropout (float): Dropout probability. Default: 0.5

    Example:
        >>> model = BiTCNForecaster(
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

    def _create_model(self) -> BiTCN:
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

        return BiTCN(
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
            # Model architecture params
            hidden_size=hp.pop("hidden_size", 16),
            dropout=hp.pop("dropout", 0.5),
            accelerator=self._accelerator,
            enable_progress_bar=self._enable_progress_bar,
            **hp,
        )
