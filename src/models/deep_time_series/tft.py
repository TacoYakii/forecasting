"""
TFT forecaster: NeuralForecast Temporal Fusion Transformer wrapper.

TFT is an attention-based architecture for interpretable multi-horizon
forecasting. It uses variable selection networks, static enrichment, and
temporal self-attention to produce probabilistic predictions with built-in
feature importance.

Default: MQLoss (quantile regression). Configurable via loss_type hyperparameter.

Reference:
    Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon
    Time Series Forecasting", International Journal of Forecasting, 2021.
"""

from neuralforecast.models import TFT
from src.core.base_deep_model import BaseDeepModel
from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register_model(name="tft")
class TFTForecaster(BaseDeepModel):
    """
    Temporal Fusion Transformer probabilistic forecaster.

    Wraps NeuralForecast TFT with the unified BaseDeepModel interface.
    Produces QuantileForecastResult output.

    Exogenous variable support:
        - futr_exog: Supported.
        - hist_exog: Supported. TFT's Variable Selection Network compresses
          historical features into encoder hidden state, so hist-only variables
          can be used without future values.

    Loss selection (via hyperparameter dict):
        loss_type (str): "quantile" (default), "distribution", "implicit_quantile"
        distribution (str): For loss_type="distribution". Default: "StudentT"

    Model-specific hyperparameters:
        hidden_size (int): Hidden layer dimension. Default: 128
        n_head (int): Number of attention heads. Default: 4
        dropout (float): Dropout probability. Default: 0.1
        attn_dropout (float): Attention dropout. Default: 0.0
        n_rnn_layers (int): Number of RNN layers. Default: 1

    Example:
        >>> model = TFTForecaster(
        ...     dataset=df, y_col="power",
        ...     hyperparameter={
        ...         "input_size": 168,
        ...         "prediction_length": 48,
        ...         "loss_type": "distribution",       # or "quantile", "implicit_quantile"
        ...         "distribution": "Normal",
        ...     }
        ... )
        >>> model.fit()
        >>> result = model.predict()
        >>> result.to_distribution(6).ppf(0.9)
    """

    SUPPORTS_HIST_EXOG = True

    def _create_model(self) -> TFT:
        hp = dict(self._model_hp)

        # Build exogenous feature lists (futr/hist split)
        futr_exog = self.futr_cols or None
        hist_exog = self.hist_cols or None

        loss, valid_loss = self._create_loss(
            default_loss_type="quantile",
            default_distribution="StudentT",
        )

        return TFT(
            h=self._prediction_length,
            input_size=self._input_size,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=self._max_steps,
            batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            early_stop_patience_steps=self._early_stop,
            scaler_type=self._scaler_type,
            futr_exog_list=futr_exog,
            hist_exog_list=hist_exog,
            # Model architecture params
            hidden_size=hp.pop("hidden_size", 128),
            n_head=hp.pop("n_head", 4),
            dropout=hp.pop("dropout", 0.1),
            attn_dropout=hp.pop("attn_dropout", 0.0),
            n_rnn_layers=hp.pop("n_rnn_layers", 1),
            accelerator=self._accelerator,
            **hp,
        )
