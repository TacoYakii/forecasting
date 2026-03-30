"""
DeepAR forecaster: NeuralForecast DeepAR wrapper.

DeepAR is an autoregressive recurrent network for probabilistic time series
forecasting. It models the conditional distribution of future values given
past context, producing probabilistic predictions.

Default: DistributionLoss("StudentT"). Configurable via loss_type hyperparameter.

Reference:
    Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive
    Recurrent Networks", International Journal of Forecasting, 2020.
"""

from neuralforecast.models import DeepAR
from src.core.base_deep_model import BaseDeepModel
from src.models.machine_learning.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register_model(name="deepar")
class DeepARForecaster(BaseDeepModel):
    """
    DeepAR probabilistic forecaster.

    Wraps NeuralForecast DeepAR with the unified BaseDeepModel interface.
    Produces QuantileForecastResult output.

    Loss selection (via hyperparameter dict):
        loss_type (str): "distribution" (default), "quantile", "implicit_quantile"
        distribution (str): For loss_type="distribution". Default: "StudentT"

    Model-specific hyperparameters:
        lstm_n_layers (int): Number of LSTM layers. Default: 2
        lstm_hidden_size (int): LSTM hidden dimension. Default: 128
        lstm_dropout (float): LSTM dropout probability. Default: 0.1
        trajectory_samples (int): Monte Carlo samples for prediction. Default: 100
        decoder_hidden_layers (int): Extra MLP decoder layers. Default: 0
        decoder_hidden_size (int): MLP decoder hidden size. Default: 0

    Example:
        >>> model = DeepARForecaster(
        ...     dataset=df, y_col="power",
        ...     hyperparameter={
        ...         "input_size": 168,
        ...         "prediction_length": 48,
        ...         "loss_type": "implicit_quantile",  # or "distribution", "quantile"
        ...     }
        ... )
        >>> model.fit()
        >>> result = model.predict()
        >>> result.quantile(0.9, h=6)
    """

    def _create_model(self) -> DeepAR:
        hp = dict(self._model_hp)

        # Build exogenous feature lists
        feat_cols = self._get_feature_cols(self.dataset)
        futr_exog = feat_cols if feat_cols else None

        loss, valid_loss = self._create_loss(
            default_loss_type="distribution",
            default_distribution="StudentT",
        )

        return DeepAR(
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
            # Model architecture params
            lstm_n_layers=hp.pop("lstm_n_layers", 2),
            lstm_hidden_size=hp.pop("lstm_hidden_size", 128),
            lstm_dropout=hp.pop("lstm_dropout", 0.1),
            trajectory_samples=hp.pop("trajectory_samples", 100),
            decoder_hidden_layers=hp.pop("decoder_hidden_layers", 0),
            decoder_hidden_size=hp.pop("decoder_hidden_size", 0),
            accelerator=self._accelerator,
            **hp,
        )
