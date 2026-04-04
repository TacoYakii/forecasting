"""Chronos forecaster: Amazon Chronos pretrained time series model wrapper.

Chronos is a family of pretrained time series forecasting models based on
language model architectures. It tokenizes time series values and generates
probabilistic forecasts via autoregressive sampling.

Supports both Chronos v1 (T5 encoder-decoder) and Chronos-Bolt (faster,
encoder-only) model variants, all loaded from HuggingFace Hub.

Fine-tuning is currently supported for **Chronos v1 only** with the
``full`` strategy. Bolt fine-tuning requires an inference-path change
(Bolt outputs quantiles, not samples) and is deferred to a future release.

Reference:
    Ansari et al., "Chronos: Learning the Language of Time Series", 2024.
    Amazon, "Chronos-2: From Univariate to Universal Forecasting", 2025.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.core.base_foundation_model import (
    BaseFoundationModel,
    FineTuneStrategy,
)
from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register_model(name="chronos")
class ChronosForecaster(BaseFoundationModel):
    """Chronos probabilistic forecaster.

    Wraps Amazon Chronos pretrained models with the BaseFoundationModel
    interface. Produces SampleForecastResult or QuantileForecastResult.

    Chronos models do not support exogenous variables -- exog_cols are
    ignored.

    Fine-tuning support:
        - ``fine_tune_strategy="full"``: Full parameter fine-tuning via
          HuggingFace Trainer. Chronos v1 (T5) only.
        - Bolt models raise ``NotImplementedError`` for fine-tuning.

    Model-specific hyperparameters (passed via hyperparameter dict):
        model_name_or_path (str): HuggingFace model ID. REQUIRED.
        context_length (int): Context window size. Default: 512
        prediction_length (int): Forecast horizon. Default: 48
        n_samples (int): Number of forecast samples. Default: 100
        output_type (str): "samples" or "quantiles". Default: "samples"
        level (list[int]): Confidence levels. Default: [80, 90]
        torch_dtype (str): Torch dtype. Default: "float32"

    Example:
        >>> model = ChronosForecaster(
        ...     hyperparameter={
        ...         "model_name_or_path": "amazon/chronos-t5-small",
        ...         "prediction_length": 48,
        ...         "fine_tune_strategy": "full",
        ...     }
        ... )
        >>> model.fit(dataset=df, y_col="power")
        >>> result = model.forecast()
    """

    def _is_bolt(self) -> bool:
        """Check whether the loaded pipeline is a Chronos-Bolt model."""
        from chronos import ChronosBoltPipeline

        return isinstance(self._pipeline, ChronosBoltPipeline)

    # ------------------------------------------------------------------
    # Pretrained loading
    # ------------------------------------------------------------------

    def _load_pretrained(self) -> None:
        """Load Chronos pipeline from HuggingFace Hub or local path."""
        from chronos import ChronosPipeline

        dtype_str = self._model_hp.get("torch_dtype", "float32")
        torch_dtype = getattr(torch, dtype_str, torch.float32)

        load_path = self._fine_tuned_model_path or self._model_name_or_path
        self._pipeline = ChronosPipeline.from_pretrained(
            load_path,
            device_map=self._device,
            dtype=torch_dtype,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_samples(
        self,
        context_y: np.ndarray,
        prediction_length: int,
        context_X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate forecast samples via Chronos pipeline.

        Args:
            context_y: Context target values, shape (context_length,).
            prediction_length: Number of steps to forecast.
            context_X: Ignored -- Chronos has no exogenous support.

        Returns:
            np.ndarray of shape (n_samples, prediction_length).
        """
        context_tensor = torch.tensor(
            context_y, dtype=torch.float32
        ).unsqueeze(0)  # (1, context_length)

        with torch.no_grad():
            forecast = self._pipeline.predict(
                context_tensor,
                prediction_length=prediction_length,
                num_samples=self._n_samples,
            )
        # forecast shape: (1, n_samples, prediction_length)
        return forecast.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def _fine_tune_model(self) -> None:
        """Fine-tune Chronos v1 on training data.

        Reads ``self._fine_tune_strategy`` and dispatches accordingly.
        Currently only ``"full"`` strategy is supported, and only for
        Chronos v1 (T5-based) models.

        Raises:
            NotImplementedError: If model is Bolt or strategy is not
                ``"full"``.

        Example:
            >>> model = ChronosForecaster(
            ...     hyperparameter={
            ...         "model_name_or_path": "amazon/chronos-t5-tiny",
            ...         "fine_tune_strategy": "full",
            ...     }
            ... )
            >>> model.fit(dataset=df, y_col="power")
        """
        if self._is_bolt():
            raise NotImplementedError(
                "Fine-tuning is not yet supported for Chronos-Bolt "
                "models. Bolt outputs quantiles, which is incompatible "
                "with the current _predict_samples() interface. "
                "Use a Chronos v1 (T5) model instead."
            )

        if self._fine_tune_strategy != FineTuneStrategy.FULL:
            raise NotImplementedError(
                f"Chronos v1 only supports fine_tune_strategy='full', "
                f"got '{self._fine_tune_strategy}'."
            )

        self._fine_tune_full()

    def _fine_tune_full(self) -> None:
        """Full-parameter fine-tuning for Chronos v1 via HF Trainer."""
        from transformers import Trainer, TrainingArguments

        # --- Train/val split ---
        n = len(self.y)
        split_idx = int(n * (1 - self._ft_val_ratio))
        train_y = self.y[:split_idx]
        val_y = self.y[split_idx:]

        tokenizer = self._pipeline.tokenizer
        pred_len = tokenizer.config.prediction_length

        train_ds = _ChronosFineTuneDataset(
            train_y,
            context_length=self._context_length,
            prediction_length=pred_len,
            tokenizer=tokenizer,
        )
        if len(train_ds) < 10:
            warnings.warn(
                f"Chronos fine-tuning: only {len(train_ds)} training "
                f"windows generated. Consider using more data or "
                f"reducing context_length.",
                stacklevel=2,
            )

        # Validation dataset (may be None if too short)
        min_val_len = self._context_length + pred_len
        val_ds = None
        if len(val_y) >= min_val_len:
            val_ds = _ChronosFineTuneDataset(
                val_y,
                context_length=self._context_length,
                prediction_length=pred_len,
                tokenizer=tokenizer,
            )

        # --- Trainer setup ---
        inner_model = self._pipeline.inner_model
        inner_model.train()

        # Resolve mixed precision dtype
        fp16, bf16 = False, False
        if self._ft_mixed_precision:
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    bf16 = True
                else:
                    fp16 = True

        training_args = TrainingArguments(
            output_dir=".tmp_chronos_ft",
            num_train_epochs=self._ft_epochs,
            per_device_train_batch_size=self._ft_batch_size,
            gradient_accumulation_steps=self._ft_grad_accum,
            learning_rate=self._ft_lr,
            fp16=fp16,
            bf16=bf16,
            eval_strategy="epoch" if val_ds else "no",
            save_strategy="no",
            logging_steps=50,
            load_best_model_at_end=bool(val_ds),
            metric_for_best_model="eval_loss" if val_ds else None,
            report_to="none",
            disable_tqdm=False,
        )

        callbacks = []
        if val_ds and self._ft_patience > 0:
            from transformers import EarlyStoppingCallback

            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self._ft_patience,
                )
            )

        trainer = Trainer(
            model=inner_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            callbacks=callbacks,
        )

        trainer.train()

        # Ensure pipeline uses the fine-tuned weights (in-place)
        inner_model.eval()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save_model_specific(self, model_path: Path) -> Path:
        """Save Chronos model state.

        Calls base class to write JSON config, then saves fine-tuned
        weights if applicable.

        Args:
            model_path: Base path without extension.

        Returns:
            Path to saved JSON config.
        """
        # Save fine-tuned weights before writing JSON
        if self._fine_tune_strategy is not None:
            ft_dir = model_path.parent / f"{model_path.stem}_ft"
            ft_dir.mkdir(parents=True, exist_ok=True)
            self._pipeline.inner_model.save_pretrained(ft_dir)
            self._fine_tuned_model_path = str(ft_dir)

        return super()._save_model_specific(model_path)

    def _load_model_specific(self, model_path: Path) -> None:
        """Load Chronos model state.

        Restores base metadata, then loads pretrained (or fine-tuned)
        pipeline.

        Args:
            model_path: Base path without extension.
        """
        super()._load_model_specific(model_path)
        # _load_pretrained() already called by super() and uses
        # self._fine_tuned_model_path if set.


# ------------------------------------------------------------------
# Internal dataset for Chronos v1 fine-tuning
# ------------------------------------------------------------------


class _ChronosFineTuneDataset(torch.utils.data.Dataset):
    """Sliding-window dataset for Chronos v1 fine-tuning.

    Each item is a dict with ``input_ids``, ``attention_mask``, and
    ``labels`` — ready for HuggingFace Trainer.

    Args:
        y: 1-D target array.
        context_length: Encoder context window size.
        prediction_length: Decoder target length (from tokenizer config).
        tokenizer: ``MeanScaleUniformBins`` tokenizer from the pipeline.
        stride: Window stride. Defaults to ``prediction_length``.

    Example:
        >>> ds = _ChronosFineTuneDataset(y, 512, 64, tokenizer)
        >>> item = ds[0]  # dict with input_ids, attention_mask, labels
    """

    def __init__(
        self,
        y: np.ndarray,
        context_length: int,
        prediction_length: int,
        tokenizer,
        stride: Optional[int] = None,
    ):
        self._y = y.astype(np.float32)
        self._ctx_len = context_length
        self._pred_len = prediction_length
        self._tokenizer = tokenizer
        self._stride = stride or prediction_length

        window_size = context_length + prediction_length
        self._starts = list(
            range(0, len(y) - window_size + 1, self._stride)
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> dict:
        start = self._starts[idx]
        chunk = self._y[start: start + self._ctx_len + self._pred_len]

        context = torch.tensor(
            chunk[: self._ctx_len], dtype=torch.float32
        ).unsqueeze(0)  # (1, ctx_len)
        label = torch.tensor(
            chunk[self._ctx_len:], dtype=torch.float32
        ).unsqueeze(0)  # (1, pred_len)

        token_ids, attn_mask, scale = (
            self._tokenizer.context_input_transform(context)
        )
        label_ids, _ = self._tokenizer.label_input_transform(
            label, scale
        )

        return {
            "input_ids": token_ids.squeeze(0),
            "attention_mask": attn_mask.squeeze(0).long(),
            "labels": label_ids.squeeze(0),
        }
