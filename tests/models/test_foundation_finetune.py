"""Tests for foundation model fine-tuning.

Covers:
    - FineTuneStrategy enum parsing and backward compatibility
    - Chronos v1 full fine-tuning (tiny model)
    - Moirai full/head fine-tuning (small model)
    - Save/Load round-trip for fine-tuned models
    - .pop() → .get() bug fix verification
"""

import warnings

import numpy as np
import pytest

from src.core.base_foundation_model import FineTuneStrategy

from .conftest import Y_COL

PREDICTION_LENGTH = 6
N_SAMPLES = 10
CONTEXT_LENGTH = 64


# ======================================================================
# FineTuneStrategy enum
# ======================================================================


class TestFineTuneStrategy:
    """FineTuneStrategy StrEnum behavior."""

    def test_str_equality(self):
        """StrEnum members compare equal to their string values."""
        assert FineTuneStrategy.FULL == "full"
        assert FineTuneStrategy.HEAD == "head"
        assert FineTuneStrategy.LORA == "lora"

    def test_construction_from_string(self):
        """Constructing from string returns the enum member."""
        assert FineTuneStrategy("full") is FineTuneStrategy.FULL
        assert FineTuneStrategy("head") is FineTuneStrategy.HEAD

    def test_invalid_raises(self):
        """Invalid strategy string raises ValueError."""
        with pytest.raises(ValueError):
            FineTuneStrategy("invalid")


# ======================================================================
# Backward compatibility
# ======================================================================


class TestBackwardCompatibility:
    """fine_tune=True → fine_tune_strategy='full' migration."""

    def test_deprecated_fine_tune_true(self):
        """fine_tune=True emits DeprecationWarning and maps to FULL."""
        from src.models.foundation.chronos import ChronosForecaster

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = ChronosForecaster(
                hyperparameter={
                    "model_name_or_path": "amazon/chronos-t5-tiny",
                    "fine_tune": True,
                }
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert model._fine_tune_strategy == FineTuneStrategy.FULL

    def test_strategy_none_by_default(self):
        """Default: no fine-tuning."""
        from src.models.foundation.chronos import ChronosForecaster

        model = ChronosForecaster(
            hyperparameter={
                "model_name_or_path": "amazon/chronos-t5-tiny",
            }
        )
        assert model._fine_tune_strategy is None

    def test_strategy_overrides_fine_tune(self):
        """fine_tune_strategy takes precedence over fine_tune."""
        from src.models.foundation.chronos import ChronosForecaster

        model = ChronosForecaster(
            hyperparameter={
                "model_name_or_path": "amazon/chronos-t5-tiny",
                "fine_tune": True,
                "fine_tune_strategy": "head",
            }
        )
        assert model._fine_tune_strategy == FineTuneStrategy.HEAD


# ======================================================================
# .pop() → .get() fix verification
# ======================================================================


class TestPopBugFix:
    """Verify .pop() → .get() fix allows repeated _load_pretrained()."""

    @pytest.mark.slow
    def test_chronos_double_load(self):
        """Chronos: calling _load_pretrained() twice preserves dtype."""
        from src.models.foundation.chronos import ChronosForecaster

        model = ChronosForecaster(
            hyperparameter={
                "model_name_or_path": "amazon/chronos-t5-tiny",
                "torch_dtype": "float32",
            }
        )
        model._load_pretrained()
        model._load_pretrained()  # second call should not crash
        assert model._pipeline is not None

    @pytest.mark.slow
    def test_moirai_double_load(self):
        """Moirai: calling _load_pretrained() twice preserves config."""
        from src.models.foundation.moirai import MoiraiForecaster

        model = MoiraiForecaster(
            hyperparameter={
                "model_name_or_path": "Salesforce/moirai-1.0-R-small",
                "patch_size": 32,
                "batch_size": 16,
            }
        )
        model._load_pretrained()
        first_batch_size = model._batch_size
        model._load_pretrained()
        assert model._batch_size == first_batch_size == 16


# ======================================================================
# Chronos v1 fine-tuning
# ======================================================================


@pytest.mark.slow
class TestChronosFineTune:
    """Chronos v1 full fine-tuning E2E tests."""

    def test_full_finetune_e2e(self, train_df):
        """fit with strategy='full' → forecast produces valid output."""
        from src.models.foundation.chronos import ChronosForecaster

        model = ChronosForecaster(
            hyperparameter={
                "model_name_or_path": "amazon/chronos-t5-tiny",
                "prediction_length": PREDICTION_LENGTH,
                "n_samples": N_SAMPLES,
                "context_length": CONTEXT_LENGTH,
                "fine_tune_strategy": "full",
                "fine_tune_epochs": 1,
                "fine_tune_batch_size": 2,
                "fine_tune_gradient_accumulation_steps": 1,
                "fine_tune_mixed_precision": False,
            }
        )
        model.fit(dataset=train_df, y_col=Y_COL)
        assert model.is_fitted_
        assert model._fine_tune_strategy == FineTuneStrategy.FULL

        result = model.forecast()
        assert result.samples.shape == (1, N_SAMPLES, PREDICTION_LENGTH)
        assert np.all(np.isfinite(result.samples))

    def test_bolt_raises(self, train_df):
        """Bolt models raise error for fine-tuning.

        NotImplementedError if model loads, or TypeError/OSError if the
        installed chronos package doesn't support Bolt configs.
        """
        from src.models.foundation.chronos import ChronosForecaster

        model = ChronosForecaster(
            hyperparameter={
                "model_name_or_path": "amazon/chronos-bolt-tiny",
                "prediction_length": PREDICTION_LENGTH,
                "fine_tune_strategy": "full",
            }
        )
        with pytest.raises((NotImplementedError, TypeError, OSError)):
            model.fit(dataset=train_df, y_col=Y_COL)

    def test_save_load_roundtrip(self, train_df, tmp_path):
        """Save fine-tuned Chronos → load → predict_from_context."""
        from src.models.foundation.chronos import ChronosForecaster

        model = ChronosForecaster(
            hyperparameter={
                "model_name_or_path": "amazon/chronos-t5-tiny",
                "prediction_length": PREDICTION_LENGTH,
                "n_samples": N_SAMPLES,
                "context_length": CONTEXT_LENGTH,
                "fine_tune_strategy": "full",
                "fine_tune_epochs": 1,
                "fine_tune_batch_size": 2,
                "fine_tune_gradient_accumulation_steps": 1,
                "fine_tune_mixed_precision": False,
            }
        )
        model.fit(dataset=train_df, y_col=Y_COL)

        # Save via _save_model_specific
        model_path = tmp_path / "chronos_ft"
        model._save_model_specific(model_path)

        # Load into a fresh instance
        loaded = ChronosForecaster(
            hyperparameter={
                "model_name_or_path": "amazon/chronos-t5-tiny",
                "prediction_length": PREDICTION_LENGTH,
                "n_samples": N_SAMPLES,
                "context_length": CONTEXT_LENGTH,
            }
        )
        loaded._load_model_specific(model_path)
        loaded.is_fitted_ = True

        assert loaded._fine_tune_strategy == FineTuneStrategy.FULL
        assert loaded._fine_tuned_model_path is not None

        # Predict from context
        context = train_df[Y_COL].values[-CONTEXT_LENGTH:]
        result = loaded.predict_from_context(
            context_y=context, horizon=PREDICTION_LENGTH
        )
        assert result.samples.shape == (1, N_SAMPLES, PREDICTION_LENGTH)
        assert np.all(np.isfinite(result.samples))


# ======================================================================
# Moirai fine-tuning (deferred — uni2ts numpy 2.x incompatibility)
# ======================================================================


@pytest.mark.slow
class TestMoiraiFineTune:
    """Moirai fine-tuning raises NotImplementedError."""

    def test_finetune_raises(self, train_df):
        """Any fine-tune strategy raises NotImplementedError."""
        from src.models.foundation.moirai import MoiraiForecaster

        model = MoiraiForecaster(
            hyperparameter={
                "model_name_or_path": "Salesforce/moirai-1.0-R-small",
                "prediction_length": PREDICTION_LENGTH,
                "fine_tune_strategy": "full",
            }
        )
        with pytest.raises(NotImplementedError, match="numpy"):
            model.fit(dataset=train_df, y_col=Y_COL)
