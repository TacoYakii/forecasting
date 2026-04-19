"""Configuration for hierarchical reconciliation models."""

from dataclasses import dataclass

from src.core.config import BaseConfig


@dataclass
class ReconciliationConfig(BaseConfig):
    """Architecture config for reconciliation strategies."""

    # === Strategy selection ===
    projection: str = "sparse"
    coherence: str = "ranked"
    combine: str = "weighted"

    # === Projection options ===
    mint_mode: str = "MINT_SHRINK"
    init_P: str = "ols"
    constraint_P: str = "linear"

    # === Sampling ===
    n_samples: int = 1000
    mc_samples: int = 5000
    q_start: float = 0.001
    q_end: float = 0.999

    def validate(self) -> None:
        valid_projections = ("topdown", "bottomup", "mint", "sparse")
        valid_coherences = ("ranked", "empirical_copula")
        valid_combines = ("linear", "weighted", "angular", "mean_shift")

        if self.projection not in valid_projections:
            raise ValueError(
                f"projection must be one of {valid_projections}, "
                f"got {self.projection!r}"
            )
        if self.coherence not in valid_coherences:
            raise ValueError(
                f"coherence must be one of {valid_coherences}, "
                f"got {self.coherence!r}"
            )
        if self.combine not in valid_combines:
            raise ValueError(
                f"combine must be one of {valid_combines}, "
                f"got {self.combine!r}"
            )

        # Structural/mint projections require linear combine
        if self.projection in ("topdown", "bottomup", "mint"):
            if self.coherence == "ranked" and self.combine != "linear":
                raise ValueError(
                    f"projection={self.projection!r} with coherence='ranked' "
                    f"requires combine='linear', got {self.combine!r}"
                )

        # Sparse projection cannot use linear combine
        if self.projection == "sparse" and self.combine == "linear":
            raise ValueError(
                "projection='sparse' cannot use combine='linear'. "
                "Use 'weighted' or 'angular'."
            )

        # Empirical copula requires mean_shift combine
        if self.coherence == "empirical_copula" and self.combine != "mean_shift":
            raise ValueError(
                "coherence='empirical_copula' requires combine='mean_shift', "
                f"got {self.combine!r}"
            )
