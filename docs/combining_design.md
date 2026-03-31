# Forecast Combining Module Design

## 1. Overview

여러 모형의 `ForecastResult` 객체를 받아, horizon별로 `Distribution`을 추출하고,
combining 전략에 따라 하나의 결합 예측(combined forecast)을 생성하는 모듈.

### 핵심 설계 원칙

- **Combiner는 Distribution만 다룬다** — ForecastResult에서 Distribution 추출은 공통 메서드가 담당
- **Combining 로직만 서브클래스가 구현** — 추출/검증/결과 조립은 베이스 클래스 공통
- **Horizon별 독립 처리** — 모형별 강점이 horizon에 따라 다르므로, 가중치/파라미터를 horizon별로 학습

### 모듈 위치

```
src/models/combining/
├── __init__.py
├── base.py              # BaseCombiner (추상 베이스)
├── angular.py           # AngularCombiner (angular combining)
└── equal_weight.py      # EqualWeightCombiner (동일 가중 baseline)
```

> `src/models/combining/`은 이미 디렉터리로 존재함. 기존 `PARK/Combine/def_combine.py`의
> 핵심 로직을 `src/models/combining/` 하위 모듈로 리팩터링.

---

## 2. Data Flow

```
사용자 입력:
  results_train: List[BaseForecastResult]  # M개 모형, 각 (N_train, H)
  results_test:  List[BaseForecastResult]  # M개 모형, 각 (N_test, H)
  observed:      np.ndarray                # (N_train, H) — training period 관측값

                         ┌─────────────────────────┐
                         │     BaseCombiner         │
                         ├─────────────────────────┤
  results_train ────────►│  extract_distributions() │
                         │  horizon h=1..H 루프     │
                         │    ↓                     │
                         │  List[Distribution] (M개)│
                         │    ↓                     │
                         │  _fit_horizon(h, ...)    │◄── 서브클래스 구현
                         │    → 파라미터 저장        │
                         └─────────────────────────┘

                         ┌─────────────────────────┐
  results_test ─────────►│  extract_distributions() │
                         │  horizon h=1..H 루프     │
                         │    ↓                     │
                         │  List[Distribution] (M개)│
                         │    ↓                     │
                         │  _combine_distributions()│◄── 서브클래스 구현
                         │    → (N_test, Q)         │
                         │    ↓                     │
                         │  조립: SampleForecastResult
                         │    (N_test, Q, H)        │
                         └─────────────────────────┘
```

### Shape 추적

| 단계 | Shape | 설명 |
|------|-------|------|
| `ForecastResult` | `(N, H)` | 모형별 전체 결과 |
| `to_distribution(h)` | `Distribution(T=N)` | 특정 horizon의 Distribution |
| `dist.ppf(quantile_levels)` | `(N, Q)` | 고정 quantile levels에서 결정론적 추출 |
| `extract_distributions(h)` | `List[Distribution]` len=M | M개 모형의 같은 horizon Distribution |
| `_combine_distributions(h)` | `(N, Q)` | 결합된 값 (cross-quantile mixing 가능) |
| 최종 결과 | `SampleForecastResult(N, Q, H)` | H개 horizon 조립 (Q를 sample 차원으로 사용) |

---

## 3. Class Design

### 3.1 BaseCombiner

```python
# src/models/combining/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.core.forecast_distribution import (
    EmpiricalDistribution,
    ParametricDistribution,
)
from src.core.forecast_results import (
    BaseForecastResult,
    SampleForecastResult,
)

Distribution = Union[ParametricDistribution, EmpiricalDistribution]


class BaseCombiner(ABC):
    """여러 모형의 ForecastResult를 결합하는 추상 베이스 클래스.

    공통 역할:
        - ForecastResult 리스트의 유효성 검증 (horizon 일치)
        - 공통 basis_index 추출 및 결과 정렬 (_align_results)
        - horizon별 Distribution 추출 (extract_distributions)
        - fit/combine 호출 시 horizon loop 관리
        - 최종 SampleForecastResult 조립

    서브클래스 역할:
        - _fit_horizon(): 특정 horizon에서 combining 파라미터 학습
        - _combine_distributions(): 학습된 파라미터로 Distribution 결합

    Attributes:
        n_quantiles (int): combining에 사용할 quantile level 수 (default: 1000).
        quantile_levels (np.ndarray): 고정 quantile levels, shape (n_quantiles,).
            (0, 1) 구간을 균등 분할. 모든 모형에서 동일한 level로 추출.
        fitted_params_ (Dict[int, Any]): horizon별 학습된 파라미터.

    Args:
        n_quantiles: quantile level 수. 클수록 분포 근사가 정밀해짐.

    Example:
        >>> combiner = AngularCombiner(n_quantiles=99)
        >>> combiner.fit(train_results, observed)
        >>> combined = combiner.combine(test_results)
        >>> combined.to_distribution(6).ppf([0.1, 0.5, 0.9])
    """

    def __init__(
        self,
        n_quantiles: int = 99,
    ):
        self.n_quantiles = n_quantiles
        self.quantile_levels = np.linspace(0, 1, n_quantiles + 2)[1:-1]  # (0, 1) 개구간
        self.is_fitted_ = False
        self.fitted_params_: Dict[int, Any] = {}
        self._horizon: Optional[int] = None

    # ── Validation & Alignment ──

    def _validate_results(
        self, results: List[BaseForecastResult]
    ) -> None:
        """ForecastResult 리스트의 호환성 검증.

        검증 항목:
            1. 최소 2개 이상의 모형
            2. 모든 모형의 horizon이 동일

        Note:
            basis_index 일치는 검증하지 않음. futr_exog 사용 모형과
            미사용 모형의 예측 길이가 다를 수 있으므로, _align_results()에서
            공통 index를 추출하여 정렬함.

        Raises:
            ValueError: 검증 실패 시.
        """
        if len(results) < 2:
            raise ValueError(
                f"최소 2개 모형 필요, {len(results)}개 제공됨."
            )

        H = results[0].horizon
        for i, r in enumerate(results[1:], 1):
            if r.horizon != H:
                raise ValueError(
                    f"Horizon 불일치: results[0].horizon={H}, "
                    f"results[{i}].horizon={r.horizon}"
                )

    def _align_results(
        self,
        results: List[BaseForecastResult],
    ) -> List[BaseForecastResult]:
        """모든 ForecastResult를 공통 basis_index로 정렬.

        futr_exog를 사용하는 모형과 미사용 모형의 예측 구간이 다를 수
        있으므로, 모든 results의 basis_index 교집합을 구한 뒤 각 result를
        공통 index로 슬라이싱한다.

        내부적으로 BaseForecastResult.reindex(common_idx)를 호출.

        Args:
            results: M개 모형의 ForecastResult 리스트.

        Returns:
            List[BaseForecastResult]: 공통 index로 정렬된 결과 리스트.
                이미 모든 basis_index가 동일하면 원본 그대로 반환.

        Raises:
            ValueError: 공통 index가 비어있는 경우.

        Example:
            >>> # ARIMA (futr_exog 사용): basis_index 길이 300
            >>> # Chronos (exog 없음):    basis_index 길이 350
            >>> aligned = combiner._align_results([arima_result, chronos_result])
            >>> len(aligned[0])  # 공통 구간 <= 300
        """
        common_idx = results[0].basis_index
        for r in results[1:]:
            common_idx = common_idx.intersection(r.basis_index)

        if len(common_idx) == 0:
            raise ValueError(
                "공통 basis_index가 비어있습니다. "
                "모형들의 예측 구간이 전혀 겹치지 않습니다."
            )

        # 이미 모든 index가 동일하면 불필요한 복사 방지
        all_equal = all(
            r.basis_index.equals(common_idx) for r in results
        )
        if all_equal:
            return results

        return [r.reindex(common_idx) for r in results]

    # ── Distribution Extraction (공통) ──

    def extract_distributions(
        self,
        results: List[BaseForecastResult],
        h: int,
    ) -> List[Distribution]:
        """특정 horizon에 대해 각 모형의 Distribution을 추출.

        Args:
            results: M개 모형의 ForecastResult 리스트.
            h: forecast horizon (1-indexed).

        Returns:
            List[Distribution]: 길이 M, 각 Distribution의 T = N (basis times).
        """
        return [r.to_distribution(h) for r in results]

    # ── Fit (Training Period) ──

    def fit(
        self,
        results: List[BaseForecastResult],
        observed: pd.DataFrame,
    ) -> "BaseCombiner":
        """Training period ForecastResult들로 combining 파라미터 학습.

        각 horizon h에 대해:
            1. _align_results(results) → 공통 index로 정렬
            2. extract_distributions(results, h) → List[Distribution]
            3. _fit_horizon(h, dists, observed_h) → 파라미터 저장

        Args:
            results: M개 모형의 training period ForecastResult.
                     각 (N_train, H) shape. basis_index가 다를 수 있음.
            observed: 관측값 DataFrame, shape (N_total, H), index가
                      basis times를 포함. 공통 index에 맞춰 자동 정렬됨.

        Returns:
            Self: method chaining 지원.
        """
        self._validate_results(results)
        results = self._align_results(results)

        H = results[0].horizon
        self._horizon = H
        common_idx = results[0].basis_index

        # observed도 공통 index에 맞춰 정렬
        observed_aligned = observed.loc[common_idx].values  # (N_common, H)

        for h in range(1, H + 1):
            dists = self.extract_distributions(results, h)
            observed_h = observed_aligned[:, h - 1]  # (N_common,)
            self.fitted_params_[h] = self._fit_horizon(h, dists, observed_h)

        self.is_fitted_ = True
        return self

    # ── Combine (Test Period) ──

    def combine(
        self,
        results: List[BaseForecastResult],
    ) -> SampleForecastResult:
        """Test period ForecastResult들을 결합하여 SampleForecastResult 반환.

        입력은 고정 quantile levels에서 결정론적으로 추출하지만,
        combining 방법에 따라 cross-quantile mixing이 발생할 수 있으므로
        (예: angular combining의 degree > 0) 결과는 sample로 취급.

        각 horizon h에 대해:
            1. _align_results(results) → 공통 index로 정렬
            2. extract_distributions(results, h) → List[Distribution]
            3. _combine_distributions(h, dists) → (N_common, Q)
        결과를 조립하여 SampleForecastResult(N_common, Q, H) 반환.

        Args:
            results: M개 모형의 test period ForecastResult.
                     각 (N_test, H) shape. basis_index가 다를 수 있음.

        Returns:
            SampleForecastResult: shape (N_common, Q, H).
                Q = n_quantiles. cross-quantile mixing으로 인해
                원래 quantile level 대응이 깨질 수 있으므로 sample로 반환.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        if not self.is_fitted_:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")

        self._validate_results(results)
        results = self._align_results(results)

        H = results[0].horizon
        basis_index = results[0].basis_index

        combined_per_h = []  # List of (N, Q)
        for h in range(1, H + 1):
            dists = self.extract_distributions(results, h)
            combined_h = self._combine_distributions(h, dists)
            # combined_h shape: (N, Q)
            combined_per_h.append(combined_h)

        # Stack: List of (N, Q) → (N, Q, H)
        samples_all = np.stack(combined_per_h, axis=2)

        return SampleForecastResult(
            samples=samples_all,
            basis_index=basis_index,
        )

    # ── Abstract Methods (서브클래스 구현) ──

    @abstractmethod
    def _fit_horizon(
        self,
        h: int,
        distributions: List[Distribution],
        observed: np.ndarray,
    ) -> Any:
        """특정 horizon에서 combining 파라미터 학습.

        Args:
            h: forecast horizon (1-indexed).
            distributions: M개 모형의 Distribution, 각 T=N_train.
            observed: 해당 horizon의 관측값, shape (N_train,).

        Returns:
            학습된 파라미터 (Any). fitted_params_[h]에 저장됨.
        """
        ...

    @abstractmethod
    def _combine_distributions(
        self,
        h: int,
        distributions: List[Distribution],
    ) -> np.ndarray:
        """학습된 파라미터로 Distribution들을 결합.

        Args:
            h: forecast horizon (1-indexed).
            distributions: M개 모형의 Distribution, 각 T=N_test.

        Returns:
            np.ndarray: 결합된 quantile values, shape (N, Q).
                Q = len(self.quantile_levels).
        """
        ...
```

### 3.2 AngularCombiner

```python
# src/models/combining/angular.py

from typing import Any, Dict, List, Optional, Union
from itertools import product

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from src.core.forecast_distribution import (
    EmpiricalDistribution,
    ParametricDistribution,
)
from src.utils.metrics_jax.crps import crps_numerical_j

from .base import BaseCombiner, Distribution


class AngularCombineConfig:
    """Angular combining 최적화 설정.

    Attributes:
        BETA_MIN/MAX: 가중치 파라미터 beta 탐색 범위.
        ANGLE_MIN/MAX: angular combining degree 탐색 범위.
        GRID_POINTS: grid search 해상도.
        OPTIMIZATION_TOLERANCE: L-BFGS-B 수렴 기준 (ftol).
        GRADIENT_TOLERANCE: L-BFGS-B gradient 기준 (gtol).
        MAX_ITERATIONS: L-BFGS-B 최대 반복 횟수.
    """

    RANDOM_SEED = 99
    BETA_MIN = 0.2
    BETA_MAX = 5.0
    ANGLE_MIN = 0.0
    ANGLE_MAX = np.pi / 2
    GRID_POINTS = 10
    OPTIMIZATION_TOLERANCE = 1e-10
    GRADIENT_TOLERANCE = 1e-8
    MAX_ITERATIONS = 1000


class AngularCombiner(BaseCombiner):
    """Angular forecast combining.

    Baran & Lerch (2018) angular combining을
    ForecastResult/Distribution 인터페이스에 맞게 구현.

    학습 파라미터: [beta, degree] per horizon
        - beta: 가중치 계산에 사용 (w = 1/loss^beta)
        - degree: angular combining 강도 (0이면 단순 가중 평균)

    최적화: 2단계
        1. Grid search (beta × degree 그리드)
        2. L-BFGS-B local optimization (JAX gradient 사용)

    Attributes:
        config (AngularCombineConfig): 최적화 설정.
        base_losses_ (Dict[int, np.ndarray]): horizon별 모형별 base CRPS loss.

    Args:
        n_quantiles: combining 결과 quantile 수 (default: 99).
        seed: random seed.
        config: 최적화 설정. None이면 기본값 사용.

    Example:
        >>> combiner = AngularCombiner(n_quantiles=99)
        >>> combiner.fit(train_results, observed)
        >>> combined = combiner.combine(test_results)
    """

    def __init__(
        self,
        n_quantiles: int = 99,
        config: Optional[AngularCombineConfig] = None,
    ):
        super().__init__(n_quantiles=n_quantiles)
        self.config = config or AngularCombineConfig()
        self.base_losses_: Dict[int, np.ndarray] = {}

    # ── Distribution → Quantile Values 변환 (내부 헬퍼) ──

    def _distributions_to_quantiles(
        self,
        distributions: List[Distribution],
    ) -> jnp.ndarray:
        """M개 Distribution에서 고정 quantile levels로 값을 추출하여 (N, M, Q) 배열로 변환.

        모든 모형에서 동일한 self.quantile_levels를 사용하므로
        결정론적이고 재현 가능한 결과를 보장.

        Args:
            distributions: M개 Distribution, 각 T=N.

        Returns:
            jnp.ndarray: shape (N, M, Q), Q = len(self.quantile_levels).
                         quantile level 순서대로 정렬되어 있음.
        """
        quantiles_list = []
        for dist in distributions:
            q = dist.ppf(self.quantile_levels)  # (N, Q)
            quantiles_list.append(q)

        # Stack: (M, N, Q) → transpose → (N, M, Q)
        stacked = np.stack(quantiles_list, axis=0)  # (M, N, Q)
        stacked = stacked.transpose(1, 0, 2)        # (N, M, Q)
        return jnp.array(stacked)

    # ── Base CRPS Loss 계산 ──

    def _compute_base_losses(
        self,
        distributions: List[Distribution],
        observed: np.ndarray,
    ) -> np.ndarray:
        """각 모형의 CRPS loss를 계산.

        고정 quantile levels에서 추출한 quantile values로 CRPS를 계산.

        Args:
            distributions: M개 Distribution.
            observed: 관측값, shape (N,).

        Returns:
            np.ndarray: shape (M,), 모형별 평균 CRPS.
        """
        losses = []
        for dist in distributions:
            q_values = dist.ppf(self.quantile_levels)  # (N, Q)
            crps_vals = jax.vmap(
                crps_numerical_j, in_axes=(0, 0, None)
            )(jnp.array(observed), jnp.array(q_values), True)
            losses.append(float(jnp.mean(crps_vals)))
        return np.array(losses)

    # ── Combining 핵심 로직 (기존 Combiner.combine 포팅) ──

    @staticmethod
    def _angular_combine_core(
        dataset: jnp.ndarray,
        weight: jnp.ndarray,
        beta: float,
        degree: float,
        key: jax.Array,
    ) -> jnp.ndarray:
        """Angular combining 핵심 연산.

        Args:
            dataset: (N, M, Q), 고정 quantile levels에서 추출한 값.
                     quantile level 순으로 정렬되어 있음.
            weight: (M,), combining 가중치.
            beta: 가중치 파라미터 (이 함수에서는 사용하지 않음, weight에 이미 반영).
            degree: angular combining 강도.
            key: JAX random key.

        Returns:
            jnp.ndarray: (N, Q), 결합된 quantile values.
        """
        N, M, Q = dataset.shape
        quantile_values = jnp.linspace(0, 1, Q)

        if degree < 1e-3:
            # 단순 가중 평균
            # dataset: (N, M, Q) → transpose → (N, Q, M)
            res = jnp.sum(
                dataset.transpose(0, 2, 1) * weight, axis=2
            )  # (N, Q)
        else:
            # Angular combining
            indices = jax.random.choice(
                key=key,
                a=M,
                shape=(N, Q),
                replace=True,
                p=weight,
            )

            s0 = (dataset + quantile_values / degree).transpose(0, 2, 1)
            s = jnp.take_along_axis(s0, indices[..., None], axis=2).squeeze(-1)
            s_sorted = jnp.sort(s, axis=1)
            res = s_sorted - jnp.arange(Q) / (Q * degree)

        return res  # (N, Q)

    # ── Abstract Method 구현 ──

    def _fit_horizon(
        self,
        h: int,
        distributions: List[Distribution],
        observed: np.ndarray,
    ) -> np.ndarray:
        """특정 horizon에서 [beta, degree] 최적화.

        1단계: Grid search로 초기값 탐색
        2단계: L-BFGS-B로 local optimization

        Returns:
            np.ndarray: 최적 [beta, degree].
        """
        # Distribution → quantile dataset
        dataset = self._distributions_to_quantiles(distributions)  # (N, M, Q)
        base_losses = self._compute_base_losses(distributions, observed)
        self.base_losses_[h] = base_losses

        observed_jnp = jnp.array(observed)
        base_losses_jnp = jnp.array(base_losses)
        key = jax.random.key(self.config.RANDOM_SEED)

        # Objective: [beta, degree] → mean CRPS
        def objective(params):
            beta, degree = params[0], params[1]
            w = 1.0 / (base_losses_jnp ** beta)
            weight = w / jnp.sum(w)

            key_local = jax.random.key(self.config.RANDOM_SEED)
            combined = self._angular_combine_core(
                dataset, weight, beta, degree, key_local,
            )
            crps = jnp.mean(
                jax.vmap(crps_numerical_j, in_axes=(0, 0, None))(
                    observed_jnp, combined, True,
                )
            )
            return crps

        obj_grad = jax.grad(objective)

        # 1단계: Grid search
        cfg = self.config
        beta_grid = np.linspace(cfg.BETA_MIN, cfg.BETA_MAX, cfg.GRID_POINTS)
        angle_grid = np.linspace(cfg.ANGLE_MIN, cfg.ANGLE_MAX, cfg.GRID_POINTS)

        best_loss = float("inf")
        best_params = np.array([1.0, 0.0])

        for beta, angle in product(beta_grid, angle_grid):
            params = jnp.array([beta, angle])
            loss = float(objective(params))
            if loss < best_loss:
                best_loss = loss
                best_params = np.array([beta, angle])

        # 2단계: Local optimization
        def scipy_obj_grad(params):
            p = jnp.array(params)
            loss = float(objective(p))
            grad = np.array(obj_grad(p))
            return loss, grad

        result = minimize(
            fun=scipy_obj_grad,
            x0=best_params,
            jac=True,
            bounds=[
                (cfg.BETA_MIN, cfg.BETA_MAX),
                (cfg.ANGLE_MIN, cfg.ANGLE_MAX),
            ],
            method="L-BFGS-B",
            options={
                "ftol": cfg.OPTIMIZATION_TOLERANCE,
                "gtol": cfg.GRADIENT_TOLERANCE,
                "maxiter": cfg.MAX_ITERATIONS,
            },
        )

        return result.x  # np.ndarray [beta, degree]

    def _combine_distributions(
        self,
        h: int,
        distributions: List[Distribution],
    ) -> np.ndarray:
        """학습된 [beta, degree]로 Distribution들을 결합.

        Returns:
            np.ndarray: (N, Q).
        """
        params = self.fitted_params_[h]  # [beta, degree]
        beta, degree = params[0], params[1]

        dataset = self._distributions_to_quantiles(distributions)
        base_losses = jnp.array(self.base_losses_[h])

        w = 1.0 / (base_losses ** beta)
        weight = w / jnp.sum(w)

        key = jax.random.key(self.config.RANDOM_SEED)
        combined = self._angular_combine_core(
            dataset, weight, beta, degree, key,
        )
        return np.array(combined)  # (N, Q)
```

### 3.3 EqualWeightCombiner

```python
# src/models/combining/equal_weight.py

from typing import Any, List, Optional

import numpy as np

from .base import BaseCombiner, Distribution


class EqualWeightCombiner(BaseCombiner):
    """단순 동일 가중 샘플 평균 combiner (baseline).

    모든 모형에서 동일 수의 샘플을 뽑은 뒤 element-wise 평균.
    학습 파라미터 없음.

    Example:
        >>> combiner = EqualWeightCombiner(n_quantiles=99)
        >>> combiner.fit(train_results, observed)  # no-op
        >>> combined = combiner.combine(test_results)
    """

    def _fit_horizon(
        self,
        h: int,
        distributions: List[Distribution],
        observed: np.ndarray,
    ) -> None:
        """학습할 파라미터 없음."""
        return None

    def _combine_distributions(
        self,
        h: int,
        distributions: List[Distribution],
    ) -> np.ndarray:
        """동일 가중 quantile 평균.

        각 Distribution에서 동일한 quantile levels로 값을 추출하고 평균.

        Returns:
            np.ndarray: (N, Q).
        """
        M = len(distributions)
        q_sum = np.zeros(
            (len(distributions[0]), self.n_quantiles),
            dtype=float,
        )

        for dist in distributions:
            q = dist.ppf(self.quantile_levels)  # (N, Q)
            q_sum += q

        return q_sum / M  # (N, Q)
```

### 3.4 `__init__.py`

```python
# src/models/combining/__init__.py

from .base import BaseCombiner
from .angular import AngularCombiner, AngularCombineConfig
from .equal_weight import EqualWeightCombiner

__all__ = [
    "BaseCombiner",
    "AngularCombiner",
    "AngularCombineConfig",
    "EqualWeightCombiner",
]
```

---

## 4. Interface Specification

### 4.1 BaseCombiner Public API

| Method | Signature | 설명 |
|--------|-----------|------|
| `__init__` | `(n_quantiles=99, seed=None)` | 공통 초기화 |
| `fit` | `(results: List[BaseForecastResult], observed: pd.DataFrame) → Self` | Training period 파라미터 학습 |
| `combine` | `(results: List[BaseForecastResult]) → SampleForecastResult` | Test period 결합. 입력은 고정 quantile levels, 출력은 sample (cross-quantile mixing 가능) |
| `extract_distributions` | `(results, h: int) → List[Distribution]` | horizon별 Distribution 추출 |

### 4.2 서브클래스 Abstract Methods

| Method | Signature | 설명 |
|--------|-----------|------|
| `_fit_horizon` | `(h, distributions: List[Distribution], observed: np.ndarray) → Any` | horizon별 파라미터 학습 |
| `_combine_distributions` | `(h, distributions: List[Distribution]) → np.ndarray (N, Q)` | 파라미터로 결합 |

### 4.3 Input/Output 규약

**fit() 입력:**
- `results`: `List[BaseForecastResult]` — M개 모형, 같은 `horizon`. `basis_index`는 달라도 됨 (내부에서 공통 index로 정렬)
- `observed`: `pd.DataFrame` shape `(N_total, H)`, index가 basis times를 포함 — 공통 index에 맞춰 자동 정렬됨

**combine() 입력:**
- `results`: `List[BaseForecastResult]` — M개 모형 (fit 때와 같은 모형 수 + 순서)

**combine() 출력:**
- `SampleForecastResult` shape `(N_common, Q, H)` — Q = n_quantiles, N_common은 모형들의 공통 basis_index 길이. cross-quantile mixing으로 원래 level 대응이 깨질 수 있으므로 sample로 취급

---

## 5. 기존 코드와의 관계

### 5.1 `PARK/Combine/` → `src/models/combining/`

기존 참조 코드: `PARK/Combine/def_combine.py`

| 기존 (`PARK/Combine/`) | 신규 (`src/models/combining/`) | 변경사항 |
|------------------------|-------------------------------|---------|
| `fuse_cdfs_horizontal()` | `BaseCombiner` (quantile averaging 기반) | 동일 quantile levels에서 ppf 추출 후 가중 평균 — `Q_H(u) = Σ w_i Q_i(u)` |
| `fuse_cdfs_vertical()` | `EqualWeightCombiner._combine_distributions()` (linear pool) | `F_V(x) = Σ w_i F_i(x)` — ppf 값의 가중 평균 |
| `fuse_pdf_angular_averaging()` | `AngularCombiner._angular_combine_core()` | θ 기반 H/V 혼합: `Q_G = (1-α)Q_H + α Q_V`, geodesic interpolation |
| `fuse_pdf_linear_pool()` | — (필요시 LinearPoolCombiner로 확장) | 단순 PDF 가중합 |
| `_interp_to_common_grid()` + `_safe_row_normalize()` | `dist.ppf(quantile_levels)` | PDF→CDF→quantile 보간 대신 Distribution 인터페이스의 ppf로 직접 추출 |
| `_pdf_to_cdf()` + `np.searchsorted()` (quantile 역변환) | `dist.ppf(quantile_levels)` | CDF 수동 역변환 대신 Distribution.ppf() 사용 |
| `tools.load_dataset()` + index intersection | `BaseCombiner._align_results()` + `BaseForecastResult.reindex()` | basis_index 교집합 추출 및 정렬 |
| `fuse_with_recent_theta_timewise_global()` (θ 최적화) | `AngularCombiner._fit_horizon()` | grid search + L-BFGS-B 최적화, horizon별 분리 |

**핵심 차이: PDF grid 기반 → Quantile function 기반**

기존 `PARK/Combine/`은 PDF를 이산 grid (`x_grid`) 위에서 직접 조작:
- PDF → CDF → quantile 역변환 → 가중 평균 → CDF → PDF (복잡한 보간 체인)
- `_interp_to_common_grid`, `_safe_row_normalize`, `_centers_to_edges` 등 수치 안정화 필요

신규 `src/models/combining/`은 Distribution 인터페이스의 `ppf()`로 직접 추출:
- `dist.ppf(quantile_levels)` → 고정 quantile levels에서 결정론적 값 추출
- 보간/정규화/grid 정렬 불필요, Distribution이 내부적으로 처리

### 5.2 Core 변경: `BaseForecastResult.reindex()`

Combining에서 futr_exog 유무에 따른 basis_index 차이를 처리하기 위해
`BaseForecastResult`에 `reindex(idx)` 메서드를 추가한다.

```python
# src/core/forecast_results.py — BaseForecastResult에 추가

def reindex(self, idx: pd.Index) -> "BaseForecastResult":
    """주어진 index에 해당하는 행만 추출한 새 인스턴스 반환.

    Args:
        idx: 추출할 basis_index의 부분집합.

    Returns:
        동일 타입의 새 ForecastResult, basis_index = idx.

    Raises:
        KeyError: idx에 self.basis_index에 없는 값이 포함된 경우.

    Example:
        >>> common_idx = result_a.basis_index.intersection(result_b.basis_index)
        >>> result_a_aligned = result_a.reindex(common_idx)
    """
    ...
```

서브클래스별 구현:

| 서브클래스 | 슬라이싱 대상 |
|-----------|-------------|
| `ParametricForecastResult` | `params[k][mask, :]` for each k |
| `QuantileForecastResult` | `quantiles_data[q][mask, :]` for each q |
| `SampleForecastResult` | `samples[mask, :, :]` |

구현 방식: `self.basis_index.get_indexer(idx)`로 positional indices를 구한 뒤
내부 배열을 fancy indexing으로 슬라이싱.

### 5.3 기존 아키텍처와의 정합성

- **ForecastResult**: `reindex()` 메서드 추가. `to_distribution(h)` 그대로 사용
- **Distribution**: 변경 없음. `sample()`, `ppf()` 그대로 사용
- **Runner**: 변경 없음. Runner가 만든 ForecastResult를 Combiner에 전달
- **BaseModel**: Combiner는 BaseModel을 상속하지 않음 (모형이 아니라 후처리기)

---

## 6. Usage Examples

### 6.1 기본 사용

```python
from src.models.combining import AngularCombiner

# 각 모형의 Runner에서 ForecastResult 획득 (이미 완료된 상태)
# arima_train:   ParametricForecastResult (N_train, H)
# ngboost_train: ParametricForecastResult (N_train, H)
# chronos_train: SampleForecastResult     (N_train, n_samples, H)

train_results = [arima_train, ngboost_train, chronos_train]
test_results  = [arima_test, ngboost_test, chronos_test]

# observed: np.ndarray (N_train, H)
combiner = AngularCombiner(n_quantiles=99)
combiner.fit(train_results, observed)

combined = combiner.combine(test_results)
# combined: SampleForecastResult (N_test, 1000, H)

# 활용
dist_h6 = combined.to_distribution(6)   # EmpiricalDistribution (T=N_test)
dist_h6.ppf([0.1, 0.5, 0.9])           # (N_test, 3)
```

### 6.2 Baseline 비교

```python
from src.models.combining import AngularCombiner, EqualWeightCombiner

# Angular combining
angular = AngularCombiner(n_quantiles=99)
angular.fit(train_results, observed)
combined_angular = angular.combine(test_results)

# Equal weight (baseline)
equal = EqualWeightCombiner(n_quantiles=99)
equal.fit(train_results, observed)  # no-op
combined_equal = equal.combine(test_results)
```

### 6.3 특정 horizon만 확인

```python
combiner = AngularCombiner(n_quantiles=99)
combiner.fit(train_results, observed)
combined = combiner.combine(test_results)

# horizon 1 (단기) vs horizon 24 (장기) 비교
for h in [1, 24]:
    dist = combined.to_distribution(h)
    print(f"Horizon {h}: mean CRPS = {compute_crps(dist, obs_h):.4f}")
    print(f"  학습된 파라미터: {combiner.fitted_params_[h]}")
```

---

## 7. Implementation Checklist

### Phase 0: Core 확장 — `BaseForecastResult.reindex()`
0. `src/core/forecast_results.py` — `BaseForecastResult`에 `reindex(idx)` 추가
   - `BaseForecastResult.reindex()`: abstract 또는 base 구현 (get_indexer + fancy indexing)
   - `ParametricForecastResult.reindex()`: params dict 슬라이싱
   - `QuantileForecastResult.reindex()`: quantiles_data dict 슬라이싱
   - `SampleForecastResult.reindex()`: samples 3D array 슬라이싱
   - 단위 테스트: `tests/core/test_forecast_results_reindex.py`

### Phase 1: BaseCombiner + EqualWeightCombiner
1. `src/models/combining/base.py` — BaseCombiner 구현
   - `_validate_results()`: horizon 검증 (basis_index 일치는 검증하지 않음)
   - `_align_results()`: 공통 basis_index 추출 + `reindex()` 호출
   - `extract_distributions()`: `to_distribution(h)` 호출
   - `fit()`: align → horizon loop + `_fit_horizon()` 호출
   - `combine()`: align → horizon loop + `_combine_distributions()` + `SampleForecastResult` 조립
2. `src/models/combining/equal_weight.py` — EqualWeightCombiner 구현
3. `src/models/combining/__init__.py` — 공개 API export
4. 단위 테스트: `tests/models/combining/test_base.py`
   - 검증 로직 테스트 (horizon 불일치, basis_index 불일치)
   - EqualWeightCombiner end-to-end 테스트 (합성 데이터)

### Phase 2: AngularCombiner
5. `src/models/combining/angular.py` — AngularCombiner 구현
   - `_distributions_to_samples()`: Distribution → sorted sample array
   - `_compute_base_losses()`: CRPS 기반 모형별 loss
   - `_angular_combine_core()`: angular combining 핵심 연산 (static method)
   - `_fit_horizon()`: grid search + L-BFGS-B
   - `_combine_distributions()`: 학습된 파라미터로 결합
6. 단위 테스트: `tests/models/combining/test_angular.py`
   - 기존 `PARK/Combine/def_combine.py`의 `fuse_pdf_angular_averaging()`와 결과 비교 검증

### Phase 3: 통합 및 정리
7. `PARK/Combine/def_combine.py` 대비 결과 일관성 검증
8. CLAUDE.md Architecture 섹션 업데이트
