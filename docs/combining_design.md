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

> `src/models/combining/`은 이미 디렉터리로 존재함. 기존 `etc/combining/angular.py`의
> 핵심 로직을 `src/models/combining/angular.py`로 리팩터링.

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
                         │    → (N_test, n_samples) │
                         │    ↓                     │
                         │  조립: SampleForecastResult
                         │    (N_test, n_samples, H)│
                         └─────────────────────────┘
```

### Shape 추적

| 단계 | Shape | 설명 |
|------|-------|------|
| `ForecastResult` | `(N, H)` | 모형별 전체 결과 |
| `to_distribution(h)` | `Distribution(T=N)` | 특정 horizon의 Distribution |
| `dist.sample(n_samples)` | `(N, n_samples)` | Distribution에서 샘플 추출 |
| `extract_distributions(h)` | `List[Distribution]` len=M | M개 모형의 같은 horizon Distribution |
| `_combine_distributions(h)` | `(N, n_samples)` | 결합된 샘플 |
| 최종 결과 | `SampleForecastResult(N, n_samples, H)` | H개 horizon 조립 |

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
        - ForecastResult 리스트의 유효성 검증 (basis_index 일치, horizon 일치)
        - horizon별 Distribution 추출 (extract_distributions)
        - fit/combine 호출 시 horizon loop 관리
        - 최종 SampleForecastResult 조립

    서브클래스 역할:
        - _fit_horizon(): 특정 horizon에서 combining 파라미터 학습
        - _combine_distributions(): 학습된 파라미터로 Distribution 결합

    Attributes:
        n_samples (int): combining 결과의 샘플 수 (default: 1000).
        fitted_params_ (Dict[int, Any]): horizon별 학습된 파라미터.

    Args:
        n_samples: 결합 결과 생성 시 사용할 샘플 수.
        seed: 재현성을 위한 random seed.

    Example:
        >>> combiner = AngularCombiner(n_samples=1000, seed=42)
        >>> combiner.fit(train_results, observed)
        >>> combined = combiner.combine(test_results)
        >>> combined.to_distribution(6).ppf([0.1, 0.5, 0.9])
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.is_fitted_ = False
        self.fitted_params_: Dict[int, Any] = {}
        self._horizon: Optional[int] = None

    # ── Validation ──

    def _validate_results(
        self, results: List[BaseForecastResult]
    ) -> None:
        """ForecastResult 리스트의 호환성 검증.

        검증 항목:
            1. 최소 2개 이상의 모형
            2. 모든 모형의 horizon이 동일
            3. 모든 모형의 basis_index가 동일

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

        base_idx = results[0].basis_index
        for i, r in enumerate(results[1:], 1):
            if not r.basis_index.equals(base_idx):
                raise ValueError(
                    f"basis_index 불일치: results[0] vs results[{i}]"
                )

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
        observed: np.ndarray,
    ) -> "BaseCombiner":
        """Training period ForecastResult들로 combining 파라미터 학습.

        각 horizon h에 대해:
            1. extract_distributions(results, h) → List[Distribution]
            2. _fit_horizon(h, dists, observed_h) → 파라미터 저장

        Args:
            results: M개 모형의 training period ForecastResult.
                     각 (N_train, H) shape.
            observed: 관측값 배열, shape (N_train, H).
                      observed[:, h-1]이 horizon h의 관측값.

        Returns:
            Self: method chaining 지원.
        """
        self._validate_results(results)

        H = results[0].horizon
        self._horizon = H

        for h in range(1, H + 1):
            dists = self.extract_distributions(results, h)
            observed_h = observed[:, h - 1]  # (N_train,)
            self.fitted_params_[h] = self._fit_horizon(h, dists, observed_h)

        self.is_fitted_ = True
        return self

    # ── Combine (Test Period) ──

    def combine(
        self,
        results: List[BaseForecastResult],
    ) -> SampleForecastResult:
        """Test period ForecastResult들을 결합하여 SampleForecastResult 반환.

        각 horizon h에 대해:
            1. extract_distributions(results, h) → List[Distribution]
            2. _combine_distributions(h, dists) → (N_test, n_samples)
        결과를 stack하여 (N_test, n_samples, H) 반환.

        Args:
            results: M개 모형의 test period ForecastResult.
                     각 (N_test, H) shape.

        Returns:
            SampleForecastResult: shape (N_test, n_samples, H).

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        if not self.is_fitted_:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")

        self._validate_results(results)

        H = results[0].horizon
        N = len(results[0].basis_index)
        basis_index = results[0].basis_index

        samples_per_h = []
        for h in range(1, H + 1):
            dists = self.extract_distributions(results, h)
            combined_h = self._combine_distributions(h, dists)
            # combined_h shape: (N, n_samples)
            samples_per_h.append(combined_h)

        # Stack: List of (N, n_samples) → (N, n_samples, H)
        samples_all = np.stack(samples_per_h, axis=2)

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
            np.ndarray: 결합된 샘플, shape (N, n_samples).
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
        n_samples: combining 결과 샘플 수 (default: 1000).
        seed: random seed.
        config: 최적화 설정. None이면 기본값 사용.

    Example:
        >>> combiner = AngularCombiner(n_samples=1000)
        >>> combiner.fit(train_results, observed)
        >>> combined = combiner.combine(test_results)
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seed: Optional[int] = None,
        config: Optional[AngularCombineConfig] = None,
    ):
        super().__init__(n_samples=n_samples, seed=seed)
        self.config = config or AngularCombineConfig()
        self.base_losses_: Dict[int, np.ndarray] = {}

    # ── Distribution → Sample 변환 (내부 헬퍼) ──

    def _distributions_to_samples(
        self,
        distributions: List[Distribution],
    ) -> jnp.ndarray:
        """M개 Distribution에서 샘플을 추출하여 (N, M, n_samples) 배열로 변환.

        각 Distribution에서 self.n_samples만큼 샘플을 뽑고, 정렬 후 스택.

        Args:
            distributions: M개 Distribution, 각 T=N.

        Returns:
            jnp.ndarray: shape (N, M, n_samples), sorted along sample axis.
        """
        samples_list = []
        for dist in distributions:
            s = dist.sample(self.n_samples, seed=self.seed)  # (N, n_samples)
            s = np.sort(s, axis=1)
            samples_list.append(s)

        # Stack: (M, N, n_samples) → transpose → (N, M, n_samples)
        stacked = np.stack(samples_list, axis=0)  # (M, N, n_samples)
        stacked = stacked.transpose(1, 0, 2)      # (N, M, n_samples)
        return jnp.array(stacked)

    # ── Base CRPS Loss 계산 ──

    def _compute_base_losses(
        self,
        distributions: List[Distribution],
        observed: np.ndarray,
    ) -> np.ndarray:
        """각 모형의 CRPS loss를 계산.

        Args:
            distributions: M개 Distribution.
            observed: 관측값, shape (N,).

        Returns:
            np.ndarray: shape (M,), 모형별 평균 CRPS.
        """
        losses = []
        for dist in distributions:
            samples = dist.sample(self.n_samples, seed=self.seed)  # (N, n_samples)
            crps_vals = jax.vmap(
                crps_numerical_j, in_axes=(0, 0, None)
            )(jnp.array(observed), jnp.array(samples), True)
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
            dataset: (N, M, n_samples), sorted along sample axis.
            weight: (M,), combining 가중치.
            beta: 가중치 파라미터 (이 함수에서는 사용하지 않음, weight에 이미 반영).
            degree: angular combining 강도.
            key: JAX random key.

        Returns:
            jnp.ndarray: (N, n_samples), 결합된 샘플.
        """
        N, M, n_samples = dataset.shape
        quantile_values = jnp.linspace(0, 1, n_samples)

        if degree < 1e-3:
            # 단순 가중 평균
            # dataset: (N, M, n_samples) → transpose → (N, n_samples, M)
            res = jnp.sum(
                dataset.transpose(0, 2, 1) * weight, axis=2
            )  # (N, n_samples)
        else:
            # Angular combining
            indices = jax.random.choice(
                key=key,
                a=M,
                shape=(N, n_samples),
                replace=True,
                p=weight,
            )

            s0 = (dataset + quantile_values / degree).transpose(0, 2, 1)
            s = jnp.take_along_axis(s0, indices[..., None], axis=2).squeeze(-1)
            s_sorted = jnp.sort(s, axis=1)
            res = s_sorted - jnp.arange(n_samples) / (n_samples * degree)

        return res  # (N, n_samples)

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
        # Distribution → sample dataset
        dataset = self._distributions_to_samples(distributions)  # (N, M, n_samples)
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
            np.ndarray: (N, n_samples).
        """
        params = self.fitted_params_[h]  # [beta, degree]
        beta, degree = params[0], params[1]

        dataset = self._distributions_to_samples(distributions)
        base_losses = jnp.array(self.base_losses_[h])

        w = 1.0 / (base_losses ** beta)
        weight = w / jnp.sum(w)

        key = jax.random.key(self.config.RANDOM_SEED)
        combined = self._angular_combine_core(
            dataset, weight, beta, degree, key,
        )
        return np.array(combined)  # (N, n_samples)
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
        >>> combiner = EqualWeightCombiner(n_samples=1000)
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
        """동일 가중 샘플 평균.

        각 Distribution에서 n_samples만큼 뽑고 평균.

        Returns:
            np.ndarray: (N, n_samples).
        """
        M = len(distributions)
        samples_sum = np.zeros(
            (len(distributions[0]), self.n_samples),
            dtype=float,
        )

        for dist in distributions:
            s = dist.sample(self.n_samples, seed=self.seed)  # (N, n_samples)
            s = np.sort(s, axis=1)
            samples_sum += s

        return samples_sum / M  # (N, n_samples)
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
| `__init__` | `(n_samples=1000, seed=None)` | 공통 초기화 |
| `fit` | `(results: List[BaseForecastResult], observed: np.ndarray) → Self` | Training period 파라미터 학습 |
| `combine` | `(results: List[BaseForecastResult]) → SampleForecastResult` | Test period 결합 |
| `extract_distributions` | `(results, h: int) → List[Distribution]` | horizon별 Distribution 추출 |

### 4.2 서브클래스 Abstract Methods

| Method | Signature | 설명 |
|--------|-----------|------|
| `_fit_horizon` | `(h, distributions: List[Distribution], observed: np.ndarray) → Any` | horizon별 파라미터 학습 |
| `_combine_distributions` | `(h, distributions: List[Distribution]) → np.ndarray (N, n_samples)` | 파라미터로 결합 |

### 4.3 Input/Output 규약

**fit() 입력:**
- `results`: `List[BaseForecastResult]` — M개 모형, 모두 같은 `basis_index`와 `horizon`
- `observed`: `np.ndarray` shape `(N, H)` — `observed[:, h-1]`이 horizon h의 관측값

**combine() 입력:**
- `results`: `List[BaseForecastResult]` — M개 모형 (fit 때와 같은 모형 수 + 순서)

**combine() 출력:**
- `SampleForecastResult` shape `(N_test, n_samples, H)`

---

## 5. 기존 코드와의 관계

### 5.1 `etc/combining/angular.py` → `src/models/combining/angular.py`

| 기존 (`etc/`) | 신규 (`src/`) | 변경사항 |
|--------------|--------------|---------|
| `Combiner.__init__(simulated_forecast: List[DataFrame], base_forecast_loss)` | `AngularCombiner.__init__(n_samples, seed, config)` | DataFrame 대신 Distribution에서 샘플 추출 |
| `Combiner._stack_and_sort_fc()` | `_distributions_to_samples(distributions)` | Distribution.sample() 사용 |
| `Combiner._calculate_weight_from_loss(beta)` | `_angular_combine_core()` 내부 | 동일 로직 유지 |
| `Combiner.combine(combine_parameter)` | `_angular_combine_core(dataset, weight, beta, degree, key)` | 순수 함수로 분리 |
| `AngularCombine.fit()` | `_fit_horizon(h, dists, observed)` | horizon별 분리, Distribution 인터페이스 |
| `AngularCombine._align_obs_fc()` | `BaseCombiner._validate_results()` | basis_index 검증으로 대체 |

### 5.2 기존 아키텍처와의 정합성

- **ForecastResult**: 변경 없음. `to_distribution(h)` 그대로 사용
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
combiner = AngularCombiner(n_samples=1000, seed=42)
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
angular = AngularCombiner(n_samples=1000, seed=42)
angular.fit(train_results, observed)
combined_angular = angular.combine(test_results)

# Equal weight (baseline)
equal = EqualWeightCombiner(n_samples=1000, seed=42)
equal.fit(train_results, observed)  # no-op
combined_equal = equal.combine(test_results)
```

### 6.3 특정 horizon만 확인

```python
combiner = AngularCombiner(n_samples=1000)
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

### Phase 1: BaseCombiner + EqualWeightCombiner
1. `src/models/combining/base.py` — BaseCombiner 구현
   - `_validate_results()`: basis_index, horizon 검증
   - `extract_distributions()`: `to_distribution(h)` 호출
   - `fit()`: horizon loop + `_fit_horizon()` 호출
   - `combine()`: horizon loop + `_combine_distributions()` + `SampleForecastResult` 조립
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
   - 기존 `etc/combining/angular.py`와 결과 비교 검증

### Phase 3: 통합 및 정리
7. `etc/combining/angular.py`에 deprecation 주석 추가
8. CLAUDE.md Architecture 섹션 업데이트
