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
├── equal_weight.py      # EqualWeightCombiner (동일 가중 baseline)
├── horizontal.py        # HorizontalCombiner (Quantile Averaging, quantile 가중 평균)
└── vertical.py          # VerticalCombiner (Linear Pool, CDF 가중 평균)
```

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
from joblib import Parallel, delayed

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
        n_quantiles (int): combining에 사용할 quantile level 수 (default: 99).
        quantile_levels (np.ndarray): 고정 quantile levels, shape (n_quantiles,).
            (0, 1) 구간을 균등 분할. 모든 모형에서 동일한 level로 추출.
        weights_ (Dict[int, Dict[str, float]]): horizon별 모형별 가중치.
            fit() 완료 시 초기화. key는 horizon (1-indexed),
            value는 {model_name: weight} dict.
            예: {1: {'ARIMA': 0.4, 'DeepAR': 0.35, 'TFT': 0.25}, ...}

    Args:
        n_quantiles: quantile level 수. 클수록 분포 근사가 정밀해짐.
        n_jobs: fit() 시 horizon별 병렬 worker 수 (joblib).
            1이면 순차 실행, -1이면 모든 CPU 코어 사용.

    Example:
        >>> combiner = HorizontalCombiner(n_quantiles=99, n_jobs=4)
        >>> combiner.fit(train_results, observed)
        >>> combiner.weights_[1]  # {'ARIMA': 0.4, 'DeepAR': 0.35, 'TFT': 0.25}
        >>> combined = combiner.combine(test_results)
        >>> combined.to_distribution(6).ppf([0.1, 0.5, 0.9])
    """

    def __init__(
        self,
        n_quantiles: int = 99,
        n_jobs: int = 1,
    ):
        self.n_quantiles = n_quantiles
        self.n_jobs = n_jobs
        self.quantile_levels = np.linspace(0, 1, n_quantiles + 2)[1:-1]  # (0, 1) 개구간
        self.is_fitted_ = False
        self.weights_: Dict[int, Dict[str, float]] = {}
        self._horizon: Optional[int] = None
        self._model_names: Optional[List[str]] = None

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
        """Training period ForecastResult들로 combining 가중치 학습.

        각 horizon h에 대해:
            1. _align_results(results) → 공통 index로 정렬
            2. extract_distributions(results, h) → List[Distribution]
            3. _fit_horizon(h, dists, observed_h) → 가중치 (M,) 반환
            4. weights_[h] = {model_name: weight} dict로 저장

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
        self._model_names = [r.model_name for r in results]
        common_idx = results[0].basis_index

        # observed도 공통 index에 맞춰 정렬
        observed_aligned = observed.loc[common_idx].values  # (N_common, H)

        def _fit_single(h):
            dists = self.extract_distributions(results, h)
            observed_h = observed_aligned[:, h - 1]  # (N_common,)
            return h, self._fit_horizon(h, dists, observed_h)  # (M,)

        fit_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single)(h) for h in range(1, H + 1)
        )

        for h, weight_array in fit_results:
            self.weights_[h] = dict(zip(self._model_names, weight_array))

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
    ) -> np.ndarray:
        """특정 horizon에서 모형별 가중치 학습.

        Args:
            h: forecast horizon (1-indexed).
            distributions: M개 모형의 Distribution, 각 T=N_train.
            observed: 해당 horizon의 관측값, shape (N_train,).

        Returns:
            np.ndarray: 모형별 가중치, shape (M,). 합이 1.
                Base의 fit()에서 {model_name: weight} dict로 변환하여
                self.weights_[h]에 저장.
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

### 3.2 HorizontalCombiner (Quantile Averaging)

```python
# src/models/combining/horizontal.py

from typing import Any, List, Optional

import numpy as np

from .base import BaseCombiner, Distribution


class HorizontalCombiner(BaseCombiner):
    """Quantile function 가중 평균 (Quantile Averaging).

    Q_combined(u) = Σ w_i Q_i(u)

    같은 quantile level u에서 각 모형의 quantile 값을 가중 평균.
    가중치는 fit() 시 CRPS 기반으로 학습하거나, 사용자 지정 가능.

    Attributes:
        weights_ (Dict[int, np.ndarray]): horizon별 학습된 가중치, shape (M,).

    Args:
        n_quantiles: combining 결과 quantile 수 (default: 99).
        n_jobs: horizon별 병렬 worker 수 (default: 1).
        weights: 사용자 지정 가중치. None이면 CRPS 기반 학습.

    Example:
        >>> combiner = HorizontalCombiner(n_quantiles=99, n_jobs=-1)
        >>> combiner.fit(train_results, observed)
        >>> combined = combiner.combine(test_results)
    """

    def __init__(
        self,
        n_quantiles: int = 99,
        n_jobs: int = 1,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(n_quantiles=n_quantiles, n_jobs=n_jobs)
        self._user_weights = weights

    def _fit_horizon(
        self,
        h: int,
        distributions: List[Distribution],
        observed: np.ndarray,
    ) -> np.ndarray:
        """CRPS 기반 가중치 학습.

        w_i = 1 / CRPS_i, 정규화하여 합이 1이 되도록.
        사용자 지정 가중치가 있으면 그대로 사용.

        Returns:
            np.ndarray: 가중치, shape (M,).
        """
        if self._user_weights is not None:
            return self._user_weights

        # TODO: CRPS 기반 가중치 학습 구현
        ...

    def _combine_distributions(
        self,
        h: int,
        distributions: List[Distribution],
    ) -> np.ndarray:
        """Quantile function 가중 평균.

        Q_combined(u) = Σ w_i Q_i(u)
        같은 quantile level에서 가중 평균이므로 cross-quantile mixing 없음.

        Returns:
            np.ndarray: (N, Q).
        """
        weights = np.array(list(self.weights_[h].values()))  # (M,)

        q_sum = np.zeros(
            (len(distributions[0]), self.n_quantiles),
            dtype=float,
        )
        for w, dist in zip(weights, distributions):
            q = dist.ppf(self.quantile_levels)  # (N, Q)
            q_sum += w * q

        return q_sum  # (N, Q)
```

### 3.3 VerticalCombiner (Linear Pool)

```python
# src/models/combining/vertical.py

from typing import Any, List, Optional

import numpy as np

from .base import BaseCombiner, Distribution


class VerticalCombiner(BaseCombiner):
    """CDF 가중 평균 (Linear Pool).

    F_combined(x) = Σ w_i F_i(x)

    같은 x 값에서 각 모형의 CDF를 가중 평균.
    Horizontal과 달리 결합 분포가 원래 분포들보다 넓어지는 특성이 있어
    calibration이 부족한 경우에 유리.

    구현 방식:
        각 Distribution에서 quantile values를 추출한 뒤,
        공통 x grid 위에서 CDF를 구성하고 가중 평균 → 역변환.

    Attributes:
        weights_ (Dict[int, np.ndarray]): horizon별 학습된 가중치, shape (M,).

    Args:
        n_quantiles: combining 결과 quantile 수 (default: 99).
        n_jobs: horizon별 병렬 worker 수 (default: 1).
        weights: 사용자 지정 가중치. None이면 CRPS 기반 학습.

    Example:
        >>> combiner = VerticalCombiner(n_quantiles=99, n_jobs=-1)
        >>> combiner.fit(train_results, observed)
        >>> combined = combiner.combine(test_results)
    """

    def __init__(
        self,
        n_quantiles: int = 99,
        n_jobs: int = 1,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(n_quantiles=n_quantiles, n_jobs=n_jobs)
        self._user_weights = weights

    def _fit_horizon(
        self,
        h: int,
        distributions: List[Distribution],
        observed: np.ndarray,
    ) -> np.ndarray:
        """CRPS 기반 가중치 학습.

        Returns:
            np.ndarray: 가중치, shape (M,).
        """
        if self._user_weights is not None:
            return self._user_weights

        # TODO: CRPS 기반 가중치 학습 구현
        ...

    def _combine_distributions(
        self,
        h: int,
        distributions: List[Distribution],
    ) -> np.ndarray:
        """CDF 가중 평균 후 quantile 역변환.

        1. 각 Distribution에서 quantile values 추출 → 공통 x grid 구성
        2. 공통 x grid에서 각 Distribution의 CDF 값 계산
        3. CDF 가중 평균: F_combined(x) = Σ w_i F_i(x)
        4. 결합 CDF에서 quantile 역변환

        Returns:
            np.ndarray: (N, Q).
        """
        weights = self.fitted_params_[h]  # (M,)

        # TODO: CDF 가중 평균 + quantile 역변환 구현
        ...
```

### 3.4 EqualWeightCombiner

```python
# src/models/combining/equal_weight.py

from typing import Any, List, Optional

import numpy as np

from .base import BaseCombiner, Distribution


class EqualWeightCombiner(BaseCombiner):
    """단순 동일 가중 평균 combiner (baseline).

    모든 모형에 1/M 동일 가중치 부여. 학습 파라미터 없음.

    Example:
        >>> combiner = EqualWeightCombiner(n_quantiles=99)
        >>> combiner.fit(train_results, observed)
        >>> combiner.weights_[1]  # {'ARIMA': 0.333, 'DeepAR': 0.333, 'TFT': 0.333}
        >>> combined = combiner.combine(test_results)
    """

    def _fit_horizon(
        self,
        h: int,
        distributions: List[Distribution],
        observed: np.ndarray,
    ) -> np.ndarray:
        """동일 가중치 1/M 반환."""
        M = len(distributions)
        return np.full(M, 1.0 / M)

    def _combine_distributions(
        self,
        h: int,
        distributions: List[Distribution],
    ) -> np.ndarray:
        """동일 가중 quantile 평균.

        각 Distribution에서 동일한 quantile levels로 값을 추출하고
        weights_[h]의 가중치로 평균.

        Returns:
            np.ndarray: (N, Q).
        """
        weights = np.array(list(self.weights_[h].values()))  # (M,)

        q_sum = np.zeros(
            (len(distributions[0]), self.n_quantiles),
            dtype=float,
        )

        for w, dist in zip(weights, distributions):
            q = dist.ppf(self.quantile_levels)  # (N, Q)
            q_sum += w * q

        return q_sum  # (N, Q)
```

### 3.5 `__init__.py`

```python
# src/models/combining/__init__.py

from .base import BaseCombiner
from .equal_weight import EqualWeightCombiner
from .horizontal import HorizontalCombiner
from .vertical import VerticalCombiner

__all__ = [
    "BaseCombiner",
    "EqualWeightCombiner",
    "HorizontalCombiner",
    "VerticalCombiner",
]
```

---

## 4. Interface Specification

### 4.1 BaseCombiner Public API

| Method | Signature | 설명 |
|--------|-----------|------|
| `__init__` | `(n_quantiles=99, n_jobs=1)` | 공통 초기화. n_jobs: horizon별 병렬 worker 수 |
| `fit` | `(results: List[BaseForecastResult], observed: pd.DataFrame) → Self` | Training period 파라미터 학습 |
| `combine` | `(results: List[BaseForecastResult]) → SampleForecastResult` | Test period 결합. 고정 quantile levels에서 추출 후 결합 |
| `extract_distributions` | `(results, h: int) → List[Distribution]` | horizon별 Distribution 추출 |

### 4.2 서브클래스 Abstract Methods

| Method | Signature | 설명 |
|--------|-----------|------|
| `_fit_horizon` | `(h, distributions: List[Distribution], observed: np.ndarray) → np.ndarray (M,)` | horizon별 가중치 학습. Base가 weights_[h]에 {model_name: weight} dict로 저장 |
| `_combine_distributions` | `(h, distributions: List[Distribution]) → np.ndarray (N, Q)` | 파라미터로 결합 |

### 4.3 Input/Output 규약

**fit() 입력:**
- `results`: `List[BaseForecastResult]` — M개 모형, 같은 `horizon`. `basis_index`는 달라도 됨 (내부에서 공통 index로 정렬)
- `observed`: `pd.DataFrame` shape `(N_total, H)`, index가 basis times를 포함 — 공통 index에 맞춰 자동 정렬됨

**combine() 입력:**
- `results`: `List[BaseForecastResult]` — M개 모형 (fit 때와 같은 모형 수 + 순서)

**combine() 출력:**
- `SampleForecastResult` shape `(N_common, Q, H)` — Q = n_quantiles, N_common은 모형들의 공통 basis_index 길이

---

## 5. Core 변경: `BaseForecastResult` 확장

### 5.1 `model_name` 속성 추가

Combiner가 가중치를 모형 이름으로 저장/조회할 수 있도록
`BaseForecastResult.__init__`에 `model_name: str` 파라미터를 추가한다.
Runner가 ForecastResult 생성 시 모형 클래스 이름을 자동으로 설정.

### 5.2 `reindex()` 메서드 추가

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

### 기존 아키텍처와의 정합성

- **ForecastResult**: `model_name` 속성 + `reindex()` 메서드 추가. `to_distribution(h)` 그대로 사용
- **Distribution**: 변경 없음. `sample()`, `ppf()` 그대로 사용
- **Runner**: ForecastResult 생성 시 `model_name` 설정
- **BaseModel**: Combiner는 BaseModel을 상속하지 않음 (모형이 아니라 후처리기)

---

## 6. Usage Examples

### 6.1 기본 사용

```python
from src.models.combining import HorizontalCombiner

# 각 모형의 Runner에서 ForecastResult 획득 (이미 완료된 상태)
# arima_train:   ParametricForecastResult (N_train, H)
# ngboost_train: ParametricForecastResult (N_train, H)
# chronos_train: SampleForecastResult     (N_train, n_samples, H)

train_results = [arima_train, ngboost_train, chronos_train]
test_results  = [arima_test, ngboost_test, chronos_test]

# observed: pd.DataFrame (N_total, H)
combiner = HorizontalCombiner(n_quantiles=99)
combiner.fit(train_results, observed)

# 학습된 가중치 확인
combiner.weights_[1]
# {'ARIMA': 0.45, 'DeepAR': 0.30, 'Chronos': 0.25}

combined = combiner.combine(test_results)
# combined: SampleForecastResult (N_test, 99, H)

# 활용
dist_h6 = combined.to_distribution(6)   # EmpiricalDistribution (T=N_test)
dist_h6.ppf([0.1, 0.5, 0.9])           # (N_test, 3)
```

### 6.2 방법론 비교

```python
from src.models.combining import (
    EqualWeightCombiner,
    HorizontalCombiner,
    VerticalCombiner,
)

combiners = {
    "equal_weight": EqualWeightCombiner(n_quantiles=99),
    "horizontal": HorizontalCombiner(n_quantiles=99),
    "vertical": VerticalCombiner(n_quantiles=99),
}

for name, combiner in combiners.items():
    combiner.fit(train_results, observed)
    combined = combiner.combine(test_results)
    print(f"{name}: combined shape = {combined.samples.shape}")
```

---

## 7. Implementation Checklist

### Phase 0: Core 확장 — `BaseForecastResult` ✅
0. `src/core/forecast_results.py`
   - [x] `BaseForecastResult.__init__`에 `model_name: str` 파라미터 추가
   - [x] `BaseForecastResult.reindex(idx)` 추가 (get_indexer + fancy indexing)
   - [x] `ParametricForecastResult.reindex()`: params dict 슬라이싱
   - [x] `QuantileForecastResult.reindex()`: quantiles_data dict 슬라이싱
   - [x] `SampleForecastResult.reindex()`: samples 3D array 슬라이싱
   - [x] Runner에서 ForecastResult 생성 시 `model_name` 설정
   - [x] 단위 테스트: `tests/core/test_forecast_results_reindex.py`

### Phase 1: BaseCombiner + EqualWeightCombiner ✅
1. `src/models/combining/base.py` — BaseCombiner 구현
   - [x] `_validate_results()`: horizon 검증 (basis_index 일치는 검증하지 않음)
   - [x] `_align_results()`: 공통 basis_index 추출 + `reindex()` 호출
   - [x] `_deduplicate_names()`: 중복 model_name에 `_0`, `_1` suffix 부여
   - [x] `extract_distributions()`: `to_distribution(h)` 호출
   - [x] `fit()`: align → horizon loop + `_fit_horizon()` 호출
   - [x] `combine()`: align → horizon loop + `_combine_distributions()` + `SampleForecastResult` 조립
2. [x] `src/models/combining/equal_weight.py` — EqualWeightCombiner 구현
3. [x] `src/models/combining/__init__.py` — 공개 API export
4. [x] 단위 테스트: `tests/models/combining/test_base.py`
   - 검증 로직 테스트 (horizon 불일치, basis_index 불일치)
   - EqualWeightCombiner end-to-end 테스트 (합성 데이터)
   - 중복 model_name 처리 테스트

### Phase 2: HorizontalCombiner + VerticalCombiner ✅
5. [x] `src/models/combining/horizontal.py` — HorizontalCombiner 구현
   - `_fit_horizon()`: quantile-based CRPS → inverse-CRPS 가중치 학습
   - `_combine_distributions()`: quantile function 가중 평균
6. [x] `src/models/combining/vertical.py` — VerticalCombiner 구현
   - `_fit_horizon()`: quantile-based CRPS → inverse-CRPS 가중치 학습
   - `_combine_distributions()`: CDF 가중 평균 + np.interp 기반 quantile 역변환
7. [x] 단위 테스트: `tests/models/combining/test_horizontal.py`, `test_vertical.py`
