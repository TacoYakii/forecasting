# TODO

## Done

### ForecastResult / Distribution 역할 분리
- 설계 문서: `docs/forecast_result_distribution_design.md`
- [x] Distribution에 `interval()` 메서드 추가
- [x] ForecastResult에서 `mean()`, `std()`, `quantile()`, `interval()` 제거
- [x] ForecastResult `to_dataframe()` 내부 구현을 `to_distribution(h)` 경유로 변경
- [x] 테스트 코드 일괄 수정
- [x] Docstring example 업데이트

## Planned

### 하드코딩된 절대경로 제거 (`src/data/download/`)
- `ECMWF.py`: `metadata_root`, `sv_dir`
- `KMA.py`: `__main__` 블록 내 경로들
- `elevation.py`: 모듈 레벨 + `__main__` 블록 경로 (옛 프로젝트 경로 `projects/windpower_forecasting/meta/`)
- [x] 프로젝트 루트 기준 상대경로로 변환

### Data Pipeline Refactoring (`src/data/`)
- Download (KMA, ECMWF), NWP preprocessing (GRIB2/TXT readers, validators, derived variables), Training data builder
- [ ] Design document
- [ ] Implementation

### Hierarchical Forecasting Refactoring (`src/pipelines/`)
- `BaseForecastRunner` -> `HierarchyForecastCoordinator` -> `HierarchyForecastSampler`, S matrix
- [ ] Design document
- [ ] Implementation

### Evaluation Refactoring (`src/utils/`)
- CRPS, Loss Registry (RandomCRPSLoss, QuantileCRPSLoss, PinballLoss)
- [x] 설계 문서: `docs/crps_refactoring_design.md`
- [x] 구현: `pinball_loss`, `crps_quantile`, `crps` dispatch → `src/utils/metrics/crps.py`
- [x] combining 중복 제거, Codex 리뷰 3회 반복 완료

### CKD 리팩토링
- `src/models/conditional_kernel_density/` 모듈 재설계
- [ ] 설계 문서 작성
- [ ] 구현

### Runner model_name 설정
- Runner에서 model_name을 설정 가능하게 변경
- [ ] 구현

### Combining 모듈 설계
- 다중 모형 예측 결합 (forecast combination) 모듈
- [ ] 설계 문서 작성
- [ ] 구현

### Combining 객체 save/load
- Combining 모듈의 학습된 가중치 등 직렬화/역직렬화 지원
- [ ] 구현

### Quantile Crossing Repair (`src/core/forecast_results.py`)
- **원인**: MQLoss/IQLoss 기반 모델(DeepAR, TFT 등)이 각 quantile level을 독립적으로 예측하므로 Q(τ_i) > Q(τ_j) (τ_i < τ_j) 발생 가능. Parametric(`ppf`는 항상 monotone)이나 Sample(`np.percentile`도 항상 monotone)에서는 불가능하며, `QuantileForecastResult`에서만 발생하는 문제.
- **영향**: Crossing된 quantiles는 유효한 CDF를 형성하지 못함. 평가(CRPS, PIT), 시각화, VerticalCombiner의 CDF 보간 등 downstream 전체에 영향.
- **해결**: Isotonic regression (PAVA 알고리즘) — 단조 증가 제약 하에서 원본과의 L2 거리를 최소화하는 최적 projection. crossing이 있는 구간만 가중 평균으로 평탄화하고, 이미 monotone한 값은 보존. `sklearn.isotonic.IsotonicRegression`으로 O(Q) 구현.
- **위치**: Combiner가 아닌 `QuantileForecastResult` 생성 시점에서 처리 (downstream 전체 커버)
- [ ] 구현

### Dead Code 삭제 (완료)
- [x] `src/utils/evaluate_module.py` — `src/utils/metrics/`가 대체
- [x] `src/utils/metrics_jax/` — production 미사용, Numba가 대체
- [x] `src/utils/visualization/empirical_distribution.py` — 빈 파일

---

## Utils 재설계

> 아래 항목들은 `src/utils/` 내 미사용/레거시 모듈을 현 시스템에 맞게 재설계하는 작업.

### 1. simulate.py → 삭제
- `ParametricDistribution.sample()` (`src/core/forecast_distribution.py`)이 이미 동일 기능 제공
- `simulate.py`의 개별 분포 함수(normal, laplace 등)는 `ParametricDistribution`이 대체
- 파라미터 clamping은 `DISTRIBUTION_REGISTRY`의 `clamp` 필드가 담당
- [ ] 삭제 (+ `etc/` 레거시 참조 확인)

### 2. Transformation 모듈 재설계 (`src/utils/transformation.py`)
- 현재: sklearn `PowerTransformer` 래퍼 (Box-Cox/Yeo-Johnson), DataFrame 구조 보존
- 재설계 방향: 시계열 전처리 파이프라인에서 사용 가능한 형태로 통합
  - 학습 데이터에 fit → 예측/평가 시 transform/inverse_transform
  - ForecastResult의 값을 역변환하는 인터페이스 필요
  - Runner 또는 Trainer에서 전처리 단계로 삽입 가능한 구조
- [ ] 설계 문서
- [ ] 구현

### 3. PIT 재설계 (`src/utils/metrics/pit.py` + `src/utils/visualization/pit.py`)
- 현재: `pit_get_values(forecast_samples, observations)` — raw samples (N, M) 입력만 지원
- 재설계 방향:
  - ForecastResult 객체를 직접 받아서 horizon별 PIT 계산
    - Parametric → CDF(y; params)로 직접 계산
    - Sample → 경험적 CDF로 계산 (현재 방식)
    - Quantile → 보간으로 CDF 근사
  - horizon별 PIT histogram + KS test 시각화 (subplot per horizon)
  - KS test: PIT 값의 경험적 CDF vs U(0,1) 최대 차이 검정
    - p > α → 예측 분포가 잘 calibrate됨
    - p ≤ α → over/under-dispersed
- [ ] 설계 문서
- [ ] 구현

### 4. Hyperparameter Optimizer 재설계 (`src/utils/hyperparameter_optimizer/`)
- 현재: Optuna 기반 `ModelOptimizer` + dataclass params (NGBoost, CatBoost, XGBoost)
  - `set_dataset.py`에 의존 (dataset 분할) — `set_dataset.py`는 여기서만 사용됨
  - 현 시스템의 Runner/Trainer/LossRegistry와 연결되지 않음
- 재설계 방향:
  - Runner/Trainer 인터페이스와 통합 (모형별 HP 공간 정의 → Optuna study 실행)
  - CKD는 이미 별도 `CKDOptunaTrainer` 존재 — 패턴 통일 필요
  - `set_dataset.py`는 HP optimizer와 함께 삭제/통합 (Runner가 데이터 분할 담당)
- [ ] 설계 문서
- [ ] 구현

### 5. 시각화 모듈 재설계 (`src/utils/visualization/`)
- 현재 파일: `pit.py`, `plot_average_performance.py`, `plot_hierarchy.py`
- 재설계 방향:
  - **PIT 시각화**: 위 #3에서 함께 처리
  - **Average Performance**: ForecastResult + `crps()` dispatch 결과를 입력으로 모형 간 성능 비교 scatter plot
  - **Hierarchy**: Reconciliation 리팩토링(`src/pipelines/`) 시 함께 재설계
    - S matrix 기반 aggregation level별 비교 시각화
    - 현 코드의 helper 함수 (`get_aggregation_metadata`, `get_dynamic_horizons`) 활용
- [ ] PIT 시각화 (#3과 함께)
- [ ] Average Performance 재설계
- [ ] Hierarchy 시각화 (Reconciliation 리팩토링과 함께)
