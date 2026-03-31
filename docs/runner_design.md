# Runner 설계

## 설계 원칙

| 객체 | 역할 | 대상 모형 |
|------|------|-----------|
| **RollingRunner** | 시계열 rolling 평가 오케스트레이터 | Statistical, Deep, Foundation |
| **PerHorizonRunner** | Horizon별 독립 모형 학습·예측 오케스트레이터 | ML (XGBoost, CatBoost, NGBoost 등) |

Runner는 **모형 외부**에서 평가를 조율한다. 모형은 fit/forecast만 담당하고, rolling 루프·결과 집계·외생변수 슬라이싱은 모두 Runner가 처리한다.

```
Runner
  ├─ RollingRunner (N회 반복, axis=0 concat)
  │     ├─ StatefulPredictor  → forecast() → update_state()
  │     └─ ContextPredictor   → predict_from_context()
  │
  └─ PerHorizonRunner (H개 독립 모형, axis=1 stack)
        └─ BaseForecaster     → fit() → forecast()
```

---

## Protocol — 모형 인터페이스

Runner는 구체 클래스가 아닌 **Protocol**에 의존한다. 모형은 해당 Protocol을 만족하기만 하면 Runner에 투입할 수 있다.

### StatefulPredictor

내부 상태를 갱신하며 예측하는 모형 (ARIMA-GARCH 계열).

| 메서드/속성 | 시그니처 | 설명 |
|-------------|---------|------|
| `is_fitted_` | `bool` | 학습 완료 여부 |
| `forecast()` | `(horizon, x_future?) → (mu, sigma)` | 현재 상태에서 H-step 예측 |
| `update_state()` | `(y_new, x_new?) → None` | 실측값 1개로 상태 갱신 |

### ContextPredictor

Context window로부터 예측하는 모형 (Deep, Foundation). 내부 상태 변경 없음.

| 메서드/속성 | 시그니처 | 설명 |
|-------------|---------|------|
| `is_fitted_` | `bool` | 학습 완료 여부 |
| `predict_from_context()` | `(context_y, horizon, **kwargs) → ForecastResult` | Context window 기반 예측 |

`predict_from_context()`에 전달되는 kwargs:

| 키 | 설명 |
|----|------|
| `context_index` | Context 구간의 시간 인덱스 |
| `context_X` | 과거 외생변수 (futr + hist 또는 exog) |
| `future_X` | 미래 외생변수 (futr만, hist 제외) |
| `future_index` | 예측 구간의 시간 인덱스 |

---

## RollingRunner

### 파라미터

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `model` | `StatefulPredictor \| ContextPredictor` | 학습 완료된 모형 |
| `dataset` | `pd.DataFrame` | 전체 DataFrame (train + test), 시간 인덱스 정렬 |
| `y_col` | `str` | 타겟 컬럼명 |
| `forecast_period` | `(start, end)` | 평가 구간 |
| `exog_cols` | `List[str]?` | Statistical/Foundation 모형용 외생변수 |
| `futr_cols` | `List[str]?` | Deep 모형 미래 외생변수 (NWP 등) |
| `hist_cols` | `List[str]?` | Deep 모형 과거 전용 외생변수 (SCADA 등) |

### `run()` 메서드

```python
def run(
    horizon: int,
    method: str = "forecast",        # StatefulPredictor 전용
    method_kwargs: dict | None = None,
    dist_name: str = "normal",
    show_progress: bool = True,
) -> ParametricForecastResult | SampleForecastResult | QuantileForecastResult
```

| 인자 | 설명 |
|------|------|
| `horizon` | 각 basis time에서의 예측 수평 (H) |
| `method` | StatefulPredictor의 호출 메서드명 (`"forecast"`, `"simulate_paths"` 등) |
| `method_kwargs` | 모형 메서드에 전달할 추가 kwargs (`n_paths`, `seed` 등) |
| `dist_name` | 출력 분포 이름 |
| `show_progress` | tqdm 진행률 표시 여부 |

### Rolling 전략

#### StatefulPredictor 경로

```
for t in forecast_period:
    1. x_future ← exog_cols[t:t+H] 슬라이싱 (부족 시 zero-pad)
    2. output ← model.{method}(horizon, x_future, **kwargs)
    3. 결과 기록
    4. model.update_state(y_actual[t], x_current[t])

→ 결과 concat (axis=0) → ForecastResult (N, H)
```

- `seed`가 `method_kwargs`에 있으면 매 step마다 `base_seed + t`로 증가시켜 재현성 보장

#### ContextPredictor 경로

```
for t in forecast_period:
    1. context_data ← dataset[:current_time] (현재 시점 미포함)
    2. context_y, context_index 추출
    3. 외생변수 분기:
       - Deep (futr_cols/hist_cols):
           context_X ← futr + hist (과거 전체)
           future_X  ← futr만 (hist 제외 → leakage 방지)
       - Foundation (exog_cols):
           context_X ← exog_cols (과거만)
           future_X  ← None
    4. output ← model.predict_from_context(context_y, horizon, ...)
    5. 결과 기록

→ 결과 concat (axis=0) → ForecastResult (N, H)
```

- 미래 외생변수가 horizon보다 짧으면 마지막 행을 반복하여 padding
- PyTorch Lightning 로거를 `ERROR` 레벨로 억제하여 진행률 바 중복 방지

### 결과 집계 (`_collect_results`)

개별 모형 출력 `(1, H)` 또는 `(1, n_samples, H)`을 axis=0으로 연결하여 `(N, H)` 또는 `(N, n_samples, H)` 생성.

| 입력 타입 | 집계 방식 | 출력 shape |
|-----------|----------|------------|
| `ParametricForecastResult` | `params` dict 각 key별 `np.concatenate(axis=0)` | (N, H) per key |
| `SampleForecastResult` | `samples` 배열 `np.concatenate(axis=0)` | (N, n_samples, H) |
| `QuantileForecastResult` | `quantiles_data` dict 각 q별 `np.concatenate(axis=0)` | (N, H) per q |

---

## PerHorizonRunner

`BaseModel`을 상속하여 로깅·디렉토리 관리·모형 저장 기능을 활용한다.

### 파라미터

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `data_dir` | `str \| Path` | `horizon_1.csv`, ..., `horizon_H.csv` 위치 |
| `model_name` | `str` | MODEL_REGISTRY 키 (`"xgboost"`, `"ngboost"`, `"catboost"`, `"lr"`, `"gbm"`, `"pgbm"`) |
| `y_col` | `str` | 타겟 컬럼명 |
| `exog_cols` | `List[str]?` | 피처 컬럼 (None = y_col 제외 전체) |
| `training_period` | `(start, end)` | 학습 구간, 모든 horizon 공유 |
| `forecast_period` | `(start, end)` | 예측 구간, 모든 horizon 공유 |
| `hyperparameter` | `dict?` | 모든 horizon 모형에 전달 |
| `horizons` | `List[int]?` | 명시적 horizon 리스트 (None = `horizon_*.csv` 자동 탐색) |
| `dist_name` | `str` | 출력 분포 이름 (기본: `"normal"`) |
| `n_jobs` | `int` | 병렬 학습 수 (1=순차, -1=전체 코어) |

### 메서드

| 메서드 | 반환 | 설명 |
|--------|------|------|
| `fit()` | `Self` | 전체 horizon 모형 학습 (순차 또는 joblib 병렬) |
| `forecast()` | `ParametricForecastResult (N_common, H)` | 통합 다중 horizon 예측 |
| `forecast_horizon(h)` | `ParametricForecastResult (T, 1)` | 단일 horizon 예측 |

### 학습·예측 흐름

#### `fit()`

```
1. horizons 결정 (명시 또는 data_dir에서 자동 탐색)
2. for h in horizons:
     a. horizon_{h}.csv 로드 → training_period 슬라이싱
     b. MODEL_REGISTRY.get(model_name) → 모형 인스턴스 생성
     c. model.fit()
     d. self._models[h] = model

   (n_jobs != 1이면 joblib.Parallel로 병렬 실행)
```

#### `forecast()`

```
1. for h in horizons:
     a. forecast_period 슬라이싱
     b. model.forecast(X, index) → ParametricForecastResult (T, 1)

2. common_idx ← 모든 horizon의 forecast_index 교집합 (N_common)
3. 각 horizon 결과에서 common_idx 행만 추출, axis=1 방향으로 조립
   → ParametricForecastResult (N_common, H)
```

- `N_common`이 최대 horizon 인덱스의 90% 미만이면 경고 로그 출력

### 데이터 탐색

`data_dir`에서 `horizon_(\d+).csv` 패턴을 glob하여 자동으로 horizon 리스트를 구성한다. CSV는 `basis_time` 컬럼을 DatetimeIndex로 파싱한다.

### 저장·로드

| 메서드 | 설명 |
|--------|------|
| `_save_model_specific()` | 각 horizon 모형의 `save_model()` 호출 |
| `_load_model_specific()` | horizon별 디렉토리에서 모형 바이너리 로드 |

저장 구조 상세: `docs/runner_persistence_design.md`

---

## 외생변수 처리

### 분류 체계

| 범주 | 정의 | 예시 |
|------|------|------|
| **futr_exog** | 예측 시점에 미래값이 알려진 변수 | NWP 풍속, 기온 예보 |
| **hist_exog** | 과거 관측값만 사용 가능한 변수 | SCADA 실측 풍속 |

### 모형 패밀리별 인터페이스

| 패밀리 | Runner 파라미터 | 이유 |
|--------|----------------|------|
| Statistical (GARCH) | `exog_cols` | futr만 사용 (hist는 leakage) |
| ML (CatBoost 등) | `exog_cols` | CSV에서 피처 엔지니어링 완료 |
| Deep (DeepAR, TFT) | `futr_cols` + `hist_cols` | NeuralForecast가 별도 리스트 요구 |
| Foundation (Moirai) | `exog_cols` | `past_feat_dynamic_real`로 활용 (context만) |
| Foundation (Chronos) | — | 외생변수 미지원 |

### Leakage 방지

Deep 모형 경로에서 `future_X`에는 `futr_cols`만 포함하고 `hist_cols`는 제외한다. 이는 예측 시점에 hist 변수의 미래값을 알 수 없기 때문이다.

```
과거 구간 (context):  futr + hist  →  context_X
미래 구간 (horizon):  futr만       →  future_X   (hist 제외)
```

---

## Use Cases

### Statistical 모형 rolling 예측

```python
model = ArimaGarchForecaster(dataset=train_df, y_col="power",
                              exog_cols=["nwp_wspd"]).fit()

runner = RollingRunner(
    model, dataset=full_df, y_col="power",
    forecast_period=("2023-07-01", "2023-12-31"),
    exog_cols=["nwp_wspd"],
)
result = runner.run(horizon=24, dist_name="normal")
# ParametricForecastResult (N, 24)
```

### Statistical 모형 시뮬레이션 경로

```python
result = runner.run(
    horizon=24,
    method="simulate_paths",
    method_kwargs={"n_paths": 100, "seed": 42},
)
# SampleForecastResult (N, 100, 24)
```

### Deep 모형 (DeepAR) rolling 예측

```python
model = DeepARForecaster(
    dataset=train_df, y_col="power",
    futr_cols=["nwp_wspd", "nwp_temp"],
    hist_cols=["observed_wspd"],
).fit()

runner = RollingRunner(
    model, dataset=full_df, y_col="power",
    forecast_period=("2023-07-01", "2023-12-31"),
    futr_cols=["nwp_wspd", "nwp_temp"],
    hist_cols=["observed_wspd"],
)
result = runner.run(horizon=24)
# ParametricForecastResult 또는 QuantileForecastResult (N, 24)
```

### Foundation 모형 (Chronos) rolling 예측

```python
model = ChronosForecaster(dataset=train_df, y_col="power").fit()

runner = RollingRunner(
    model, dataset=full_df, y_col="power",
    forecast_period=("2023-07-01", "2023-12-31"),
)
result = runner.run(horizon=24)
# SampleForecastResult (N, n_samples, 24)
```

### ML 모형 per-horizon 학습·예측

```python
runner = PerHorizonRunner(
    data_dir="data/training_dataset/sinan/w100002/",
    model_name="ngboost",
    y_col="forecast_time_observed_KPX_pwr",
    training_period=("2020-01-01", "2022-12-31"),
    forecast_period=("2023-01-01", "2023-06-30"),
    n_jobs=-1,
)
runner.fit()
result = runner.forecast()
# ParametricForecastResult (N_common, H)
```

### 단일 horizon 예측

```python
result_h6 = runner.forecast_horizon(h=6)
# ParametricForecastResult (T, 1)
```

### 결과에서 Distribution 추출

```python
# 모든 Runner 결과에 공통 적용
for h in range(1, H + 1):
    dist = result.to_distribution(h)
    crps = compute_crps(dist, actual[:, h-1])

dist = result.to_distribution(h=6)
lower, upper = dist.interval(coverage=0.9)
q90 = dist.ppf(0.9)
```

### CSV 저장

```python
df = result.to_dataframe(h=1)     # 단일 horizon
df_all = result.to_dataframe()    # 전체 horizon (MultiIndex columns)
```
