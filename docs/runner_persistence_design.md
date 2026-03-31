# Runner Persistence Design

## 1. Overview

Runner가 예측 실행 시 config와 ForecastResult를 `save_dir`에 저장하고,
이후 로드하여 분석할 수 있는 기능.

현재 상태:
- `RollingRunner`: 저장 기능 없음 (stateless orchestrator)
- `PerHorizonRunner`: 모형/메타데이터만 저장, 예측 결과는 미저장

### 모형 저장 정책

모형 파일(`model_config.yaml` + 모형 바이너리)은 **`save_model=True`** 옵션으로
Runner에게 명시적으로 요청한 경우에만 저장된다.

- `save_model=True` → Runner가 내부적으로 `model.save_model()` 호출
  → `model_config.yaml` + 모형 파일이 `save_dir/model/`에 저장
- `save_model=False` → 모형 파일 미저장, 예측 결과만 저장

이는 `BaseModel.save_model()` 내부에서 `save_model_config()` + `_save_model_specific()`을
함께 호출하는 구조와 일치한다. `fit()`만으로는 디스크에 아무것도 남지 않는다.

---

## 2. 저장 구조

### RollingRunner (단일 모델)

Statistical, Deep, Foundation 모형은 하나의 모델로 rolling 예측을 수행한다.

```
save_dir/
├── runner_config.yaml          # Runner 설정 (runner_type, forecast_period 등)
├── forecast_result/            # ForecastResult 직렬화
│   ├── params.npz              # numpy 배열 (loc, scale 등) — Parametric/Quantile
│   ├── samples.npz             # numpy 배열 (N, n_samples, H) — Sample
│   ├── metadata.yaml           # result_type, dist_name, shape, param_keys
│   └── basis_index.csv         # DatetimeIndex
└── model/                      # save_model=True인 경우에만 생성
    ├── model_config.yaml       # ModelConfig (하이퍼파라미터, dataset_setting)
    └── {ModelName}_model.{ext} # 모형 바이너리 (pkl, json, joblib, pt, nf/)
```

### PerHorizonRunner (horizon별 독립 모델)

ML 모형은 horizon별 독립 모델을 학습하므로, 모형 저장 시 horizon별 서브디렉토리가 생성된다.

```
save_dir/
├── runner_config.yaml          # Runner 설정 (runner_type, model_name, horizons 등)
├── forecast_result/            # 통합 ForecastResult (N, H)
│   ├── params.npz              # numpy 배열 (loc, scale 등)
│   ├── metadata.yaml           # result_type, dist_name, shape, param_keys
│   └── basis_index.csv         # DatetimeIndex
└── model/                      # save_model=True인 경우에만 생성
    ├── horizon_1/
    │   ├── model_config.yaml
    │   └── {ModelName}_model.{ext}
    ├── horizon_2/
    │   ├── model_config.yaml
    │   └── {ModelName}_model.{ext}
    ├── ...
    └── horizon_H/
        ├── model_config.yaml
        └── {ModelName}_model.{ext}
```

### runner_config.yaml

```yaml
runner_type: RollingRunner
model_name: ArimaGarchForecaster
y_col: power
exog_cols: [nwp_wspd, nwp_temp]
forecast_period: ["2023-07-01", "2023-12-31"]
horizon: 24
save_model: true
created: "2025-03-31 14:00:00"
```

### metadata.yaml

```yaml
result_type: ParametricForecastResult  # or QuantileForecastResult, SampleForecastResult
dist_name: normal                       # ParametricForecastResult만
param_keys: [loc, scale]                # npz 내 배열 이름
shape: [180, 24]                        # (N, H)
basis_index_file: basis_index.csv       # DatetimeIndex 저장
```

---

## 3. API

### RollingRunner

#### 저장

```python
# Statistical / Deep / Foundation 모형
model = ArimaGarchForecaster(dataset=train_df, y_col="power", ...).fit()

runner = RollingRunner(
    model=model,
    dataset=full_df,
    y_col="power",
    forecast_period=("2023-07-01", "2023-12-31"),
    exog_cols=FUTR_COLS,
    save_dir="res/arima_garch/exp_0",
    save_model=True,  # 모형 파일도 함께 저장 (기본: False)
)
result = runner.run(horizon=24)
# → runner_config.yaml + forecast_result/ 자동 저장
# → save_model=True이면 model/ 에 model_config.yaml + 모형 바이너리 저장
```

#### 로드

```python
from src.core.forecast_results import load_forecast_result

# 예측 결과만 로드 (분석용)
result = load_forecast_result("res/arima_garch/exp_0/forecast_result/")
result.to_distribution(h=6).ppf([0.1, 0.5, 0.9])

# 모형까지 로드 (재예측용, save_model=True로 저장한 경우)
model = ArimaGarchForecaster(dataset=train_df, y_col="power")
model.load_model("res/arima_garch/exp_0/model/ArimaGarchForecaster_model")
```

### PerHorizonRunner

#### 저장

```python
# ML 모형 (horizon별 독립 학습 + 예측)
runner = PerHorizonRunner(
    data_dir="data/training_dataset/sinan/w100002/",
    model_name="ngboost",
    y_col="forecast_time_observed_KPX_pwr",
    training_period=("2020-01-01", "2022-12-31"),
    forecast_period=("2023-01-01", "2023-06-30"),
    save_dir="res/ngboost/exp_0",
    save_model=True,  # horizon별 모형 전부 저장 (기본: False)
)
runner.fit()
result = runner.forecast()
# → runner_config.yaml + forecast_result/ 자동 저장
# → save_model=True이면 model/horizon_1/ ~ model/horizon_H/ 각각 저장
```

#### 로드

```python
from src.core.forecast_results import load_forecast_result

# 예측 결과만 로드 (분석용) — 통합 (N, H) 결과
result = load_forecast_result("res/ngboost/exp_0/forecast_result/")
result.to_distribution(h=6).ppf([0.1, 0.5, 0.9])

# 특정 horizon 모형 로드 (재예측용, save_model=True로 저장한 경우)
from src.models.machine_learning.registry import MODEL_REGISTRY
model_cls = MODEL_REGISTRY.get("ngboost")
model = model_cls(dataset=horizon_6_df, y_col="forecast_time_observed_KPX_pwr")
model.load_model("res/ngboost/exp_0/model/horizon_6/NGBoostForecaster_model")
```

---

## 4. ForecastResult별 직렬화

| Result 타입 | 저장 내용 | 포맷 |
|-------------|----------|------|
| `ParametricForecastResult` | `params` dict (loc, scale, df 등) | `params.npz` |
| `QuantileForecastResult` | `quantiles_data` dict (q별 배열) | `params.npz` |
| `SampleForecastResult` | `samples` 배열 (N, n_samples, H) | `samples.npz` |

공통: `basis_index` → `basis_index.csv`

---

## 5. Implementation Checklist

### Phase 1: ForecastResult save/load
1. `src/core/forecast_results.py` — 각 Result 클래스에 `save(path)` / `load(path)` 메서드
2. `src/core/forecast_results.py` — `load_forecast_result(path)` 팩토리 함수

### Phase 2: Runner 저장
3. `src/core/runner.py` — `RollingRunner`에 `save_dir`, `save_model` 파라미터 추가
4. `src/core/runner.py` — `run()` 후 `runner_config.yaml` + `forecast_result/` 자동 저장
5. `src/core/runner.py` — `save_model=True`이면 `model.save_model(save_dir / "model" / ...)` 호출
6. `src/core/runner.py` — `PerHorizonRunner`에도 동일 적용

### Phase 3: 테스트
7. 저장/로드 round-trip 테스트 (save → load → 값 일치 확인)
8. `save_model=False` 시 model/ 디렉토리 미생성 확인
9. `save_model=True` 시 `model_config.yaml` + 모형 바이너리 존재 확인
