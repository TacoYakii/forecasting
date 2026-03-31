# Exogenous Variable Classification Design

## 1. Overview

시계열 예측에서 외생변수(exogenous variable)는 **예측 시점에 미래 값의 가용 여부**에 따라
두 가지로 분류된다. 현재 코드베이스는 `x_cols` 하나로 모든 외생변수를 통칭하며,
분류 정보가 없어 Runner/Model 간 암묵적 가정에 의존하고 있다.

이 문서는 exog 2분류 개념을 정리하고, 각 모형 패밀리에서 어떻게 적용되는지,
그리고 리팩토링 시 실제 변경이 필요한 부분을 정리한다.

> **Note:** NeuralForecast는 `stat_exog` (시간 불변 정적 변수)도 지원하지만,
> 본 프로젝트는 항상 단일 발전단지 단위로 데이터셋을 구성하므로 stat_exog는 사용하지 않는다.

### 외생변수 2분류

| 분류 | 정의 | 풍력 예측 예시 |
|------|------|----------------|
| **`futr_exog`** | 예측 시점에 미래 값이 **알려진** 변수 | NWP 예보 (풍속, 온도, 기압) |
| **`hist_exog`** | 과거 값만 관측 가능, 미래 값 **미지** | 실측 기상, SCADA 센서 |

---

## 2. 현재 상태 분석

### 2.1 `x_cols`의 현재 사용 흐름

```
BaseForecaster.__init__(x_cols)
  → prepare_dataset(): self.X = dataset[x_cols].to_numpy()
  → model_info에 x_cols 저장

Runner가 x_cols로 forecast_data에서 feature 추출
  → 모형의 forecast/predict_from_context에 numpy 배열로 전달
```

### 2.2 모형별 현재 exog 처리

| 모형 패밀리 | Runner | 현재 exog 파라미터 | 실제 의미 |
|-------------|--------|-------------------|-----------|
| **Statistical** (GARCH) | RollingRunner | `x_future: (H, n_exog)` + `x_new: (n_exog,)` | 전부 futr_exog (NWP) |
| **ML** (CatBoost 등) | PerHorizonRunner | `X: (T, n_features)` | feature engineering에서 futr/hist 혼재 |
| **Deep** (DeepAR, TFT) | RollingRunner | `future_X: (H, n_feat)` + `context_X: (ctx, n_feat)` | 전부 futr_exog로 취급 |
| **Foundation** (Chronos) | RollingRunner | 무시 | exog 미지원 |
| **Foundation** (Moirai) | RollingRunner | `context_X` → `past_feat_dynamic_real` | hist_exog처럼 동작 |

### 2.3 실제 데이터 컬럼 매핑

Continuous dataset (`data/training_data_new/continuous/farm_level.csv`) 기준:

| 컬럼 | 분류 | 이유 |
|------|------|------|
| `ECMWF_forecast_u10`, `_u100`, `_u200` | **futr_exog** | ECMWF NWP 예보 — 미래 값 알려짐 |
| `ECMWF_forecast_v10`, `_v100`, `_v200` | **futr_exog** | 〃 |
| `ECMWF_forecast_wdir10`, `_wdir100`, `_wdir200` | **futr_exog** | 〃 |
| `ECMWF_forecast_wspd10`, `_wspd100`, `_wspd200` | **futr_exog** | 〃 |
| `KMA_forecast_*` (풍속, 온도, 습도, 기압 등) | **futr_exog** | KMA NWP 예보 |
| `observed_wspd` | **hist_exog** | 실측 풍속 — 과거만 관측 가능 |
| `observed_KPX_pwr` | **target (y)** | 예측 대상 |

Deep 모형에서의 사용:
- `ECMWF_forecast_*`, `KMA_forecast_*` → `futr_exog_list` → 인코더(과거) + 디코더(미래) 입력
- `observed_wspd` → `hist_exog_list` → 인코더(과거)에만 입력, 미래 값은 전달하지 않음

### 2.4 핵심 문제

1. **futr/hist 구분 없음**: `x_cols` 하나로 통칭하여, Runner가 모든 x_cols의 미래 값을
   forecast_data에서 추출하려고 시도 → hist_exog는 미래 값이 없으므로 의미 없는 패딩 발생
2. **모형별 암묵적 가정**: Statistical은 x_future를 무조건 futr_exog로 가정,
   Deep은 모든 x_cols를 `futr_exog_list`에 전달
---

## 3. 설계 방향

별도 Config 클래스는 만들지 않는다. 모형 패밀리에 따라 적절한 인자명으로 `List[str]`을
직접 받는다.

### 3.1 모형별 exog 인터페이스

futr/hist 구분이 **필요한 모형**과 **불필요한 모형**이 있다.

| 모형 | exog 인자 | 이유 |
|------|-----------|------|
| **Statistical** (GARCH) | `exog_cols` | NWP + 관측값 모두 exog로 사용. forecast에는 NWP(미래), update_state에는 관측값(현재) — 시점별 슬라이싱은 Runner가 처리 |
| **ML** (CatBoost 등) | `exog_cols` | per-horizon CSV에 feature engineering 완료 상태. futr/hist 구분 의미 없음 |
| **Deep** (DeepAR, TFT) | `futr_cols` + `hist_cols` | NeuralForecast가 인코더/디코더 입력을 구분하므로 분리 필요 |
| **Foundation** (Moirai) | `exog_cols` | `past_feat_dynamic_real`만 지원 — 과거 context만 사용하므로 futr/hist 구분 의미 없음 |
| **Foundation** (Chronos) | 없음 | exog 미지원 |

### 3.2 BaseForecaster

기존 `x_cols`를 `exog_cols`로 rename. Statistical/ML은 이것만 사용.

```python
class BaseForecaster(BaseModel):
    def __init__(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        exog_cols: Optional[List[str]] = None,   # 기존 x_cols → rename
        ...
    ):
```

### 3.3 BaseDeepModel

Deep 모형만 `futr_cols` + `hist_cols`로 분리. NeuralForecast가 구분을 요구하기 때문.

```python
class BaseDeepModel(BaseForecaster):
    def __init__(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        futr_cols: Optional[List[str]] = None,
        hist_cols: Optional[List[str]] = None,
        ...
    ):
        # BaseForecaster에는 exog_cols = futr_cols + hist_cols 전달
        super().__init__(
            dataset=dataset,
            y_col=y_col,
            exog_cols=(futr_cols or []) + (hist_cols or []),
            ...
        )
        self.futr_cols = futr_cols or []
        self.hist_cols = hist_cols or []
```

### 3.4 BaseFoundationModel

Foundation 모형은 `exog_cols`로 통합. 모델 내부에서 지원 방식에 맞게 처리.

- **Moirai**: `exog_cols` 전체를 `past_feat_dynamic_real`로 전달 (과거 context만 사용)
- **Chronos**: `exog_cols` 무시 (exog 미지원)

### 3.5 Runner

```python
class RollingRunner:
    def __init__(
        self,
        model,
        dataset: pd.DataFrame,
        y_col: str,
        forecast_period: Tuple,
        exog_cols: Optional[List[str]] = None,   # Statistical/Foundation용
        futr_cols: Optional[List[str]] = None,   # Deep용
        hist_cols: Optional[List[str]] = None,    # Deep용
    ):
```

- **StatefulPredictor** (GARCH): `exog_cols`로 슬라이싱, forecast에 미래 구간, update_state에 현재 시점 전달
- **ContextPredictor — Deep**: `futr_cols` / `hist_cols` 분리
  - context에는 futr+hist 모두 전달
  - future에는 futr만 전달 (hist는 미래 값 없음)
- **ContextPredictor — Foundation**: `exog_cols`로 통합 전달
  - Moirai: context 구간만 `past_feat_dynamic_real`로 사용
  - Chronos: 무시

### 3.3 Deep 모형 (BaseDeepModel)

NeuralForecast는 학습/예측 시 exog 분류를 명시적으로 구분한다.

#### 학습 시: 하나의 DataFrame에 모든 변수를 컬럼으로

```python
# NeuralForecast 학습 데이터 형식
train_df = pd.DataFrame({
    "unique_id": "target",
    "ds": time_index,
    "y": power_values,           # 타겟
    "nwp_ws": [...],             # futr_exog 컬럼
    "nwp_temp": [...],           # futr_exog 컬럼
    "obs_ws": [...],             # hist_exog 컬럼
})

# 모델 생성 시 리스트로 분류를 알려줌
model = DeepAR(
    h=48,
    futr_exog_list=["nwp_ws", "nwp_temp"],   # 이 컬럼은 미래 값 있음
    hist_exog_list=["obs_ws"],                # 이 컬럼은 과거만
    ...
)

nf = NeuralForecast(models=[model], freq="h")
nf.fit(df=train_df)
```

NeuralForecast는 리스트를 보고 내부적으로 각 변수를 다르게 처리:
- `futr_exog_list` 변수 → 인코더(과거) + 디코더(미래) 모두 입력
- `hist_exog_list` 변수 → 인코더(과거)에만 입력, 디코더에는 안 들어감

#### hist_exog가 인코더에서 활용되는 방식

DeepAR, TFT 등의 인코더-디코더 구조에서 hist_exog의 역할:

```
시간축:  ──────── 과거 (context) ────────┃──── 미래 (horizon) ────
                                        ┃
인코더 입력:  y_past + futr + hist       ┃
              ↓                          ┃
         hidden state (h)                ┃
              ↓                          ┃
디코더 입력:          h + futr만          ┃→  예측값
```

인코더는 과거 구간의 모든 정보(y, futr_exog, hist_exog)를 **hidden state로 압축**한다.
디코더는 이 hidden state + futr_exog만으로 미래를 예측한다.

hist_exog(e.g. `observed_wspd`)는 미래 예측에 직접 입력되는 것이 아니라,
**과거 패턴 인식의 보조 정보**로 hidden state에 반영된다:

- 인코더가 과거 48시간의 `observed_wspd` 패턴을 보고
  "최근 풍속이 증가 추세", "NWP 예보와 실측의 괴리가 크다" 등의 정보를 hidden state에 담음
- 디코더는 그 hidden state + 미래 NWP 예보를 결합해서 발전량 예측

따라서 hist_exog의 미래 값은 **아예 필요하지 않으며**, `futr_df`에 포함되지 않는다.

#### 예측 시: `futr_df`에 futr_exog의 미래 값만 전달

```python
# futr_exog만 미래 H스텝 값을 제공
futr_df = pd.DataFrame({
    "unique_id": "target",
    "ds": future_time_index,       # 미래 H스텝의 시간
    "nwp_ws": future_ws_values,    # NWP 예보값
    "nwp_temp": future_temp_values,
    # obs_ws 없음 — hist_exog는 미래 값이 없으므로
})

forecast = nf.predict(futr_df=futr_df)
```

#### 현재 코드의 문제

현재 `BaseDeepModel`은 모든 `x_cols`를 `futr_exog_list`에만 전달:
```python
# 현재 코드 (base_deep_model.py)
feat_cols = self._get_feature_cols(self.dataset)
futr_exog = feat_cols if feat_cols else None
model = DeepAR(..., futr_exog_list=futr_exog)  # hist/stat 구분 없음
```

변경 후:
```python
model = DeepAR(
    ...,
    futr_exog_list=self.futr_cols or None,
    hist_exog_list=self.hist_cols or None,
)
```

### 3.6 모형별 exog 흐름 요약

| | 학습 시 exog | 예측 시 미래 구간 | 예측 시 과거/현재 구간 |
|---|---|---|---|
| **Statistical** | futr만 | futr만 (`x_future`) | futr만 (`x_new`) |
| **ML** | futr + lag(y) | futr만 (CSV에서 보장) | N/A (per-horizon) |
| **Deep** | futr + hist | futr만 (`futr_df`) | futr + hist (context) |
| **Foundation** | futr + hist | 전달 없음 | futr + hist (context) |

- Statistical은 학습에도 hist를 포함하지 않는다. forecast 시 hist의 미래 값을
  줄 수 없으므로, 학습에 넣으면 학습/예측 불일치가 발생하기 때문.
- Deep은 NeuralForecast의 인코더-디코더 아키텍처가 futr/hist를 분리 처리하므로,
  학습에 hist를 넣어도 예측 시에는 인코더(과거)에서만 활용된다.
- Foundation은 context(과거)만 사용하므로, 모든 exog를 넣어도 leakage가 발생하지 않는다.
- ML은 per-horizon CSV의 feature engineering 단계에서 보장해야 한다.

---

## 4. Implementation Checklist

### Phase 1: Core rename
1. `src/core/base_model.py` — `x_cols` → `exog_cols` rename
2. `src/models/statistical/`, `src/models/machine_learning/` — 동일 rename

### Phase 2: Runner
3. `src/core/runner.py` — `RollingRunner`에 `exog_cols` + `futr_cols` + `hist_cols` 수용
4. `src/core/runner.py` — `_run_stateful`에서 `exog_cols`로 슬라이싱
5. `src/core/runner.py` — `_run_context`에서 futr/hist 분리

### Phase 3: Deep 모형
6. `src/core/base_deep_model.py` — `futr_cols` + `hist_cols` 인자, super에 합산 전달
7. `src/core/base_deep_model.py` — `forecast()`, `predict_from_context()` 수정
8. `src/models/deep_time_series/deepar.py` — `_create_model()`에서 futr/hist 분리
9. `src/models/deep_time_series/tft.py` — `_create_model()`에서 futr/hist 분리

### Phase 4: Foundation 모형
10. `src/models/foundation/moirai.py` — `x_cols` → `exog_cols` rename

### Phase 5: 문서
11. `CLAUDE.md` Architecture 섹션에 exog 분류 규약 추가
