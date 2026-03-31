# Exogenous Variable Classification Design

## 1. Overview

시계열 예측에서 외생변수(exogenous variable)는 **예측 시점에 미래 값의 가용 여부**에 따라
세 가지로 분류된다. 현재 코드베이스는 `x_cols` 하나로 모든 외생변수를 통칭하며,
분류 정보가 없어 Runner/Model 간 암묵적 가정에 의존하고 있다.

이 문서는 exog 3분류 개념을 정리하고, 각 모형 패밀리에서 어떻게 적용되는지,
그리고 리팩토링 시 실제 변경이 필요한 부분을 정리한다.

### 외생변수 3분류

| 분류 | 정의 | 풍력 예측 예시 |
|------|------|----------------|
| **`futr_exog`** | 예측 시점에 미래 값이 **알려진** 변수 | NWP 예보 (풍속, 온도, 기압) |
| **`hist_exog`** | 과거 값만 관측 가능, 미래 값 **미지** | 실측 기상, SCADA 센서 |
| **`stat_exog`** | 시간 불변 상수 | 터빈 위치, 정격출력, 발전단지 ID |

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

### 2.3 핵심 문제

1. **futr/hist 구분 없음**: `x_cols` 하나로 통칭하여, Runner가 모든 x_cols의 미래 값을
   forecast_data에서 추출하려고 시도 → hist_exog는 미래 값이 없으므로 의미 없는 패딩 발생
2. **모형별 암묵적 가정**: Statistical은 x_future를 무조건 futr_exog로 가정,
   Deep은 모든 x_cols를 `futr_exog_list`에 전달
3. **stat_exog 미지원**: 정적 변수를 시계열 컬럼으로 반복 저장 (메모리 낭비)

---

## 3. 설계 방향

별도 Config 클래스는 만들지 않는다. 각 모형/Runner에서 `futr_cols`, `hist_cols`,
`stat_cols`를 `List[str]` 인자로 직접 받는 방식으로 처리한다.

### 3.1 BaseForecaster

기존 `x_cols`를 `futr_cols`로 rename. `hist_cols`, `stat_cols`는 필요한 모형에서만 추가.

```python
class BaseForecaster(BaseModel):
    def __init__(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        futr_cols: Optional[List[str]] = None,   # 기존 x_cols → rename
        ...
    ):
```

### 3.2 Runner

Runner 생성 시 `futr_cols`, `hist_cols`를 따로 받고 내부에서 올바르게 슬라이싱.

```python
class RollingRunner:
    def __init__(
        self,
        model,
        dataset: pd.DataFrame,
        y_col: str,
        forecast_period: Tuple,
        futr_cols: Optional[List[str]] = None,
        hist_cols: Optional[List[str]] = None,
    ):
```

- **StatefulPredictor** (GARCH): `futr_cols`만 `x_future`로 슬라이싱
- **ContextPredictor** (Deep/Foundation):
  - `context_X_futr` = 과거 futr_cols 데이터
  - `context_X_hist` = 과거 hist_cols 데이터
  - `future_X_futr` = 미래 futr_cols 데이터 (NWP 등)
  - hist_cols의 미래 값은 전달하지 않음

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
    "capacity": [...],           # stat_exog 컬럼 (매 행 같은 값)
})

# 모델 생성 시 리스트로 분류를 알려줌
model = DeepAR(
    h=48,
    futr_exog_list=["nwp_ws", "nwp_temp"],   # 이 컬럼은 미래 값 있음
    hist_exog_list=["obs_ws"],                # 이 컬럼은 과거만
    stat_exog_list=["capacity"],              # 이 컬럼은 시간 불변
    ...
)

nf = NeuralForecast(models=[model], freq="h")
nf.fit(df=train_df)
```

NeuralForecast는 리스트를 보고 내부적으로 각 변수를 다르게 처리:
- `futr_exog_list` 변수 → 인코더(과거) + 디코더(미래) 모두 입력
- `hist_exog_list` 변수 → 인코더(과거)에만 입력, 디코더에는 안 들어감
- `stat_exog_list` 변수 → 시간 불변 임베딩으로 처리

#### 예측 시: `futr_df`에 futr_exog의 미래 값만 전달

```python
# futr_exog만 미래 H스텝 값을 제공
futr_df = pd.DataFrame({
    "unique_id": "target",
    "ds": future_time_index,       # 미래 H스텝의 시간
    "nwp_ws": future_ws_values,    # NWP 예보값
    "nwp_temp": future_temp_values,
    # obs_ws 없음 — hist_exog는 미래 값이 없으므로
    # capacity 없음 — stat_exog는 학습 시 이미 저장됨
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
    stat_exog_list=self.stat_cols or None,
)
```

---

## 4. 모형별 영향도 분석

### 4.1 Statistical (GARCH family)

| 항목 | 영향 | 설명 |
|------|------|------|
| `GarchBase.fit()` | **없음** | `x_cols` → `futr_cols` rename만 |
| `GarchBase.forecast(x_future)` | **없음** | Runner가 futr_cols만 전달하면 동일 |
| `GarchBase.update_state(x_new)` | **없음** | 동일 |
| `GarchBase.simulate_paths(x_future)` | **없음** | 동일 |

**결론: x_cols → futr_cols rename 외 변경 없음.**
Statistical 모형은 conditional mean에 exog를 포함하므로, futr_exog만 의미가 있다.

### 4.2 Machine Learning (DeterministicForecaster)

| 항목 | 영향 | 설명 |
|------|------|------|
| `*.forecast(X, target_index)` | **없음** | X는 per-horizon CSV의 전체 feature matrix |
| `PerHorizonRunner` | **rename만** | `x_cols` → `futr_cols` |

**결론: rename만.** ML 모형은 feature engineering이 data builder에서 완료됨.
Per-horizon CSV의 컬럼이 곧 feature이므로 futr/hist 구분은 builder 단계에서 해결.

### 4.3 Deep Time Series (BaseDeepModel)

| 항목 | 영향 | 설명 |
|------|------|------|
| `BaseDeepModel.__init__` | **중간** | `futr_cols`, `hist_cols`, `stat_cols` 인자 추가 |
| `_build_nf_dataframe()` | **없음** | 모든 컬럼이 이미 DataFrame에 있음 |
| `_create_model()` (서브클래스) | **중간** | 3개 리스트 분리 전달 |
| `forecast()` | **소** | `futr_df`에 futr_cols만 포함 |
| `predict_from_context()` | **중간** | context에는 futr+hist, futr_df에는 futr만 |

**결론: ~30줄 변경.** NeuralForecast가 이미 3분류를 지원하므로 리스트만 분리 전달.

### 4.4 Foundation Models

| 모형 | 영향 | 설명 |
|------|------|------|
| **Chronos** | **없음** | exog 미지원 |
| **Moirai** | **소** | `hist_cols` → `past_feat_dynamic_real` 명시적 매핑 |

### 4.5 Runner

| Runner | 영향 | 설명 |
|--------|------|------|
| `RollingRunner.__init__` | **소** | `x_cols` → `futr_cols` + `hist_cols` |
| `_run_stateful` | **소** | `futr_cols`만 슬라이싱 (현재와 실질 동일) |
| `_run_context` | **중간** | context/future를 futr/hist로 분리 |
| `PerHorizonRunner` | **rename만** | `x_cols` → `futr_cols` |

---

## 5. 변경 요약

| 파일 | 변경 규모 | 주요 내용 |
|------|-----------|-----------|
| `src/core/base_model.py` | **소** | `x_cols` → `futr_cols` rename |
| `src/core/base_deep_model.py` | **~30줄** | `hist_cols`/`stat_cols` 인자 추가, NF 3분류 전달 |
| `src/core/runner.py` | **~20줄** | Runner futr/hist 분리 슬라이싱 |
| `src/models/statistical/_garch_base.py` | **rename만** | `x_cols` → `futr_cols` |
| `src/models/machine_learning/*.py` | **rename만** | `x_cols` → `futr_cols` |
| `src/models/deep_time_series/*.py` | **~10줄** | `_create_model()`에서 3분류 |
| `src/models/foundation/moirai.py` | **~10줄** | hist_cols 매핑 |
| `src/models/foundation/chronos.py` | **0줄** | 변경 없음 |

---

## 6. Implementation Checklist

### Phase 1: Core rename
1. `src/core/base_model.py` — `x_cols` → `futr_cols` rename
2. `src/models/` 전체 — 동일 rename 반영

### Phase 2: Runner 분리
3. `src/core/runner.py` — `RollingRunner`에 `futr_cols` + `hist_cols` 수용
4. `src/core/runner.py` — `_run_stateful`에서 futr_cols만 슬라이싱
5. `src/core/runner.py` — `_run_context`에서 futr/hist 분리

### Phase 3: Deep 모형 3분류
6. `src/core/base_deep_model.py` — `hist_cols`, `stat_cols` 인자 추가
7. `src/core/base_deep_model.py` — `_build_nf_dataframe()`, `forecast()`, `predict_from_context()` 수정
8. `src/models/deep_time_series/deepar.py` — `_create_model()`에서 3분류
9. `src/models/deep_time_series/tft.py` — `_create_model()`에서 3분류
10. `src/models/foundation/moirai.py` — hist_cols 매핑

### Phase 4: 문서
11. `CLAUDE.md` Architecture 섹션에 exog 분류 규약 추가
