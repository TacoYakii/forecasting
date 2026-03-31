# Exogenous Variable Split Tests

`test_exog_split.py`의 테스트 항목과 검증 내용.

예측 테스트는 `tests/models/plots/`에 플롯을 저장하여 시각적 확인 가능.

## 테스트 데이터 구조

실제 데이터 컬럼 매핑(`docs/exogenous_variable_design.md` Section 2.3)을 모사한 합성 데이터:

| 컬럼 | 분류 | 설명 |
|------|------|------|
| `nwp_wspd` | **futr_exog** | NWP 풍속 예보 (= 실측 + 예보 오차) |
| `nwp_temp` | **futr_exog** | NWP 온도 예보 |
| `obs_wspd` | **hist_exog** | 실측 풍속 (과거만 관측 가능) |
| `power` | **target (y)** | 풍력 발전량 |

---

## TestStatisticalExog

Statistical 모형은 `exog_cols`에 **futr만** 전달해야 한다.
hist를 넣으면 forecast 시 미래 값을 줄 수 없어 leakage 발생.

| 테스트 | 검증 내용 | 플롯 |
|--------|----------|------|
| `test_arima_garch_futr_only` | `exog_cols=FUTR_COLS`로 fit → `x_future`에 futr 데이터 전달하여 forecast → (1, H) 정상 | `statistical_arima_garch_futr.png` |
| `test_arima_garch_update_state_with_exog` | update_state(y, x_new) → forecast(x_future) 순서로 exog가 올바르게 전달되는지 | — |
| `test_arima_garch_exog_shape` | `model.exog_cols == FUTR_COLS`, `model.X.shape[1] == 2` (futr 개수) | — |

## TestMLExog

ML 모형은 `exog_cols`에 **futr + hist 전체**를 전달한다.
per-horizon CSV에서 feature engineering이 완료된 상태이므로 분리 불필요.

| 테스트 | 검증 내용 | 플롯 |
|--------|----------|------|
| `test_fit_forecast_with_exog` | LR/XGBoost가 `exog_cols=FUTR+HIST`로 fit → forecast → (T, 1) 정상, 예측값 범위 합리적 | `ml_{model_name}_exog.png` |
| `test_futr_only_vs_futr_hist` | futr만 vs futr+hist로 학습한 LR의 예측이 서로 다른지 (hist가 실제로 정보를 추가하는지) | `ml_lr_futr_vs_futr_hist.png` |
| `test_ml_exog_cols_stored` | `model.exog_cols == EXOG_COLS`, `model.X.shape[1] == 3` (futr + hist) | — |

## TestDeepExog (slow)

Deep 모형은 `futr_cols`와 `hist_cols`를 **분리**하여 전달한다.
NeuralForecast가 인코더/디코더 입력을 구분하므로 분리 필수.

| 테스트 | 검증 내용 | 플롯 |
|--------|----------|------|
| `test_deepar_futr_hist_split` | `model.futr_cols == FUTR_COLS`, `model.hist_cols == HIST_COLS`, `model.exog_cols == FUTR + HIST` (합산) | — |
| `test_deepar_futr_hist_fit_forecast` | futr+hist로 학습 → `future_X`에 **futr만** 전달 → (1, H) 정상, 예측값 범위 합리적 | `deep_deepar_futr_hist.png` |
| `test_deepar_predict_from_context` | `predict_from_context(context_X=futr+hist, future_X=futr만)` — Runner가 실제로 호출하는 경로 검증 | `deep_deepar_predict_from_context.png` |
| `test_deepar_futr_only` | hist 없이 futr만으로도 fit/forecast 정상, 예측값 범위 합리적 | `deep_deepar_futr_only.png` |

---

## 플롯 구성

각 플롯은 2-panel 구조:
- **상단**: 학습 데이터 (마지막 48h) + forecast 기간 관측값 + 예측값 (mu) + 95% CI
- **하단**: 외생변수 (futr_exog 실선, hist_exog 점선)

비교 플롯(`ml_lr_futr_vs_futr_hist.png`)은 단일 panel에 futr-only vs futr+hist 예측 비교.

---

## 실행 방법

```bash
# non-slow 테스트만 (Statistical, ML)
uv run pytest tests/models/test_exog_split.py -v -m "not slow"

# Deep 모형 포함 전체
uv run pytest tests/models/test_exog_split.py -v

# 플롯 확인
ls tests/models/plots/
```
