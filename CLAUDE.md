# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소의 코드를 다룰 때 참고하는 안내서입니다.

## 프로젝트 개요

풍력발전 확률적 예측 시스템. 다양한 모형군(통계, ML, 딥러닝, Foundation), 계층적 조정(hierarchical reconciliation), 한국전력거래소(KPX) 평가 기준을 지원합니다.

## 개발 환경 설정

```bash
uv sync          # 의존성 설치
uv run python    # 프로젝트 환경으로 실행
uv run pytest    # 테스트 실행 (있는 경우)
```

- Python 3.13 (`uv` 관리)
- 주요 의존성: NeuralForecast, Chronos, Uni2TS (Moirai), PyTorch Lightning, scikit-learn, scipy

## 아키텍처

### 핵심 추상화 (`src/core/`)

모든 모형은 **Distribution Registry** (Normal, StudentT, Gamma, Weibull 등)를 통해 네이티브 분포 파라미터(mu/std가 아닌)를 담은 **`ForecastParams`**를 반환합니다. 세 가지 결과 컨테이너가 있습니다:
- **`ParametricForecastResult`** — (N, H) 그리드 위의 parametric 분포
- **`QuantileForecastResult`** — 분위수 기반 (딥러닝 모형)
- **`SampleForecastResult`** — 몬테카를로 샘플 (Foundation 모형)

두 가지 Runner 패턴이 평가를 조율합니다:
- **`RollingRunner`** — 상태 유지 모형 (ARIMA-GARCH): fit → forecast → update_state 루프
- **`PerHorizonRunner`** — 상태 비유지 ML 모형: 예측 시점(horizon)별로 H개 독립 모형 훈련

### 모형 패밀리 (`src/models/`)

| 패밀리 | 모형 | Runner | 출력 |
|--------|------|--------|------|
| `statistical/` | ARIMA-GARCH, SARIMA, ARFIMA | RollingRunner | ParametricForecastResult |
| `machine_learning/` | CatBoost, NGBoost, PGBM, XGBoost, LR | PerHorizonRunner | ParametricForecastResult |
| `deep_time_series/` | DeepAR, TFT (NeuralForecast) | NeuralForecast wrapper | QuantileForecastResult |
| `foundation/` | Chronos, Moirai | from_pretrained() | SampleForecastResult |
| `reconciliation/` | MinT, Angular, TopDown, BottomUp | 전용 trainer | 조정된 예측 |
| `conditional_kernel_density/` | Adaptive KDE | — | 비모수 |

### 데이터 파이프라인 (`src/data/`)

다운로드 (KMA, ECMWF) → NWP 전처리 (GRIB2/TXT 리더, 검증기, 파생변수 생성) → 훈련 데이터 빌더 (연속/horizon별, 래그/외생변수 피처)

### 계층적 예측 (`src/pipelines/`)

`BaseForecastRunner` → `HierarchyForecastCoordinator` (CRPS 기준 레벨/horizon별 최적 모형 선택) → `HierarchyForecastSampler` (앙상블 데이터셋 생성). S 행렬이 집계 구조를 정의합니다.

### 평가 (`src/utils/`)

- **nMAPE**: KPX 전일시장 및 실시간시장 평가
- **CRPS/PIT**: 확률적 보정(calibration) 지표
- **Loss Registry**: 플러그 가능한 손실 함수 (RandomCRPSLoss, QuantileCRPSLoss, PinballLoss)

## 주요 패턴

- **Distribution Registry**: 분포 이름 → scipy 분포 + 파라미터명을 매핑하는 확장 가능한 딕셔너리. 모멘트 매칭에는 `mu_std_to_dist_params()`를 사용
- **Configuration**: dataclass 기반 설정, JSON 직렬화 지원 (`BaseConfig`, `BaseTrainerConfig`, `HierarchyForecastConfig`)
- **실험 디렉토리**: `res/{ModelClass}/{exp_num}/` 하위에 자동 증가, 로그·메타데이터·직렬화 모형 저장
- **Reconciliation**: 모든 방법은 S 행렬 (n_total, n_bottom)을 사용하며 `torch.nn.Module` 하위 클래스로 구현

## 관례

- 코드와 docstring은 영어; 도메인 특화 맥락(KPX 시장 유형, nMAPE)에서 한국어 용어 사용
- 모형은 `ForecastParams`를 통해 네이티브 분포 파라미터를 반환하며, raw mu/std를 직접 반환하지 않음
- `etc/`는 설정 파일 및 보조 스크립트 포함; `res/`는 실험 결과 저장 (둘 다 gitignore 대상)
- `notebooks/`의 노트북은 탐색과 테스트용이며, 프로덕션 코드 아님
