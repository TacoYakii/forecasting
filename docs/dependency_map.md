# Dependency Map

`src/` 디렉토리에서 실제로 import하는 모든 외부 라이브러리와 사용 위치 정리.

> 기준일: 2026-03-31  
> 범위: `src/**/*.py` 내 top-level import 기준

---

## 외부 라이브러리 (Third-party)

### 핵심 과학 컴퓨팅

| 패키지 | pyproject.toml | 사용 파일 수 | 사용 모듈 |
|--------|:-:|:-:|-----------|
| **numpy** | - | 46 | `core/`, `data/`, `datasets/`, `models/` (전 패밀리), `pipelines/`, `trainers/`, `utils/` |
| **pandas** | - | 53 | `core/`, `data/`, `datasets/`, `models/` (전 패밀리), `pipelines/`, `utils/` |
| **torch** | - | 25 | `core/` (loss, reconciliation_model, trainer), `datasets/`, `models/` (CKD, foundation, ML일부, reconciliation), `trainers/`, `utils/` (loss, evaluate, inference) |
| **scipy** | - | 5 | `core/forecast_distribution`, `models/statistical/_garch_base`, `utils/metrics/` (crps, pit), `utils/simulate` |

### ML 모형 라이브러리

| 패키지 | pyproject.toml | 사용 파일 수 | 사용 모듈 |
|--------|:-:|:-:|-----------|
| **neuralforecast** | O | 3 | `core/base_deep_model`, `models/deep_time_series/` (deepar, tft) |
| **catboost** | - | 1 | `models/machine_learning/catboost_model` |
| **ngboost** | - | 2 | `models/machine_learning/ngboost_model`, `utils/hyperparameter_optimizer/params` |
| **xgboost** | - | 1 | `models/machine_learning/xgboost_model` |
| **pgbm** | - | 1 | `models/machine_learning/pgboost_model` |
| **scikit-learn** (`sklearn`) | - | 3 | `models/machine_learning/` (gboost_model, lr_model), `utils/transformation` |

### Foundation 모형

| 패키지 | pyproject.toml | 사용 파일 수 | 사용 모듈 |
|--------|:-:|:-:|-----------|
| **chronos-forecasting** | O | 1 | `models/foundation/chronos` (lazy import) |
| **uni2ts** | O | 1 | `models/foundation/moirai` (lazy import) |
| **gluonts** | - | 1 | `models/foundation/moirai` (lazy import, uni2ts 의존성) |

### 기상/지리 데이터

| 패키지 | pyproject.toml | 사용 파일 수 | 사용 모듈 |
|--------|:-:|:-:|-----------|
| **metpy** | O | 1 | `data/nwp_preprocess/processors/deriver` |
| **xarray** | - | 1 | `data/nwp_preprocess/readers/grib2_reader` |
| **ecmwfapi** | - | 1 | `data/download/ECMWF` |
| **folium** | - | 1 | `data/download/location_map` |

### 유틸리티

| 패키지 | pyproject.toml | 사용 파일 수 | 사용 모듈 |
|--------|:-:|:-:|-----------|
| **tqdm** | - | 12 | `core/runner`, `data/` (download, nwp_preprocess, training_data_builder), `pipelines/hierarchy/`, `utils/evaluate_reconciliation` |
| **joblib** | - | 3 | `models/machine_learning/` (gboost_model, lr_model, ngboost_model) |
| **optuna** | - | 2 | `trainers/ckd`, `utils/hyperparameter_optimizer/opt_hyperparameter` |
| **requests** | - | 2 | `data/download/` (elevation, KMA) |
| **python-dotenv** | O | 2 | `data/download/` (elevation, KMA) — API 키 로드 |
| **matplotlib** | - | 4 | `models/conditional_kernel_density/plotting`, `utils/visualization/` (pit, plot_average_performance, plot_hierarchy) |
| **PyYAML** (`yaml`) | - | 1 | `data/nwp_preprocess/config` |

### 고성능/특수 컴퓨팅

| 패키지 | pyproject.toml | 사용 파일 수 | 사용 모듈 |
|--------|:-:|:-:|-----------|
| **numba** | - | 1 | `utils/metrics/crps` |
| **jax** | - | 1 | `utils/metrics_jax/crps` |

---

## pyproject.toml에 선언되었지만 미사용

| 패키지 | 비고 |
|--------|------|
| **ipywidgets** | `src/` 내 import 없음 |
| **jaxtyping** | `src/` 내 import 없음 |
| **hydra-core** | `src/` 내 import 없음 |

---

## 표준 라이브러리 (stdlib)

참고용으로 사용 중인 표준 라이브러리도 기록합니다.

| 패키지 | 사용 파일 수 | 주요 용도 |
|--------|:-:|-----------|
| `pathlib` | 44 | 경로 관리 |
| `logging` | 17 | 로깅 |
| `json` | 14 | 설정/결과 직렬화 |
| `dataclasses` | 13 | Config 클래스 정의 |
| `typing` | 다수 | 타입 힌트 |
| `pickle` | 9 | 모형/결과 직렬화 |
| `abc` | 7 | 추상 클래스 |
| `datetime` | 5 | 날짜 처리 |
| `re` | 5 | 정규표현식 |
| `os` | 4 | 환경변수, 파일시스템 |
| `time` | 3 | 다운로드 딜레이 |
| `concurrent.futures` | 3 | 병렬 처리 |
| `multiprocessing` | 2 | 병렬 처리 |
| `math` | 2 | 수학 연산 |
| `copy` | 2 | 깊은 복사 |
| `collections` | 1 | 특수 컨테이너 |
| `shutil` | 1 | 디렉토리 복사 |
| `sys` | 1 | 시스템 경로 |
| `threading` | 1 | 다운로드 스레딩 |
