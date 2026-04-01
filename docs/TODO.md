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
- [ ] Design document
- [ ] Implementation

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
