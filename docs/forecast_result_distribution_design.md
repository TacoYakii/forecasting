# ForecastResult & Distribution 설계

## 설계 원칙

| 객체 | 차원 | 역할 |
|------|------|------|
| **ForecastResult** | (N, H) 또는 (N, n_samples, H) | 다중 horizon 결과 **저장·조립·horizon 간 joint 구조 접근** |
| **Distribution** | (N,) | 단일 horizon **marginal 확률 분포 연산** |

Marginal 통계량(`mean`, `std`, `quantile`, `interval`)은 Distribution에서만 제공한다. ForecastResult에서 통계량이 필요하면 `to_distribution(h)`을 거친다.

```
ForecastResult (N, H)
       │
       └─ to_distribution(h) ──▶  Distribution (N,)
                                    ├─ mean(), std()
                                    ├─ ppf(q), cdf(x), pdf(x)
                                    ├─ interval(coverage)
                                    └─ sample(n)
```

---

## ForecastResult

### 공통 (BaseForecastResult)

| 메서드/속성 | 설명 |
|-------------|------|
| `basis_index` | 예측 origin 시간 인덱스 (N,) |
| `horizon` | 최대 예측 수평 H |
| `to_distribution(h)` | 단일 horizon Distribution 추출 (1-indexed) |
| `to_dataframe(h=None)` | DataFrame 변환. 내부적으로 `to_distribution(h)` 호출 |

### ParametricForecastResult

| 메서드/속성 | 설명 |
|-------------|------|
| `dist_name` | 분포 이름 ("normal", "studentT" 등) |
| `params` | 분포 파라미터 dict, 각 (N, H) |
| `to_distribution(h)` | → `ParametricDistribution` |

### QuantileForecastResult

| 메서드/속성 | 설명 |
|-------------|------|
| `quantile_levels` | 저장된 분위수 레벨 리스트 |
| `quantiles_data` | 분위수 데이터 dict, 각 (N, H) |
| `to_distribution(h)` | → `EmpiricalDistribution` (quantile 기반) |

### SampleForecastResult

| 메서드/속성 | 설명 |
|-------------|------|
| `samples` | raw 샘플 배열 (N, n_samples, H) |
| `n_samples` | 샘플 수 |
| `path(basis_idx)` | 특정 origin의 전체 horizon 경로 (n_samples, H) |
| `to_distribution(h)` | → `EmpiricalDistribution` (sample 기반) |

`path()`는 horizon 간 joint 구조 접근이므로 ForecastResult에만 존재한다. `to_distribution(h)`을 거치면 단일 horizon marginal만 남아 path 정보가 소실된다.

---

## Distribution

### 공통 인터페이스

| 메서드 | 반환 shape | 설명 |
|--------|-----------|------|
| `mean()` | (N,) | 분포 평균 |
| `std()` | (N,) | 분포 표준편차 |
| `ppf(q)` | (N,) 또는 (N, Q) | 분위수 함수 |
| `cdf(x)` | (N,) | 누적분포함수 |
| `pdf(x)` | (N,) | 확률밀도함수 |
| `sample(n)` | (N, n) | 샘플 생성 |
| `interval(coverage)` | (lower, upper) 각 (N,) | 예측 구간 |
| `to_dataframe()` | DataFrame | mu, std 컬럼 |

### ParametricDistribution

scipy frozen distribution 기반 해석적 계산. `dist_name`과 `params` dict로 구성.

### EmpiricalDistribution

sorted samples 또는 quantile levels/values 기반 비모수적 근사. `ppf()`는 `np.percentile`, `interval()`은 `ppf(alpha/2), ppf(1-alpha/2)`로 구현.

---

## Use Cases

### 모형 평가 (CRPS, nMAPE)

```python
for h in range(1, H + 1):
    dist = result.to_distribution(h)
    crps = compute_crps(dist, actual[:, h-1])
```

### 예측 구간 조회

```python
dist = result.to_distribution(h=6)
lower, upper = dist.interval(coverage=0.9)
```

### 분위수 예보

```python
dist = result.to_distribution(h=6)
q90 = dist.ppf(0.9)
quantiles = dist.ppf([0.1, 0.5, 0.9])   # (N, 3)
```

### Sample path 분석 (SampleForecastResult 전용)

```python
# Joint 구조 접근 — ForecastResult에서만 가능
paths = result.path(basis_idx=0)   # (n_samples, H)

# Marginal 통계 — Distribution 경유
dist_h6 = result.to_distribution(h=6)
median = dist_h6.ppf(0.5)
```

### CSV 저장

```python
df = result.to_dataframe(h=1)    # 단일 horizon
df_all = result.to_dataframe()   # 전체 horizon (MultiIndex columns)
```
