# Codex Battle Review Report

- **Date**: 2026-04-02
- **Scope**: Quantile Crossing Repair — `_repair_quantile_crossings()` (PAVA isotonic regression) in `QuantileForecastResult.__init__`
- **Total Iterations**: 7
- **Note**: `adversarial-review`는 Iteration 1-3에서 EISDIR 에러로 실패. 원인은 codex 플러그인의 `formatUntrackedFile()`에서 디렉터리(symlink `data/`)를 `fs.readFileSync()`로 읽으려 한 것. `stat.isDirectory()` 체크 추가로 수정 후 Iteration 5부터 정상 실행.

---

## Iteration 1

### Codex Review
1-D 배열(단일 horizon) 입력 시 `_repair_quantile_crossings()`가 crash하는 regression 지적. `np.stack(..., axis=1)`이 2-D를 생성하고 `N, _, H = stacked.shape`에서 unpacking 실패.

### Codex Adversarial Review
EISDIR 에러로 실행 실패.

### My Analysis
| # | Issue | Judgment | Reason |
|---|-------|---------|--------|
| 1 | 1-D quantile 배열에서 `_repair_quantile_crossings` crash | Agree | `_get_horizon()`에 1-D 분기가 존재하므로 방어 코드 필요 |

### Changes Made
- `src/core/forecast_results.py`: `_repair_quantile_crossings()`에 `is_1d` 분기 추가 — 1-D 입력을 `(N, Q, 1)`로 확장 후 처리, 반환 시 1-D 복원
- `tests/core/test_quantile_crossing_repair.py`: `test_1d_single_horizon_crossing` 테스트 추가

---

## Iteration 2

### Codex Review
1-D 배열로 `QuantileForecastResult`를 생성하면 생성은 성공하지만, `to_distribution()`과 `_reindex_positions()`가 2-D를 가정하므로 이후 사용 시 crash한다는 지적.

### Codex Adversarial Review
EISDIR 에러로 실행 실패.

### My Analysis
| # | Issue | Judgment | Reason |
|---|-------|---------|--------|
| 1 | 1-D 배열 후속 메서드 crash | Disagree | 이 문제는 repair 변경 이전부터 존재하는 기존 코드의 한계. `_repair_quantile_crossings`는 입력 차원을 보존하므로 regression이 아님. |

### Rebuttals Sent
- **1-D 후속 crash**: 기존 코드에서도 `to_distribution()`의 `[:, h_idx]` 인덱싱이 1-D에서 실패함. Pre-existing issue.

---

## Iteration 3

### Codex Review
"actionable한 regression을 발견하지 못했다" — 리뷰 종료.

### Codex Adversarial Review
EISDIR 에러로 실행 실패. 원인 분석 → codex 플러그인 `scripts/lib/git.mjs` 수정.

### Changes Made (플러그인)
- `scripts/lib/git.mjs:138`: `stat.isDirectory()` 체크 추가 → EISDIR 방지

---

## Iteration 4 (EISDIR 수정 후 첫 adversarial-review)

### Codex Review
Iteration 3에서 이미 종료.

### Codex Adversarial Review
**Verdict: needs-attention** — 1-D 입력이 constructor를 통과하지만 후속 메서드에서 crash. **해결 제안**: constructor에서 1-D를 `(N, 1)`로 정규화.

### My Analysis
| # | Issue | Judgment | Reason |
|---|-------|---------|--------|
| 1 | 1-D → 2-D 정규화 필요 | Agree | Pre-existing으로 반론했으나 정규화 제안이 타당. Constructor에서 정규화하면 기존 불일치까지 해결. |

### Changes Made
- `src/core/forecast_results.py`:
  - `QuantileForecastResult.__init__`: 1-D → `(N, 1)` 자동 정규화
  - `_get_horizon()`: 1-D 분기 제거 (항상 2-D 보장)
  - `_repair_quantile_crossings()`: `is_1d` 분기 제거 (항상 2-D 입력)
- `tests/core/test_quantile_crossing_repair.py`: `test_1d_input_normalizes_and_works` 추가 (constructor → `to_distribution` → `reindex` 통합)

---

## Iteration 5

### Codex Review
통과 — "internally consistent, no regression".

### Codex Adversarial Review
**Verdict: needs-attention** — 2건:
1. [high] `np.stack`으로 전체 텐서 복사 → OOM/latency
2. [medium] NaN 셀 skip → 불완전한 distribution

### My Analysis
| # | Issue | Judgment | Reason |
|---|-------|---------|--------|
| 1 | 전체 텐서 복사 OOM | Partially agree | 벤치마크: N=10000에서 85ms/190MB. 현실적 N≤1000에서 10ms/20MB로 문제없지만, fast path에서 불필요한 stacking은 최적화 가능 |
| 2 | NaN 셀 bypass | Disagree | Pre-existing behavior. NaN을 reject하면 rolling forecast 사용 불가. Repair scope가 아닌 입력 검증 정책 |

### Changes Made
- `src/core/forecast_results.py`: fast path를 stacking 없이 인접 level 간 비교로 최적화 → N=10000에서 85ms → 6ms

---

## Iteration 6

### Codex Review
통과 — "no discrete regressions".

### Codex Adversarial Review
**Verdict: needs-attention** — 2건:
1. [high] NaN bypass (반복 2회차)
2. [medium] 모듈 레벨 `IsotonicRegression` singleton → thread safety

### My Analysis
| # | Issue | Judgment | Reason |
|---|-------|---------|--------|
| 1 | NaN bypass | Disagree | 반론 유지 (pre-existing, scope 외) |
| 2 | Singleton thread safety | Agree | `fit_transform`이 내부 상태 변경, concurrent construction에서 race 가능 |

### Changes Made
- `src/core/forecast_results.py`: 모듈 레벨 `_ISO_REG` 제거 → 함수 로컬 `iso = IsotonicRegression(...)` 생성

---

## Iteration 7

### Codex Review
통과 — "no introduced bug worth fixing before merging".

### Codex Adversarial Review
NaN bypass 반복 3회차 → **교착 상태**.

### Deadlock Resolution: NaN Policy
- **Issue**: NaN-containing quantile 셀이 repair를 bypass하고 downstream에 전달됨
- **My Position**: Pre-existing behavior. Repair scope 외. NaN reject은 rolling forecast 사용성 파괴. 별도 입력 검증 정책으로 분리 필요.
- **Codex Position**: NaN 셀이 invalid distribution을 생성하므로 fail-fast 또는 explicit handling 필요.
- **Final Status**: **Unresolved** — NaN 정책은 `QuantileForecastResult` 전체의 입력 검증 설계 문제로, 별도 후속 과제로 등록.

---

## Summary
- **Total issues raised**: 5 (distinct)
  - 1-D crash: 1건 (Iteration 1)
  - 1-D 후속 메서드: 1건 (Iteration 2-4, 최종 수용)
  - OOM/latency: 1건 (Iteration 5)
  - Singleton thread safety: 1건 (Iteration 6)
  - NaN bypass: 1건 (Iteration 5-7, 3회 반복)
- **Accepted & fixed**: 4
  - 1-D crash 방어 → constructor 정규화로 통합
  - Fast path 최적화 (stacking 회피)
  - Singleton → 함수 로컬 생성
- **Rebutted & dismissed**: 0
- **Unresolved (deadlocked)**: 1 (NaN policy — 별도 후속 과제)
- **Rescue interventions**: 0
- **Key improvements**:
  - `QuantileForecastResult`가 1-D 단일 horizon 입력을 `(N, 1)`로 자동 정규화
  - Fast path에서 full stacking 회피 (N=10000: 85ms → 6ms)
  - `IsotonicRegression` thread-safe (함수 로컬 인스턴스)
  - Codex 플러그인 EISDIR 버그 수정
