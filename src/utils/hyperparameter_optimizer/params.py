from dataclasses import dataclass, field
from typing import List 
import ngboost.distns


@dataclass
class ExampleParams:
    # 1. 정수 범위: tuple[int, int] = (min, max)
    n_estimators: tuple[int, int] = (50, 500)
    max_depth: tuple[int, int] = (3, 12)
    min_samples_split: tuple[int, int] = (2, 20)

    # 2. 실수 범위: tuple[float, float] = (min, max)
    learning_rate: tuple[float, float] = (0.01, 0.3)
    subsample: tuple[float, float] = (0.6, 1.0)
    colsample_bytree: tuple[float, float] = (0.6, 1.0)

    # 3. 문자열 선택지: List[str] = [choice1, choice2, ...]
    booster: List[str] = field(default_factory=lambda: ['gbtree', 'gblinear', 'dart'])
    objective: List[str] = field(default_factory=lambda: ['reg:squarederror', 'reg:gamma', 'reg:tweedie'])
    eval_metric: List[str] = field(default_factory=lambda: ['rmse', 'mae', 'mape'])

    # 4. 로그 스케일 실수: tuple[float, float, bool] = (min, max, log_scale)
    learning_rate_log: tuple[float, float, bool] = (0.001, 1.0, True)
    reg_alpha: tuple[float, float, bool] = (1e-8, 10.0, True)
    reg_lambda: tuple[float, float, bool] = (1e-8, 10.0, True)

    # 5. 스텝이 있는 정수: tuple[int, int, int] = (min, max, step)
    max_leaves: tuple[int, int, int] = (10, 100, 5)  # 10, 15, 20, ..., 100
    num_boost_round: tuple[int, int, int] = (100, 1000, 50)  # 100, 150, 200, ...

    # 6. 정수 선택지: List[int] = [choice1, choice2, ...]
    num_parallel_tree: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    max_bin: List[int] = field(default_factory=lambda: [64, 128, 256, 512])

    # 7. 실수 선택지: List[float] = [choice1, choice2, ...]
    dropout_rate: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    gamma: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.5, 1.0, 2.0])

    # 8. 불리언 값: bool = True (True/False 중에서 선택됨)
    enable_categorical: bool = True
    use_gpu: bool = True
    early_stopping: bool = True
    
    # 9. class 값: List[type]
    Dist: List[type] = field(default_factory=lambda: [ngboost.distns.Normal, ngboost.distns.T, ngboost.distns.LogNormal])
    
@dataclass
class NGBoostParams:
    # 못찾았음, 일단 이것만이라도 
    # Distribution selection
    Dist: List[type] = field(default_factory=lambda: [ngboost.distns.Normal, ngboost.distns.T, ngboost.distns.LogNormal])
    
@dataclass
class CatBoostParams:
    # https://catboost.ai/docs/en/concepts/parameter-tuning
    iterations: tuple[int, int] = (100, 1000)
    depth: tuple[int, int] = (4, 10)
    learning_rate: tuple[float, float] = (0.01, 0.3)
    l2_leaf_reg: tuple[float, float] = (1, 10)


@dataclass
class XGBoostParams:
    # https://xgboost.readthedocs.io/en/stable/parameter.html 
    booster: List[str] = field(default_factory=lambda: ['gbtree', 'gblinear', 'dart'])
    reg_alpha: tuple[float, float, bool]= (1e-8, 1.0, True) 
    reg_lambda: tuple[float, float, bool] = (1e-8, 1.0, True)  # lambda처럼 예약어와 곂칠경우 앞에 "_"를 추가 
    subsample: tuple[float, float] = (0.2, 1.0) 
    colsample_bytree: tuple[float, float] = (0.2, 1.0)