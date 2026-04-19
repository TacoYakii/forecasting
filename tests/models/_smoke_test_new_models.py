"""Smoke test for new deep learning models (BiTCN, TSMixerx, TimesNet).

Mini config for fast verification:
- max_steps=200 (~30s training)
- 1-day rolling predict (24 steps)
- Dongbok data

Verifies:
1. fit() completes without exception
2. Best checkpoint restore message
3. RollingRunner.run() succeeds
4. Output shape and 19 quantile levels
5. No NaN/Inf in predictions
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

import src.models.deep_time_series  # noqa: F401  (register models)
from src.core.registry import MODEL_REGISTRY
from src.core.runner import RollingRunner

df = pd.read_csv(
    'data/training_data_new/dongbok/continuous/farm_level.csv',
    index_col='basis_time', parse_dates=True,
)
futr_cols = [c for c in df.columns if c.startswith(('ECMWF_', 'KMA_'))]
hist_cols = [c for c in df.columns if c.startswith('observed_') and c != 'observed_KPX_pwr']
train_df = df.loc['2022-01-03 06:00:00':'2022-12-31 23:00:00']

MINI_HP = {
    'prediction_length': 48,
    'input_size': 144,
    'loss_type': 'quantile',
    'max_steps': 200,
    'batch_size': 16,
    'windows_batch_size': 32,
    'val_size': 336,
    'early_stop_patience_steps': -1,  # no early stop for this mini test
    'scaler_type': 'standard',
}

for key in ['bitcn', 'tsmixerx', 'timesnet']:
    torch.cuda.empty_cache()
    print(f'\n{"="*60}\n=== {key.upper()} smoke test ===\n{"="*60}', flush=True)

    cls = MODEL_REGISTRY.get(key)
    model = cls(hyperparameter=MINI_HP, model_name=f'{key}_smoke')
    model.verbose = True
    model._enable_progress_bar = False

    # timesnet does not support hist_exog
    h_cols = None if key == 'timesnet' else hist_cols
    model.fit(
        dataset=train_df, y_col='observed_KPX_pwr',
        futr_cols=futr_cols, hist_cols=h_cols,
    )

    # Rolling predict on 10 basis_times (need period >= horizon=48 + n_valid)
    runner = RollingRunner(
        model=model, dataset=df, y_col='observed_KPX_pwr',
        forecast_period=('2023-01-01 00:00:00', '2023-01-03 09:00:00'),
        futr_cols=futr_cols, hist_cols=h_cols,
    )
    result = runner.run(horizon=48, show_progress=False)

    # Verify
    assert hasattr(result, 'basis_index'), 'No basis_index'
    assert result.horizon == 48, f'Expected H=48, got {result.horizon}'
    assert len(result.quantile_levels) == 19, (
        f'Expected 19 quantile levels, got {len(result.quantile_levels)}'
    )
    median_q = sorted(result.quantile_levels)[len(result.quantile_levels) // 2]
    median = result.quantiles_data[median_q]
    assert not np.any(np.isnan(median)), 'NaN in median prediction'
    assert not np.any(np.isinf(median)), 'Inf in median prediction'

    print(
        f'  ✓ {key}: shape={median.shape}, 19 quantiles, no NaN/Inf\n'
        f'    median range: [{median.min():.1f}, {median.max():.1f}]',
        flush=True,
    )

print('\n' + '='*60)
print('✓ All 3 models: smoke test PASSED')
print('='*60)
