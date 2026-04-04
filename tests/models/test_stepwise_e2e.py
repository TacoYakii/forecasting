"""E2E tests: stepwise order selection -> RollingRunner forecast.

Verifies the full pipeline (stepwise fit -> rolling forecast) works
for all three distributions: normal, studentT, skewStudentT.
"""

import numpy as np
import pytest

from src.models.statistical import StepwiseOrderSelector
from src.core.runner import RollingRunner

from .conftest import Y_COL, EXOG_COLS, FORECAST_START, FORECAST_END

HORIZON = 6


@pytest.fixture(params=["normal", "studentT", "skewStudentT"])
def e2e_result(request, train_df, full_df):
    """Run stepwise + RollingRunner for each distribution."""
    dist = request.param

    selector = StepwiseOrderSelector(
        "arima", ic="aicc", max_p=2, max_q=2, d=0,
        distribution=dist, verbose=False,
    )
    best = selector.select(train_df, y_col=Y_COL, exog_cols=EXOG_COLS)

    runner = RollingRunner(
        model=best,
        dataset=full_df,
        y_col=Y_COL,
        forecast_period=(FORECAST_START, FORECAST_END),
        exog_cols=EXOG_COLS,
    )
    result = runner.run(horizon=HORIZON, show_progress=False)

    return {
        "dist": dist,
        "selector": selector,
        "model": best,
        "result": result,
    }


class TestStepwiseE2E:
    """Full pipeline: stepwise -> RollingRunner -> ForecastResult."""

    def test_model_is_fitted(self, e2e_result):
        assert e2e_result["model"].is_fitted_

    def test_result_shape(self, e2e_result):
        r = e2e_result["result"]
        N = len(r.basis_index)
        H = r.horizon
        assert N > 0
        assert H == HORIZON

    def test_ic_finite(self, e2e_result):
        model = e2e_result["model"]
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)
        assert np.isfinite(model.aicc)

    def test_distribution_params(self, e2e_result):
        dist = e2e_result["dist"]
        model = e2e_result["model"]

        if dist == "normal":
            assert model._dist_params == {}
        elif dist == "studentT":
            assert "df" in model._dist_params
            assert model._dist_params["df"] > 2.0
        elif dist == "skewStudentT":
            assert "df" in model._dist_params
            assert "skew" in model._dist_params

    def test_forecast_result_to_distribution(self, e2e_result):
        r = e2e_result["result"]
        d = r.to_distribution(1)  # 1-indexed
        assert np.all(np.isfinite(d.mean()))

    def test_search_history_populated(self, e2e_result):
        sel = e2e_result["selector"]
        assert len(sel.search_history) > 0
        assert any(r.converged for r in sel.search_history)
