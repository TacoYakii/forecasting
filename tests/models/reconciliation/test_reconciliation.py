"""Tests for numpy/scipy-based hierarchical reconciliation."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.reconciliation import (
    HierarchicalReconciliation,
    ReconciliationConfig,
    ReconciliationData,
    ReconciliationFitConfig,
)
from src.models.reconciliation.combine import (
    angular_combine_core,
    apply_P_constraint_np,
    pack_sparse_params,
    sample_angular_random_numbers,
    unpack_sparse_params,
)
from src.models.reconciliation.fit import angular_objective, make_crn
from src.models.reconciliation.projection import BottomUpProjection, TopDownProjection
from src.models.reconciliation.utils import (
    build_padded_adjacency,
    extract_tree_from_S,
    get_sample_covariance,
    get_shrinkage_estimator,
)


B = 8
N_SAMPLES = 64


@pytest.fixture()
def S_simple():
    return np.array([[1, 1], [1, 0], [0, 1]], dtype=float)


@pytest.fixture()
def S_three_level():
    return np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _make_forecast_3d(S, batch=B, n_samples=N_SAMPLES, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, S.shape[0], n_samples))


def _make_forecast_2d(S, batch=B, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, S.shape[0]))


def _make_prob_data(S, batch=24, n_samples=N_SAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(range(batch))
    forecast = rng.standard_normal((batch, S.shape[0], n_samples))
    observed = rng.standard_normal((batch, S.shape[0]))
    return ReconciliationData(keys=keys, forecast=forecast, observed=observed)


class TestConfigValidation:
    @pytest.mark.parametrize(
        "proj, coh, comb",
        [
            ("topdown", "ranked", "linear"),
            ("bottomup", "ranked", "linear"),
            ("mint", "ranked", "linear"),
            ("mint", "empirical_copula", "mean_shift"),
            ("sparse", "ranked", "weighted"),
            ("sparse", "ranked", "angular"),
            ("sparse", "empirical_copula", "mean_shift"),
        ],
    )
    def test_valid_combinations(self, proj, coh, comb):
        ReconciliationConfig(projection=proj, coherence=coh, combine=comb).validate()

    def test_sparse_linear_rejected(self):
        with pytest.raises(ValueError, match="sparse"):
            ReconciliationConfig(projection="sparse", combine="linear").validate()


class TestForwardAndCoherence:
    def _check_coherence(self, S_np, out):
        n_low = S_np.shape[1]
        if out.ndim == 3:
            bottom = out[:, -n_low:, :]
            reconstructed = np.einsum("nl,tlq->tnq", S_np, bottom)
        else:
            bottom = out[:, -n_low:]
            reconstructed = np.einsum("nl,tl->tn", S_np, bottom)
        np.testing.assert_allclose(out, reconstructed, atol=1e-6)

    @pytest.mark.parametrize(
        "proj, coh, comb",
        [
            ("topdown", "ranked", "linear"),
            ("bottomup", "ranked", "linear"),
            ("sparse", "ranked", "weighted"),
            ("sparse", "ranked", "angular"),
        ],
    )
    def test_reconcile_3d_shape(self, S_simple, proj, coh, comb):
        model = HierarchicalReconciliation(
            S_simple,
            ReconciliationConfig(projection=proj, coherence=coh, combine=comb, n_samples=N_SAMPLES),
        )
        data = ReconciliationData(keys=list(range(B)), forecast=_make_forecast_3d(S_simple), observed=np.zeros((B, 3)))
        out = model.reconcile(data)
        assert out.shape == data.forecast.shape
        self._check_coherence(S_simple, out)

    def test_reconcile_2d_shape(self, S_simple):
        model = HierarchicalReconciliation(
            S_simple,
            ReconciliationConfig(projection="bottomup", coherence="ranked", combine="linear"),
        )
        data = ReconciliationData(keys=list(range(B)), forecast=_make_forecast_2d(S_simple), observed=np.zeros((B, 3)))
        out = model.reconcile(data)
        assert out.shape == data.forecast.shape
        self._check_coherence(S_simple, out)

    def test_three_level_shape(self, S_three_level):
        model = HierarchicalReconciliation(S_three_level, ReconciliationConfig())
        data = ReconciliationData(keys=list(range(B)), forecast=_make_forecast_3d(S_three_level), observed=np.zeros((B, 7)))
        out = model.reconcile(data)
        assert out.shape == (B, 7, N_SAMPLES)
        self._check_coherence(S_three_level, out)


class TestProjectionStrategies:
    def test_topdown_P_structure(self):
        proj = TopDownProjection(2, 3)
        P = proj.get_P()
        assert P.shape == (2, 3)
        np.testing.assert_allclose(P[:, 0], [1.0, 1.0])
        np.testing.assert_allclose(P[:, 1:], 0.0)

    def test_bottomup_P_structure(self):
        proj = BottomUpProjection(2, 3)
        P = proj.get_P()
        assert P.shape == (2, 3)
        np.testing.assert_allclose(P[:, 0], [0.0, 0.0])
        np.testing.assert_allclose(P[:, 1:], np.eye(2))

    def test_sparse_P_softmax_sums_to_one(self, S_simple):
        model = HierarchicalReconciliation(S_simple, ReconciliationConfig(constraint_P="linear"))
        P = model.projection.get_P()
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("mode", ["OLS", "WLS", "MINT_SAMPLE", "MINT_SHRINK"])
    def test_mint_all_modes(self, S_simple, mode):
        model = HierarchicalReconciliation(
            S_simple,
            ReconciliationConfig(projection="mint", coherence="ranked", combine="linear", mint_mode=mode),
        )
        base_train_data = _make_prob_data(S_simple, batch=40, n_samples=1, seed=123)
        model.fit(base_train_data=base_train_data)
        P = model.projection.get_P()
        assert P.shape == (2, 3)
        assert np.all(np.isfinite(P))


class TestEmpiricalCopula:
    def test_copula_setup_and_reconcile(self, S_simple):
        model = HierarchicalReconciliation(
            S_simple,
            ReconciliationConfig(projection="sparse", coherence="empirical_copula", combine="mean_shift"),
        )
        base_train_data = _make_prob_data(S_simple, batch=30, seed=7)
        test_data = _make_prob_data(S_simple, batch=8, seed=8)
        model.fit(base_train_data=base_train_data)
        out = model.reconcile(test_data)
        assert out.shape == test_data.forecast.shape
        assert np.all(np.isfinite(out))


class TestAngularAndFitHelpers:
    def test_angle_defaults(self, S_simple):
        model = HierarchicalReconciliation(S_simple, ReconciliationConfig(combine="angular", n_samples=N_SAMPLES))
        np.testing.assert_allclose(model.combine.get_angle_vector(), 45.0)

    def test_pack_unpack_roundtrip(self, S_simple):
        model = HierarchicalReconciliation(S_simple, ReconciliationConfig(combine="angular"))
        packed = pack_sparse_params(model.projection.P_raw, model.parent_mask, model.combine.get_angle_vector())
        P_raw, angles = unpack_sparse_params(packed, model.parent_mask, with_angles=True)
        np.testing.assert_allclose(P_raw, model.projection.P_raw)
        np.testing.assert_allclose(angles, model.combine.get_angle_vector())

    def test_apply_P_constraint_linear(self, S_simple):
        model = HierarchicalReconciliation(S_simple, ReconciliationConfig(combine="angular"))
        logits = np.random.randn(*model.projection.P_raw.shape)
        P = apply_P_constraint_np(logits, model.parent_mask, "linear")
        for i in range(model.num_low):
            valid = model.parent_mask[i]
            np.testing.assert_allclose(P[i, valid].sum(), 1.0, atol=1e-6)
            assert np.all(P[i, valid] >= 0)

    def test_make_crn_deterministic(self):
        u1, v1 = make_crn(2, 10, 100, 42)
        u2, v2 = make_crn(2, 10, 100, 42)
        np.testing.assert_array_equal(u1, u2)
        np.testing.assert_array_equal(v1, v2)

    def test_angular_core_shape_and_determinism(self, S_simple):
        model = HierarchicalReconciliation(S_simple, ReconciliationConfig(combine="angular", n_samples=N_SAMPLES))
        forecast = np.sort(_make_forecast_3d(S_simple, batch=5), axis=-1)
        P = model.projection.get_P()
        u, v = sample_angular_random_numbers(model.num_low, 5, 200, seed=99)
        out1 = angular_combine_core(
            P, model.combine.get_angle_vector(), model.combine.q_val, forecast,
            model.padded_parents, model.parent_mask, u, v,
        )
        out2 = angular_combine_core(
            P, model.combine.get_angle_vector(), model.combine.q_val, forecast,
            model.padded_parents, model.parent_mask, u, v,
        )
        assert out1.shape == (5, model.num_low, N_SAMPLES)
        np.testing.assert_array_equal(out1, out2)

    def test_angular_objective_returns_scalar(self, S_simple):
        model = HierarchicalReconciliation(S_simple, ReconciliationConfig(combine="angular", n_samples=N_SAMPLES))
        recon_val_data = _make_prob_data(S_simple, batch=12, seed=19)
        fit_config = ReconciliationFitConfig()
        u, v = make_crn(model.num_low, 12, model.combine.mc_samples, 42)
        x0 = pack_sparse_params(model.projection.P_raw, model.parent_mask, model.combine.get_angle_vector())
        loss = angular_objective(x0, model, recon_val_data, fit_config, u, v)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_sparse_weighted_fit_updates_P(self, S_simple):
        model = HierarchicalReconciliation(S_simple, ReconciliationConfig(combine="weighted"))
        before = model.projection.P_raw.copy()
        recon_val_data = _make_prob_data(S_simple, batch=16, seed=31)
        model.fit(recon_val_data=recon_val_data, fit_config=ReconciliationFitConfig(maxiter=20))
        assert not np.allclose(before, model.projection.P_raw)


class TestSaveLoad:
    def test_sparse_roundtrip(self, S_simple, tmp_path):
        model = HierarchicalReconciliation(S_simple, ReconciliationConfig())
        data = _make_prob_data(S_simple, batch=8, seed=55)
        out_before = model.reconcile(data)
        model.save_pretrained(tmp_path / "model")
        loaded = HierarchicalReconciliation.from_pretrained(S_simple, tmp_path / "model")
        out_after = loaded.reconcile(data)
        np.testing.assert_allclose(out_before, out_after, atol=1e-6)


class TestUtils:
    def test_sample_covariance_2d(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 100))
        W = get_sample_covariance(X)
        assert W.shape == (3, 3)
        assert np.all(np.diag(W) > 0)
        np.testing.assert_allclose(W, W.T, atol=1e-6)

    def test_sample_covariance_3d(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3, 20))
        W = get_sample_covariance(X)
        assert W.shape == (3, 3, 20)

    def test_shrinkage_estimator_2d(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 100))
        W = get_shrinkage_estimator(X)
        assert W.shape == (3, 3)
        assert np.all(np.diag(W) > 0)

    def test_extract_tree_simple(self, S_simple):
        tree = extract_tree_from_S(S_simple)
        assert 0 in tree
        assert {c[0] for c in tree[0]} == {1, 2}

    def test_build_padded_adjacency(self, S_simple):
        max_parents, padded, mask = build_padded_adjacency(S_simple)
        assert max_parents >= 1
        assert padded.shape == mask.shape
