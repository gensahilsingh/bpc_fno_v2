"""Tests for the full forward-inverse pipeline (BPC_FNO_A)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from bpc_fno.models.bpc_fno_a import BPC_FNO_A


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GRID_SIZE: int = 8
_BATCH_SIZE: int = 2
_LATENT_DIM: int = 32
_C_HIDDEN: int = 16
_N_SENSORS_TOTAL: int = 12  # small for testing
_N_TIMESTEPS: int = 10


def _make_config() -> SimpleNamespace:
    """Build a minimal config for BPC_FNO_A with small dimensions."""

    model_ns = SimpleNamespace(
        n_fno_hidden=_C_HIDDEN,
        n_fno_layers=2,
        n_fno_modes=4,
        n_geometry_channels=4,
        latent_dim=_LATENT_DIM,
        n_sensors_total=_N_SENSORS_TOTAL,
        n_output_timesteps=_N_TIMESTEPS,
        grid_size=_GRID_SIZE,
        decoder_base_channels=64,
        n_decoder_upsamples=1,  # 4->8 (matches _GRID_SIZE=8)
        use_bilinear_decoder=False,
    )

    def _get(self: SimpleNamespace, key: str, default: object = None) -> object:
        return getattr(self, key, default)

    model_ns.get = _get.__get__(model_ns, type(model_ns))

    return SimpleNamespace(model=model_ns)


@pytest.fixture()
def config() -> SimpleNamespace:
    return _make_config()


@pytest.fixture()
def model(config: SimpleNamespace) -> BPC_FNO_A:
    torch.manual_seed(0)
    return BPC_FNO_A(config)


@pytest.fixture()
def sample_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    N = _GRID_SIZE
    T = _N_TIMESTEPS
    B_obs = torch.randn(_BATCH_SIZE, _N_SENSORS_TOTAL, _N_TIMESTEPS)
    return {
        "J_i": torch.randn(_BATCH_SIZE, 3, T, N, N, N),
        "geometry": torch.randn(_BATCH_SIZE, 4, N, N, N),
        "B_obs": B_obs,
        "B_mig": B_obs.clone(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestForwardThenInverse:
    """Run J_i through forward, then inverse, verify reconstruction."""

    def test_forward_then_inverse(
        self, model: BPC_FNO_A, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        model.eval()
        result = model(sample_batch)

        # Verify all expected keys exist
        assert "B_pred" in result
        assert "J_i_hat" in result
        assert "mu" in result
        assert "log_var" in result
        assert "z" in result

        # Verify shapes
        B = _BATCH_SIZE
        N = _GRID_SIZE
        T = _N_TIMESTEPS
        assert result["B_pred"].shape == (B, _N_SENSORS_TOTAL, T)
        assert result["J_i_hat"].shape == (B, 3, T, N, N, N)
        assert result["mu"].shape == (B, _LATENT_DIM)
        assert result["log_var"].shape == (B, _LATENT_DIM)
        assert result["z"].shape == (B, _LATENT_DIM)

    def test_forward_accepts_legacy_b_mig_key(
        self, model: BPC_FNO_A, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        model.eval()
        legacy_batch = {
            "J_i": sample_batch["J_i"],
            "geometry": sample_batch["geometry"],
            "B_mig": sample_batch["B_mig"],
        }
        result = model(legacy_batch)
        assert result["B_pred"].shape == (
            _BATCH_SIZE, _N_SENSORS_TOTAL, _N_TIMESTEPS
        )

    def test_reconstruct_method(
        self, model: BPC_FNO_A, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        result = model.reconstruct(
            B_obs=sample_batch["B_obs"],
            geometry=sample_batch["geometry"],
            n_samples=3,
        )
        assert result["J_i_mean"].shape == (
            _BATCH_SIZE, 3, _N_TIMESTEPS, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE
        )
        assert result["J_i_std"].shape == result["J_i_mean"].shape
        # With n_samples > 1, std should be non-zero (different posterior samples)
        # (Could be zero if model is degenerate, but very unlikely with random init)


class TestSharedWeights:
    """Verify geometry_encoder and fno_backbone are same objects in forward and inverse."""

    def test_shared_weights(self, model: BPC_FNO_A) -> None:
        # The geometry_encoder should be the exact same object
        assert model.forward_pino.geometry_encoder is model.geometry_encoder
        assert model.inverse_encoder.geometry_encoder is model.geometry_encoder

        # The FNO backbone should be the exact same object
        assert model.forward_pino.fno_backbone is model.fno_backbone
        assert model.inverse_encoder.fno_backbone is model.fno_backbone

    def test_shared_weights_same_parameters(self, model: BPC_FNO_A) -> None:
        """Parameter objects should have the same id."""
        fwd_geo_params = list(model.forward_pino.geometry_encoder.parameters())
        inv_geo_params = list(model.inverse_encoder.geometry_encoder.parameters())

        for p_fwd, p_inv in zip(fwd_geo_params, inv_geo_params):
            assert p_fwd is p_inv, "Shared parameters should be identical objects."


class TestPhase1Only:
    """Verify forward_only() works independently."""

    def test_phase1_only(
        self, model: BPC_FNO_A, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        B_pred = model.forward_only(
            J_i=sample_batch["J_i"],
            geometry=sample_batch["geometry"],
        )
        assert B_pred.shape == (_BATCH_SIZE, _N_SENSORS_TOTAL, _N_TIMESTEPS)

    def test_phase1_gradients(
        self, model: BPC_FNO_A, sample_batch: dict[str, torch.Tensor]
    ) -> None:
        """Gradients should flow through forward_only."""
        J_i = sample_batch["J_i"].clone().requires_grad_(True)
        B_pred = model.forward_only(J_i, sample_batch["geometry"])
        loss = B_pred.pow(2).mean()
        loss.backward()

        assert J_i.grad is not None
        assert J_i.grad.abs().sum() > 0


class TestLatentSampling:
    """Verify reparameterization produces different samples in train mode,
    same in eval mode."""

    def test_latent_sampling_train(self, model: BPC_FNO_A) -> None:
        model.train()
        mu = torch.zeros(_BATCH_SIZE, _LATENT_DIM)
        log_var = torch.zeros(_BATCH_SIZE, _LATENT_DIM)  # std = 1

        z1 = model.inverse_encoder.sample_z(mu, log_var)
        z2 = model.inverse_encoder.sample_z(mu, log_var)

        # In train mode, samples should differ (reparameterization trick)
        assert not torch.allclose(z1, z2, atol=1e-6), (
            "Samples should differ in train mode due to stochastic sampling."
        )

    def test_latent_sampling_eval(self, model: BPC_FNO_A) -> None:
        model.eval()
        mu = torch.randn(_BATCH_SIZE, _LATENT_DIM)
        log_var = torch.zeros(_BATCH_SIZE, _LATENT_DIM)

        z1 = model.inverse_encoder.sample_z(mu, log_var)
        z2 = model.inverse_encoder.sample_z(mu, log_var)

        # In eval mode, sample_z returns mu directly
        torch.testing.assert_close(z1, z2)
        torch.testing.assert_close(z1, mu)


class TestParameterGroups:
    """Verify get_parameter_groups returns all model params exactly once."""

    def test_parameter_groups(self, model: BPC_FNO_A) -> None:
        groups = model.get_parameter_groups()

        assert set(groups.keys()) == {
            "fno_shared",
            "forward_head",
            "inverse_head",
            "decoder",
        }

        # Collect all param ids from groups
        group_param_ids: list[int] = []
        for name, params in groups.items():
            for p in params:
                group_param_ids.append(id(p))

        # Collect all model param ids
        model_param_ids = {id(p) for p in model.parameters()}

        # Every model parameter should appear in exactly one group
        group_param_id_set = set(group_param_ids)
        assert group_param_id_set == model_param_ids, (
            f"Parameter groups don't cover all model params. "
            f"Missing: {model_param_ids - group_param_id_set}, "
            f"Extra: {group_param_id_set - model_param_ids}"
        )

        # No duplicates
        assert len(group_param_ids) == len(group_param_id_set), (
            "Some parameters appear in multiple groups."
        )

    def test_parameter_groups_non_empty(self, model: BPC_FNO_A) -> None:
        groups = model.get_parameter_groups()
        for name, params in groups.items():
            assert len(params) > 0, f"Parameter group '{name}' is empty."
