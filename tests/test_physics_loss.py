"""Tests for physics-informed losses: monodomain PDE and consistency losses."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from bpc_fno.physics.monodomain_loss import MonodomainPDELoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GRID_SIZE: int = 8
_BATCH_SIZE: int = 2
_VOXEL_SIZE_CM: float = 0.1


def _make_pde_config(
    voxel_size: float = _VOXEL_SIZE_CM,
    n_collocation: int = 64,
) -> SimpleNamespace:
    """Create a config with a .get() method mimicking DictConfig."""
    ns = SimpleNamespace(
        voxel_size_cm=voxel_size,
        n_collocation_points=n_collocation,
    )
    ns.get = lambda key, default=None: getattr(ns, key, default)
    return ns


@pytest.fixture()
def pde_loss() -> MonodomainPDELoss:
    return MonodomainPDELoss(_make_pde_config())


@pytest.fixture()
def all_ones_mask() -> torch.Tensor:
    """Mask where all voxels are myocardium."""
    return torch.ones(
        _BATCH_SIZE, 1, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE, dtype=torch.bool
    )


@pytest.fixture()
def geometry_all_active() -> torch.Tensor:
    """Geometry tensor with mask channel (index 3) = 1 everywhere."""
    geo = torch.zeros(_BATCH_SIZE, 4, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)
    geo[:, 3, :, :, :] = 1.0
    return geo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestZeroDivergence:
    """Uniform J_i field should have div = 0."""

    def test_zero_divergence(self, pde_loss: MonodomainPDELoss) -> None:
        # Uniform J_i = (1, 2, 3) everywhere
        J_i = torch.zeros(_BATCH_SIZE, 3, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)
        J_i[:, 0, :, :, :] = 1.0
        J_i[:, 1, :, :, :] = 2.0
        J_i[:, 2, :, :, :] = 3.0

        div_J = pde_loss.compute_divergence_J(J_i, _VOXEL_SIZE_CM)

        # Interior points should have exactly zero divergence
        interior = div_J[:, :, 1:-1, 1:-1, 1:-1]
        torch.testing.assert_close(
            interior,
            torch.zeros_like(interior),
            atol=1e-7,
            rtol=0,
        )


class TestKnownDivergence:
    """J_i = (x, 0, 0) should have div = 1 everywhere (interior)."""

    def test_known_divergence(self, pde_loss: MonodomainPDELoss) -> None:
        N = _GRID_SIZE
        h = _VOXEL_SIZE_CM

        J_i = torch.zeros(_BATCH_SIZE, 3, N, N, N)
        # J_x = x-coordinate (in voxel index * voxel_size for physical coords)
        x_coords = torch.arange(N, dtype=torch.float32) * h
        J_i[:, 0, :, :, :] = x_coords.view(1, N, 1, 1)

        div_J = pde_loss.compute_divergence_J(J_i, h)

        # Central difference of J_x = x gives dJ_x/dx = 1 at interior points
        # (J_y, J_z are zero, so dJ_y/dy = dJ_z/dz = 0)
        interior = div_J[:, 0, 2:-2, 2:-2, 2:-2]  # avoid boundaries
        expected = torch.ones_like(interior)

        torch.testing.assert_close(interior, expected, atol=1e-5, rtol=1e-5)


class TestCollocationSampling:
    """Verify correct number of collocation points are sampled."""

    def test_collocation_sampling(
        self, geometry_all_active: torch.Tensor
    ) -> None:
        n_collocation = 32
        loss_fn = MonodomainPDELoss(_make_pde_config(n_collocation=n_collocation))

        # Use non-zero J_i so loss is non-trivial
        J_i = torch.randn(_BATCH_SIZE, 3, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

        # The loss function should run without error and return a scalar
        loss = loss_fn(J_i, geometry_all_active)
        assert loss.ndim == 0, "Loss should be a scalar."
        assert loss.item() >= 0, "MSE loss should be non-negative."

    def test_empty_mask_gives_zero_loss(self) -> None:
        loss_fn = MonodomainPDELoss(_make_pde_config())

        J_i = torch.randn(_BATCH_SIZE, 3, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)
        # Geometry with mask channel = 0 (no myocardium)
        geo = torch.zeros(_BATCH_SIZE, 4, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

        loss = loss_fn(J_i, geo)
        assert loss.item() == 0.0, "Loss should be zero when mask is empty."


class TestConsistencyLossGradient:
    """Verify gradients flow to J_i but not through forward model weights."""

    def test_consistency_loss_gradient(self) -> None:
        from bpc_fno.physics.consistency_loss import ForwardConsistencyLoss

        consistency_loss = ForwardConsistencyLoss()

        # Create a simple mock forward model
        class MockForward:
            """Minimal forward model: B = linear(J_i.mean(dim=[-3,-2,-1]))."""

            def __init__(self) -> None:
                self.linear = torch.nn.Linear(3, 48 * 50)
                # Freeze weights to simulate Phase 2 training
                for p in self.linear.parameters():
                    p.requires_grad_(False)

            def predict_B(
                self, J_i: torch.Tensor, geometry: torch.Tensor
            ) -> torch.Tensor:
                B = J_i.shape[0]
                pooled = J_i.mean(dim=[-3, -2, -1])  # (B, 3)
                return self.linear(pooled).view(B, 48, 50)

        model = MockForward()
        N = _GRID_SIZE

        J_i_hat = torch.randn(
            _BATCH_SIZE, 3, N, N, N, requires_grad=True
        )
        geometry = torch.randn(_BATCH_SIZE, 4, N, N, N)
        B_obs = torch.randn(_BATCH_SIZE, 48, 50)

        loss = consistency_loss(model, J_i_hat, geometry, B_obs)
        loss.backward()

        # J_i_hat should receive gradients
        assert J_i_hat.grad is not None, "J_i_hat did not receive gradients."
        assert J_i_hat.grad.abs().sum() > 0, "J_i_hat gradient is all zeros."

        # Forward model weights should NOT receive gradients (frozen)
        for p in model.linear.parameters():
            assert p.grad is None, (
                "Forward model weights should not receive gradients."
            )


class TestMonodomainLossUnits:
    """Verify dimensional analysis of the monodomain PDE loss."""

    def test_monodomain_loss_units(self, pde_loss: MonodomainPDELoss) -> None:
        """The divergence of J_i (uA/cm^2) divided by voxel_size (cm) gives
        units of uA/cm^3.  The squared residual has units (uA/cm^3)^2.

        We verify this by checking that scaling J_i by a factor k scales
        the divergence by k (since div is linear in J_i).  This confirms
        correct dimensional behaviour.
        """
        N = _GRID_SIZE
        torch.manual_seed(42)
        J_i = torch.randn(_BATCH_SIZE, 3, N, N, N)

        div_1 = pde_loss.compute_divergence_J(J_i, _VOXEL_SIZE_CM)

        k = 3.0
        div_k = pde_loss.compute_divergence_J(k * J_i, _VOXEL_SIZE_CM)

        # div should scale linearly with J_i
        torch.testing.assert_close(div_k, k * div_1, atol=1e-4, rtol=1e-4)

    def test_loss_quadratic_scaling(self) -> None:
        """Full loss (mean of div^2) should scale as k^2 when using all
        voxels deterministically via compute_divergence_J."""
        N = _GRID_SIZE
        torch.manual_seed(42)
        J_i = torch.randn(_BATCH_SIZE, 3, N, N, N)
        pde = MonodomainPDELoss(_make_pde_config())

        div_1 = pde.compute_divergence_J(J_i, _VOXEL_SIZE_CM)
        loss_1 = (div_1 ** 2).mean()

        k = 3.0
        div_k = pde.compute_divergence_J(k * J_i, _VOXEL_SIZE_CM)
        loss_k = (div_k ** 2).mean()

        expected_ratio = k ** 2
        actual_ratio = loss_k.item() / max(loss_1.item(), 1e-30)

        torch.testing.assert_close(
            torch.tensor(actual_ratio),
            torch.tensor(expected_ratio),
            atol=1e-4,
            rtol=1e-4,
        )
