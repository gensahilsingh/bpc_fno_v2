from __future__ import annotations

import torch

from bpc_fno.utils.checkpointing import load_checkpoint, save_checkpoint, validate_checkpoint


def _run_one_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _assert_optimizer_states_equal(
    actual: dict, expected: dict
) -> None:
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for param_id, expected_state in expected["state"].items():
        actual_state = actual["state"][param_id]
        assert actual_state.keys() == expected_state.keys()
        for key, expected_value in expected_state.items():
            actual_value = actual_state[key]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def test_checkpoint_round_trip_restores_optimizer_scheduler_and_extra_state(
    tmp_path,
) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )
    _run_one_optimizer_step(model, optimizer, scheduler)

    saved_model_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    saved_opt_state = optimizer.state_dict()
    saved_sched_state = scheduler.state_dict()

    path = tmp_path / "round_trip.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=7,
        phase="forward",
        metrics={"val_loss": 0.123},
        extra_state={"best_val_loss": 0.111},
        path=path,
    )

    new_model = torch.nn.Linear(4, 2)
    new_optimizer = torch.optim.SGD(
        new_model.parameters(), lr=0.01, momentum=0.0
    )
    new_scheduler = torch.optim.lr_scheduler.StepLR(
        new_optimizer, step_size=1, gamma=0.1
    )

    meta = load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        path=path,
        strict=True,
    )

    assert meta["epoch"] == 7
    assert meta["phase"] == "forward"
    assert meta["metrics"]["val_loss"] == 0.123
    assert meta["extra_state"]["best_val_loss"] == 0.111
    _assert_optimizer_states_equal(new_optimizer.state_dict(), saved_opt_state)
    assert new_scheduler.state_dict() == saved_sched_state
    for key, expected in saved_model_state.items():
        assert torch.equal(new_model.state_dict()[key], expected)

    validated = validate_checkpoint(path)
    assert validated["has_optimizer_state"] is True
    assert validated["has_scheduler_state"] is True
    assert validated["extra_state"]["best_val_loss"] == 0.111


def test_checkpoint_without_optimizer_or_scheduler_is_supported(tmp_path) -> None:
    model = torch.nn.Linear(3, 1)
    path = tmp_path / "model_only.pt"

    save_checkpoint(
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=3,
        phase="joint",
        metrics={"score": 1.0},
        path=path,
    )

    new_model = torch.nn.Linear(3, 1)
    meta = load_checkpoint(
        model=new_model,
        optimizer=None,
        scheduler=None,
        path=path,
        strict=True,
    )

    assert meta["epoch"] == 3
    assert meta["phase"] == "joint"
    assert meta["metrics"]["score"] == 1.0
