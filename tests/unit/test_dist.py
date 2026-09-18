"""Tests for dist.py."""

import logging
from typing import Any
from typing import Callable

import pytest
from pytest_mock import MockerFixture
from torch.distributed import device_mesh

from minifsdp import dist


@pytest.fixture
def world_size() -> int:
    """Return default world size fixture."""
    return 2


def _mock_fn(
    rank: int, world_size: int, mesh: device_mesh.DeviceMesh, msg: str
) -> None:
    """Mock worker function that logs the message."""
    logging.info(f"rank {rank}: {msg}")


def test_run_multiprocess(
    world_size: int,
    caplog: pytest.LogCaptureFixture,
    mocker: MockerFixture,
) -> None:
    """Test run_multiprocess executes worker and passes arguments."""
    caplog.set_level(logging.INFO)
    msg = "hello"

    def fake_spawn(
        worker_fn: Callable[..., None],
        args: tuple[Any, ...],
        nprocs: int,
        join: bool,
    ) -> None:
        ws, init_method, fn, user_args = args
        assert nprocs == world_size
        assert join is True
        assert ws == world_size
        assert init_method.startswith("file://")
        assert fn is _mock_fn
        assert user_args == (msg,)
        worker_fn(0, 1, init_method, fn, user_args)

    mock_spawn = mocker.patch("torch.multiprocessing.spawn", side_effect=fake_spawn)

    dist.run_multiprocess(_mock_fn, world_size, msg)

    mock_spawn.assert_called_once()
    assert msg in caplog.text
