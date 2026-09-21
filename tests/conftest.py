"""Unit-test conftest: distributed test helpers."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh

from minifsdp import dist


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._fc1 = nn.Linear(4, 8, bias=False)
        self._fc2 = nn.Linear(8, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self._fc1(x)
        return self._fc2(x)


@pytest.fixture
def model() -> nn.Module:
    """Return a simple two-layer model."""
    torch.manual_seed(42)
    return SimpleModel()


def _distributed_wrapper(
    rank: int,
    world_size: int,
    mesh: DeviceMesh,
    fn: Callable[..., None],
    args: tuple[Any, ...],
) -> None:
    """Module-level worker: unpickles fn and args, then calls fn(mesh, *args).

    Must be defined at module scope so that ``mp.spawn`` (which uses pickle) can
    serialise it.  The harness-constructed ``mesh`` is forwarded as the first argument
    so the callable can use it if needed.
    """
    fn(mesh, *args)


@pytest.fixture
def world_size() -> int:
    return 2


@pytest.fixture
def run_distributed(world_size: int) -> Callable[..., None]:
    """Fixture: run a callable inside the multiprocessing harness.

    The callable receives the harness-constructed ``mesh`` as its first
    argument, followed by any user-supplied positional arguments.

    Example::

        def _body(mesh: DeviceMesh, model: nn.Module) -> None:
            config = sharding.FSDP2Config(mesh=mesh)
            ...

        def test_foo(run_distributed, model: nn.Module) -> None:
            run_distributed(_body, model)
    """

    def _run(fn: Callable[..., None], *args: Any) -> None:
        dist.run_multiprocess(_distributed_wrapper, world_size, fn, args)

    return _run
