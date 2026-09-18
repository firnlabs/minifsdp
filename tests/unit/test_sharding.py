"""Tests for sharding.py."""

import pytest
from torch import nn
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule

from minifsdp import sharding


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
    return SimpleModel()


# module-level so mp.spawn can pickle it.
def _assert_shard_model_default_config(mesh: DeviceMesh, model: nn.Module) -> None:
    config = sharding.FSDP2Config(mesh=mesh)
    sharded = sharding.shard_model(model, config)
    assert isinstance(sharded, FSDPModule)


def test_shard_model_default_config(run_distributed, model: nn.Module) -> None:
    """Test that shard_model wraps a model with FSDP2."""
    run_distributed(_assert_shard_model_default_config, model)
