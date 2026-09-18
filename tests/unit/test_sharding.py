"""Tests for sharding.py."""

import pytest
from torch import nn
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule

from minifsdp import sharding
from tests.unit.conftest import distributed


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._fc1 = nn.Linear(4, 8, bias=False)
        self._fc2 = nn.Linear(8, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self._fc1(x)
        return self._fc2(x)


@pytest.fixture
def model() -> nn.Module:
    return SimpleModel()


@distributed(world_size=2)
def test_shard_model_default_config(
    rank: int, world_size: int, mesh: DeviceMesh, model: nn.Module
):
    _ = rank  # unused
    _ = world_size  # unused
    config = sharding.FSDP2Config(mesh=mesh)
    sharded = sharding.shard_model(model, config)

    assert isinstance(sharded, FSDPModule)
