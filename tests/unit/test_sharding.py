"""Tests for sharding.py."""

import pytest
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from minifsdp import sharding
from tests.conftest import DeepModel
from tests.conftest import DeepModelBlock


def _assert_shard_model(
    mesh: DeviceMesh, model: DeepModel, target_module_classes
) -> None:
    config = sharding.FSDP2Config(
        mesh=mesh, target_module_classes=target_module_classes
    )

    sharded = sharding.shard_model(model, config)

    assert isinstance(sharded, FSDPModule)
    assert isinstance(sharded, DeepModel)

    for block in sharded.blocks:
        if target_module_classes is None:
            assert not isinstance(block, FSDPModule)
        else:
            assert isinstance(block, FSDPModule)
    assert not isinstance(model.head, FSDPModule)


@pytest.mark.parametrize(
    "target_module_classes",
    [
        None,
        {
            DeepModelBlock,
        },
    ],
)
def test_shard_model(target_module_classes, run_distributed, deep_model: DeepModel):
    run_distributed(_assert_shard_model, deep_model, target_module_classes)


def _assert_dist_inspect_tensors(mesh: DeviceMesh, simple_model: nn.Module) -> None:
    config = sharding.FSDP2Config(mesh=mesh)
    sharded = sharding.shard_model(simple_model, config)
    assert isinstance(sharded, FSDPModule)

    sharded_stats = sharding.inspect_dtensors(sharded)
    for (s_name, stats), (p_name, p) in zip(
        sharded_stats.items(), sharded.named_parameters(), strict=True
    ):
        assert isinstance(p, DTensor)
        local_p = p.to_local()
        assert s_name == p_name
        assert stats.is_dtensor is True
        assert stats.global_shape == p.shape
        assert stats.global_numel == p.numel()
        assert stats.local_shape == tuple(local_p.shape)
        assert stats.local_numel == local_p.numel()
        assert stats.placements == p.placements
        assert stats.placements is not None


def test_inspect_dtensors(run_distributed, simple_model: nn.Module):
    # pre-sharding
    init_stats = sharding.inspect_dtensors(simple_model)
    for (s_name, stats), (p_name, p) in zip(
        init_stats.items(), simple_model.named_parameters(), strict=True
    ):
        assert s_name == p_name
        assert stats.is_dtensor is False
        assert stats.global_shape == p.shape
        assert stats.global_numel == p.numel()
        assert stats.local_shape == p.shape
        assert stats.local_numel == p.numel()
        assert stats.placements is None

    run_distributed(_assert_dist_inspect_tensors, simple_model)
