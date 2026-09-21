"""Tests for sharding.py."""

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from minifsdp import sharding


def _assert_shard_model_default_config(mesh: DeviceMesh, model: nn.Module) -> None:
    config = sharding.FSDP2Config(mesh=mesh)
    sharded = sharding.shard_model(model, config)
    assert isinstance(sharded, FSDPModule)


def test_shard_model_default_config(run_distributed, model: nn.Module) -> None:
    """Test that shard_model wraps a model with FSDP2."""
    run_distributed(_assert_shard_model_default_config, model)


def _assert_dist_inspect_tensors(mesh: DeviceMesh, model: nn.Module) -> None:
    config = sharding.FSDP2Config(mesh=mesh)
    sharded = sharding.shard_model(model, config)
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


def test_inspect_dtensors(run_distributed, model: nn.Module):
    # pre-sharding
    init_stats = sharding.inspect_dtensors(model)
    for (s_name, stats), (p_name, p) in zip(
        init_stats.items(), model.named_parameters(), strict=True
    ):
        assert s_name == p_name
        assert stats.is_dtensor is False
        assert stats.global_shape == p.shape
        assert stats.global_numel == p.numel()
        assert stats.local_shape == p.shape
        assert stats.local_numel == p.numel()
        assert stats.placements is None

    run_distributed(_assert_dist_inspect_tensors, model)
