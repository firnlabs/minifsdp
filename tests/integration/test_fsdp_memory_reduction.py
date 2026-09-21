"""Tests parameter memory reduction per process through FSDP."""

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from minifsdp import dist
from minifsdp import sharding
from tests.conftest import DeepModel
from tests.conftest import DeepModelBlock


def _memory_profile_worker(
    rank: dist.Rank, world_size: dist.WorldSize, mesh: DeviceMesh
):
    model_dim = 16
    num_layers = 2
    fsdp_model = DeepModel(dim=model_dim, num_layers=num_layers)

    config = sharding.FSDP2Config(
        mesh=mesh,
        target_module_classes={
            DeepModelBlock,
        },
    )

    sharded_model = sharding.shard_model(fsdp_model, config)

    single_model = DeepModel(dim=model_dim, num_layers=num_layers)

    for (f_name, f_param), (s_name, s_param) in zip(
        sharded_model.named_parameters(), single_model.named_parameters(), strict=True
    ):
        assert f_name == s_name
        s_param_count = s_param.numel()
        if "blocks" in f_name:
            assert isinstance(f_param, DTensor)
            f_param_count = f_param.to_local().numel()
            exp_param_count = s_param_count // world_size
        else:
            # layer is not sharded (i.e. `head`)
            assert "head" in f_name
            f_param_count = f_param.numel()
            exp_param_count = s_param_count
        assert f_param_count == exp_param_count, (
            f"rank {rank}, param {f_name}: {f_param_count=} vs. {exp_param_count=}"
        )


def test_fsdp_memory_profile(world_size: dist.WorldSize):
    dist.run_multiprocess(_memory_profile_worker, world_size)
