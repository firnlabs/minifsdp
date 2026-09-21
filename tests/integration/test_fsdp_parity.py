"""Tests FSDP model parity with single-process model."""

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from minifsdp import dist
from minifsdp import sharding
from tests.conftest import DeepModel
from tests.conftest import DeepModelBlock


def _parity_worker(
    rank: dist.Rank, world_size: dist.WorldSize, mesh: DeviceMesh
) -> None:
    random_seed = 42  # initial FSDP and single-process models with identical seeds
    torch.manual_seed(random_seed)
    model_dim = 4
    num_layers = 10
    fsdp_model = DeepModel(dim=model_dim, num_layers=num_layers)
    config = sharding.FSDP2Config(
        mesh=mesh,
        target_module_classes={
            DeepModelBlock,
        },
    )
    sharded_model = sharding.shard_model(fsdp_model, config)

    batch_size = 8
    torch.manual_seed(random_seed)
    x_global = torch.randn(batch_size, model_dim)
    y_global = x_global.sum(-1, keepdim=True)
    learning_rate = 0.01

    # FSDP model shard training step
    local_batch = batch_size // world_size
    x_local = x_global[rank * local_batch : (rank + 1) * local_batch]
    y_local = y_global[rank * local_batch : (rank + 1) * local_batch]

    sharded_opt = torch.optim.AdamW(sharded_model.parameters(), lr=learning_rate)
    sharded_criterion = nn.MSELoss()
    sharded_opt.zero_grad()
    sharded_y_pred = sharded_model(x_local)
    sharded_loss = sharded_criterion(sharded_y_pred, y_local)
    sharded_loss.backward()
    sharded_opt.step()

    # full_tensor() needs to be run on all ranks to collect all the data needed
    f_params = {}
    for n, p in sharded_model.named_parameters():
        assert isinstance(p.data, DTensor)
        assert isinstance(p.grad, DTensor)
        f_params[n] = (p.data.full_tensor(), p.grad.full_tensor())

    if rank == 0:  # run the parity test on one worker only
        # train single-process model as reference
        torch.manual_seed(random_seed)
        single_model = DeepModel(dim=model_dim, num_layers=num_layers)
        single_opt = torch.optim.AdamW(single_model.parameters(), lr=learning_rate)
        single_criterion = nn.MSELoss()
        single_opt.zero_grad()
        single_y_pred = single_model(x_global)
        single_loss = single_criterion(single_y_pred, y_global)
        single_loss.backward()
        single_opt.step()

        # assertions
        for (s_name, s_param), (f_name, (f_data, f_grad)) in zip(
            single_model.named_parameters(), f_params.items(), strict=True
        ):
            assert s_name == f_name
            torch.testing.assert_close(s_param.data, f_data, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(s_param.grad, f_grad, rtol=1e-4, atol=1e-4)


def test_fsdp_parity(world_size: int):
    dist.run_multiprocess(_parity_worker, world_size)
