"""Demonstrates memory profiling of an FSDP sharded model against a single model."""

from typing import Any

from torch import distributed as torch_dist
from torch import nn
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from minifsdp import dist
from minifsdp import sharding


class Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        return self.fc2(x)


class MultiBlockModel(nn.Module):
    def __init__(self, *, dim: int, num_layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim=dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _profile_worker(
    rank: dist.Rank,
    world_size: dist.WorldSize,
    mesh: DeviceMesh,
    model: MultiBlockModel,
    target_module_classes: set[type[nn.Module]] | None = None,
) -> None:
    config = sharding.FSDP2Config(
        mesh=mesh, target_module_classes=target_module_classes
    )
    sharded_model = sharding.shard_model(model, config)

    global_param_count = sum((p.numel() for p in model.parameters()))
    block_param_count = sum((p.numel() for p in model.blocks[0].parameters()))
    param = next(model.parameters())
    bytes_per_elem = param.element_size()
    dtype = param.dtype
    local_param_count = sum(
        p.to_local().numel() if isinstance(p, DTensor) else p.numel()
        for p in sharded_model.parameters()
    )
    total_param_bytes = global_param_count * bytes_per_elem
    sharding_ratio = local_param_count / global_param_count
    block_layer_peak_count = local_param_count + block_param_count
    peak_memory_reduction = global_param_count / block_layer_peak_count

    rank_stats = {
        "rank": rank,
        "local_param_count": local_param_count,
        "sharding_ratio": sharding_ratio,
        "block_layer_peak_count": block_layer_peak_count,
        "peak_memory_reduction": peak_memory_reduction,
    }

    gathered_stats: list[dict[str, Any] | None] = [None] * world_size
    torch_dist.all_gather_object(gathered_stats, rank_stats)

    if rank == 0:
        print(
            f"Total global parameters: {global_param_count}"
            f"\nTotal parameter memory: {total_param_bytes / 1024:.2f} KB in {dtype}"
        )
        for stats in gathered_stats:
            assert stats is not None
            print(
                f"\nRank {stats['rank']}:"
                f"\n\tLocal parameters: {stats['local_param_count']}"
                f"\n\tSharding ratio: {stats['sharding_ratio']:.2f}"
                "\n\tPeak forward param overhead (1/N*Phi + Phi_block):"
                f" {stats['block_layer_peak_count']}"
                f"\n\tPeak memory reduction: {stats['peak_memory_reduction']:.2f}"
            )


def main():
    world_size = 2
    model_dim = 16
    num_layers = 4
    model = MultiBlockModel(dim=model_dim, num_layers=num_layers)
    target_module_classes = {
        Block,
    }

    print(f"Config: {world_size=}, {model_dim=}, {num_layers=}")
    dist.run_multiprocess(_profile_worker, world_size, model, target_module_classes)


if __name__ == "__main__":
    main()
