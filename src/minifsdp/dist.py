"""Routines for distributed execution."""

import os
import tempfile
from typing import Any
from typing import Callable
from typing import Concatenate
from typing import ParamSpec

from torch import distributed as dist
from torch import multiprocessing as mp
from torch.distributed import device_mesh

P = ParamSpec("P")
type WorldSize = int
type Rank = int
DeviceMesh = device_mesh.DeviceMesh


def _worker_wrapper(
    rank: Rank,
    world_size: WorldSize,
    init_method: str,
    fn: Callable[Concatenate[Rank, WorldSize, DeviceMesh, P], None],
    args: tuple[Any, ...],
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        mesh = device_mesh.init_device_mesh("cpu", (world_size,))
        fn(rank, world_size, mesh, *args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_multiprocess(
    fn: Callable[Concatenate[Rank, WorldSize, DeviceMesh, P], None],
    world_size: WorldSize = 2,
    *args: Any,
) -> None:
    """Spawns distributed worker processes with Gloo backend and DeviceMesh.

    Creates an isolated temporary file rendezvous, initializes a 1D CPU
    DeviceMesh, executes `fn(rank, world_size, mesh, *args)` across spawned
    workers, and guarantees clean process group teardown.

    Args:
        fn: Target function invoked by each rank.
        world_size: Number of concurrent worker processes to spawn.
        *args: Positional arguments forwarded to `fn`.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    init_method = f"file://{temp_file.name}"
    all_args = [world_size, init_method]
    all_args += args
    try:
        mp.spawn(
            _worker_wrapper,
            args=(world_size, init_method, fn, args),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
