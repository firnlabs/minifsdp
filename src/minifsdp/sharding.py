"""PyTorch FSDP2 model sharding and distributed parameter inspection."""

from dataclasses import dataclass
from dataclasses import field
import functools
from typing import Any

from torch import nn
from torch.distributed import fsdp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor


@dataclass
class FSDP2Config:
    """Configuration options for PyTorch FSDP2 model sharding.

    Attributes:
        mesh: 1D or 2D device mesh topology for distributed execution.
        target_module_classes: Specific module types to shard individually before
            root-level sharding.
        reshard_after_forward: If True, frees gathered full parameters after forward
            pass to minimize memory footprint.
        mp_policy: Mixed precision communication and computation policy.
    """

    mesh: DeviceMesh | None = None
    target_module_classes: set[type[nn.Module]] | None = None
    reshard_after_forward: bool = True
    mp_policy: fsdp.MixedPrecisionPolicy = field(
        default_factory=fsdp.MixedPrecisionPolicy
    )


def shard_model(model: nn.Module, config: FSDP2Config | None = None) -> nn.Module:
    """Applies PyTorch FSDP2 `fully_shard` to target submodules and the root model.

    Mutates modules in-place by attaching FSDP communication hooks to any submodules
    matching `target_module_classes`, followed by the root module.

    Args:
        model: Target module to shard.
        config: FSDP2 configuration settings.

    Returns:
        The sharded `FSDPModule` root model.

    Raises:
        ValueError: If `model` is already an `FSDPModule`.
    """
    if isinstance(model, fsdp.FSDPModule):
        raise ValueError("model is already FSDP sharded.")

    if config is None:
        config = FSDP2Config()

    shard_fn = functools.partial(
        fsdp.fully_shard,
        mesh=config.mesh,
        reshard_after_forward=config.reshard_after_forward,
        mp_policy=config.mp_policy,
    )

    if config.target_module_classes is not None:
        for layer in reversed(list(model.modules())):
            if type(layer) in config.target_module_classes and not isinstance(
                layer, FSDPModule
            ):
                shard_fn(layer)

    shard_fn(model)

    return model


@dataclass(frozen=True)
class ParamStats:
    """Metadata and local/global shape statistics for a model parameter."""

    is_dtensor: bool
    global_shape: tuple[int, ...]
    global_numel: int
    local_shape: tuple[int, ...]
    local_numel: int
    placements: Any


def inspect_dtensors(model: nn.Module) -> dict[str, ParamStats]:
    """Extracts distributed tensor metadata and shard statistics for all parameters.

    Args:
        model: Module containing parameters to inspect.

    Returns:
        Mapping of parameter names to `ParamStats` records.
    """
    stats = {}

    for n, p in model.named_parameters():
        is_dtensor = isinstance(p, DTensor)
        local_p = p.to_local() if is_dtensor else p
        placements = p.placements if is_dtensor and hasattr(p, "placements") else None
        p_stats = ParamStats(
            is_dtensor=is_dtensor,
            global_shape=p.shape,
            global_numel=p.numel(),
            local_shape=local_p.shape,
            local_numel=local_p.numel(),
            placements=placements,
        )
        stats[n] = p_stats

    return stats
