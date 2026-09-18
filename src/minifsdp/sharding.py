"""Model sharding."""

from dataclasses import dataclass
from dataclasses import field
import functools
from typing import Any

from torch import nn
from torch.distributed import fsdp
from torch.distributed.device_mesh import DeviceMesh


@dataclass
class FSDP2Config:
    mesh: DeviceMesh | None = None
    target_module_classes: set[type[nn.Module]] | None = None
    reshard_after_forward: bool = True
    mp_policy: fsdp.MixedPrecisionPolicy = field(
        default_factory=fsdp.MixedPrecisionPolicy
    )


def shard_model(model: nn.Module, config: FSDP2Config | None = None) -> nn.Module:
    """Applies FSDP2 fully_shard to the root model."""
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

    for layer in model.children():
        if (
            config.target_module_classes is not None
            and type(layer) in config.target_module_classes
        ):
            shard_fn(layer)
    shard_fn(model)

    return model


def inspect_dtensors(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Extracts sharded parameter metadata.

    Metadata: (is_dtensor, placements, global_shape, local_shape, local_numel).
    """
    raise NotImplementedError
