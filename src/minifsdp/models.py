"""Model abstractions and paired actor-reference architectures."""

import torch
from torch import nn
from torch import Tensor

from minifsdp import sharding


class ActorReference:
    """Paired actor and frozen reference model wrapper for RL post-training.

    Both modules are sharded via PyTorch FSDP2. The reference module is
    automatically set to evaluation mode (`eval()`) with frozen parameters
    (`requires_grad = False`).
    """

    def __init__(
        self,
        actor: nn.Module,
        reference: nn.Module,
        config: sharding.FSDP2Config | None = None,
    ) -> None:
        """Initializes and shards actor and reference models.

        Args:
            actor: Trainable policy module.
            reference: Frozen reference policy module.
            config: Optional FSDP2 sharding configuration.
        """
        actor = sharding.shard_model(actor, config)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad = False
        reference = sharding.shard_model(reference, config)
        self._actor = actor
        self._reference = reference

    @property
    def actor(self) -> nn.Module:
        return self._actor

    @property
    def reference(self) -> nn.Module:
        return self._reference

    def forward_actor(self, *args, **kwargs) -> Tensor:
        """Executes forward pass through the trainable actor model."""
        return self.actor(*args, **kwargs)

    @torch.no_grad()
    def forward_reference(self, *args, **kwargs) -> Tensor:
        """Executes forward pass through reference model, returning detached output."""
        out = self.reference(*args, **kwargs)
        return out.detach()
