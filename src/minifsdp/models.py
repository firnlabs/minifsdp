"""Model utilities."""

from torch import nn
from torch import Tensor

from minifsdp import sharding


class ActorReference:
    """Dual actor-reference model wrapper."""

    def __init__(
        self,
        actor: nn.Module,
        reference: nn.Module,
        config: sharding.FSDP2Config | None = None,
    ) -> None:
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
        """Runs the forward pass on the actor model."""
        return self.actor(*args, **kwargs)

    def forward_reference(self, *args, **kwargs) -> Tensor:
        """Runs the forward pass on the reference model.

        Ensures that the output is detached.
        """
        out = self.reference(*args, **kwargs)
        return out.detach()
