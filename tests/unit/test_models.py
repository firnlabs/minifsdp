import copy

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.special import Tensor

from minifsdp import models
from minifsdp import sharding


@pytest.fixture
def actor(simple_model) -> nn.Module:
    return simple_model


@pytest.fixture
def reference(simple_model) -> nn.Module:
    return copy.deepcopy(simple_model)


def _assert_init(mesh: DeviceMesh, actor: nn.Module, reference: nn.Module):
    config = sharding.FSDP2Config(mesh=mesh)
    actor_reference = models.ActorReference(actor, reference, config)
    assert isinstance(actor_reference.actor, nn.Module)
    assert isinstance(actor_reference.reference, nn.Module)
    assert isinstance(actor_reference.actor, FSDPModule)
    assert isinstance(actor_reference.reference, FSDPModule)
    for p in actor_reference.actor.parameters():
        assert p.requires_grad

    for p in actor_reference.reference.parameters():
        assert not p.requires_grad
        assert p.grad is None


def test_init(
    actor: nn.Module,
    reference: nn.Module,
    run_distributed,
):
    run_distributed(_assert_init, actor, reference)


def _assert_forward_actor(mesh: DeviceMesh, actor: nn.Module, reference: nn.Module):
    config = sharding.FSDP2Config(mesh=mesh)
    actor_reference = models.ActorReference(actor, reference, config)
    x = torch.rand(2, 4)

    out = actor_reference.forward_actor(x)

    assert out.shape == (2, 1)
    assert out.grad_fn is not None


def test_forward_actor(
    actor: nn.Module,
    reference: nn.Module,
    run_distributed,
):
    run_distributed(_assert_forward_actor, actor, reference)


def _assert_forward_reference(mesh: DeviceMesh, actor: nn.Module, reference: nn.Module):
    config = sharding.FSDP2Config(mesh=mesh)
    actor_reference = models.ActorReference(actor, reference, config)
    # enable requires_grad to test torch.no_grad behavior in forward_reference
    x = torch.rand(2, 4, requires_grad=True)

    # verify torch.is_grad_enabled() is False during forward pass (via torch.no_grad)
    grad_enabled: list[bool] = []

    actor_reference.reference.register_forward_hook(
        lambda *args, **kwargs: grad_enabled.append(torch.is_grad_enabled())
    )

    # intercept any tensors saved for backward pass
    saved_tensors: list[Tensor] = []

    def pack(tensor: Tensor) -> Tensor:
        saved_tensors.append(tensor)
        return tensor

    def unpack(tensor: Tensor) -> Tensor:
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = actor_reference.forward_reference(x)

    assert out.shape == (2, 1)
    assert out.grad_fn is None
    assert grad_enabled[0] is False
    assert len(saved_tensors) == 0


def test_forward_reference(
    actor: nn.Module,
    reference: nn.Module,
    run_distributed,
):
    run_distributed(_assert_forward_reference, actor, reference)
