"""Tests for models.py."""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule

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


def _assert_forward_actor(
    mesh: DeviceMesh, actor: nn.Module, reference: nn.Module, model_type: str
):
    config = sharding.FSDP2Config(mesh=mesh)
    actor_reference = models.ActorReference(actor, reference, config)
    x = torch.rand(2, 4)

    match model_type:
        case "actor":
            out = actor_reference.forward_actor(x)
        case "reference":
            out = actor_reference.forward_reference(x)
        case _:
            raise ValueError("Unsupported model_type: {model_type}")

    assert out.shape == (2, 1)
    match model_type:
        case "actor":
            assert out.grad_fn is not None
        case "reference":
            assert out.grad_fn is None


@pytest.mark.parametrize("model_type", ["actor", "reference"])
def test_forward_actor_reference(
    model_type: str,
    actor: nn.Module,
    reference: nn.Module,
    run_distributed,
):
    run_distributed(_assert_forward_actor, actor, reference, model_type)
