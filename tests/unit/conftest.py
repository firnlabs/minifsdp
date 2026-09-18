<<<<<<< HEAD
"""Unit-test conftest: distributed test helpers."""

from collections.abc import Callable
from typing import Any

import pytest

from minifsdp import dist


def _distributed_wrapper(
    rank: int,
    world_size: int,
    mesh: dist.DeviceMesh,
    fn: Callable[..., None],
    args: tuple[Any, ...],
) -> None:
    """Module-level worker: unpickles fn and args, then calls fn(mesh, *args).

    Must be defined at module scope so that ``mp.spawn`` (which uses pickle) can
    serialise it.  The harness-constructed ``mesh`` is forwarded as the first argument
    so the callable can use it if needed.
    """
    fn(mesh, *args)


@pytest.fixture
def run_distributed() -> Callable[..., None]:
    """Fixture: run a callable inside the multiprocessing harness.

    The callable receives the harness-constructed ``mesh`` as its first
    argument, followed by any user-supplied positional arguments.

    Example::

        def _body(mesh: DeviceMesh, model: nn.Module) -> None:
            config = sharding.FSDP2Config(mesh=mesh)
            ...

        def test_foo(run_distributed, model: nn.Module) -> None:
            run_distributed(_body, model)
    """

    def _run(fn: Callable[..., None], *args: Any, world_size: int = 2) -> None:
        dist.run_multiprocess(_distributed_wrapper, world_size, fn, args)

    return _run
=======
"""Unit-test conftest: distributed test decorator."""

from collections.abc import Callable
import functools
import inspect
import sys
import types
from typing import Any
from typing import cast

from minifsdp import dist

# Parameters injected by the dist harness; invisible to pytest.
_HARNESS_PARAMS = frozenset({"rank", "world_size", "mesh"})

# A plain Python function (has __name__, __qualname__, __signature__, …).
_Fn = types.FunctionType


def distributed(world_size: int = 2) -> Callable[[_Fn], _Fn]:
    """Decorator: run a test body inside ``dist.run_multiprocess``.

    Write the test function with its full signature — harness params
    (``rank``, ``world_size``, ``mesh``) plus any pytest fixtures.  The
    decorator strips the harness params from the visible signature so pytest
    injects only the fixtures, then calls ``dist.run_multiprocess`` with the
    injected fixture values at run time.

    Example::

        @distributed(world_size=2)
        def test_func(rank: int, world_size: int, mesh: DeviceMesh, model: nn.Module):
            ...
    """

    def decorator(fn: _Fn) -> _Fn:
        sig = inspect.signature(fn)
        fixture_params = [
            p for name, p in sig.parameters.items() if name not in _HARNESS_PARAMS
        ]
        fixture_names = [p.name for p in fixture_params]

        # mp.spawn pickles functions by (module, qualname). After decoration
        # the original name in the module points to the wrapper, so we
        # register fn under a mangled private name and patch its identity to
        # match, giving pickle a valid round-trip.
        module = sys.modules[fn.__module__]
        mangled = f"_distributed__{fn.__name__}"
        setattr(module, mangled, fn)
        fn.__name__ = mangled
        fn.__qualname__ = mangled

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            # Pytest may call the wrapper with keyword args; normalise to
            # positional so they can be forwarded through *args.
            bound = dict(zip(fixture_names, args, strict=False))
            bound.update(kwargs)
            positional = [bound[name] for name in fixture_names]
            dist.run_multiprocess(fn, world_size, *positional)

        # functools.wraps returns _Wrapped; cast to FunctionType so ty knows
        # that __name__, __qualname__, and __signature__ are all writable.
        fn_wrapper = cast(_Fn, wrapper)

        # Restore a human-readable name on the wrapper for pytest output.
        fn_wrapper.__name__ = fn.__name__.removeprefix("_distributed__")
        fn_wrapper.__qualname__ = fn.__qualname__.removeprefix("_distributed__")
        fn_wrapper.__signature__ = inspect.Signature(fixture_params)  # ty: ignore[unresolved-attribute]
        return fn_wrapper

    return decorator
>>>>>>> 069ea59 (test: add distributed decorator)
