from pathlib import Path

import pytest
from torch.distributed import device_mesh

from minifsdp import dist


@pytest.fixture
def world_size() -> int:
    """Return default world size fixture."""
    return 2


def _mock_fn(
    rank: int, world_size: int, mesh: device_mesh.DeviceMesh, out_dir: str, msg: str
) -> None:
    """Mock worker function that writes rank and message to a file."""
    Path(out_dir, f"rank_{rank}.txt").write_text(f"{rank}:{msg}")


def test_run_multiprocess(world_size: int, tmp_path: Path) -> None:
    """Test run_multiprocess executes worker and passes arguments correctly."""
    msg = "hello"

    dist.run_multiprocess(_mock_fn, world_size, str(tmp_path), msg)

    for rank in range(world_size):
        assert (tmp_path / f"rank_{rank}.txt").read_text() == f"{rank}:{msg}"
