# miniFSDP: Composable layer-wise sharding with PyTorch FSDP2

[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](#testing)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A lightweight distributed training library providing modular abstractions over PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard`), layer-wise sharding policies, DTensor parameter inspection, and CPU-based testing harnesses.

---

## Getting started

Prerequisites: Python 3.14+ and [uv](https://github.com/astral-sh/uv).

Install dependencies:

```bash
uv sync
```

### Profiling sharded vs unsharded model

Run the distributed memory profiling script to inspect DTensor parameter sharding and peak memory reduction across ranks:

```bash
uv run python benchmarks/fsdp_profile_memory.py
```

---

## Directory structure

```text
.
├── src/minifsdp/
│   ├── dist.py                 # CPU-based distributed running harness
│   ├── models.py               # Actor-reference wrapper for RL alignment workloads
│   ├── sharding.py             # FSDP2 layer-wise wrapping and DTensor inspection
├── benchmarks/
│   └── fsdp_profile_memory.py  # Memory profiling and sharding reduction benchmark
├── tests/                      # Unit and integration tests
├── pyproject.toml              # uv Python configuration
└── uv.lock                     # uv lock file
```

---

## Testing

Run the complete test suite:

```bash
uv run pytest
```

---

## Contributing

Contributions, feedback, and bug reports are welcome. To contribute:

1. Fork the repository and create a feature branch (e.g. `git checkout -b feat/my-feature`).
2. Verify all tests pass and linting succeeds (`uv run pytest && uv run ruff check`).
3. Submit a pull request describing the rationale, changes made, and invariants preserved.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
