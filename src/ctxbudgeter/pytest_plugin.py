"""pytest plugin for ctxbudgeter.

Activated automatically when ctxbudgeter is installed (via the ``pytest11`` entry
point in pyproject.toml). Provides:

- ``--ctxbudgeter-update-golden`` flag — refresh golden snapshots in this run
- ``ctxbudgeter_golden`` fixture — convenient GoldenPack constructor for the
  current test, defaulting to ``<test_dir>/golden/<test_name>.json``

Use like this::

    def test_my_pack(ctxbudgeter_golden):
        compiled = build_pack().compile()
        ctxbudgeter_golden().check(compiled)        # default path
        ctxbudgeter_golden("special.json").check(c) # override
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from .testing import GoldenPack


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ctxbudgeter-update-golden",
        action="store_true",
        default=False,
        help="Update ctxbudgeter golden snapshots in this run.",
    )


@pytest.fixture
def ctxbudgeter_update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--ctxbudgeter-update-golden"))


@pytest.fixture
def ctxbudgeter_golden(request: pytest.FixtureRequest, ctxbudgeter_update_golden: bool) -> Callable[..., GoldenPack]:
    """Factory that constructs a GoldenPack for the current test.

    Default location: ``<test_file_dir>/golden/<test_function_name>.json``.
    """
    test_path = Path(request.node.fspath)
    test_dir = test_path.parent
    default_name = f"{request.node.name}.json"

    def make(path: str | Path | None = None) -> GoldenPack:
        if path is None:
            full_path = test_dir / "golden" / default_name
        else:
            p = Path(path)
            full_path = p if p.is_absolute() else (test_dir / p)
        return GoldenPack(full_path, update=ctxbudgeter_update_golden)

    return make
