from __future__ import annotations

from pathlib import Path
import sys

from setuptools import Extension, setup

try:
    import numpy
except ImportError as exc:  # pragma: no cover - build-time guard
    raise SystemExit("numpy is required to build the native river solver.") from exc

try:
    import pybind11
except ImportError as exc:  # pragma: no cover - build-time guard
    raise SystemExit(
        "pybind11 is required to build the native river solver. "
        "Install it first, e.g. `./poker-rl/bin/pip install pybind11`."
    ) from exc


ROOT = Path(__file__).resolve().parent


def extension() -> Extension:
    extra_compile_args = ["/std:c++17", "/O2"] if sys.platform == "win32" else ["-std=c++17", "-O3"]
    return Extension(
        "hulhe_bot._native_river",
        sources=[str(ROOT / "native" / "river_solver.cpp")],
        include_dirs=[pybind11.get_include(), numpy.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )


setup(
    name="hulhe-bot-native-river",
    version="0.1.0",
    description="Optional native exact river solver for hulhe_bot",
    ext_modules=[extension()],
)
