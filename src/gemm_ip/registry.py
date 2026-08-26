"""Target registry for gemm-ip-gen.

Connects the thin framework core to the available hardblock targets. Under the
one-tool-per-hardblock model there is a single target today, ``tensor_slice``
(tool: Catapult). This module owns the target enumeration and the loaders that
reach into the active target's generator modules by putting the target directory
on ``sys.path`` and importing by module name.
"""

import sys as _sys
from pathlib import Path


# Registered targets (one tool per hardblock; tensor_slice's tool is Catapult).
# Adding a hardblock = one new directory under src/targets/ plus its name here.
TARGETS = ("tensor_slice",)


def _target_dir(target):
    return str(Path(__file__).resolve().parent.parent / "targets" / target)


def _tensor_slice_dir():
    return _target_dir("tensor_slice")


def _on_path(target):
    tdir = _target_dir(target)
    if tdir not in _sys.path:
        _sys.path.insert(0, tdir)
    return tdir


def _targets_root():
    return str(Path(__file__).resolve().parent.parent / "targets")


def load_target(target="tensor_slice"):
    """Import and return the Target object for *target* (see targets/base.py)."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}'; known targets: {TARGETS}")
    troot = _targets_root()
    if troot not in _sys.path:
        _sys.path.insert(0, troot)  # so `flow` can import `base`
    _on_path(target)                # so `flow` can import its siblings
    import flow  # noqa: F401 — the active target's flow.py exposes TARGET
    return flow.TARGET


def load_target_package(target="tensor_slice"):
    """Import and return the package-generation module for *target*."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}'; known targets: {TARGETS}")
    _on_path(target)
    import package  # noqa: F401 — the active target's package.py
    return package


def load_catapult_rtl_generator():
    """Import generate_grid_verilog from targets/tensor_slice/rtl.py."""
    _on_path("tensor_slice")
    from rtl import generate_grid_verilog  # noqa: F811
    return generate_grid_verilog
