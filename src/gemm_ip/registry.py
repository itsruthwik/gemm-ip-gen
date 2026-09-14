"""Target registry for gemm-ip-gen.

Connects the thin framework core to the available hardblock targets. Each target
is a subpackage under ``src/targets/`` with a plain, importable name and normal
relative imports internally -- no ``sys.path`` tricks, no ``sys.modules``
bookkeeping. This module owns the target enumeration and the (target, tool)
resolution: there is exactly one user-facing ``generic`` target name, backed by
two concrete implementations (``v_generic`` for Vitis, ``c_generic`` for
Catapult); ``tensor_slice`` and ``mvau`` are each welded to a single tool.
"""

import importlib


# User-facing target names.
TARGETS = ("tensor_slice", "generic", "mvau")

# (target, tool) -> the concrete implementation package under src/targets/.
_IMPLS = {
    ("generic", "vitis"): "v_generic",
    ("generic", "catapult"): "c_generic",
    ("tensor_slice", "catapult"): "tensor_slice",
    ("mvau", "vitis"): "mvau",
}

# The tool a target resolves to when the caller doesn't specify one.
_DEFAULT_TOOL = {
    "generic": "vitis",
    "tensor_slice": "catapult",
    "mvau": "vitis",
}


def _resolve_impl(target, tool):
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}'; known targets: {TARGETS}")
    if tool is None:
        tool = _DEFAULT_TOOL[target]
    impl = _IMPLS.get((target, tool))
    if impl is None:
        supported = sorted(t for (trg, t) in _IMPLS if trg == target)
        raise ValueError(
            f"target '{target}' does not support tool '{tool}'; "
            f"'{target}' supports: {supported}")
    return impl, tool


def supported_tools(target):
    """Return the tuple of tools *target* supports."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}'; known targets: {TARGETS}")
    return tuple(sorted(t for (trg, t) in _IMPLS if trg == target))


def load_target(target="generic", tool=None):
    """Import and return the Target object for (*target*, *tool*) (see targets/base.py)."""
    impl, _tool = _resolve_impl(target, tool)
    flow = importlib.import_module(f"targets.{impl}.flow")
    return flow.TARGET


def load_target_package(target="generic", tool=None):
    """Import and return the package-generation module for (*target*, *tool*)."""
    impl, _tool = _resolve_impl(target, tool)
    return importlib.import_module(f"targets.{impl}.package")


def load_catapult_rtl_generator():
    """Import generate_grid_verilog from targets/tensor_slice/rtl.py."""
    from targets.tensor_slice.rtl import generate_grid_verilog
    return generate_grid_verilog
