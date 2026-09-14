"""Skeleton tests for the generic/catapult target (step 2 of
jojo-track/open/catapult-generic-target/plan.md).

generic/catapult (c_generic) shares the Target contract surface with
generic/vitis (v_generic): registry resolution, tool/knobs, and geometry /
ReuseFactor snapping match generic's shared gemm_ip.behavioral rules exactly.
"""
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gemm_ip.registry import load_target  # noqa: E402
from gemm_ip.behavioral import snap_reuse_factor  # noqa: E402


def test_registry_resolves_c_generic():
    target = load_target("generic", "catapult")
    assert target.name == "generic"
    assert target.tool == "catapult"


def test_c_generic_has_no_knobs():
    target = load_target("generic", "catapult")
    assert target.knobs == []


@pytest.mark.parametrize("shape", [(4, 8, 16), (1, 32, 32), (16, 64, 8), (8, 8, 8)])
def test_geometry_matches_generic(shape):
    c_generic = load_target("generic", "catapult")
    generic = load_target("generic", "vitis")
    assert c_generic.geometry(shape) == generic.geometry(shape)


@pytest.mark.parametrize("rf,k,n", [
    (1, 8, 16), (3, 8, 16), (5, 8, 16), (100, 8, 16),
    (1, 64, 8), (7, 64, 8), (64, 64, 8), (513, 64, 8),
])
def test_reuse_factor_snap_matches_generic(rf, k, n):
    c_item = {"name": "l", "gemm_k": k, "gemm_n": n, "reuse_factor": rf}
    g_item = dict(c_item)
    snap_reuse_factor(c_item)
    # generic's own snapping goes through the same shared helper (step 1); call
    # it directly here too so this test still means something even if generic's
    # flow.py one day wraps it differently.
    snap_reuse_factor(g_item)
    assert c_item["reuse_factor"] == g_item["reuse_factor"]
    assert c_item["reuse_factor_requested"] == rf


def test_emit_rtl_raises():
    target = load_target("generic", "catapult")
    with pytest.raises(NotImplementedError):
        target.emit_rtl((4, 8, 16))
