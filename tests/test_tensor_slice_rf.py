"""Tests for tensor_slice ReuseFactor (RF) legalization and the pass-2
package.py plumbing built on top of it.

RF is purely "number of passes over K" -- never a cycle count or an
initiation interval. See geometry.resolve_reuse_factor / package
pass 2 for the full model.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SCRATCH_ROOT = Path("/mnt/vault0/rsunketa/atlas/temp_space/ts-rf/pytest")


def _load_tensor_slice_module(modname):
    """Load a tensor_slice sibling module under a unique name, sidestepping
    the geometry/package name collision with other targets on sys.path
    (see test_tensor_slice_operand_guard.py's loader for the same trick)."""
    tdir = str(_SRC / "targets" / "tensor_slice")
    saved_path = list(sys.path)
    saved_modules = {k: sys.modules.get(k) for k in ("geometry", "package", "rtl", "golden")}
    sys.path.insert(0, tdir)
    for stale in ("geometry", "package", "rtl", "golden"):
        sys.modules.pop(stale, None)
    try:
        spec = importlib.util.spec_from_file_location(f"_ts_{modname}", Path(tdir) / f"{modname}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = saved_path
        for stale, prev in saved_modules.items():
            if prev is None:
                sys.modules.pop(stale, None)
            else:
                sys.modules[stale] = prev


_geom = _load_tensor_slice_module("geometry")
_pkg = _load_tensor_slice_module("package")

resolve_reuse_factor = _geom.resolve_reuse_factor
k_chunks = _geom.k_chunks
multipliers = _geom.multipliers
latency_first_out = _geom.latency_first_out
generate_catapult_pkg = _pkg.generate_catapult_pkg


@pytest.fixture
def scratch_dir(tmp_path):
    """A unique scratch dir under temp_space/ts-rf/pytest, cleaned up after."""
    d = _SCRATCH_ROOT / tmp_path.name
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# k=24 -> k_chunks = ceil(24/8) = 3
K24_CHUNKS = 3


def test_resolve_reuse_factor_legal_set_k24():
    for rf in range(1, K24_CHUNKS + 1):
        resolved = resolve_reuse_factor(24, rf)
        assert resolved["k_chunks"] == K24_CHUNKS
        assert resolved["passes"] == resolved["reuse_factor"] == resolved["effective_reuse"]
        assert resolved["warnings"] == []
        assert resolved["k_spatial"] * resolved["passes"] >= K24_CHUNKS


def test_rf1_gives_full_k_spatial():
    resolved = resolve_reuse_factor(24, 1)
    assert resolved["k_spatial"] == K24_CHUNKS
    assert resolved["passes"] == 1
    assert resolved["reuse_factor"] == 1


def test_rf_at_kchunks_gives_chunked():
    resolved = resolve_reuse_factor(24, K24_CHUNKS)
    assert resolved["k_spatial"] == 1
    assert resolved["passes"] == K24_CHUNKS
    assert resolved["reuse_factor"] == K24_CHUNKS


def test_rf_out_of_range_warns_and_clamps():
    resolved = resolve_reuse_factor(24, K24_CHUNKS + 5, name="mylayer")
    assert len(resolved["warnings"]) == 1
    msg = resolved["warnings"][0]
    assert msg == (
        f"WARNING: Invalid ReuseFactor={K24_CHUNKS + 5} for layer mylayer. "
        f"Using ReuseFactor={K24_CHUNKS} instead. Valid ReuseFactor(s): 1..{K24_CHUNKS}."
    )
    assert resolved["reuse_factor"] == K24_CHUNKS
    assert resolved["k_spatial"] == 1


def test_silent_lower_landing_k784_rf50():
    resolved = resolve_reuse_factor(784, 50)
    assert resolved["k_chunks"] == 98
    assert resolved["k_spatial"] == 2
    assert resolved["passes"] == 49
    assert resolved["warnings"] == []


def test_padding_arithmetic_k24_rf2():
    resolved = resolve_reuse_factor(24, 2)
    assert resolved["k_spatial"] == 2
    assert resolved["passes"] == 2
    assert resolved["k_chunks_pad"] == 4


def test_multipliers_sanity():
    # 64 INT8 MACs per row-tile x col-tile x k_spatial partition.
    assert multipliers(8, 8, 1) == 64
    assert multipliers(16, 8, 1) == 128
    assert multipliers(8, 8, 3) == 192


def test_latency_first_out_endpoints_match_old_formulas():
    m, k, n = 8, 24, 8
    kc = K24_CHUNKS
    input_beats = max(m, n)
    # Chunked endpoint (k_spatial == 1): old formula was
    # k_chunks * max(M,N) + max(0, K+N - k_chunks*max(M,N)).
    total_beats_chunked = kc * input_beats
    expected_chunked = total_beats_chunked + max(0, k + n - total_beats_chunked)
    assert latency_first_out(m, k, n, 1) == expected_chunked
    # Full-K endpoint (k_spatial == k_chunks, passes == 1): old formula was
    # max(M,N) + max(0, K+N - max(M,N)).
    expected_full_k = input_beats + max(0, k + n - input_beats)
    assert latency_first_out(m, k, n, kc) == expected_full_k


def _random_weight_matrix(k, n, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    return rng.randint(-64, 64, size=(k, n)).astype(int)


def _with_tensor_slice_on_path(fn, *args, **kwargs):
    """Run *fn* with the tensor_slice dir first on sys.path and its sibling
    module names (geometry/package/rtl/golden) cleared, so gemm_ip.weights'
    ``from golden import ...`` (etc.) resolves to tensor_slice's own siblings
    instead of whatever other target's same-named module another test left in
    sys.modules (the tensor_slice/mvau collision noted in
    test_tensor_slice_operand_guard.py's loader)."""
    tdir = str(_SRC / "targets" / "tensor_slice")
    saved_path = list(sys.path)
    saved_modules = {k: sys.modules.get(k) for k in ("geometry", "package", "rtl", "golden")}
    sys.path.insert(0, tdir)
    for stale in ("geometry", "package", "rtl", "golden"):
        sys.modules.pop(stale, None)
    try:
        return fn(*args, **kwargs)
    finally:
        sys.path[:] = saved_path
        for stale, prev in saved_modules.items():
            if prev is None:
                sys.modules.pop(stale, None)
            else:
                sys.modules[stale] = prev


@pytest.mark.parametrize("reuse_factor,expect_replay_size", [
    (2, 2),
    (1, 1),
    (3, 3),
])
def test_generate_catapult_pkg_reuse_factor_weight_stationary(scratch_dir, reuse_factor, expect_replay_size):
    m, k, n = 8, 24, 8
    weight_matrix = _random_weight_matrix(k, n)
    name = f"rf{reuse_factor}_layer"
    _with_tensor_slice_on_path(
        generate_catapult_pkg,
        m, k, n, name, str(scratch_dir),
        interface="stream", output_precision="fixed<10,4>",
        reuse_factor=reuse_factor,
        input_precision="fixed<8,2>", weight_precision="fixed<8,2>",
        weight_matrix=weight_matrix,
    )
    pkg_dir = scratch_dir / name
    header = (pkg_dir / f"{name}_gemm_ip.h").read_text()
    assert (pkg_dir / f"{name}_core.v").is_file()
    # passes == expect_replay_size at k=24 (k_chunks=3) for rf in {1,2,3}:
    #   rf=1 -> passes=1, rf=2 -> passes=2, rf=3 -> passes=3.
    # Slot 0 (fed directly from the stream) is never stored, so the buffer
    # holds only the passes-1 replayed passes, each of the m A rows wide.
    # At passes == 1 (full-K, rf=1) there is nothing to replay: no a_replay
    # buffer is declared at all.
    if expect_replay_size == 1:
        assert "a_replay" not in header
    else:
        assert f"a_replay[{expect_replay_size - 1}][{m}]" in header


def test_generate_catapult_pkg_manifest_fields_via_flow(scratch_dir):
    """The batch manifest path (flow.normalize_config + gen_integration_manifest)
    carries the new RF fields end to end."""
    import flow as _flow_mod  # loaded relative to the tensor_slice dir below

    tdir = str(_SRC / "targets" / "tensor_slice")
    saved_path = list(sys.path)
    saved_modules = {k: sys.modules.get(k) for k in ("geometry", "package", "rtl", "golden", "flow", "base")}
    sys.path.insert(0, str(_SRC / "targets"))
    sys.path.insert(0, tdir)
    for stale in ("geometry", "package", "rtl", "golden", "flow", "base"):
        sys.modules.pop(stale, None)
    try:
        spec = importlib.util.spec_from_file_location("_ts_flow", Path(tdir) / "flow.py")
        flow_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flow_mod)
        target = flow_mod.TARGET
        items = [{"name": "l0", "m": 8, "k": 24, "n": 8, "reuse_factor": 2}]
        norm = target.normalize_config(items)
        manifest = json.loads(target.integration_manifest(norm))
    finally:
        sys.path[:] = saved_path
        for stale, prev in saved_modules.items():
            if prev is None:
                sys.modules.pop(stale, None)
            else:
                sys.modules[stale] = prev

    core = manifest["cores"][0]
    for key in ("reuse_factor_requested", "reuse_factor", "effective_reuse",
                "k_spatial", "k_passes", "k_chunks_pad", "multipliers"):
        assert key in core
    assert core["reuse_factor_requested"] == 2
    assert core["k_spatial"] == 2
    assert core["k_passes"] == 2
    assert core["k_chunks_pad"] == 4


# NOTE: no csim g++ compile check here. run_rtl_tests.py only exercises the
# behavioral Verilog (iverilog) path, not a g++ build of the generated C++
# header/wrapper -- there is no existing cheap pattern in this repo's test
# suite for compiling the Catapult-facing csim header, and standing one up
# (ac_datatypes include path, etc.) is out of scope for this pass.
