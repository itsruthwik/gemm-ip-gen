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


# ── fold-M (FoldAxis="m") ────────────────────────────────────────────────────

resolve_fold_m = _geom.resolve_fold_m

# m=20 -> grid_rows = ceil(20/8) = 3
M20_GRID_ROWS = 3


def test_resolve_fold_m_legal_set_m20():
    for rf in range(1, M20_GRID_ROWS + 1):
        resolved = resolve_fold_m(20, rf)
        assert resolved["grid_rows"] == M20_GRID_ROWS
        assert resolved["m_passes"] == resolved["reuse_factor"]
        assert resolved["warnings"] == []
        assert resolved["mg"] * resolved["m_passes"] >= M20_GRID_ROWS


def test_resolve_fold_m_rf1_is_single_frame():
    resolved = resolve_fold_m(20, 1)
    assert resolved["mg"] == M20_GRID_ROWS
    assert resolved["m_passes"] == 1
    assert resolved["reuse_factor"] == 1
    assert resolved["grid_rows_pad"] == M20_GRID_ROWS


def test_resolve_fold_m_out_of_range_warns_and_clamps():
    resolved = resolve_fold_m(20, M20_GRID_ROWS + 5, name="mylayer")
    assert len(resolved["warnings"]) == 1
    msg = resolved["warnings"][0]
    assert msg == (
        f"WARNING: Invalid ReuseFactor={M20_GRID_ROWS + 5} for layer mylayer. "
        f"Using ReuseFactor={M20_GRID_ROWS} instead. Valid ReuseFactor(s): 1..{M20_GRID_ROWS}."
    )
    assert resolved["reuse_factor"] == M20_GRID_ROWS
    assert resolved["mg"] == 1


def test_resolve_fold_m_silent_lower_landing():
    # grid_rows(100) = 13; rf=6 -> mg=ceil(13/6)=3, m_passes=ceil(13/3)=5 < 6:
    # the request lands lower than asked, silently (no warning -- only an
    # out-of-range request above grid_rows warns).
    assert resolve_fold_m(100, 1)["grid_rows"] == 13
    resolved = resolve_fold_m(100, 6)
    assert resolved["mg"] == 3
    assert resolved["m_passes"] == 5
    assert resolved["reuse_factor"] == 5
    assert resolved["warnings"] == []


def test_resolve_fold_m_padding_arithmetic():
    # grid_rows(20) = 3, rf=2 -> mg = ceil(3/2) = 2, m_passes = ceil(3/2) = 2,
    # grid_rows_pad = 4 (one padding row tile in the last frame).
    resolved = resolve_fold_m(20, 2)
    assert resolved["mg"] == 2
    assert resolved["m_passes"] == 2
    assert resolved["grid_rows_pad"] == 4


def test_resolve_reuse_factor_fold_axis_m_pins_k_fields_at_rf1():
    resolved = resolve_reuse_factor(24, 2, fold_axis="m", m=20)
    kc = K24_CHUNKS
    assert resolved["k_spatial"] == kc
    assert resolved["passes"] == 1
    assert resolved["k_chunks_pad"] == kc
    assert resolved["effective_reuse"] == 1
    assert resolved["mg"] == 2
    assert resolved["m_passes"] == 2
    assert resolved["reuse_factor"] == 2


def test_flow_normalize_config_fold_axis_m_fields(scratch_dir):
    tdir = str(_SRC / "targets" / "tensor_slice")
    saved_path = list(sys.path)
    saved_modules = {k: sys.modules.get(k) for k in ("geometry", "package", "rtl", "golden", "flow", "base")}
    sys.path.insert(0, str(_SRC / "targets"))
    sys.path.insert(0, tdir)
    for stale in ("geometry", "package", "rtl", "golden", "flow", "base"):
        sys.modules.pop(stale, None)
    try:
        spec = importlib.util.spec_from_file_location("_ts_flow_m", Path(tdir) / "flow.py")
        flow_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flow_mod)
        target = flow_mod.TARGET
        items = [{"name": "l0", "m": 20, "k": 24, "n": 16, "reuse_factor": 2, "fold_axis": "m"}]
        norm = target.normalize_config(items)
        manifest = json.loads(target.integration_manifest(norm))
        k_items = [{"name": "l1", "m": 20, "k": 24, "n": 16, "reuse_factor": 2}]
        k_norm = target.normalize_config(k_items)
        k_manifest = json.loads(target.integration_manifest(k_norm))
        # The runner emits the dict form (name -> layer); the shared normalizer
        # must forward fold_axis through that path too.
        d_norm = target.normalize_config({"l2": {
            "type": "Gemm", "n_in": 24, "n_out": 16, "gemm_m": 20,
            "reuse_factor": 2, "fold_axis": "m"}})
    finally:
        sys.path[:] = saved_path
        for stale, prev in saved_modules.items():
            if prev is None:
                sys.modules.pop(stale, None)
            else:
                sys.modules[stale] = prev

    item = norm[0]
    assert item["fold_axis"] == "m"
    assert d_norm[0]["fold_axis"] == "m" and d_norm[0]["m_passes"] == 2
    assert item["m_groups"] == 2
    assert item["m_passes"] == 2
    assert item["grid_rows_pad"] == 4
    assert item["k_spatial"] == K24_CHUNKS
    assert item["k_passes"] == 1

    core = manifest["cores"][0]
    assert core["fold_axis"] == "m"
    assert core["m_groups"] == 2
    assert core["m_passes"] == 2
    assert core["grid_rows_pad"] == 4

    # Default axis ("k") reports fold_axis="k", m_groups=grid_rows, m_passes=1.
    k_core = k_manifest["cores"][0]
    assert k_core["fold_axis"] == "k"
    assert k_core["m_groups"] == M20_GRID_ROWS
    assert k_core["m_passes"] == 1


def test_generate_catapult_pkg_fold_axis_m_rf1_byte_identical_to_k(scratch_dir):
    """RF=1 under fold_axis='m' is a single frame: the generated core, header
    and RTL are byte-identical to the fold_axis='k' package for the same shape
    (only the manifest's fold fields differ)."""
    import filecmp
    m, k, n = 20, 24, 16
    for name, axis in (("foldm_rf1", "m"), ("foldk_rf1", "k")):
        _with_tensor_slice_on_path(
            generate_catapult_pkg, m, k, n, name, str(scratch_dir),
            interface="stream", reuse_factor=1, fold_axis=axis,
        )
    dm, dk = scratch_dir / "foldm_rf1", scratch_dir / "foldk_rf1"
    for fm in sorted(dm.iterdir()):
        fk = dk / fm.name.replace("foldm_rf1", "foldk_rf1")
        assert fk.exists(), fm.name
        tm = fm.read_text().replace("foldm_rf1", "X").replace("FOLDM_RF1", "X")
        tk = fk.read_text().replace("foldk_rf1", "X").replace("FOLDK_RF1", "X")
        assert tm == tk, fm.name


def test_generate_catapult_pkg_fold_axis_k_byte_identical_to_default(scratch_dir):
    """fold_axis defaults to 'k' and must not change phase 1 output at all."""
    m, k, n = 9, 17, 10
    name_default = "nofoldaxis"
    name_k = "explicit_k"
    _with_tensor_slice_on_path(
        generate_catapult_pkg, m, k, n, name_default, str(scratch_dir),
        interface="stream", reuse_factor=2,
    )
    _with_tensor_slice_on_path(
        generate_catapult_pkg, m, k, n, name_k, str(scratch_dir),
        interface="stream", reuse_factor=2, fold_axis="k",
    )
    for ext in ("_gemm_ip.h", "_core.v"):
        a = (scratch_dir / name_default / f"{name_default}{ext}").read_text()
        b = (scratch_dir / name_k / f"{name_k}{ext}").read_text()
        a = a.replace(name_default.upper(), name_k.upper()).replace(name_default, name_k)
        assert a == b
