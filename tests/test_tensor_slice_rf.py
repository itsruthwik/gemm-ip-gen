"""Tests for tensor_slice ReuseFactor (RF) legalization and the pass-2
package.py plumbing built on top of it.

RF is purely "number of passes over K" -- never a cycle count or an
initiation interval. See geometry.resolve_reuse_factor / package
pass 2 for the full model.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from targets.tensor_slice import geometry as _geom  # noqa: E402
from targets.tensor_slice import package as _pkg  # noqa: E402

_SCRATCH_ROOT = Path("/mnt/vault0/rsunketa/atlas/temp_space/ts-rf/pytest")

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
    """Run *fn* directly. Kept only so the other test modules that import this
    helper (test_tensor_slice_requant_shift.py, test_tensor_slice_unit_tb.py)
    don't need their own call-site changes -- tensor_slice's siblings are now
    plain relative imports, so no sys.path/sys.modules juggling is needed."""
    return fn(*args, **kwargs)


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

    from targets.tensor_slice.flow import TARGET as target
    items = [{"name": "l0", "m": 8, "k": 24, "n": 8, "reuse_factor": 2}]
    norm = target.normalize_config(items)
    manifest = json.loads(target.integration_manifest(norm))

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
    from targets.tensor_slice.flow import TARGET as target
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


# ── fold-N (FoldAxis="n") ────────────────────────────────────────────────────

resolve_fold_n = _geom.resolve_fold_n

# n=20 -> grid_cols = ceil(20/8) = 3
N20_GRID_COLS = 3


def test_resolve_fold_n_legal_set_n20():
    for rf in range(1, N20_GRID_COLS + 1):
        resolved = resolve_fold_n(20, rf)
        assert resolved["grid_cols"] == N20_GRID_COLS
        assert resolved["n_passes"] == resolved["reuse_factor"]
        assert resolved["warnings"] == []
        assert resolved["cg"] * resolved["n_passes"] >= N20_GRID_COLS


def test_resolve_fold_n_rf1_is_single_frame():
    resolved = resolve_fold_n(20, 1)
    assert resolved["cg"] == N20_GRID_COLS
    assert resolved["n_passes"] == 1
    assert resolved["reuse_factor"] == 1
    assert resolved["grid_cols_pad"] == N20_GRID_COLS


def test_resolve_fold_n_out_of_range_warns_and_clamps():
    resolved = resolve_fold_n(20, N20_GRID_COLS + 5, name="mylayer")
    assert len(resolved["warnings"]) == 1
    msg = resolved["warnings"][0]
    assert msg == (
        f"WARNING: Invalid ReuseFactor={N20_GRID_COLS + 5} for layer mylayer. "
        f"Using ReuseFactor={N20_GRID_COLS} instead. Valid ReuseFactor(s): 1..{N20_GRID_COLS}."
    )
    assert resolved["reuse_factor"] == N20_GRID_COLS
    assert resolved["cg"] == 1


def test_resolve_fold_n_silent_lower_landing():
    # grid_cols(100) = 13; rf=6 -> cg=ceil(13/6)=3, n_passes=ceil(13/3)=5 < 6:
    # the request lands lower than asked, silently (no warning -- only an
    # out-of-range request above grid_cols warns).
    assert resolve_fold_n(100, 1)["grid_cols"] == 13
    resolved = resolve_fold_n(100, 6)
    assert resolved["cg"] == 3
    assert resolved["n_passes"] == 5
    assert resolved["reuse_factor"] == 5
    assert resolved["warnings"] == []


def test_resolve_fold_n_padding_arithmetic():
    # grid_cols(20) = 3, rf=2 -> cg = ceil(3/2) = 2, n_passes = ceil(3/2) = 2,
    # grid_cols_pad = 4 (one padding column tile in the last group).
    resolved = resolve_fold_n(20, 2)
    assert resolved["cg"] == 2
    assert resolved["n_passes"] == 2
    assert resolved["grid_cols_pad"] == 4


def test_resolve_reuse_factor_fold_axis_n_pins_k_fields_at_rf1():
    resolved = resolve_reuse_factor(24, 2, fold_axis="n", n=20)
    kc = K24_CHUNKS
    assert resolved["k_spatial"] == kc
    assert resolved["passes"] == 1
    assert resolved["k_chunks_pad"] == kc
    assert resolved["cg"] == 2
    assert resolved["n_passes"] == 2
    assert resolved["reuse_factor"] == 2
    # Unlike fold-M (effective_reuse pinned at 1), fold-N's effective_reuse
    # tracks the pass count: each frame reuses the array once per group.
    assert resolved["effective_reuse"] == 2


def test_flow_normalize_config_fold_axis_n_fields(scratch_dir):
    from targets.tensor_slice.flow import TARGET as target
    items = [{"name": "l0", "m": 20, "k": 24, "n": 20, "reuse_factor": 2, "fold_axis": "n"}]
    norm = target.normalize_config(items)
    manifest = json.loads(target.integration_manifest(norm))
    k_items = [{"name": "l1", "m": 20, "k": 24, "n": 20, "reuse_factor": 2}]
    k_norm = target.normalize_config(k_items)
    k_manifest = json.loads(target.integration_manifest(k_norm))
    d_norm = target.normalize_config({"l2": {
        "type": "Gemm", "n_in": 24, "n_out": 20, "gemm_m": 20,
        "reuse_factor": 2, "fold_axis": "n"}})

    item = norm[0]
    assert item["fold_axis"] == "n"
    assert d_norm[0]["fold_axis"] == "n" and d_norm[0]["n_passes"] == 2
    assert item["n_groups"] == 2
    assert item["n_passes"] == 2
    assert item["grid_cols_pad"] == 4
    assert item["core_cols"] == 16
    assert item["k_spatial"] == K24_CHUNKS
    assert item["k_passes"] == 1

    core = manifest["cores"][0]
    assert core["n_groups"] == 2
    assert core["n_passes"] == 2
    assert core["grid_cols_pad"] == 4

    # Default axis ("k") reports n_groups=grid_cols, n_passes=1, core_cols=n.
    k_core = k_manifest["cores"][0]
    assert k_core["n_groups"] == N20_GRID_COLS
    assert k_core["n_passes"] == 1


def test_generate_catapult_pkg_fold_axis_n_rf1_byte_identical_to_k(scratch_dir):
    """RF=1 under fold_axis='n' is a single frame: the generated core, header
    and RTL are byte-identical to the fold_axis='k' package for the same shape
    (only the manifest's fold fields differ)."""
    m, k, n = 20, 24, 16
    for name, axis in (("foldn_rf1", "n"), ("foldk_rf1b", "k")):
        _with_tensor_slice_on_path(
            generate_catapult_pkg, m, k, n, name, str(scratch_dir),
            interface="stream", reuse_factor=1, fold_axis=axis,
        )
    dn, dk = scratch_dir / "foldn_rf1", scratch_dir / "foldk_rf1b"
    for fn_ in sorted(dn.iterdir()):
        fk = dk / fn_.name.replace("foldn_rf1", "foldk_rf1b")
        assert fk.exists(), fn_.name
        tn = fn_.read_text().replace("foldn_rf1", "X").replace("FOLDN_RF1", "X")
        tk = fk.read_text().replace("foldk_rf1b", "X").replace("FOLDK_RF1B", "X")
        assert tn == tk, fn_.name


def test_generate_catapult_pkg_fold_axis_n_generates_full_package(scratch_dir):
    """A multi-frame fold-N package (rf > 1) still produces a complete,
    well-formed package (core/header/inst/tb/tcl all present and non-empty)."""
    m, k, n = 12, 24, 32
    name = "foldn_manifest"
    pkg = _with_tensor_slice_on_path(
        generate_catapult_pkg, m, k, n, name, str(scratch_dir),
        interface="stream", reuse_factor=4, fold_axis="n",
    )
    pkg_dir = scratch_dir / name
    for f in (f"{name}_core.v", "nnet_types.h", f"{name}_gemm_ip.h",
              f"{name}_inst.cpp", f"{name}_tb.cpp", "run_catapult.tcl"):
        assert (pkg_dir / f).is_file() and (pkg_dir / f).stat().st_size > 0, f



def test_generate_catapult_pkg_fold_axis_n_rom_read_offsets_by_group(scratch_dir):
    """Under fold-N with baked weights the C model must read frame g's column
    group from the ROM (the RTL's group counter does); a plain per-pass index
    would feed group 0 on every frame and csim would disagree with cosim."""
    m, k, n = 12, 24, 32
    W = _random_weight_matrix(k, n)
    for name, axis in (("romn2", "n"), ("romk1", "k")):
        _with_tensor_slice_on_path(
            generate_catapult_pkg, m, k, n, name, str(scratch_dir),
            interface="stream", reuse_factor=2 if axis == "n" else 1, fold_axis=axis,
            input_precision="fixed<8,2>", weight_precision="fixed<8,2>",
            output_precision="fixed<16,6>", weight_matrix=W,
        )
    hdr_n = (scratch_dir / "romn2" / "romn2_gemm_ip.h").read_text()
    hdr_k = (scratch_dir / "romk1" / "romk1_gemm_ip.h").read_text()
    assert "B_ROM[_bidx + _grp * 16]" in hdr_n
    assert "_grp = (_grp + 1) % 2;" in hdr_n
    assert "_grp" not in hdr_k
