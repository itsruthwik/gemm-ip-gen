"""Runtime-B (two-operand) entry: verify it accepts hls4ml's B-stream format.

hls4ml's two-operand contract (nnet_gemm_stream.h gemm_stream) streams B
FIRST, in the layout selected by the manifest's ``weight_layout``
(``rtl.generate_core``'s ``b_row_major``): column-major (one gemm_k-high beat
per real N column, element k of column n == B[k][n]) or row-major (one
gemm_n-wide beat per real K row, element n of row k == B[k][n]). The
generated ``<name>_gemm_stream_runtime_b`` entry passes each beat straight to
the ccore as one raw-bit word -- no C++ reorder/buffer -- so these tests
check its static_asserts/port width match the layout, and (when g++ +
Catapult's ac_types headers are available) that the generated C++ TB compiles
and matches golden's arithmetic for both layouts.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from targets.cmvu import package as pkg  # noqa: E402
from targets.cmvu import geometry as geo_mod  # noqa: E402

_AC_INCLUDE = "/home/tools/siemens/catapult/Mgc_home/shared/include"


def _geo(m, k, n, kfold=1, nfold=1, name="rb"):
    return geo_mod.resolve_geometry(m, k, n, kfold, nfold, name)


# ── structural checks ──────────────────────────────────────────────────────


def test_runtime_b_entry_reads_column_major_beats():
    m, k, n = 4, 16, 8
    geo = _geo(m, k, n, name="rb")
    header = pkg.gen_runtime_b_header("rb", m, k, n, None, shift=4, geo=geo,
                                      b_row_major=False)

    # Column-major: one K-high beat per real N column.
    assert "static_assert(b_T::size == 16" in header
    assert "ac_int<128, false>" in header  # k*8 = 128-bit b_beat port
    assert "b_stream.read()" in header
    # No native-tile repack buffer anymore -- raw beat passed straight to ccore.
    assert "b_native[" not in header
    assert "B_raw[" not in header


def test_runtime_b_entry_reads_row_major_beats():
    m, k, n = 4, 16, 8
    geo = _geo(m, k, n, name="rb")
    header = pkg.gen_runtime_b_header("rb", m, k, n, None, shift=4, geo=geo,
                                      b_row_major=True)

    # Row-major: one N-wide beat per real K row.
    assert "static_assert(b_T::size == 8" in header
    assert "ac_int<64, false>" in header  # n*8 = 64-bit b_beat port
    assert "b_stream.read()" in header
    assert "b_native[" not in header
    assert "B_raw[" not in header


def test_runtime_b_row_major_load_schedule_has_drain_stalls():
    # N_PASSES>1: expect a stall (false entries) worked into the load window.
    m, k, n = 4, 8, 16
    geo = _geo(m, k, n, kfold=1, nfold=2, name="rb_np2")
    assert geo["n_passes"] > 1
    header = pkg.gen_runtime_b_header("rb_np2", m, k, n, None, shift=4,
                                      geo=geo, b_row_major=True)
    assert "false" in header


def test_runtime_b_column_major_load_schedule_has_no_stalls():
    m, k, n = 4, 16, 8
    geo = _geo(m, k, n, name="rb")
    header = pkg.gen_runtime_b_header("rb", m, k, n, None, shift=4, geo=geo,
                                      b_row_major=False)
    load_line = [l for l in header.splitlines() if "_load_valid[" in l][0]
    assert "false" not in load_line


# ── C++ compile-and-run check ──────────────────────────────────────────────


def _gxx():
    return shutil.which("g++")


def _have_ac_headers():
    return (Path(_AC_INCLUDE) / "ac_int.h").is_file()


def _compile_and_run(tmp_path, name, header, tb):
    (tmp_path / "nnet_types.h").write_text(pkg.gen_nnet_types_header())
    (tmp_path / f"{name}_gemm_ip.h").write_text(header)
    src = tmp_path / f"{name}_tb.cpp"
    src.write_text(tb)
    exe = tmp_path / f"{name}_tb"
    r = subprocess.run(
        ["g++", "-std=c++11", f"-I{_AC_INCLUDE}", "-I" + str(tmp_path),
         str(src), "-o", str(exe)],
        capture_output=True, text=True)
    assert r.returncode == 0, f"compile failed:\n{r.stdout}\n{r.stderr}"
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0 and "CMVU CSIM: PASS" in r.stdout, \
        f"runtime-B hls4ml-format TB failed:\n{r.stdout}\n{r.stderr}"


@pytest.mark.skipif(not _gxx() or not _have_ac_headers(),
                    reason="g++ or Catapult ac_types headers not available")
def test_runtime_b_hls4ml_column_major_tb_compiles_and_matches_golden(tmp_path):
    m, k, n = 3, 12, 9
    geo = _geo(m, k, n, name="rb")
    rng = np.random.default_rng(7)
    W = rng.integers(-128, 128, size=(k, n))
    header = pkg.gen_runtime_b_header("rb", m, k, n, None, shift=4, geo=geo,
                                      b_row_major=False)
    tb = pkg.gen_runtime_b_tb("rb", m, k, n, W, None, shift=4, geo=geo,
                              b_row_major=False)
    _compile_and_run(tmp_path, "rb", header, tb)


@pytest.mark.skipif(not _gxx() or not _have_ac_headers(),
                    reason="g++ or Catapult ac_types headers not available")
def test_runtime_b_hls4ml_row_major_tb_compiles_and_matches_golden(tmp_path):
    m, k, n = 3, 12, 9
    geo = _geo(m, k, n, name="rb_row")
    rng = np.random.default_rng(11)
    W = rng.integers(-128, 128, size=(k, n))
    header = pkg.gen_runtime_b_header("rb_row", m, k, n, None, shift=4,
                                      geo=geo, b_row_major=True)
    tb = pkg.gen_runtime_b_tb("rb_row", m, k, n, W, None, shift=4, geo=geo,
                              b_row_major=True)
    _compile_and_run(tmp_path, "rb_row", header, tb)


# ── multi-call (per-call B reload) ──────────────────────────────────────────


@pytest.mark.skipif(not _gxx() or not _have_ac_headers(),
                    reason="g++ or Catapult ac_types headers not available")
@pytest.mark.parametrize("b_row_major", [False, True])
def test_runtime_b_multi_call_tb_compiles_and_matches_golden(tmp_path, b_row_major):
    """hls4ml streams a NEW B ahead of every call's A rows; the entry's
    `static ccore` persists across calls (like real hardware) but its C++
    state is fully reset each call (`ccore.reset_state()` at entry). Calls
    the entry 3x in a row with independently random B/A each time -- the
    scenario the RTL wrapper's load FSM has to get right too (see
    test_cmvu_rtl.py's Icarus regression for that side)."""
    m, k, n = 3, 12, 9
    geo = _geo(m, k, n, name="rbmc")
    header = pkg.gen_runtime_b_header("rbmc", m, k, n, None, shift=4, geo=geo,
                                      b_row_major=b_row_major)
    tb = pkg.gen_runtime_b_multi_call_tb("rbmc", m, k, n, None, shift=4,
                                         geo=geo, seed=13,
                                         b_row_major=b_row_major, n_calls=3)
    _compile_and_run(tmp_path, "rbmc", header, tb)
