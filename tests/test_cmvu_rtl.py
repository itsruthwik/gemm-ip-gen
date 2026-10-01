"""Unit tests for the cmvu RTL/TB generators (structural, no simulator)."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Package-qualified import (see test_cmvu_geometry.py) to avoid a bare
# `geometry`/`golden`/`rtl` sys.modules collision with the mvau tests.
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from targets.cmvu import geometry as g  # noqa: E402
from targets.cmvu import golden as gold  # noqa: E402
from targets.cmvu import rtl as cmvu_rtl  # noqa: E402


def _B(k, n, seed=3):
    return np.random.default_rng(seed).integers(-128, 128, size=(k, n))


# ── generate_core ─────────────────────────────────────────────────────────────

def test_temporal_core_structure():
    B = _B(8, 8)
    core = cmvu_rtl.generate_core(2, 8, 8, 2, 1, B, shift=3,
                                  module_name="cmvu_core")
    assert "module cmvu_core" in core
    assert "K_PASSES  = 2" in core
    assert "N_GROUPS  = 1" in core
    assert "SHIFT_AMT = 5'd3" in core
    assert "#(.CASCADE_EN(1'b0)) u_blk_r0_c0" in core
    # one init literal per physical bank word (8 words), in both guard
    # branches; exclude the (also 256-bit, since BIAS_WIDTH=32*N_PHYS) BIAS_WORD
    # localparam line, which is not a weight-tile init.
    tile_256_lines = [ln for ln in core.splitlines()
                      if "= 256'h" in ln and "BIAS_WORD" not in ln]
    assert len(tile_256_lines) == 16
    assert "`ifdef __ICARUS__" in core
    assert "force u_blk_r0_c0.u_w_mem.bank0[0]" in core
    assert "release u_blk_r0_c0.u_w_mem.bank0[0]" in core


def test_core_bakes_slot_map():
    # K=8 KFold=2 -> k_passes=2; slot 1 is (k_pass=1, n_group=0)
    B = _B(8, 8)
    core = cmvu_rtl.generate_core(1, 8, 8, 2, 1, B, module_name="cmvu_core")
    tiles = gold.weight_tiles(B, 8, 8, 2, 1)
    want = gold.hex_literal(gold.pack_tile(tiles[1][0]), 32)
    assert want in core
    assert gold.hex_literal(gold.pack_tile(tiles[0][0]), 32) in core


def test_spatial_core_structure():
    # K=8/N=16 fully spatial -> 2 cascade columns x 2 broadcast rows.
    B = _B(8, 16)
    core = cmvu_rtl.generate_core(1, 8, 16, 1, 1, B, module_name="cmvu_core")
    assert "K_SPATIAL = 2" in core and "N_SPATIAL = 2" in core
    assert "cmvu_mode1 u_blk_r1_c1" in core
    # column-1 alignment skew + row-1 broadcast forwarding register
    assert "u_skew_c1_s0" in core
    assert "u_bcast_r1_c0" in core
    # cascade chain: head cascade_in left dangling (CASCADE_EN=0), column 1
    # consumes column 0; the tail's cascade_out is left dangling too.
    assert "#(.CASCADE_EN(1'b0)) u_blk_r0_c0" in core
    assert ".cascade_in()," in core
    assert ".cascade_in(casc_blk_r1_c0)" in core
    assert ".cascade_out()," in core
    assert "{256{1'b0}}" not in core
    # output de-skew for row 0 (ns-1 hops) and aligned done
    assert "u_yds_r0_s0" in core
    assert "wire done_align" in core
    # Reset is sync at the wrapper boundary: no wrapper-owned register has an
    # async reset, and the delay lines are wrapper registers, not the
    # vendored (async-reset) regbank.
    assert "posedge rst" not in core
    assert "cmvu_regbank" not in core
    # non-tail cascade stages reset every pass; tails own framing + bias
    assert ".acc_first(bus_r0_c0[B_VALID])" in core
    assert ".acc_first(bus_r0_c1[B_FIRST])" in core
    # bias lanes are BIAS_WIDTH (32, accumulator scale); the y path stays
    # RESULT_WIDTH (16). bias_group_w = N_PHYS*BIAS_WIDTH = 8*32 = 256.
    assert ".bias_in(bus_r0_c1[B_BIAS_LO + 0 +: BIAS_W])" in core
    assert ".bias_in(bus_r1_c1[B_BIAS_LO + 256 +: BIAS_W])" in core
    assert ".bias_in('0)" in core
    # effective width: default W = RESULT_WIDTH (32) -> out_w encoded as 31
    assert "OUT_W     = 32" in core and "OUT_W_ENC = 5'd31" in core
    assert ".out_w(OUT_W_ENC)" in core
    assert "res_row[gs*OUT_W +: OUT_W] = res_buf[gs*32 +: OUT_W]" in core


def test_single_block_leaves_both_cascade_pins_dangling():
    # K=N=one tile -> one block, both chain endpoints: cascade_in and
    # cascade_out are left dangling, never tied to 0/1.
    core = cmvu_rtl.generate_core(1, 4, 8, 1, 1, _B(4, 8),
                                  module_name="cmvu_core")
    assert "#(.CASCADE_EN(1'b0)) u_blk_r0_c0" in core
    assert ".cascade_in()," in core and ".cascade_out()," in core
    assert "casc_blk" not in core
    assert "{256{1'b0}}" not in core


def test_spatial_blocks_bake_their_own_tiles():
    B = _B(8, 16)
    core = cmvu_rtl.generate_core(1, 8, 16, 1, 1, B, module_name="cmvu_core")
    tiles = gold.weight_tiles(B, 8, 16, 2, 2)
    # block (row 1, col 0) owns logical tile (kt=0, nt=1)
    assert gold.hex_literal(gold.pack_tile(tiles[0][1]), 32) in core
    # block (row 0, col 1) owns logical tile (kt=1, nt=0)
    assert gold.hex_literal(gold.pack_tile(tiles[1][0]), 32) in core
    assert "u_blk_r1_c0.u_w_mem" in core and "u_blk_r0_c1.u_w_mem" in core


def test_bad_shift_is_error():
    with pytest.raises(ValueError, match="shift must be"):
        cmvu_rtl.generate_core(1, 4, 8, 1, 1, _B(4, 8), shift=32)


def test_bias_word_baked():
    B = np.zeros((4, 8), dtype=np.int64)
    codes = [1, -1, 2, -2, 3, -3, 4, -4]
    core = cmvu_rtl.generate_core(1, 4, 8, 1, 1, B, bias_codes=codes,
                                  module_name="cmvu_core")
    want = gold.hex_literal(gold.pack_res_row(codes, width=g.BIAS_WIDTH), 32)
    assert f"256'h{want}" in core


# ── generate_tb ───────────────────────────────────────────────────────────────

def test_tb_structure_and_golden_rows():
    B = _B(8, 8)
    tb = gold.generate_tb(2, 8, 8, 2, 1, B, shift=3, module_name="cmvu_core",
                          seed=11)
    assert "module cmvu_core_tb" in tb
    assert "cmvu_core dut" in tb
    assert "a_mem[0] =" in tb and "a_mem[1] =" in tb
    assert "e_mem[0] =" in tb and "e_mem[1] =" in tb
    assert "ALL_PASS" in tb


# ── runtime-B (two-operand) core ──────────────────────────────────────────────

def test_runtime_b_core_structure():
    B = _B(8, 8)
    core = cmvu_rtl.generate_core(4, 8, 8, 2, 1, B, shift=3,
                                  module_name="cmvu_core", runtime_b=True)
    assert "b_beat" in core and "b_valid" in core
    # Column-major load: every block in the target block-row has its own
    # write port (parallel writes across K_SPATIAL columns), gated by the
    # registered target row (tgt_r), not a per-column loop.
    assert "b_beat_r" in core and "tgt_r" in core
    assert ".w_we(b_valid_r" in core
    assert ".b_in(b_in_c" in core
    # no baked weight-tile init in runtime-B mode (BIAS_WORD is still a
    # 256-bit localparam now that BIAS_WIDTH=32, so exclude that line)
    tile_256_lines = [ln for ln in core.splitlines()
                      if "256'h" in ln and "BIAS_WORD" not in ln]
    assert tile_256_lines == []
    assert "force " not in core


def test_const_core_has_no_b_ports():
    B = _B(8, 8)
    core = cmvu_rtl.generate_core(4, 8, 8, 2, 1, B, shift=3,
                                  module_name="cmvu_core")
    assert "b_beat" not in core and "b_valid" not in core
    assert "256'h" in core


# ── clock gating ──────────────────────────────────────────────────────────────

def test_gclk_gate_is_glitch_free():
    """`en` is combinational (registered control logic) and can change while
    clk is high; a plain `clk & en` AND gate glitches (an extra rising edge
    on gclk whenever en rises mid-high-phase). The gate must sample en on
    the falling edge instead, so it is already stable for the whole high
    phase of clk."""
    B = _B(8, 8)
    core = cmvu_rtl.generate_core(2, 8, 8, 2, 1, B, shift=3,
                                  module_name="cmvu_core")
    assert "wire gclk = clk & en;" not in core
    assert "always @(negedge clk) begin" in core
    assert "en_n <= en;" in core
    assert "wire gclk = clk & (en_n | rst);" in core


def test_generate_tb_en_gaps_structure():
    B = _B(8, 8)
    tb = gold.generate_tb(2, 8, 8, 2, 1, B, shift=3, module_name="cmvu_core",
                          seed=11, en_gaps=True)
    assert "task automatic step;" in tb
    assert "$random(en_seed)" in tb
    assert "while (out_valid) @(posedge clk);" in tb
    # default (no gaps) TB is unaffected
    tb0 = gold.generate_tb(2, 8, 8, 2, 1, B, shift=3, module_name="cmvu_core",
                           seed=11)
    assert "task automatic step;" not in tb0
    assert "en_seed" not in tb0


def test_generate_runtime_b_tb_en_gaps_structure():
    B = _B(8, 8)
    tb = gold.generate_runtime_b_tb(2, 8, 8, 2, 1, B, shift=3,
                                    module_name="cmvu_core", seed=11,
                                    en_gaps=True)
    assert "task automatic step;" in tb
    assert "$random(en_seed)" in tb


@pytest.mark.skipif(shutil.which("iverilog") is None,
                    reason="iverilog not on PATH")
def test_en_gap_regression_icarus(tmp_path):
    """Regression for the gclk glitch: with en gaps, `clk & en` over-advances
    the vendored blocks relative to the wrapper's own ena-gated registers,
    corrupting results. Run one shape through Icarus with en gaps and check
    it still passes against the fixed (glitch-free) gate."""
    rtl_dir = g.vendored_rtl_dir()
    if rtl_dir is None:
        pytest.skip("cmvu vendored block RTL not found")
    rtl_files = [str((rtl_dir / f).resolve()) for f in g.VENDORED_SV]

    B = _B(8, 8, seed=5)
    codes, _ = gold.bias_codes(np.array([1.5, -2.0, 0.0, 3.25, -1.0, 0.5,
                                         2.0, -0.5]), 3)
    core = cmvu_rtl.generate_core(4, 8, 8, 2, 1, B, bias_codes=codes,
                                  shift=3, module_name="cmvu_core")
    tb = gold.generate_tb(4, 8, 8, 2, 1, B, bias_codes=codes, shift=3,
                          module_name="cmvu_core", seed=21, en_gaps=True)
    (tmp_path / "cmvu_core.v").write_text(core)
    (tmp_path / "cmvu_core_tb.v").write_text(tb)

    build = subprocess.run(
        ["iverilog", "-g2012", "-o", "simv", "-s", "cmvu_core_tb",
         "cmvu_core_tb.v", "cmvu_core.v", *rtl_files],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr

    sim = subprocess.run(["vvp", "simv"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    out = sim.stdout + sim.stderr
    assert sim.returncode == 0 and "ALL_PASS" in out and "ERROR:" not in out, out


# ── runtime-B multi-call re-arm ────────────────────────────────────────────


def test_runtime_b_core_rearms_loading_after_first_call():
    """hls4ml streams a NEW B ahead of EVERY frame's A rows (a fresh K for QK,
    a fresh V for aV), but the wrapper is persistent hardware: the load FSM
    re-arms whenever its target slot set is empty, not just once at reset,
    or only the first frame's B is ever seen."""
    B = _B(8, 8)
    core = cmvu_rtl.generate_core(4, 8, 8, 2, 1, B, shift=3,
                                  module_name="cmvu_core", runtime_b=True)
    assert "call_done" not in core
    assert "assign arm = !loading && !loading_d && !set_full[wr_set];" in core
    assert "loading <= 1'b1; b_valid_r <= 1'b0;" in core.split(
        "if (arm) begin", 1)[1][:200]


@pytest.mark.parametrize("k,kf,dbuf", [(8, 2, True), (32, 8, False)])
def test_runtime_b_slot_sets(k, kf, dbuf):
    # Two slot sets (double buffering) when two copies of the layer's tiles
    # fit in the block's 8 slots; one set otherwise.
    core = cmvu_rtl.generate_core(4, k, 8, kf, 1, None, shift=3,
                                  module_name="cmvu_core", runtime_b=True)
    slots = g.resolve_geometry(4, k, 8, kf, 1)["slots_per_block"]
    assert f"SET_BASE = {slots + slots % 2};" in core
    assert f"DBUF = 1'b{1 if dbuf else 0};" in core
    assert "rd_set*SET_BASE + np*K_PASSES + kp" in core


def test_generate_runtime_b_multi_call_tb_structure():
    tb = gold.generate_runtime_b_multi_call_tb(
        4, 8, 8, 2, 1, shift=3, module_name="cmvu_core", seed=11, n_calls=3)
    assert "callc" in tb
    assert "a_mem [0:3-1][0:4-1]" in tb or "a_mem [0:2][0:3]" in tb
    assert 'ALL_PASS (%0d calls x 4 rows)' in tb


@pytest.mark.skipif(shutil.which("iverilog") is None,
                    reason="iverilog not on PATH")
@pytest.mark.parametrize("row_major", [False, True])
def test_runtime_b_multi_call_icarus(tmp_path, row_major):
    """3 consecutive calls, each with an independently random B and A,
    against the SAME persistent dut instance -- the regression for the
    'loading only ever re-arms at reset' bug (single-call TBs never call the
    entry twice, so they can't catch it)."""
    rtl_dir = g.vendored_rtl_dir()
    if rtl_dir is None:
        pytest.skip("cmvu vendored block RTL not found")
    rtl_files = [str((rtl_dir / f).resolve()) for f in g.VENDORED_SV]

    m, k, n, kf, nf, shift = 8, 8, 10, 1, 1, 4
    core = cmvu_rtl.generate_core(m, k, n, kf, nf, None, shift=shift,
                                  module_name="cmvu_core", runtime_b=True,
                                  b_row_major=row_major, result_width=16)
    tb = gold.generate_runtime_b_multi_call_tb(
        m, k, n, kf, nf, shift=shift, module_name="cmvu_core", seed=31,
        result_width=16, b_row_major=row_major, n_calls=3)
    (tmp_path / "cmvu_core.v").write_text(core)
    (tmp_path / "cmvu_core_tb.v").write_text(tb)

    build = subprocess.run(
        ["iverilog", "-g2012", "-o", "simv", "-s", "cmvu_core_tb",
         "cmvu_core_tb.v", "cmvu_core.v", *rtl_files],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr

    sim = subprocess.run(["vvp", "simv"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    out = sim.stdout + sim.stderr
    assert sim.returncode == 0 and "ALL_PASS" in out and "ERROR:" not in out, out


# ── sustained backpressure ──────────────────────────────────────────────────


def test_backpressure_decls_structure():
    decls = gold._backpressure_decls(7)
    assert "bp_remain" in decls and "bp_gate" in decls
    assert "task automatic step;" in decls


def test_en_gaps_and_backpressure_are_mutually_exclusive():
    with pytest.raises(ValueError):
        gold._gap_mode(7, en_gaps=True, backpressure=True)


@pytest.mark.skipif(shutil.which("iverilog") is None,
                    reason="iverilog not on PATH")
def test_backpressure_regression_icarus_const_weight(tmp_path):
    """Long (20-50 cycle) disabled bursts, instead of en_gaps' single-cycle
    gate -- reliably lands mid-row and mid-drain, not just between rows."""
    rtl_dir = g.vendored_rtl_dir()
    if rtl_dir is None:
        pytest.skip("cmvu vendored block RTL not found")
    rtl_files = [str((rtl_dir / f).resolve()) for f in g.VENDORED_SV]

    B = _B(8, 8, seed=5)
    codes, _ = gold.bias_codes(np.array([1.5, -2.0, 0.0, 3.25, -1.0, 0.5,
                                         2.0, -0.5]), 3)
    core = cmvu_rtl.generate_core(4, 8, 8, 2, 1, B, bias_codes=codes,
                                  shift=3, module_name="cmvu_core")
    tb = gold.generate_tb(4, 8, 8, 2, 1, B, bias_codes=codes, shift=3,
                          module_name="cmvu_core", seed=21,
                          backpressure=True)
    (tmp_path / "cmvu_core.v").write_text(core)
    (tmp_path / "cmvu_core_tb.v").write_text(tb)

    build = subprocess.run(
        ["iverilog", "-g2012", "-o", "simv", "-s", "cmvu_core_tb",
         "cmvu_core_tb.v", "cmvu_core.v", *rtl_files],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr

    sim = subprocess.run(["vvp", "simv"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    out = sim.stdout + sim.stderr
    assert sim.returncode == 0 and "ALL_PASS" in out and "ERROR:" not in out, out


@pytest.mark.skipif(shutil.which("iverilog") is None,
                    reason="iverilog not on PATH")
@pytest.mark.parametrize("row_major", [False, True])
def test_backpressure_regression_icarus_runtime_b(tmp_path, row_major):
    """Backpressure bursts landing mid runtime-B load phase and mid-drain
    (row-major N_PASSES=2), not just between rows."""
    rtl_dir = g.vendored_rtl_dir()
    if rtl_dir is None:
        pytest.skip("cmvu vendored block RTL not found")
    rtl_files = [str((rtl_dir / f).resolve()) for f in g.VENDORED_SV]

    m, k, n, kf, nf, shift = 4, 8, 16, 1, 2, 5
    W = _B(k, n, seed=17)
    core = cmvu_rtl.generate_core(m, k, n, kf, nf, W, shift=shift,
                                  module_name="cmvu_core", runtime_b=True,
                                  b_row_major=row_major, result_width=16)
    tb = gold.generate_runtime_b_tb(m, k, n, kf, nf, W, shift=shift,
                                    module_name="cmvu_core", seed=23,
                                    result_width=16, b_row_major=row_major,
                                    backpressure=True)
    (tmp_path / "cmvu_core.v").write_text(core)
    (tmp_path / "cmvu_core_tb.v").write_text(tb)

    build = subprocess.run(
        ["iverilog", "-g2012", "-o", "simv", "-s", "cmvu_core_tb",
         "cmvu_core_tb.v", "cmvu_core.v", *rtl_files],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr

    sim = subprocess.run(["vvp", "simv"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    out = sim.stdout + sim.stderr
    assert sim.returncode == 0 and "ALL_PASS" in out and "ERROR:" not in out, out


@pytest.mark.skipif(shutil.which("iverilog") is None,
                    reason="iverilog not on PATH")
def test_backpressure_regression_icarus_multi_call(tmp_path):
    """Backpressure combined with multi-call: bursts landing across the
    call boundary (mid re-arm) as well as mid-row/mid-load."""
    rtl_dir = g.vendored_rtl_dir()
    if rtl_dir is None:
        pytest.skip("cmvu vendored block RTL not found")
    rtl_files = [str((rtl_dir / f).resolve()) for f in g.VENDORED_SV]

    m, k, n, kf, nf, shift = 8, 8, 10, 1, 1, 4
    core = cmvu_rtl.generate_core(m, k, n, kf, nf, None, shift=shift,
                                  module_name="cmvu_core", runtime_b=True,
                                  b_row_major=True, result_width=16)
    tb = gold.generate_runtime_b_multi_call_tb(
        m, k, n, kf, nf, shift=shift, module_name="cmvu_core", seed=33,
        result_width=16, b_row_major=True, n_calls=3, backpressure=True)
    (tmp_path / "cmvu_core.v").write_text(core)
    (tmp_path / "cmvu_core_tb.v").write_text(tb)

    build = subprocess.run(
        ["iverilog", "-g2012", "-o", "simv", "-s", "cmvu_core_tb",
         "cmvu_core_tb.v", "cmvu_core.v", *rtl_files],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr

    sim = subprocess.run(["vvp", "simv"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    out = sim.stdout + sim.stderr
    assert sim.returncode == 0 and "ALL_PASS" in out and "ERROR:" not in out, out


# ── Back-to-back rows ─────────────────────────────────────────────────────────
# The wrapper takes the next row on the current row's last pass, so rows issue
# every K_PASSES*N_GROUPS cycles with no idle cycle between them. The other
# Icarus TBs send one row and wait for its result, so they never exercise it.

@pytest.mark.parametrize("m,k,n,kf,nf,backpressure", [
    (8, 8, 16, 1, 1, False),    # fully spatial: a new row every cycle
    (8, 8, 16, 2, 2, False),    # 2 K passes x 2 N groups
    (6, 6, 10, 2, 2, False),    # K and N tails
    (5, 16, 8, 4, 1, False),    # K passes only
    (8, 8, 16, 2, 2, True),     # long en gaps mid-row
])
def test_streaming_rows_back_to_back_icarus(tmp_path, m, k, n, kf, nf,
                                            backpressure):
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog not on PATH")
    rtl_dir = g.vendored_rtl_dir()
    if rtl_dir is None:
        pytest.skip("cmvu vendored block RTL not found")
    rtl_files = [str((rtl_dir / f).resolve()) for f in g.VENDORED_SV]
    B = _B(k, n)
    core = cmvu_rtl.generate_core(m, k, n, kf, nf, B, shift=4,
                                  module_name="cmvu_core", result_width=16)
    tb = gold.generate_streaming_tb(m, k, n, kf, nf, B, shift=4,
                                    module_name="cmvu_core", seed=21,
                                    result_width=16, backpressure=backpressure)
    (tmp_path / "cmvu_core.v").write_text(core)
    (tmp_path / "cmvu_core_tb.v").write_text(tb)
    build = subprocess.run(
        ["iverilog", "-g2012", "-o", "simv", "-s", "cmvu_core_tb",
         "cmvu_core_tb.v", "cmvu_core.v", *rtl_files],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr
    sim = subprocess.run(["vvp", "simv"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    out = sim.stdout + sim.stderr
    assert sim.returncode == 0 and "ALL_PASS" in out and "ERROR:" not in out, out


# ── Column-major runtime-B with more than two K passes ───────────────────────
# Passes beyond the live pair are staged in the wrapper's tile store and
# drained as row-major writes after each n-group; multi-call + backpressure
# covers the drain across calls and under long en gaps.

@pytest.mark.parametrize("m,k,n,kf,nf", [
    (6, 12, 16, 3, 2),   # K_PASSES=3: odd base slot for n-group 1 -> single live tile
    (4, 32, 16, 4, 2),   # K_PASSES=4 x N_PASSES=2, 2 cascade blocks: all 8 slots
])
def test_staged_k_passes_multi_call_backpressure_icarus(tmp_path, m, k, n, kf, nf):
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog not on PATH")
    rtl_dir = g.vendored_rtl_dir()
    if rtl_dir is None:
        pytest.skip("cmvu vendored block RTL not found")
    rtl_files = [str((rtl_dir / f).resolve()) for f in g.VENDORED_SV]
    core = cmvu_rtl.generate_core(m, k, n, kf, nf, None, shift=3,
                                  module_name="cmvu_core", runtime_b=True,
                                  result_width=16)
    tb = gold.generate_runtime_b_multi_call_tb(
        m, k, n, kf, nf, shift=3, module_name="cmvu_core", seed=17,
        result_width=16, n_calls=3, backpressure=True)
    (tmp_path / "cmvu_core.v").write_text(core)
    (tmp_path / "cmvu_core_tb.v").write_text(tb)
    build = subprocess.run(
        ["iverilog", "-g2012", "-o", "simv", "-s", "cmvu_core_tb",
         "cmvu_core_tb.v", "cmvu_core.v", *rtl_files],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr
    sim = subprocess.run(["vvp", "simv"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=300)
    out = sim.stdout + sim.stderr
    assert sim.returncode == 0 and "ALL_PASS" in out and "ERROR:" not in out, out


# ── Double-buffered runtime-B, frames streamed back to back ──────────────────
# With two slot sets the next frame's B loads while the current frame
# computes, so the frame interval is max(M * slots, load + 2): the compute
# (the reuse factor) whenever the load fits under it.

@pytest.mark.parametrize("m,k,n,kf,nf,row_major", [
    (8, 16, 8, 2, 1, False),   # 2 slots, paired column writes
    (8, 12, 8, 3, 1, False),   # 3 slots (odd): set 1 on an even base, staging
    (8, 8, 16, 2, 2, True),    # row-major, 4 slots
    (8, 4, 8, 1, 1, False),    # 1 slot: load-bound (load + 2 > M * slots)
    (6, 12, 16, 3, 2, False),  # 6 slots: one set, no double buffering
])
@pytest.mark.parametrize("backpressure", [False, True])
def test_runtime_b_frames_stream_icarus(tmp_path, m, k, n, kf, nf, row_major,
                                        backpressure):
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog not on PATH")
    rtl_dir = g.vendored_rtl_dir()
    if rtl_dir is None:
        pytest.skip("cmvu vendored block RTL not found")
    rtl_files = [str((rtl_dir / f).resolve()) for f in g.VENDORED_SV]
    geo = g.resolve_geometry(m, k, n, kf, nf)
    slots = geo["slots_per_block"]
    dbuf = slots + slots % 2 + slots <= g.MEM_TILES
    load = len(g.runtime_b_load_schedule(k, n, geo, row_major))
    interval = max(m * slots, load + 2) if dbuf else None
    core = cmvu_rtl.generate_core(m, k, n, kf, nf, None, shift=4,
                                  module_name="cmvu_core", runtime_b=True,
                                  b_row_major=row_major, result_width=16)
    tb = gold.generate_streaming_runtime_b_tb(
        m, k, n, kf, nf, shift=4, module_name="cmvu_core", seed=3,
        result_width=16, b_row_major=row_major, n_frames=5,
        backpressure=backpressure, frame_interval=interval)
    (tmp_path / "cmvu_core.v").write_text(core)
    (tmp_path / "cmvu_core_tb.v").write_text(tb)
    build = subprocess.run(
        ["iverilog", "-g2012", "-o", "simv", "-s", "cmvu_core_tb",
         "cmvu_core_tb.v", "cmvu_core.v", *rtl_files],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr
    sim = subprocess.run(["vvp", "simv"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=300)
    out = sim.stdout + sim.stderr
    assert sim.returncode == 0 and "ALL_PASS" in out and "ERROR:" not in out, out
