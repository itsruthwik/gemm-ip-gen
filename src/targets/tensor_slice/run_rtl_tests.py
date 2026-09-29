#!/usr/bin/env python3
"""
RTL-level test for the tensor-slice GEMM wrapper.

For a spread of GEMM sizes (tiled and non-8-multiple), this generates the
behavioral combined-core wrapper RTL (``rtl``) and a
self-checking testbench (``golden``), writes them under
``tb/generated/``, then compiles + runs each with Icarus Verilog and checks the
testbench's ``ALL_PASS`` marker.

This is the RTL-level regression: it exercises the wrapper protocol + the
structural GEMM datapath end-to-end. Single-branch RTL: iverilog compiles
with ``-DSYNTHESIS`` and links the ``tensor_slice_int8_atlas`` black-box model, so
the structural wrapper (the one true core) is what is simulated.

Run it via the repo venv (numpy is needed by the generators):

    ./run_rtl_tests.sh                          # default size set x default seeds
    ./run_rtl_tests.sh --cases 8x8x8 16x16x16   # specific sizes
    ./run_rtl_tests.sh --seeds 1 7              # specific seeds
    ./run_rtl_tests.sh --keep                   # keep tb/generated/*.v for inspection
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from .rtl import generate_combined_core_verilog, generate_k_spatial_combined_core_verilog
from .golden import generate_tb, generate_tb_with_data, _gen_catapult_tb, pack_a_chunk, pack_b_chunk, \
    pack_bias, pack_c_row, two_stage_reference, hex_literal
from .geometry import k_chunks as _k_chunks, resolve_reuse_factor, resolve_fold_m, resolve_fold_n, \
    feed_beats as _feed_beats
from gemm_ip.weights import build_weight_rom_k_spatial, build_weight_rom_fold_n, build_weight_rom_combined_fold

GEN_DIR = HERE / "tb" / "generated"

# A spread of shapes: single-tile, multi-row-tile, multi-col-tile, multi-tile,
# multi-K-chunk, and non-8-multiple tails. Each case is (m, k, n, rf); rf=None
# means "rf = k_chunks", i.e. today's chunked endpoint (k_spatial=1) -- the
# original 10 shapes, unchanged generator path. The extra cases below exercise
# the general ReuseFactor fold: (8,24,8,2)/(9,17,10,2)/(16,40,16,2) are
# multi-pass K-spatial (9x17x10 pads its last pass); (8,16,8,1) and
# (24,16,8,1) legalize to full-K (rf=1 -> k_spatial=k_chunks); (16,40,16,5)
# legalizes to chunked (rf=k_chunks -> k_spatial=1), same generator path as
# the default cases.
DEFAULT_CASES = [
    (8, 8, 8, None),
    (16, 8, 8, None),
    (8, 8, 16, None),
    (16, 16, 16, None),
    (8, 16, 8, None),
    (24, 16, 8, None),
    (14, 6, 6, None),
    (9, 17, 10, None),
    (5, 5, 5, None),
    (12, 10, 10, None),
    (8, 24, 8, 2),
    (9, 17, 10, 2),
    (16, 40, 16, 2),
    (8, 16, 8, 1),
    (16, 40, 16, 5),
    (24, 16, 8, 1),
]
DEFAULT_SEEDS = [1, 7, 42]

# Weight-stationary (const-weight ROM) cases, m > n: the ROM now holds only
# passes*n entries (see gemm_ip/weights.py + rtl.py _weight_rom_block), so beats
# t >= n of each pass — which only exist here because input_beats = max(m,n) = m
# > n — must read back zero via the beat_ctr/rom_base mux. One rf=1 (full-K, one
# pass) and one rf>1 (multi-pass) case, both m > n, exercise that tail.
ROM_CASES = [
    (16, 8, 8, 1),   # m > n, full-K (rf=1 -> k_spatial=k_chunks=1 pass)
    (16, 24, 8, 2),  # m > n, multi-pass K-spatial (rf=2 -> passes=k_chunks/2)
]

# Fold-M regression: (m, k, n, rf). The core is generated for M_g = 8*mg rows
# (K and N fully spatial, one K pass); the golden iverilog testbench's
# back-to-back multi-frame mode (back2back=True, num_vectors=m_passes) is the
# regression hook -- it feeds m_passes distinct frames of M_g rows each and
# checks every output row against golden in frame order, exactly how the
# wrapper RUN loop replays the same rf=1 core (confirmed by standalone probing
# of the RTL). (9,17,10,2) pads the last pass; (16,40,16,2)
# is an exact-division fold; (8,16,8,1) is the RF=1 (m_passes=1, single-frame)
# identity case; (16,40,16,5) drives RF to its grid_rows bound (5 -> legalizes
# down since grid_rows(16)=2).
FOLD_M_CASES = [
    (9, 17, 10, 2),
    (16, 40, 16, 2),
    (8, 16, 8, 1),
    (16, 40, 16, 5),
]

# Fold-N regression: (m, k, n, rf). The core is generated for (M, K, N_g)
# columns (K and M fully spatial, one K pass); the golden testbench's
# GROUP-AWARE back-to-back mode (num_vectors=n_passes, fold_n_groups=n_passes)
# feeds every frame the SAME A while frame g's B columns are group g's real
# slice of one shared full-width B, and checks frame g's output rows against
# A @ B[:, group g] (padded tail columns expected zero) -- exactly how the
# wrapper's fold-N RUN loop composes the logical N-wide GEMM back together.
# (12,24,32,2) exact division; (12,24,32,4) rf=4 pads the last group's tail
# columns; (20,24,16,2) M > N_g; (9,17,10,2) M < N_g and K > 8 (non-8-multiple
# K exercises the K-spatial tail mask together with the group counter).
FOLD_N_CASES = [
    (8, 8, 32, 2),    # single K chunk (chunked ROM layout) -- the width bug's shape
    (12, 8, 24, 3),   # single K chunk, three groups, ragged last group
    (12, 24, 32, 2),
    (12, 24, 32, 4),
    (20, 24, 16, 2),
    (9, 17, 10, 2),
]


# Combined M+N fold regression: (m, k, n, rf_m, rf_n). K stays single-pass (k_spatial=k_chunks, no K
# fold) so this isolates the M+N composition. m=33 (grid_rows=5) with rf_m=3
# legalizes to mg=2 row-tiles/group, m_passes=3 (ragged last mg block, only 1
# real row-tile); n=40 (grid_cols=5) with rf_n=3 legalizes to cg=2
# col-tiles/group, n_passes=3 (also ragged). m_passes*n_passes = 9 frames,
# neither axis an exact division -- exercises the A-slice-keyed-on-mg
# composition and grp_ctr cycling well past n_passes (today only ever tested
# at exactly n_passes frames).
COMBINED_CASES = [
    (33, 17, 40, 3, 3),
]

# Combined M+K and M+K+N fold regression (5b-ii/iii): (m, k, n, rf_m, rf_k, rf_n).
# rf_k folds K (k_spatial<k_chunks, i.e. multi-pass K-in-time) SIMULTANEOUSLY
# with M (and, for the M+K+N case, N too) via run_case's generalized "mn"
# combined mode (rf as a 3-tuple). Per 5a, K-pass handling is a WITHIN-frame
# loop in the per-group core (generate_k_spatial_combined_core_verilog),
# orthogonal to which axes fold the frame count, so composing it with M/N
# needed no rtl.py change -- only relaxing run_case's k_spatial=k_chunks
# hardcode on the "mn" path.
#
# (33, 17, 40, 3, ..., 1): M+K only (rf_n=1 -> n_passes=1, single N group).
# m=33/rf_m=3 is the same ragged M-fold as COMBINED_CASES (mg=2, m_passes=3,
# last mg block only 1 real row-tile); k=17/rf_k=2 mirrors DEFAULT_CASES'
# (9,17,10,2) multi-pass K-spatial case (k_chunks=3, rf=2 pads the last pass).
# (33, 17, 40, 3, 2, 3): same M and K fold, PLUS rf_n=3 (n=40 -> cg=2,
# n_passes=3, also ragged, same as COMBINED_CASES) -- all three axes folded
# and ragged simultaneously.
COMBINED_MK_CASES = [
    (33, 17, 40, 3, 2, 1),
]
COMBINED_MKN_CASES = [
    (33, 17, 40, 3, 2, 3),
]


def _run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True)


# Single-branch RTL: every generated core is the structural wrapper, so each
# iverilog compile links the tensor_slice_int8_atlas black-box model with
# -DSYNTHESIS. Resolved next to this file (clone tree) or under the sandbox
# final/ dir (vendor tree).
_MODEL_CANDIDATES = (
    HERE / "tensor_slice_int8_atlas.v",
    HERE.parents[3] / "final" / "tensor_slice_int8_atlas.v",
)
MODEL = next((p for p in _MODEL_CANDIDATES if p.is_file()), None)


def _compile(out_path, tb_path, rtl_path):
    """Compile one TB + structural core + black-box model (single-branch)."""
    if MODEL is None:
        raise FileNotFoundError(
            "tensor_slice_int8_atlas.v not found next to run_rtl_tests.py nor under "
            "ts-rtl-work/final/; cannot link the structural black-box model."
        )
    return _run(["iverilog", "-g2012", "-DSYNTHESIS", "-o", str(out_path),
                 str(tb_path), str(rtl_path), str(MODEL)])


# ── Two-stage requant (S1/S2/bias/out_width) regression ───────────────────────
#
# Exercises the two-stage requant directly at the
# rtl.py/golden.py level (package.py's accum_precision-driven S1/S2 derivation
# is exercised separately, at the packaging layer). Each case pins s1/s2/
# out_width/bias_codes explicitly.
REQUANT_CASES = [
    # (label, m, k, n, k_spatial, s1, s2, out_width, bias_codes)
    ("nonuniform_bias", 8, 8, 8, 1, 0, 0, 8,
     [100, -200, 300, -400, 50, -50, 0, 123]),
    ("bias_frac_gt_out", 8, 8, 8, 1, 0, 3, 8,
     [13, -13, 5, -5, 1, -1, 7, -7]),
    ("k_spatial_bias", 16, 40, 16, 2, 0, 1, 8,
     [9, -9, 17, -17, 3, -3, 21, -21, 9, -9, 17, -17, 3, -3, 21, -21]),
    ("forced_s1", 8, 8, 8, 1, 3, 1, 8,
     [11, -11, 4, -4, 2, -2, 6, -6]),
]


def run_requant_case(label, m, k, n, k_spatial, s1, s2, out_width, bias_codes, seed=1):
    """Generate a core pinned to explicit (s1, s2, out_width, bias_codes) and
    check it against golden.two_stage_reference (the folded, phase-1 sim
    golden -- see golden.py). Returns (ok, log).
    """
    stem = f"requant_{label}"
    mod = f"{stem}_wrapper"
    rtl_path = GEN_DIR / f"{stem}.v"
    tb_path = GEN_DIR / f"tb_{stem}.v"
    out_path = GEN_DIR / f"{stem}.out"

    # Bias is compile-time now (decision 4): pass the SAME bias_codes to the
    # RTL generator (baked as a ROM) and to golden's TB generator (its
    # reference model), so the DUT and the checker agree on one fixed bias.
    if k_spatial == 1:
        rtl_path.write_text(generate_combined_core_verilog(
            m, k, n, module_name=mod, s1=s1, s2=s2, out_width=out_width,
            bias_codes=bias_codes))
    else:
        rtl_path.write_text(generate_k_spatial_combined_core_verilog(
            m, k, n, module_name=mod, k_spatial=k_spatial, s1=s1, s2=s2, out_width=out_width,
            bias_codes=bias_codes))
    tb_path.write_text(generate_tb(
        m, k, n, module_name=mod, seed=seed, k_spatial=k_spatial, num_vectors=5,
        bias_codes=bias_codes, s1=s1, s2=s2, out_width=out_width, has_bias=True))

    comp = _compile(out_path, tb_path, rtl_path)
    if comp.returncode != 0:
        return False, "iverilog compile failed\n" + comp.stdout + comp.stderr
    sim = _run(["vvp", str(out_path)])
    log = sim.stdout + sim.stderr
    if sim.returncode != 0:
        return False, "vvp failed\n" + log
    if "ALL_PASS" not in log or "FAILURES=" in log:
        return False, log
    return True, log


# Real Catapult ``*_gemm_ip_stream_buffered_b`` protocol regression
# (modeled on the gemm_mha_aV_h0 av_pkg_tb). _gen_catapult_tb's
# sequential/back2back templates drive a_rows/b_cols together every beat --
# which turns out to be EXACTLY the same per-cycle cadence the real
# wrapper's RUN loop uses (weight_cols is a plain array pre-buffered by
# READ_B_COLS before this function is even called, but inside the RUN loop
# b_cols is re-driven from that array on the SAME beat index t as the row
# currently being read off a_stream, every cycle -- see
# gemm_mha_aV_h0_gemm_ip.h's gemm_ip_stream_buffered_b). This case pins that
# exact m=4/k=4/n=8 shape with several back-to-back frames (period =
# TOTAL_INPUT_BEATS+1 = 9, matching "Each frame is ONE in_valid=0 beat + 8
# in_valid beats" in the wrapper's own comment) as a permanent regression
# so any future change to the wrapper's frame capture/emit timing that
# breaks this cadence is caught here. Symmetric-only quantization scope:
# operands are signed (no zero point).
def run_buffered_b_case(label="mha_tiny_buffered_b", seed=1, num_frames=4,
                        m=4, k=4, n=8, out_width=8):
    """Buffered-B-then-streamed-A protocol regression: several back-to-back
    frames of the exact gemm_mha_aV_h0 shape with symmetric operands,
    checked row-by-row against an exact integer reference."""
    from .rtl import generate_combined_core_verilog
    from .golden import _gen_catapult_tb, pack_a_chunk, pack_b_chunk, pack_bias, pack_c_row, \
        two_stage_reference
    import numpy as np

    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    k_chunks = (k + 7) // 8
    input_beats = _feed_beats(m, n, k_chunks)

    rng = np.random.default_rng(seed)
    max_val = max(1, int((127 / max(k, 1)) ** 0.5))
    all_a_stim, all_b_stim, all_golden = [], [], []
    for _f in range(num_frames):
        A = rng.integers(-max_val, max_val + 1, size=(m, k))
        B = rng.integers(-max_val, max_val + 1, size=(k, n))
        a_stim, b_stim = [], []
        for chunk in range(k_chunks):
            for t in range(input_beats):
                a_stim.append(pack_a_chunk(A, t, chunk, grid_rows, m, k))
                b_stim.append(pack_b_chunk(B, t, chunk, grid_cols, n, k))
        all_a_stim.append(a_stim)
        all_b_stim.append(b_stim)
        golden = two_stage_reference(A.astype(np.int64), B.astype(np.int64), None,
                                     s1=0, s2=0, out_width=out_width)
        cat_golden = []
        for rt in range(grid_rows):
            for row in range(8):
                cat_golden.append(pack_c_row(golden, rt, row, grid_cols, m, n, out_width=out_width))
        all_golden.append(cat_golden)

    stem = f"bb_{label}"
    mod = f"{stem}_wrapper"
    rtl_path = GEN_DIR / f"{stem}.v"
    tb_path = GEN_DIR / f"tb_{stem}.v"
    out_path = GEN_DIR / f"{stem}.out"
    rtl_path.write_text(generate_combined_core_verilog(
        m, k, n, module_name=mod, out_width=out_width, s1=0, s2=0))
    tb_path.write_text(_gen_catapult_tb(
        m, k, n, mod, seed, all_a_stim, all_b_stim,
        [pack_bias(np.zeros(n, dtype=np.int64), grid_cols, n)] * num_frames, all_golden,
        out_width=out_width, back2back=True))

    comp = _compile(out_path, tb_path, rtl_path)
    if comp.returncode != 0:
        return False, "iverilog compile failed\n" + comp.stdout + comp.stderr
    sim = _run(["vvp", str(out_path)])
    log = sim.stdout + sim.stderr
    if sim.returncode != 0:
        return False, "vvp failed\n" + log
    if "ALL_PASS" not in log or "FAILURES=" in log:
        return False, log
    return True, log


def _full_width_catapult_tb(m, k, n, module_name, seed, all_a_stim, all_b_stim, all_golden, out_width):
    """Minimal self-checking back-to-back testbench with a CORRECT c_row/golden
    width (``grid_cols * 8 * out_width`` bits -- one full output beat, 8 lanes
    per column tile), for shapes ``_gen_catapult_tb`` cannot check every lane
    of (its own ``c_row``/``golden`` width, ``grid_cols * out_width // 8``
    bytes, is missing the "8 lanes per column tile" factor rtl.py's c_width
    actually uses -- see the K-padding zero-point regression below). Every vector runs
    back-to-back (no reset in between, like the real streaming wrapper),
    checking c_row against the full-width golden row on every out_valid beat.
    """
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    k_chunks = (k + 7) // 8
    input_beats = _feed_beats(m, n, k_chunks)
    total_input_beats = k_chunks * input_beats
    aw = grid_rows * 64
    bw = grid_cols * 64
    cw = grid_cols * 8 * out_width
    num_vectors = len(all_a_stim)

    a_init, b_init, golden_init = [], [], []
    for v in range(num_vectors):
        for t in range(total_input_beats):
            a_init.append(f"        a_stim[{v}][{t}] = {hex_literal(all_a_stim[v][t], aw // 8)};")
            b_init.append(f"        b_stim[{v}][{t}] = {hex_literal(all_b_stim[v][t], bw // 8)};")
        for row in range(m):
            golden_init.append(f"        golden[{v}][{row}] = {hex_literal(all_golden[v][row], cw // 8)};")

    return f"""\
`timescale 1ns/1ps
module tb_full_width_{module_name};
    localparam NV = {num_vectors};
    localparam TOTAL_INPUT_BEATS = {total_input_beats};
    localparam TOTAL_ROWS = {m};

    reg clk = 0, rst = 1, en = 1, preload_valid = 0, in_valid = 0;
    reg [{aw - 1}:0] a_rows = 0;
    reg [{bw - 1}:0] b_cols = 0;
    wire [{cw - 1}:0] c_row;
    wire out_valid, out_last;

    {module_name} dut(.clk(clk), .rst(rst), .en(en), .a_rows(a_rows), .b_cols(b_cols),
        .preload_valid(preload_valid), .in_valid(in_valid),
        .c_row(c_row), .out_valid(out_valid), .out_last(out_last));

    always #5 clk = ~clk;

    reg [{aw - 1}:0] a_stim [0:NV-1][0:TOTAL_INPUT_BEATS-1];
    reg [{bw - 1}:0] b_stim [0:NV-1][0:TOTAL_INPUT_BEATS-1];
    reg [{cw - 1}:0] golden [0:NV-1][0:TOTAL_ROWS-1];

    integer vec_idx, t, out_row_idx, pass_count, fail_count, cycle_ctr, out_last_count;

    initial begin
{chr(10).join(a_init)}
{chr(10).join(b_init)}
{chr(10).join(golden_init)}
    end

    initial begin
        $display("=== Full-width Catapult TB {m}x{k}x{n}  ({num_vectors} vectors) ===");
        pass_count = 0; fail_count = 0; out_last_count = 0; out_row_idx = 0;
        for (vec_idx = 0; vec_idx < NV; vec_idx = vec_idx + 1) begin
            if (vec_idx == 0) begin
                preload_valid <= 0; in_valid <= 0; rst <= 1;
                @(posedge clk); rst <= 0; en <= 1; @(posedge clk);
            end
            preload_valid <= 1; @(posedge clk); preload_valid <= 0;
            in_valid <= 1;
            a_rows <= a_stim[vec_idx][0]; b_cols <= b_stim[vec_idx][0];
            @(posedge clk);
            for (t = 1; t < TOTAL_INPUT_BEATS; t = t + 1) begin
                a_rows <= a_stim[vec_idx][t]; b_cols <= b_stim[vec_idx][t];
                @(posedge clk);
            end
            in_valid <= 0; a_rows <= 0; b_cols <= 0;
        end
        cycle_ctr = 0;
        while (out_last_count < NV && cycle_ctr < 2000) begin
            @(posedge clk); cycle_ctr = cycle_ctr + 1;
        end
        if (out_last_count != NV) begin
            $display("TIMEOUT: only %0d out_last events, expected %0d", out_last_count, NV);
            fail_count = fail_count + 1;
        end
        @(posedge clk);
        if (fail_count == 0) $display("ALL_PASS  (%0d vectors)", NV);
        else $display("FAILURES=%0d", fail_count);
        $finish;
    end

    always @(posedge clk) begin
        if (out_last) begin
            out_last_count <= out_last_count + 1;
            if (out_row_idx != TOTAL_ROWS - 1) begin
                $display("FAIL vec %0d out_last row %0d", out_last_count, out_row_idx);
                fail_count = fail_count + 1;
            end
        end
        if (out_valid) begin
            if (out_row_idx < TOTAL_ROWS) begin
                if (c_row !== golden[out_last_count][out_row_idx]) begin
                    $display("FAIL vec %0d row %0d: got %h expected %h",
                             out_last_count, out_row_idx, c_row, golden[out_last_count][out_row_idx]);
                    fail_count = fail_count + 1;
                end else pass_count = pass_count + 1;
                out_row_idx <= out_row_idx + 1;
            end
        end
        if (out_last) out_row_idx <= 0;
    end
endmodule
"""


def measure_s1_delta(label, m, k, n, s1, s2, out_width, bias_codes, seed=1, num_vectors=5):
    """Measure the S1>0 double-rounding delta (decision 7): max |two_stage -
    single_round| over a few random vectors, in output LSBs."""
    import numpy as np
    from .golden import _random_matrices, two_stage_reference, single_round_reference
    max_delta = 0
    n_mismatch = 0
    total = 0
    for v in range(num_vectors):
        seed_v = seed + v
        rng = np.random.default_rng(seed_v)
        max_val = max(1, int((127 / max(k, 1)) ** 0.5))
        A = rng.integers(-max_val, max_val + 1, size=(m, k), dtype=np.int8)
        B = rng.integers(-max_val, max_val + 1, size=(k, n), dtype=np.int8)
        two_stage = two_stage_reference(A, B, bias_codes, s1=s1, s2=s2, out_width=out_width)
        single = single_round_reference(A, B, bias_codes, s1=s1, s2=s2, out_width=out_width)
        diff = np.abs(two_stage.astype(np.int64) - single.astype(np.int64))
        max_delta = max(max_delta, int(diff.max()))
        n_mismatch += int((diff != 0).sum())
        total += diff.size
    return max_delta, n_mismatch, total


def run_case(m, k, n, seed, rf=None, weights_in_core=False, fold_axis="k",
            bias_codes=None, s1=0, s2=0, out_width=8):
    """Generate core RTL + TB for one shape/seed/ReuseFactor/axis, simulate, return (ok, log).

    Under fold_axis "m" the core is the rf=1 (full-K, full-N) core for M_g rows and
    the testbench's back-to-back multi-frame mode feeds m_passes frames -- the
    RTL-level equivalent of the wrapper's multi-frame RUN loop (see FOLD_M_CASES).
    """
    n_passes = 1
    core_n = n
    if fold_axis == "m":
        fm = resolve_fold_m(m, rf if rf is not None else 1)
        for w in fm["warnings"]:
            print(w, file=sys.stderr)
        rf_use = fm["reuse_factor"]
        core_m = fm["mg"] * 8
        num_vectors = fm["m_passes"]
        k_spatial = _k_chunks(k)
    elif fold_axis == "n":
        fn = resolve_fold_n(n, rf if rf is not None else 1)
        for w in fn["warnings"]:
            print(w, file=sys.stderr)
        rf_use = fn["reuse_factor"]
        core_m = m
        core_n = fn["cg"] * 8
        n_passes = fn["n_passes"]
        num_vectors = fn["n_passes"]
        k_spatial = _k_chunks(k)
    elif fold_axis == "mn":
        # Combined M+N / M+K+N: rf is (rf_m, rf_n)
        # -- unchanged, K stays single-pass (k_spatial=k_chunks) -- or
        # (rf_m, rf_k, rf_n) to ALSO fold K. M and N are legalized independently
        # via the same fold-M/fold-N geometry the single-axis paths use; K is
        # legalized via the same K-partition geometry the plain fold_axis="k"
        # path uses (resolve_reuse_factor). Core is sized for ONE (mg, ng) tile
        # (K folding is a WITHIN-frame pass loop in the per-group core, per 5a
        # -- orthogonal to which axes fold the frame count); TB feeds
        # m_passes*n_passes frames, A-slice keyed on mg = frame // n_passes.
        if isinstance(rf, tuple) and len(rf) == 3:
            rf_m, rf_k, rf_n = rf
        else:
            rf_m, rf_n = rf if isinstance(rf, tuple) else (rf, rf)
            rf_k = None
        fm = resolve_fold_m(m, rf_m if rf_m is not None else 1)
        for w in fm["warnings"]:
            print(w, file=sys.stderr)
        fn = resolve_fold_n(n, rf_n if rf_n is not None else 1)
        for w in fn["warnings"]:
            print(w, file=sys.stderr)
        core_m = fm["mg"] * 8
        core_n = fn["cg"] * 8
        m_passes = fm["m_passes"]
        n_passes = fn["n_passes"]
        num_vectors = m_passes * n_passes
        if rf_k is None:
            k_spatial = _k_chunks(k)
            rf_use = (fm["reuse_factor"], fn["reuse_factor"])
        else:
            rk = resolve_reuse_factor(k, rf_k)
            for w in rk["warnings"]:
                print(w, file=sys.stderr)
            k_spatial = rk["k_spatial"]
            rf_use = (fm["reuse_factor"], rk["reuse_factor"], fn["reuse_factor"])
    else:
        rf_use = rf if rf is not None else _k_chunks(k)
        resolved = resolve_reuse_factor(k, rf_use)
        for w in resolved["warnings"]:
            print(w, file=sys.stderr)
        core_m = m
        num_vectors = 10
        k_spatial = resolved["k_spatial"]

    rf_tag = "_".join(str(x) for x in rf_use) if isinstance(rf_use, tuple) else str(rf_use)
    stem = f"gemm_{m}x{k}x{n}_rf{rf_tag}_s{seed}"
    if fold_axis != "k":
        stem += f"_fold{fold_axis}"
    if weights_in_core:
        stem += "_rom"
    mod = f"{stem}_wrapper"
    rtl = GEN_DIR / f"{stem}.v"
    tb = GEN_DIR / f"tb_{stem}.v"
    out = GEN_DIR / f"{stem}.out"

    weight_rom = None
    fixed_B = None
    if weights_in_core and fold_axis in ("n", "mn"):
        import numpy as np
        rng = np.random.default_rng(seed)
        max_val = max(1, int((127 / max(k, 1)) ** 0.5))
        fixed_B = rng.integers(-max_val, max_val + 1, size=(k, n_passes * core_n), dtype=np.int8)
        # Match the production flow's builder (package.py). When K also folds in
        # time (passes > 1) alongside N (n_passes > 1) -- the combined M/K/N fold
        # -- package.py bakes the ROM chunk-major/ng-minor via
        # build_weight_rom_combined_fold (the layout the C csim, the sim-branch
        # weight feeder, and the structural rom_addr all address). fold-N alone
        # (passes == 1) keeps the group-major build_weight_rom_fold_n layout.
        passes = -(-_k_chunks(k) // k_spatial)
        if fold_axis == "mn" and n_passes > 1 and passes > 1:
            weight_rom = build_weight_rom_combined_fold(fixed_B, core_m, core_n, k, k_spatial, n_passes)
        else:
            # ROM holds every group's columns back to back (base g*core_n).
            weight_rom = build_weight_rom_fold_n(fixed_B, core_m, core_n, k, k_spatial, n_passes)
    elif weights_in_core:
        import numpy as np
        rng = np.random.default_rng(seed)
        max_val = max(1, int((127 / max(k, 1)) ** 0.5))
        fixed_B = rng.integers(-max_val, max_val + 1, size=(k, n), dtype=np.int8)
        weight_rom = build_weight_rom_k_spatial(fixed_B, core_m, n, k, k_spatial)

    n_passes_rtl = n_passes if fold_axis in ("n", "mn") else 1
    has_bias = bias_codes is not None
    # Combined M/K/N fold folds M in the wrapper (M_PASSES frames), unlike the
    # harness-side M-fold replay on the other axes -- thread m_passes/logical
    # extents through the structural core generator so the emitted core
    # actually declares M_PASSES (package.py:2313 does the same).
    rtl.write_text(generate_k_spatial_combined_core_verilog(
        core_m, k, core_n, module_name=mod, k_spatial=k_spatial, weight_rom=weight_rom,
        n_passes=n_passes_rtl, s1=s1, s2=s2, out_width=out_width, bias_codes=bias_codes,
        m_passes=(m_passes if fold_axis == "mn" else 1),
        logical_m=(m if fold_axis == "mn" else None),
        logical_n=(n if fold_axis == "mn" else None)))
    tb.write_text(generate_tb(core_m, k, core_n, module_name=mod, seed=seed, k_spatial=k_spatial,
                              num_vectors=num_vectors, back2back=(fold_axis in ("m", "n", "mn")),
                              weights_in_core=weights_in_core, fixed_B=fixed_B,
                              fold_n_groups=(n_passes if fold_axis == "n" else None),
                              fold_mn=((m_passes, n_passes) if fold_axis == "mn" else None),
                              bias_codes=bias_codes, s1=s1, s2=s2, out_width=out_width,
                              has_bias=has_bias))

    comp = _compile(out, tb, rtl)
    if comp.returncode != 0:
        return False, "iverilog compile failed\n" + comp.stdout + comp.stderr
    sim = _run(["vvp", str(out)])
    log = sim.stdout + sim.stderr
    if sim.returncode != 0:
        return False, "vvp failed\n" + log
    if "ALL_PASS" not in log or "FAILURES=" in log:
        return False, log
    return True, log


def run(cases=None, seeds=None, keep=False):
    """Generate + simulate the wrapper for each (shape, seed); return 0 if all pass.

    ``cases`` is a list of ``(m, k, n)`` tuples (default: DEFAULT_CASES); ``seeds``
    a list of ints (default: DEFAULT_SEEDS). Returns 2 if iverilog/vvp are absent.
    """
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        print("ERROR: iverilog/vvp not found on PATH", file=sys.stderr)
        return 2

    rom_cases = list(ROM_CASES) if not cases else []
    fold_m_cases = list(FOLD_M_CASES) if not cases else []
    fold_n_cases = list(FOLD_N_CASES) if not cases else []
    combined_cases = list(COMBINED_CASES) if not cases else []
    combined_mk_cases = list(COMBINED_MK_CASES) if not cases else []
    combined_mkn_cases = list(COMBINED_MKN_CASES) if not cases else []
    cases = list(cases) if cases else list(DEFAULT_CASES)
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for case in cases:
        m, k, n, rf = case if len(case) == 4 else (*case, None)
        for seed in seeds:
            ok, log = run_case(m, k, n, seed, rf=rf)
            rf_tag = f" rf={rf}" if rf is not None else ""
            tag = f"{m}x{k}x{n}{rf_tag} seed={seed}"
            print(f"{'PASS' if ok else 'FAIL'} {tag}")
            if not ok:
                print(log)
                failures.append(tag)

    for case in rom_cases:
        m, k, n, rf = case if len(case) == 4 else (*case, None)
        for seed in seeds:
            ok, log = run_case(m, k, n, seed, rf=rf, weights_in_core=True)
            rf_tag = f" rf={rf}" if rf is not None else ""
            tag = f"{m}x{k}x{n}{rf_tag} seed={seed} rom"
            print(f"{'PASS' if ok else 'FAIL'} {tag}")
            if not ok:
                print(log)
                failures.append(tag)

    for case in fold_m_cases:
        m, k, n, rf = case
        for seed in seeds:
            for rom in (False, True):
                ok, log = run_case(m, k, n, seed, rf=rf, weights_in_core=rom, fold_axis="m")
                tag = f"{m}x{k}x{n} rf={rf} seed={seed} fold_axis=m{' rom' if rom else ''}"
                print(f"{'PASS' if ok else 'FAIL'} {tag}")
                if not ok:
                    print(log)
                    failures.append(tag)

    for case in fold_n_cases:
        m, k, n, rf = case
        for seed in seeds:
            for rom in (False, True):
                ok, log = run_case(m, k, n, seed, rf=rf, weights_in_core=rom, fold_axis="n")
                tag = f"{m}x{k}x{n} rf={rf} seed={seed} fold_axis=n{' rom' if rom else ''}"
                print(f"{'PASS' if ok else 'FAIL'} {tag}")
                if not ok:
                    print(log)
                    failures.append(tag)

    # Combined M+N fold regression (5b-i checkpoint): streamed-B and
    # weight-stationary (ROM, exercises grp_ctr cycling past n_passes).
    for m, k, n, rf_m, rf_n in combined_cases:
        for seed in seeds:
            for rom in (False, True):
                ok, log = run_case(m, k, n, seed, rf=(rf_m, rf_n), weights_in_core=rom,
                                   fold_axis="mn")
                tag = f"{m}x{k}x{n} rf=({rf_m},{rf_n}) seed={seed} fold_axis=mn{' rom' if rom else ''}"
                print(f"{'PASS' if ok else 'FAIL'} {tag}")
                if not ok:
                    print(log)
                    failures.append(tag)

    # Combined M+K fold regression (5b-ii checkpoint): K folds (k_spatial <
    # k_chunks) simultaneously with M, N single-pass (rf_n=1).
    for m, k, n, rf_m, rf_k, rf_n in combined_mk_cases:
        for seed in seeds:
            for rom in (False, True):
                ok, log = run_case(m, k, n, seed, rf=(rf_m, rf_k, rf_n), weights_in_core=rom,
                                   fold_axis="mn")
                tag = f"{m}x{k}x{n} rf=({rf_m},{rf_k},{rf_n}) seed={seed} fold_axis=mk{' rom' if rom else ''}"
                print(f"{'PASS' if ok else 'FAIL'} {tag}")
                if not ok:
                    print(log)
                    failures.append(tag)

    # Combined M+K+N fold regression (5b-iii checkpoint): all three axes fold
    # simultaneously.
    for m, k, n, rf_m, rf_k, rf_n in combined_mkn_cases:
        for seed in seeds:
            for rom in (False, True):
                ok, log = run_case(m, k, n, seed, rf=(rf_m, rf_k, rf_n), weights_in_core=rom,
                                   fold_axis="mn")
                tag = f"{m}x{k}x{n} rf=({rf_m},{rf_k},{rf_n}) seed={seed} fold_axis=mkn{' rom' if rom else ''}"
                print(f"{'PASS' if ok else 'FAIL'} {tag}")
                if not ok:
                    print(log)
                    failures.append(tag)

    # Two-stage requant regression (S1/S2/bias/out_width), run unconditionally
    # (not gated on --cases -- it targets the numerics, not a shape sweep).
    requant_failures = []
    for label, cm, ck, cn, cks, cs1, cs2, cow, cbias in REQUANT_CASES:
        ok, log = run_requant_case(label, cm, ck, cn, cks, cs1, cs2, cow, cbias)
        tag = f"requant:{label} {cm}x{ck}x{cn} s1={cs1} s2={cs2} out_width={cow}"
        print(f"{'PASS' if ok else 'FAIL'} {tag}")
        if not ok:
            print(log)
            failures.append(tag)
            requant_failures.append(tag)
        if label == "forced_s1":
            max_delta, n_mismatch, n_total = measure_s1_delta(
                label, cm, ck, cn, cs1, cs2, cow, cbias)
            print(f"S1>0 delta (folded-sim vs single-round reference): "
                  f"max={max_delta} output LSB, mismatched {n_mismatch}/{n_total} elements")

    # Fold-N + non-uniform bias: several groups, bias differing ACROSS groups
    # (not just across columns within one group) -- this is the case that
    # would fail if the wrong group's bias were read at emit time (the live
    # grp_ctr vs. the frozen per-frame out_grp bug the fold-N-bias item
    # exists to avoid). (12, 24, 32, rf=4) legalizes to n_passes=4 groups.
    fn4 = resolve_fold_n(32, 4)
    core_n4 = fn4["cg"] * 8
    n_passes4 = fn4["n_passes"]
    # Group g's codes are (g+1)*1000 + local column index, alternating sign --
    # unmistakably different per group, so reading group g's row with group
    # (g+1 mod n_passes)'s bias is never accidentally correct.
    foldn_bias_codes = [
        ((-1) ** c) * ((g + 1) * 1000 + c) for g in range(n_passes4) for c in range(core_n4)
    ]
    ok, log = run_case(12, 24, 32, seed=3, rf=4, fold_axis="n", bias_codes=foldn_bias_codes)
    tag = f"requant:foldn_nonuniform_bias 12x24x32 rf=4 n_passes={n_passes4} core_n={core_n4}"
    print(f"{'PASS' if ok else 'FAIL'} {tag}")
    if not ok:
        print(log)
        failures.append(tag)
        requant_failures.append(tag)

    # Same, on the chunked K_SPATIAL=1 core (single K chunk): that emitter is
    # called with n_passes=1 (its ROMs are hoisted), and once folded group 0's
    # bias into constants for every group, so group 1's outputs were off.
    fn2 = resolve_fold_n(32, 2)
    core_n2 = fn2["cg"] * 8
    bias_k1 = [((-1) ** c) * ((g + 1) * 1000 + c)
               for g in range(fn2["n_passes"]) for c in range(core_n2)]
    ok, log = run_case(8, 8, 32, seed=3, rf=2, fold_axis="n", bias_codes=bias_k1, s2=3)
    tag = f"requant:foldn_nonuniform_bias_kspatial1 8x8x32 rf=2 n_passes={fn2['n_passes']}"
    print(f"{'PASS' if ok else 'FAIL'} {tag}")
    if not ok:
        print(log)
        failures.append(tag)
        requant_failures.append(tag)

    # Real Catapult gemm_ip_stream_buffered_b protocol regression (gemm_mha_aV_h0
    # av_pkg_tb shape: m=4/k=4/n=8, symmetric operands, back-to-back frames).
    # Symmetric-only quantization scope: the back-to-back frame cadence is
    # covered with signed operands; zero-point variants were removed.
    ok, log = run_buffered_b_case()
    tag = "buffered_b_protocol 4x4x8 symmetric (back-to-back frames)"
    print(f"{'PASS' if ok else 'FAIL'} {tag}")
    if not ok:
        print(log)
        failures.append(tag)

    if not keep:
        shutil.rmtree(GEN_DIR, ignore_errors=True)

    total = (len(cases) * len(seeds) + len(rom_cases) * len(seeds) + len(fold_m_cases) * len(seeds) * 2
             + len(fold_n_cases) * len(seeds) * 2 + len(combined_cases) * len(seeds) * 2
             + len(combined_mk_cases) * len(seeds) * 2 + len(combined_mkn_cases) * len(seeds) * 2
             + len(REQUANT_CASES) + 1
             + 1)
    if failures:
        print(f"\n{len(failures)}/{total} FAILED:")
        for t in failures:
            print(f"  {t}")
        return 1
    print(f"\nALL PASSED: {total} runs")
    return 0


def main():
    ap = argparse.ArgumentParser(description="RTL-level iverilog test for the GEMM wrapper")
    ap.add_argument("--cases", nargs="*", help="GEMM shapes as MxKxN, e.g. 8x8x8 16x16x16")
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS,
                    help="Random seeds for the self-checking testbenches")
    ap.add_argument("--keep", action="store_true",
                    help="Keep the generated RTL/TB under tb/generated/ for inspection")
    args = ap.parse_args()

    cases = []
    for c in (args.cases or []):
        try:
            m, k, n = (int(x) for x in c.lower().split("x"))
            cases.append((m, k, n))
        except Exception:
            print(f"ERROR: invalid case '{c}', expected MxKxN", file=sys.stderr)
            return 2

    return run(cases=cases or None, seeds=args.seeds, keep=args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
