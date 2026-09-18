#!/usr/bin/env python3
"""
rtl.py
Catapult RTL generator — tensor-slice grid with clk/rst/en protocol.
M x N output grid with K inner dimension.

Key design:
  - Single continuous operation: collect K beats, feed tiles in chained fashion,
    drain outputs once.
  - All masks resolved in Python, emitted as literals (no generate arithmetic)
  - Slices unrolled explicitly for portability
  - Output mux: each tile-row (r) streams 8 result rows in order,
    staggered correctly against other tile-rows
"""
import argparse
from pathlib import Path

from . import geometry as _geometry
tail_mask_hex, vm, _total_cycles = _geometry.tail_mask_hex, _geometry.vm, _geometry.total_cycles


# ── Two-stage requant, shared by every sim/synth emitter below ────────────────
#
# Phase 1 of jojo-track/open/tensor-slice-bias-in-rtl: the slice's raw K
# contraction is requantised in exactly two stages, both round-half-up + wrap
# (never saturate):
#   stage 1 (in-slice, S1): round-half-up shift by S1, wrap to 16 bits. S1 is a
#     Verilog module PARAMETER on the tensor_slice_int8 black box, not a port;
#     the sim behavioural model applies it once to the exact full sum (see the
#     "Sim vs synth branches" note in the plan -- phase 2 moves this per-K-
#     partition).
#   stage 2 (in the wrapper, S2): sum the 16-bit partials in 16-bit WRAPPING
#     arithmetic, add the bias (16-bit signed, baked at the intermediate scale
#     gemm_frac - S1), round-half-up shift by S2, wrap to out_width.
# Stage 2 is emitted as ONE Verilog function shared by every call site (chunked
# and K-spatial, sim and synth): only the two arguments (the wrapping partial
# sum and the bias value) differ per call site, never the function body.


def _bias_rom_block(bias_codes, rom_name="bias_rom", width=16):
    """Compile-time bias constant (decision 4): one *width*-bit signed lane per
    output column, rendered from the SAME codes list the C behavioral core
    bakes as a static array (``gemm_ip.biasrom``).

    Emitted as a single FLAT ``wire`` concatenation, not a ``reg`` array: VTR's
    parmys turns any initialised reg array into a single_port_ram even when every
    read index is a constant, and a memory with no clocked read then trips vpr's
    ``clk_pin`` assertion (the same failure the weight ROM's registered-address
    read exists to avoid). A flat wire with ``[idx*width +: width]`` part-selects
    is pure wiring for constant indices and a plain mux for the fold-N group
    index. Readers use ``_bias_lane(rom_name, idx_expr)``.
    """
    codes = [int(c) for c in bias_codes]
    n = len(codes)
    lanes = []
    for c in reversed(codes):  # MSB-first concatenation: lane i at [i*width +: width]
        lanes.append(f"{width}'sd{c}" if c >= 0 else f"-{width}'sd{-c}")
    body = ",\n        ".join(lanes)
    return (f"\n    // Baked bias, one {width}-bit signed lane per column (lane i at [i*{width} +: {width}]).\n"
            f"    wire [{n * width - 1}:0] {rom_name} = {{\n        {body}\n    }};\n")


def _bias_lane(rom_name, idx_expr, width=16):
    """Signed *width*-bit part-select of the flat bias wire at column *idx_expr*."""
    return f"$signed({rom_name}[({idx_expr}) * {width} +: {width}])"


def _fold_n_bias_group_decl(n_passes, reg_name="bias_grp_ctr"):
    """Independent fold-N group counter for the bias ROM (synth branches).

    Mirrors ``_weight_rom_block``'s ``grp_ctr`` (advances once per frame
    boundary, on the idle beat right after a frame's feed beats), but is
    NOT tied to weight-stationarity -- fold-N + bias must work with an
    external b_cols port too, where no weight ROM/grp_ctr exists at all.

    CAUTION (this is the bug the fold-N + bias item exists to avoid): this
    live counter tracks the group being FED, which advances as soon as a
    frame's beats stop (well before that same frame's own K-contraction
    pipeline latency + drain complete). It is NOT the right index to read
    the bias ROM with at EMIT time for THAT frame -- callers must latch a
    frozen per-frame copy (e.g. on ``preload_d``/``slice_start``, before the
    new frame's own feed can retire and advance this counter again) and use
    the frozen copy in the drain/stage-2 lookup instead of this live one.
    """
    if int(n_passes) <= 1:
        return "", ""
    decl = f"""
    // Fold-N bias group counter (independent of the weight ROM's grp_ctr --
    // must work with an external b_cols port too). Advances once per frame
    // boundary; CALLERS MUST LATCH a frozen per-frame copy for the drain
    // (see _fold_n_bias_group_decl's docstring).
    reg [15:0] {reg_name};
    reg {reg_name}_was_feeding;"""
    body = f"""
            if (!in_valid) begin
                if ({reg_name}_was_feeding) begin
                    if ({reg_name} + 16'd1 >= 16'd{n_passes}) {reg_name} <= 16'd0;
                    else {reg_name} <= {reg_name} + 16'd1;
                end
                {reg_name}_was_feeding <= 1'b0;
            end else begin
                {reg_name}_was_feeding <= 1'b1;
            end"""
    return decl, body


def _stage1_function(s1, func_name="stage1"):
    """Sim-branch stage 1: round-half-up shift the exact 32-bit sum by *s1*
    bits, then wrap to 16. ``s1 == 0`` is a true pass-through (low 16 bits of
    the raw sum, no rounding) -- decision 2 in the plan.
    """
    if s1 and s1 > 0:
        half = 1 << (s1 - 1)
        body = (
            f"            r = (x + 32'sd{half}) >>> {s1};\n"
        )
    else:
        body = "            r = x;\n"
    return f"""\
    function signed [15:0] {func_name};
        input signed [31:0] x;
        reg signed [31:0] r;
        begin
{body}            {func_name} = r[15:0];
        end
    endfunction
"""


def _stage2_function(s2, out_width, func_name="stage2"):
    """Wrapper-side stage 2, shared verbatim by every emitter: wrap-add the
    bias to the (already 16-bit-wrapping-summed) partial, round-half-up shift
    by *s2*, wrap to *out_width*. Both inputs are 16-bit signed; the addition
    ``sum_partials + bias_val`` truncates (Verilog assignment semantics) to
    the declared 16-bit ``biased`` reg, which is exactly the 16-bit wrap the
    plan calls for. ``s2 == 0`` skips the half-LSB round (identity shift).
    """
    half = (1 << (s2 - 1)) if s2 and s2 > 0 else 0
    shift = int(s2) if s2 else 0
    sext = "{16{biased[15]}}, biased"
    return (
        f"    function signed [{out_width - 1}:0] {func_name};\n"
        "        input signed [15:0] sum_partials;\n"
        "        input signed [15:0] bias_val;\n"
        "        reg signed [15:0] biased;\n"
        "        reg signed [31:0] r;\n"
        "        begin\n"
        "            biased = sum_partials + bias_val;\n"
        f"            r = ({{{sext}}} + 32'sd{half}) >>> {shift};\n"
        f"            {func_name} = r[{out_width - 1}:0];\n"
        "        end\n"
        "    endfunction\n"
    )


_ZERO_POINT_COMMENT = (
    "    // signedness mode: the tensor_slice_int8 black box is driven as signed\n"
    "    // int8. Unsigned 8-bit operands are offset by the zero point (bit-7 flip\n"
    "    // at the feed, = u-128) and corrected after accumulation (zero_point *\n"
    "    // sum of the other operand). If the slice exposes a native unsigned mode\n"
    "    // (slice_dtype pins), set the zero point to 0 and drive that mode instead."
)


def _zero_point_params_block(a_zero_point, b_zero_point):
    """Module parameter declarations + doc comment for A_ZERO_POINT/B_ZERO_POINT."""
    return (
        f"{_ZERO_POINT_COMMENT}\n"
        f"    parameter integer A_ZERO_POINT = {int(a_zero_point)};\n"
        f"    parameter integer B_ZERO_POINT = {int(b_zero_point)};\n"
    )


def _lane_flip_mask(width, zero_point):
    """Verilog literal that XORs bit 7 of every packed 8-bit lane across a
    ``width``-bit word when ``zero_point`` is nonzero (all-zero -- folds
    away -- otherwise)."""
    if not zero_point or width <= 0:
        return f"{max(width, 1)}'d0"
    mask_int = int("80" * (width // 8), 16) if width % 8 == 0 else (1 << (width - 1))
    return f"{width}'h{mask_int:x}"


def _lane_flip(zero_point):
    """Verilog XOR suffix that flips bit 7 of every packed 8-bit lane when
    ``zero_point`` is nonzero (u -> u-128, the same conversion the C++
    ``{name}_to_gemm_int8`` helper performs)."""
    return " ^ 8'h80" if zero_point else ""


def _lane_flip_mask_masked(width, zero_point, mask_signal):
    """Like ``_lane_flip_mask``, but gates every lane's bit-7 flip on that
    lane's bit in ``mask_signal`` (expected 8 bits wide, one bit per K lane
    within the current chunk -- e.g. ``current_k_mask``).

    K < 8-per-chunk cases (the common tail chunk, and any chunk in a K < 8
    GEMM) present zero-code padding on the lanes beyond the real K width.
    The plain (unconditional) ``_lane_flip_mask`` XORs bit 7 of EVERY lane,
    including those padding lanes: a padding zero code (0) becomes -128 once
    flipped, instead of staying 0. Whether that silently pollutes the
    product depends on the tensor_slice_int8 black box's own K-masking
    (``validity_mask_a_cols_b_rows``/``current_k_mask``) doing its job
    end-to-end; feeding a mathematically wrong operand code into a black box
    and relying on it never being read is fragile. Masking the flip itself
    by the SAME per-lane validity mask the slice is given keeps padding
    lanes at raw, unflipped 0 -- multiplying by 0 (not -128) regardless of
    whether/how the slice's own masking gates the MAC.
    """
    if not zero_point or width <= 0:
        return f"{max(width, 1)}'d0"
    n_lanes = width // 8
    lanes = [f"({mask_signal}[{i % 8}] ? 8'h80 : 8'h00)" for i in range(n_lanes - 1, -1, -1)]
    return "{" + ", ".join(lanes) + "}"


def generate_sim_verilog(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1,
                         s1=0, s2=0, out_width=16, weight_rom=None, emit_rom=True,
                         n_passes=1, bias_codes=None, emit_bias_rom=True,
                         bias_rom_name="bias_rom", a_zero_point=0, b_zero_point=0,
                         a_zero_point_correct=None, b_zero_point_correct=None):
    """``a_zero_point``/``b_zero_point`` always drive the feed-side bit-7 flip
    (and the A_ZERO_POINT/B_ZERO_POINT module parameters). ``a_zero_point_correct``/
    ``b_zero_point_correct`` (default: same as the respective zero point) drive
    the post-accumulation correction term ONLY; a caller that already folded a
    zero point's correction elsewhere (e.g. package.py folds a weight-stationary
    A_ZERO_POINT * colsum(B) into the compile-time bias instead) passes 0 here
    to avoid double-counting it while the feed-side flip still happens.
    """
    has_bias = bias_codes is not None
    a_flip = _lane_flip(a_zero_point)
    b_flip = _lane_flip(b_zero_point)
    _azp = int(a_zero_point if a_zero_point_correct is None else a_zero_point_correct)
    _bzp = int(b_zero_point if b_zero_point_correct is None else b_zero_point_correct)
    _zp_terms = []
    if _azp:
        _zp_terms.append(f"{_azp} * bmat[s * {k} + kk][actual_col]")
    if _bzp:
        _zp_terms.append(f"{_bzp} * amat[s * {m} + actual_row][kk]")
    if _azp and _bzp:
        _zp_terms.append(str(_azp * _bzp))
    zp_correction = "".join(f" + ({t})" for t in _zp_terms)
    bias_rom_block = _bias_rom_block(bias_codes, bias_rom_name) if (has_bias and emit_bias_rom) else ""
    # Two-stage requant (jojo-track/open/tensor-slice-bias-in-rtl, phase 1):
    # stage 1 rounds/wraps the FULL contraction (in-slice and cross-chunk
    # folded into `acc`, the whole point of the behavioural model) to 16 bits
    # once; stage 2 (shared verbatim with every other emitter) wrap-adds the
    # bias and rounds/wraps to out_width. Both stages are plain functions, no
    # saturation anywhere.
    #
    # Bias is a COMPILE-TIME constant (decision 4): no bias_cols port. When
    # has_bias, the caller (generate_combined_core_verilog et al.) has
    # already declared a `bias_rom` array (one 16-bit signed entry/column, at
    # the intermediate scale) hoisted above `ifndef SYNTHESIS`, above this
    # module's own text; this generator only REFERENCES it by name -- never
    # declares it, so sim and synth read the exact same ROM. When not
    # has_bias, the add folds away to a literal 0 (no ROM at all).
    stage_fns = _stage1_function(s1) + "\n" + _stage2_function(s2, out_width)
    # Fold-N + bias: the bias ROM holds n_passes*n entries (group g's real
    # columns at base g*n, like the weight ROM); a frame/slot's bias group is
    # NOT the live bias_grp_ctr at emit time (it may already have advanced to
    # the NEXT frame's group by then -- see _fold_n_bias_group_decl) but a
    # frozen per-slot copy latched at that slot's allocation, mirroring
    # exactly how the old (removed) bias_buf captured per-slot bias values.
    _fold_n_bias = bool(has_bias and n_passes and int(n_passes) > 1)
    if _fold_n_bias:
        _bias_expr = f"{_bias_lane(bias_rom_name, f'slot_grp[s] * {n} + actual_col')}"
    elif has_bias:
        _bias_expr = f"{_bias_lane(bias_rom_name, 'actual_col')}"
    else:
        _bias_expr = "16'sd0"
    _sat_call = f"stage2(stage1(acc), {_bias_expr})"
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    a_width = grid_rows * 64
    b_width = grid_cols * 64
    a_chunk_width = a_width
    b_chunk_width = b_width
    c_width = grid_cols * 8 * out_width
    input_beats = max(m, n)
    k_chunks = (k + 7) // 8
    passes = -(-k_chunks // k_spatial)
    full_k_spatial = k_spatial > 1 and passes == 1
    general_k_spatial = k_spatial > 1 and passes > 1
    if k_spatial > 1:
        # Narrow per-beat word: k_spatial partitions, 64 bits per K-chunk each
        # pass, no grid padding.
        a_width = 64 * k_spatial
        b_width = 64 * k_spatial
    total_input_beats = passes * input_beats
    total_output_rows = m
    # Wave-latency term: use k directly whenever there is no K-chunk padding
    # (chunked, full-K, or any other passes==1 case); the padded partial-pass
    # endpoint uses 8*k_chunks_pad instead (see geometry.latency_first_out,
    # which this mirrors).
    k_chunks_pad = passes * k_spatial
    k_term = k if (k_spatial == 1 or k_chunks_pad == k_chunks) else 8 * k_chunks_pad
    latency = max(0, k_term + n - total_input_beats)
    # First-output offset == geometry.latency_first_out(): feed beats + the
    # systolic K+N wave remainder. K-spatial folding feeds K_SPATIAL chunks per
    # pass, so its first_out drops accordingly; the C++ core and the wrapper's
    # DRAIN capture window use the same formula.
    first_out = total_input_beats + latency
    behav_name = f"{module_name}_behav_grid"
    if full_k_spatial:
        mode_comment = "Full K-spatial behavioral MxKxN GEMM. Not intended for synthesis."
    elif general_k_spatial:
        mode_comment = (
            f"K-spatial (K_SPATIAL={k_spatial}, PASSES={passes}) behavioral MxKxN GEMM. "
            "Not intended for synthesis."
        )
    else:
        mode_comment = "Chunked behavioral MxKxN GEMM. Not intended for synthesis."
    # Frames in flight: the feed of frame t+1 overlaps the compute/drain of
    # frame t, so back-to-back frames sustain a frame II of total_beats+1
    # (the preload step plus the data beats). Slot count covers the deepest
    # overlap plus one spare so an allocating frame never lands on a slot
    # that is still draining.
    #
    # op-contract note (jojo-track/open/tensor-slice-op-shadow-drain): this
    # behavioral model has no tensor_slice_int8 instance and no op/pe_reset
    # pins -- it is a pure functional reference (compute-at-emit over
    # per-frame operand partitions), so op[0]/op[1]/op[2] have no separate
    # Verilog signals here. The model already implements the CONTRACT the op
    # pins exist to express: each frame's amat/bmat partition + the acc
    # computed at emit time plays the role of the frame's shadow bank (its
    # final result is derived once, at TOTAL_ROWS-bounded emit, from operands
    # that are never touched again); a frame's slot is freed (slot_run
    # cleared) only after its own TOTAL_ROWS drain completes -- the
    # shadow_swap (op[2]) + accumulator-free semantics -- and drain never
    # extends past TOTAL_ROWS (there IS no padded tail here: unlike the
    # physical 8-row tensor-slice burst, this model only ever emits the M
    # logical rows), so drain_stop (op[1]) has nothing to truncate. Frame
    # t+1's feed overlapping frame t's compute/drain is out_ctrl (op[0])
    # released per frame rather than gated by a single grid-wide signal.
    frame_period = total_input_beats + 1
    slots = -(-(first_out + 1 + total_output_rows) // frame_period) + 1
    # Per-slot unpack: capture a_rows/b_cols into the allocating slot's amat/
    # bmat partition at beat index `cc` (== that frame's clk_cnt while
    # collecting). `ws` is the slot index resolved this cycle.
    single_unpack = f"""\
                    cc_chunk = cc / INPUT_BEATS;
                    cc_beat  = cc % INPUT_BEATS;
                    if (cc_beat < {m}) begin
                        for (lane = 0; lane < 8; lane = lane + 1) begin
                            kk = cc_chunk * 8 + lane;
                            if (kk < {k})
                                amat[ws * {m} + cc_beat][kk] = a_rows[(cc_beat / 8) * 64 + lane * 8 +: 8]{a_flip};
                        end
                    end
                    if (cc_beat < {n}) begin
                        for (lane = 0; lane < 8; lane = lane + 1) begin
                            kk = cc_chunk * 8 + lane;
                            if (kk < {k})
                                bmat[ws * {k} + kk][cc_beat] = b_cols[(cc_beat / 8) * 64 + lane * 8 +: 8]{b_flip};
                        end
                    end"""
    if full_k_spatial:
        single_unpack = f"""\
                    cc_beat = cc;
                    if (cc_beat < {m}) begin
                        for (kk = 0; kk < {k}; kk = kk + 1) begin
                            cc_chunk = kk / 8;
                            lane = kk % 8;
                            amat[ws * {m} + cc_beat][kk] = a_rows[cc_chunk * 64 + lane * 8 +: 8]{a_flip};
                        end
                    end
                    if (cc_beat < {n}) begin
                        for (kk = 0; kk < {k}; kk = kk + 1) begin
                            cc_chunk = kk / 8;
                            lane = kk % 8;
                            bmat[ws * {k} + kk][cc_beat] = b_cols[cc_chunk * 64 + lane * 8 +: 8]{b_flip};
                        end
                    end"""
    elif general_k_spatial:
        # General pass-sequenced unpack: pass q, beat t carries chunk
        # q*k_spatial+p on partition p (bits [p*64, p*64+64)); cc counts
        # beats across ALL passes (cc_pass = cc / INPUT_BEATS).
        single_unpack = f"""\
                    cc_pass = cc / INPUT_BEATS;
                    cc_beat = cc % INPUT_BEATS;
                    if (cc_beat < {m}) begin
                        for (p_idx = 0; p_idx < {k_spatial}; p_idx = p_idx + 1) begin
                            cc_chunk = cc_pass * {k_spatial} + p_idx;
                            for (lane = 0; lane < 8; lane = lane + 1) begin
                                kk = cc_chunk * 8 + lane;
                                if (kk < {k})
                                    amat[ws * {m} + cc_beat][kk] = a_rows[p_idx * 64 + lane * 8 +: 8]{a_flip};
                            end
                        end
                    end
                    if (cc_beat < {n}) begin
                        for (p_idx = 0; p_idx < {k_spatial}; p_idx = p_idx + 1) begin
                            cc_chunk = cc_pass * {k_spatial} + p_idx;
                            for (lane = 0; lane < 8; lane = lane + 1) begin
                                kk = cc_chunk * 8 + lane;
                                if (kk < {k})
                                    bmat[ws * {k} + kk][cc_beat] = b_cols[p_idx * 64 + lane * 8 +: 8]{b_flip};
                            end
                        end
                    end"""

    # Weight-stationary (const-weight): the top sim wrapper drops its external b_cols
    # port and feeds the inner behav_grid from the shared ROM (w_rom_out). The behav_grid
    # keeps its internal b_cols input, now driven by the ROM. Bias stays external.
    # Full-K is supported: b_width is already the narrow 64*k_chunks word above, and
    # the ROM holds one beat per presented input cycle (total_input_beats ==
    # input_beats under full-K), so the widened word carries every K chunk without
    # adding beats. The ROM must be built with the matching narrow full-K packer —
    # gemm_ip.weights.build_weight_rom_full_k.
    extra_int_decls = "    integer p_idx, cc_pass;\n" if general_k_spatial else ""

    _sim_ws = weight_rom is not None
    if _sim_ws:
        sim_b_cols_port = ""
        sim_b_src = "w_rom_out"
        sim_rom_block = _weight_rom_block(b_width, weight_rom, n, input_beats, n_passes=n_passes, passes=passes) if emit_rom else ""
    else:
        sim_b_cols_port = f"    input  wire [{b_width-1}:0]   b_cols,\n"
        sim_b_src = "b_cols"
        sim_rom_block = ""

    return f"""\
// Auto-generated simulation model by rtl.py
// {mode_comment}
// Dimensions: M={m}, K={k}, N={n}
`timescale 1ns/1ps

module {module_name}(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,
    input  wire [{a_width-1}:0]   a_rows,
{sim_b_cols_port}    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);
{_zero_point_params_block(a_zero_point, b_zero_point)}
{sim_rom_block}
    wire [{c_width-1}:0] behav_c_row;
    wire                 behav_out_valid;
    wire                 behav_out_last;

    {behav_name} grid (
        .clk(clk), .rst(rst), .en(en),
        .a_rows(a_rows), .b_cols({sim_b_src}),
        .preload_valid(preload_valid), .in_valid(in_valid),
        .c_row(behav_c_row), .out_valid(behav_out_valid), .out_last(behav_out_last)
    );

    always @(posedge clk) begin
        if (rst) begin
            c_row <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
        end else if (en) begin
            c_row <= behav_c_row;
            out_valid <= behav_out_valid;
            out_last <= behav_out_last;
        end
    end

endmodule

module {behav_name}(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,
    input  wire [{a_width-1}:0]   a_rows,
    input  wire [{b_width-1}:0]   b_cols,
    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);
{bias_rom_block}
    localparam integer INPUT_BEATS       = {input_beats};
    localparam integer K_CHUNKS          = {k_chunks};
    localparam integer TOTAL_INPUT_BEATS = {total_input_beats};
    localparam integer FIRST_OUT         = {first_out};
    localparam integer TOTAL_ROWS        = {total_output_rows};
    localparam integer FRAME_SLOTS       = {slots};

    // ── Frame-slot storage: up to FRAME_SLOTS frames in flight ─────────────
    // Pure behavioral scheduler (no tensor-slice structure): each frame gets
    // a private operand partition and cycle counter; results are computed at
    // emit time in full precision. Feed of frame t+1 overlaps compute/drain
    // of frame t, so back-to-back frames sustain frame II = TOTAL_INPUT_BEATS+1
    // (~one result row per cycle) while each frame keeps its own FIRST_OUT.
    reg signed [7:0]  amat [0:{slots * m - 1}][0:{k - 1}];
    reg signed [7:0]  bmat [0:{slots * k - 1}][0:{n - 1}];
    reg [31:0] cc_slot [0:{slots - 1}];
    reg        slot_run [0:{slots - 1}];
    reg [31:0] wr_slot;
    reg        feeding;
{"    reg [15:0] slot_grp [0:" + str(slots - 1) + "];" if _fold_n_bias else ""}
{"    reg [15:0] bias_grp_ctr;" if _fold_n_bias else ""}

    // ── Diagnostics (parsed by the testbench; not load-bearing) ────────────
    reg [31:0] beh_cyc;
    reg [31:0] prev_beh_start;
    reg [31:0] beh_ii_val;

    integer i, j, kk, lane, tile, s;
    integer ws, cc, scc, cc_chunk, cc_beat, out_idx, actual_row, actual_col;
{extra_int_decls}    reg signed [31:0] acc;
    reg signed [{out_width-1}:0] sat;

{stage_fns}
    // ═══════════════════════════════════════════════════════════════════════
    // Frame-slot clk_cnt schedule. A frame starts at the first in_valid call
    // after a non-in_valid call (the wrapper protocol always inserts at least
    // the preload step between frames): allocate the next slot, capture its
    // TOTAL_INPUT_BEATS beats, then emit one row per cycle for that slot's
    // cc in [FIRST_OUT+1, FIRST_OUT+1+TOTAL_ROWS) and retire it. Emission
    // windows of consecutive frames cannot overlap (TOTAL_ROWS <= the minimum
    // frame period TOTAL_INPUT_BEATS+1). Everything advances on en only, so
    // wrapper stalls keep the model aligned with the run()-call schedule.
    // ═══════════════════════════════════════════════════════════════════════
    always @(posedge clk) begin
        if (rst) begin
            for (s = 0; s < FRAME_SLOTS; s = s + 1) begin
                cc_slot[s]  <= 32'd0;
                slot_run[s] <= 1'b0;
            end
            wr_slot        <= FRAME_SLOTS - 1;
            feeding        <= 1'b0;
{"            bias_grp_ctr   <= 16'd0;" if _fold_n_bias else ""}
            c_row          <= {c_width}'d0;
            out_valid      <= 1'b0;
            out_last       <= 1'b0;
            beh_cyc        <= 32'd0;
            prev_beh_start <= 32'd0;
            beh_ii_val     <= 32'd0;
        end else if (en) begin
            beh_cyc   <= beh_cyc + 1;
            c_row     <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last  <= 1'b0;

            // ── capture into the feeding frame's slot ────────────────────
            ws = wr_slot;
            cc = 0;
            if (in_valid) begin
                if (!feeding) begin
                    ws = (wr_slot + 1) % FRAME_SLOTS;
                    if (slot_run[ws]) $display("BEH_SLOT_OVERFLOW beh_cyc=%0d", beh_cyc);
                    wr_slot      <= ws;
                    feeding      <= 1'b1;
                    slot_run[ws] <= 1'b1;
                    cc_slot[ws]  <= 32'd1;   // cc = 0 is consumed this cycle
{"                    slot_grp[ws] <= bias_grp_ctr;" if _fold_n_bias else ""}
{f"                    if (bias_grp_ctr + 16'd1 >= 16'd{n_passes}) bias_grp_ctr <= 16'd0; else bias_grp_ctr <= bias_grp_ctr + 16'd1;" if _fold_n_bias else ""}
                    if (prev_beh_start == 0) beh_ii_val <= 32'd0;
                    else                     beh_ii_val <= beh_cyc - prev_beh_start;
                    prev_beh_start <= beh_cyc;
                    $display("BEH_START beh_cyc=%0d", beh_cyc);
                    cc = 0;
                end else begin
                    cc = cc_slot[ws];
                end
                if (cc < TOTAL_INPUT_BEATS) begin
{single_unpack}
                end
            end else begin
                feeding <= 1'b0;
            end

            // ── per-slot emit (compute-at-emit) + advance / retire ───────
            // slot_run[] reads pre-edge values, so the slot allocated THIS
            // cycle (cc already set to 1 above) is naturally skipped.
            for (s = 0; s < FRAME_SLOTS; s = s + 1) begin
                if (slot_run[s]) begin
                    scc = cc_slot[s];
                    if (scc >= FIRST_OUT + 1 && scc < FIRST_OUT + 1 + TOTAL_ROWS) begin
                        out_idx    = scc - (FIRST_OUT + 1);
                        actual_row = out_idx;
                        for (tile = 0; tile < {grid_cols}; tile = tile + 1) begin
                            for (lane = 0; lane < 8; lane = lane + 1) begin
                                actual_col = tile * 8 + lane;
                                if (actual_row < {m} && actual_col < {n}) begin
                                    // Pure integer GEMM: no bias here (decision
                                    // 4 -- bias lands between stage 1 and stage
                                    // 2, not folded into the raw accumulation).
                                    acc = 32'sd0;
                                    for (kk = 0; kk < {k}; kk = kk + 1)
                                        acc = acc + (amat[s * {m} + actual_row][kk] * bmat[s * {k} + kk][actual_col]){zp_correction};
                                    sat = {_sat_call};
                                end else begin
                                    sat = {out_width}'sd0;
                                end
                                c_row[tile * {8 * out_width} + lane * {out_width} +: {out_width}] <= sat;
                            end
                        end
                        out_valid <= 1'b1;
                        out_last  <= (out_idx + 1 == TOTAL_ROWS) ? 1'b1 : 1'b0;
                    end
                    cc_slot[s] <= scc + 1;
                    if (scc + 1 >= FIRST_OUT + 1 + TOTAL_ROWS) begin
                        slot_run[s] <= 1'b0;
                        cc_slot[s]  <= 32'd0;
                        $display("BEH_II=%0d", beh_ii_val);
                        $display("BEH_DONE beh_cyc=%0d", beh_cyc);
                    end
                end
            end
        end
    end

endmodule
"""


def _weight_rom_block(b_width, weight_rom, n, input_beats, n_passes=1, passes=1):
    """Shared const-weight ROM: declaration + inline init + beat/addr counters + w_rom_out.

    Emitted once (above the `ifndef SYNTHESIS` split in the combined core) so a single
    ROM feeds both the behavioral-sim and structural-synth branches. The ROM holds only
    ``passes*n`` entries (beat ``t < n`` of each pass carries a real column; beats
    ``t >= n`` -- the tail when ``input_beats == max(m, n) > n`` -- are wasted zero
    reads, never stored). ``beat_ctr`` counts one presented input beat per pass
    (0..input_beats-1, mirrors the free-running `in_valid`, matching the order the
    external `b_cols` port received -- pack_b_chunk feed order). ``rom_addr`` is the
    registered ROM read address: it must be a register (not a combinational
    `rom_base + beat_ctr` expression) because VTR's parmys only infers a clocked
    `single_port_ram` for `w_rom` when the address is register-fed -- a combinational
    address makes it infer a clockless RAM and vpr aborts on the missing clock. The
    zero mux for the tail beats (t >= n) stays on the *output*, after the memory read,
    never on the address. ``rom_addr`` mirrors beat_ctr's advance within a pass and,
    on wrap, steps one further to land on the next pass's base: the held value at
    wrap is always `base + n - 1` (it stops advancing once beat_ctr reaches n - 1),
    so `rom_addr + 1` is exactly `base + n`, the next pass's base -- for both
    input_beats == n and input_beats > n. That makes a separate `rom_base` register
    redundant, so it is dropped.
    """
    hexw = (b_width + 3) // 4
    mask = (1 << b_width) - 1
    nbeats = len(weight_rom)
    rom_init = "\n".join(
        f"        w_rom[{i}] = {b_width}'h{(int(v) & mask):0{hexw}x};"
        for i, v in enumerate(weight_rom)
    )
    # Combined K+N fold (passes > 1 AND n_passes > 1): the ROM holds
    # passes*n_passes*n entries in the chunk-major/ng-minor layout
    # build_weight_rom_combined_fold emits -- addr = (kc*n_passes + ng)*n + t.
    # The plain single-group / fold-N feeder below advances rom_addr contiguously
    # (pass stride == n), which only matches build_weight_rom_fold_n's layout
    # when passes == 1; with passes > 1 a group's K-passes are strided by
    # n_passes*n (not contiguous), so that feeder would read another group's
    # pass-0 columns for pass > 0 -- the tensor-slice-kn-fold-cosim-mismatch bug.
    # Track the K-pass index (pass_ctr == kc) explicitly and compute the read
    # address for the NEXT beat straight from (kc, ng, t) next-state values, the
    # same registered-address trick VTR needs (mirrors the structural wrapper's
    # rom_addr in _general_synth_combined_fold, and the C csim's bbuf_src).
    if int(n_passes) > 1 and int(passes) > 1:
        return f"""
    // Weight-stationary combined K+N fold ROM (baked; no external b_cols port).
    // {len(weight_rom)} entries in chunk-major/ng-minor layout
    // (addr = (k_pass*{n_passes} + n_group)*{n} + beat); only {n} of every
    // {input_beats} beats per pass carry a real column, beats t >= {n} read zero.
    reg [{b_width - 1}:0] w_rom [0:{nbeats - 1}];
    initial begin
{rom_init}
    end
    reg [15:0] beat_ctr;
    // rom_addr must be a plain register (single driver = this block, single use =
    // w_rom index) so VTR's parmys infers a clocked single_port_ram; it is always
    // updated to the address the NEXT presented beat will read.
    reg [15:0] rom_addr;
    // K-pass index (kc) within the current frame and N-group index (ng); ng only
    // advances at real frame boundaries (grp_was_feeding gates out reset settle /
    // trailing idle beats), matching the fold-N feeder's group cadence.
    reg [15:0] pass_ctr;
    reg [15:0] grp_ctr;
    reg grp_was_feeding;
    always @(posedge clk) begin
        if (rst) begin
            beat_ctr <= 16'd0;
            rom_addr <= 16'd0;
            pass_ctr <= 16'd0;
            grp_ctr <= 16'd0;
            grp_was_feeding <= 1'b0;
        end else if (en) begin
            if (!in_valid) begin
                beat_ctr <= 16'd0;
                pass_ctr <= 16'd0;
                if (grp_was_feeding) begin
                    if (grp_ctr + 16'd1 >= 16'd{n_passes}) begin
                        rom_addr <= 16'd0;
                        grp_ctr <= 16'd0;
                    end else begin
                        // Next group's pass-0 base: (0*{n_passes} + (ng+1))*{n}.
                        rom_addr <= (grp_ctr + 16'd1) * 16'd{n};
                        grp_ctr <= grp_ctr + 16'd1;
                    end
                end
                grp_was_feeding <= 1'b0;
            end else if (beat_ctr < 16'd{input_beats - 1}) begin
                beat_ctr <= beat_ctr + 16'd1;
                // Next beat, same K-pass: (kc*{n_passes} + ng)*{n} + (t+1).
                if (beat_ctr + 16'd1 < 16'd{n})
                    rom_addr <= (pass_ctr * 16'd{n_passes} + grp_ctr) * 16'd{n} + (beat_ctr + 16'd1);
                grp_was_feeding <= 1'b1;
            end else begin
                // Pass boundary: beat 0 of the next K-pass, same group:
                // ((kc+1)*{n_passes} + ng)*{n}. (On the frame's final pass this
                // lands one pass past the group; the idle beat that always
                // follows overrides it before any in_valid beat consumes it.)
                beat_ctr <= 16'd0;
                pass_ctr <= pass_ctr + 16'd1;
                rom_addr <= ((pass_ctr + 16'd1) * 16'd{n_passes} + grp_ctr) * 16'd{n};
                grp_was_feeding <= 1'b1;
            end
        end
    end
    wire [{b_width - 1}:0] w_rom_out = (beat_ctr < 16'd{n}) ? w_rom[rom_addr] : {b_width}'d0;
"""
    # Fold-N group counter, emitted only when n_passes > 1 (the ROM then holds
    # n_passes*n entries, group g's columns at base g*n). The wrap on a frame's
    # last beat already lands rom_addr on the next group's base (base + n); the
    # idle beat between frames rewinds rom_addr to 0 only once every group has
    # been visited, otherwise it holds the wrapped value and advances the
    # counter. ``grp_was_feeding`` marks a real frame boundary (the idle beat
    # right after fed beats) so reset settle beats and trailing idle cycles
    # never advance the counter. With n_passes == 1 every fragment is empty and
    # the block is byte-for-byte the single-group text.
    if int(n_passes) > 1:
        grp_decl = """
    // Fold-N group counter: completed column-tile groups within the ROM's
    // back-to-back frame replay; grp_was_feeding marks a real frame boundary.
    reg [15:0] grp_ctr;
    reg grp_was_feeding;"""
        grp_reset = """
            grp_ctr <= 16'd0;
            grp_was_feeding <= 1'b0;"""
        grp_idle = f"""                if (grp_was_feeding) begin
                    if (grp_ctr + 16'd1 >= 16'd{n_passes}) begin
                        rom_addr <= 16'd0;
                        grp_ctr <= 16'd0;
                    end else begin
                        grp_ctr <= grp_ctr + 16'd1;
                    end
                end
                grp_was_feeding <= 1'b0;"""
        grp_feeding = """
                grp_was_feeding <= 1'b1;"""
    else:
        grp_decl = grp_reset = grp_feeding = ""
        grp_idle = "                rom_addr <= 16'd0;"
    text = f"""
    // Weight-stationary const-weight ROM (baked; no external b_cols port). One beat
    // per presented input cycle; feeds both the sim and synth branches below. Only
    // {n} of every {input_beats} beats per pass carry a real column ({nbeats} entries
    // total); beats t >= {n} read back zero.
    reg [{b_width - 1}:0] w_rom [0:{nbeats - 1}];
    initial begin
{rom_init}
    end
    reg [15:0] beat_ctr;
    // rom_addr must be a plain register whose only driver is this always block and
    // whose only use is indexing w_rom below: VTR's parmys infers w_rom as a clocked
    // single_port_ram only when the read address comes straight from a register, not
    // a combinational rom_base+beat_ctr expression (that form gets clk=unconn and
    // vpr aborts). Held flat, it always equals the current pass base + beat_ctr.
    reg [15:0] rom_addr;{grp_decl}
    always @(posedge clk) begin
        if (rst) begin
            beat_ctr <= 16'd0;
            rom_addr <= 16'd0;{grp_reset}
        end else if (en) begin
            if (!in_valid) begin
                beat_ctr <= 16'd0;
{grp_idle}
            end else if (beat_ctr < 16'd{input_beats - 1}) begin
                beat_ctr <= beat_ctr + 16'd1;
                if (beat_ctr + 16'd1 < 16'd{n}) rom_addr <= rom_addr + 16'd1;{grp_feeding}
            end else begin
                beat_ctr <= 16'd0;
                rom_addr <= rom_addr + 16'd1;{grp_feeding}
            end
        end
    end
    wire [{b_width - 1}:0] w_rom_out = (beat_ctr < 16'd{n}) ? w_rom[rom_addr] : {b_width}'d0;
"""
    return text


def _generate_general_synth_verilog(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1,
                                    feed_mode="chained", debug=False,
                                    weight_rom=None, emit_rom=True, n_passes=1,
                                    s1=0, s2=0, out_width=16, bias_codes=None, emit_bias_rom=True,
                                    bias_rom_name="bias_rom", a_zero_point=0, b_zero_point=0,
                                    a_zero_point_correct=None, b_zero_point_correct=None,
                                    m_passes=1, logical_m=None, logical_n=None):
    """Unified internal SYNTH emitter (jojo-track/open/tensor-slice-general-synth-grid,
    sub-phase 2a-ii).

    Per the confirmed tensor-slice contract, the general grid is ``k_spatial``
    copies of an ``Ms x Ns`` FLOWING grid (A left->right, B top->bottom):
    K-in-time (K passes/chunks) is a single copy accumulating internally
    across K chunks (no external reduction) -- this is the ``k_spatial == 1``
    branch below, byte-identical to the old standalone ``generate_synth_verilog``.
    K-in-space (``k_spatial > 1``) is ``k_spatial`` INDEPENDENT copies whose
    16-bit partials are reduced EXTERNALLY in the wrapper -- the branch below,
    byte-identical to the old standalone ``generate_k_spatial_synth_verilog``
    (itself already byte-identical to the k_spatial==1 branch when
    k_spatial==1, which is why that case delegates here with k_spatial=1
    rather than duplicating the branch below).

    2a-ii is a PURE refactor: fold-M/fold-N and k-spatial full-K/multi-pass
    all already fall out of the existing per-branch parameters (core_m/core_n
    sizing done by callers, ``n_passes``/``k_spatial`` sizing the grid and
    wrapper counters) exactly as before -- no new combined M+K+N folding is
    implemented here (deferred to 2b-2e).
    """
    # ── Combined-fold dispatch (2b, ADDITIVE) ────────────────────────────────
    # 2+ of {m_passes, k time-passes, n_passes} folding routes to the new,
    # separate _general_synth_combined_fold path; single-axis geometries fall
    # through to the two verified branches below UNCHANGED (byte-identical --
    # m_passes/logical_m/logical_n are simply never read on those paths).
    _k_chunks_for_dispatch = (k + 7) // 8
    _k_time_passes = -(-_k_chunks_for_dispatch // k_spatial)
    _folded_axes = (int(m_passes) > 1) + (_k_time_passes > 1) + (int(n_passes) > 1)
    if _folded_axes >= 2:
        return _general_synth_combined_fold(
            m, k, n, module_name=module_name, k_spatial=k_spatial,
            m_passes=m_passes, n_passes=n_passes, logical_m=logical_m, logical_n=logical_n,
            s1=s1, s2=s2, out_width=out_width, weight_rom=weight_rom, emit_rom=emit_rom,
            bias_codes=bias_codes, emit_bias_rom=emit_bias_rom, bias_rom_name=bias_rom_name,
        )
    if k_spatial != 1:
        return _general_synth_kspatial_branch(
            m, k, n, module_name=module_name, k_spatial=k_spatial, n_passes=n_passes,
            s1=s1, s2=s2, out_width=out_width, weight_rom=weight_rom, emit_rom=emit_rom,
            bias_codes=bias_codes, emit_bias_rom=emit_bias_rom, bias_rom_name=bias_rom_name,
        )
    # ── k_spatial == 1: K-in-time, single flowing Ms x Ns grid (chunked) ────
    # See ``generate_sim_verilog`` for the ``*_correct`` override contract:
    # ``a_zero_point``/``b_zero_point`` always drive the a_rows_q/b_cols_q
    # feed-side bit-7 flip and the A_ZERO_POINT/B_ZERO_POINT module parameters;
    # ``a_zero_point_correct``/``b_zero_point_correct`` (default: same value)
    # drive the running-sum post-accumulation correction only.
    has_bias = bias_codes is not None
    bias_rom_block = _bias_rom_block(bias_codes, bias_rom_name) if (has_bias and emit_bias_rom) else ""
    _fold_n_bias = bool(has_bias and n_passes and int(n_passes) > 1)
    _bias_grp_decl, _bias_grp_body = (
        _fold_n_bias_group_decl(n_passes) if _fold_n_bias else ("", "")
    )
    _azp = int(a_zero_point if a_zero_point_correct is None else a_zero_point_correct)
    _bzp = int(b_zero_point if b_zero_point_correct is None else b_zero_point_correct)
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8

    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 8 * out_width
    input_beats = max(m, n)
    # The slice emits 8 physical rows per tile.  The wrapper retires after M
    # logical rows and uses pe_reset to abort the masked remainder.
    logical_output_rows = m
    physical_output_rows = grid_rows * 8
    k_chunks = (k + 7) // 8
    last_k_size = k - (k_chunks - 1) * 8
    last_k_mask = tail_mask_hex(k, k_chunks - 1)

    row_mask_vals = [tail_mask_hex(m, r) for r in range(grid_rows)]
    col_mask_vals = [tail_mask_hex(n, c) for c in range(grid_cols)]

    chain_wires = []
    for r in range(grid_rows):
        for c in range(grid_cols + 1):
            chain_wires.append(f"    wire [63:0] a_chain_{r}_{c};")
    for r in range(grid_rows + 1):
        for c in range(grid_cols):
            chain_wires.append(f"    wire [63:0] b_chain_{r}_{c};")
    for r in range(grid_rows):
        for c in range(grid_cols):
            chain_wires.append(f"    wire [127:0] c_data_{r}_{c};")
            chain_wires.append(f"    wire         c_avail_{r}_{c};")

    boundary = []
    for r in range(grid_rows):
        boundary.append(f"    assign a_chain_{r}_0 = 64'b0;")
    for c in range(grid_cols):
        boundary.append(f"    assign b_chain_0_{c} = 64'b0;")

    # Slice data pins are NOT gated on the beat window (Ruthwik, 2026-09-11):
    # the registered beat word drives the boundary tiles' a_data/b_data directly
    # and the slice's own start/K-count decides what it consumes. The old
    # `in_beat_active ? word : 0` gate was the chunked cores' critical path
    # (state decode + 16-bit beat compare + a 64-bit mux fanned out to every
    # slice pin). Only the static "which boundary tile" select remains, which
    # constant-folds to wiring.
    data_wires = []
    for r in range(grid_rows):
        a_hi = (r + 1) * 64 - 1
        a_lo = r * 64
        for c in range(grid_cols):
            b_hi = (c + 1) * 64 - 1
            b_lo = c * 64
            data_wires.append(
                f"    wire [63:0] a_data_{r}_{c} = ({c} == 0) ? a_rows_q[{a_hi}:{a_lo}] : 64'b0;"
            )
            data_wires.append(
                f"    wire [63:0] b_data_{r}_{c} = ({r} == 0) ? b_cols_q[{b_hi}:{b_lo}] : 64'b0;"
            )

    inst_lines = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            inst_lines.append(f"""\
        // S1 = {s1}: in-slice stage-1 round-half-up shift (IP parameter, set out of band;
        // not passed as a Verilog override -- the VTR hard-block model has no parameters)
        (* black_box = "true" *) (* keep = "true" *) tensor_slice_int8 slice_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(slice_pe_reset),
            .start_mat_mul(slice_start),
            .done_mat_mul(done_mat_mul[{r*grid_cols+c}]),
            .a_data(a_data_{r}_{c}),
            .b_data(b_data_{r}_{c}),
            .a_data_in(a_chain_{r}_{c}),
            .b_data_in(b_chain_{r}_{c}),
            .a_data_out(a_chain_{r}_{c+1}),
            .b_data_out(b_chain_{r+1}_{c}),
            .c_data_out(c_data_{r}_{c}),
            .c_data_available(c_avail_{r}_{c}),
            .validity_mask_a_rows({vm(row_mask_vals[r])}),
            .validity_mask_a_cols_b_rows(current_k_mask),
            .validity_mask_b_cols({vm(col_mask_vals[c])}),
            .slice_dtype(2'd0), .slice_mode(1'b0), .op({{op2, op1_{r}, op0_{r}}}),
            .preload(preload_d), .no_rounding(1'b0),
            .final_mat_mul_size(current_k_size),
            .a_loc(5'd{r}),
            .b_loc(5'd{c})
        );""")

    # ── Zero-storage output collector (op[0]/op[1]/op[2] contract) ───────────
    # tensor_slice_int8 contract (jojo-track/open/tensor-slice-op-shadow-drain):
    #   op[0] out_ctrl (level): 1 HOLDS the tile's result inside the array (no
    #     burst, including after intermediate K-chunks); 0 shifts one result
    #     row/cycle onto c_data_out, qualified by c_data_available. Readout
    #     sources the SHADOW bank the op[2] pulse snapshotted, not the live
    #     accumulators.
    #   op[1] drain_stop (1-cycle pulse): terminates the remaining
    #     masked/padded tail of the current tile-row's 8-row physical burst
    #     once the logical M rows for that burst have all been taken. No
    #     effect on accumulators or shadow contents.
    #   op[2] shadow_swap (1-cycle pulse): fired grid-wide the cycle compute
    #     completes (entry into emit_phase), snapshotting every tile's final
    #     accumulators into its shadow bank and clearing the accumulators so
    #     the array is free for the next operation. pe_reset no longer does
    #     this job (see slice_pe_reset below).
    # Legacy free-run behaviour is op tied to 3'b000. All tiles finish
    # together, so column tiles of one tile-row concatenate as pure wiring;
    # the wrapper holds every tile-row and releases them one at a time, in
    # row-major order, for their 8-row bursts. No parking storage and no
    # delay pyramid (the old alignment shift-lines cost sum-of-delays x 129
    # FFs — ~6.2k on a 2x2 grid, ~29k on 4x2).
    op0_decl = []
    op1_decl = []
    for r in range(grid_rows):
        release = (f"emit_phase && (cur_row_tile == 16'd{r})"
                   if grid_rows > 1 else "emit_phase")
        op0_decl.append(f"    wire op0_{r} = !({release});")
        stop = (f"logical_output_complete && (cur_row_tile == 16'd{r})"
                if grid_rows > 1 else "logical_output_complete")
        op1_decl.append(f"    wire op1_{r} = {stop};")
    # Stage 2 (shared with every other emitter -- see _stage2_function): the
    # chunked path's slice output IS the full K contraction (single
    # partition, one term), so each lane's stage 2 call sums exactly one
    # partial. Was: a raw 16-bit c_data concat, with bias injected upstream
    # via the (now-removed) preload mux; now: the drain applies stage 2 per
    # lane, narrowing to out_width and adding the bias from the compile-time
    # bias_rom (or a literal 0 when has_bias is False -- folds the add away).
    if _fold_n_bias:
        # Frozen per-frame group (see _fold_n_bias_group_decl): latched from
        # the live counter at this frame's preload, held through its own
        # drain -- NOT the live counter itself (which may already have
        # advanced to the next frame's group by the time this frame emits).
        _bias_expr_synth = (
            lambda c, lane: _bias_lane(bias_rom_name, f"out_grp * {n} + {c * 8 + lane}"))
    elif has_bias:
        _bias_expr_synth = (lambda c, lane: _bias_lane(bias_rom_name, str(c * 8 + lane)))
    else:
        _bias_expr_synth = (lambda c, lane: "16'sd0")
    # ── Zero-point correction (streamed operands) ───────────────────────────
    # A_ZERO_POINT nonzero: correction for output column j is
    # A_ZERO_POINT * sum_k b[k][j] -- a per-column running sum of the signed
    # B codes actually fed this frame (post bit-7-flip), accumulated as the
    # slice consumes each K chunk, reset once per frame.
    # B_ZERO_POINT nonzero: correction for output row i is
    # B_ZERO_POINT * sum_k a[i][k], looked up by the row currently on the
    # output wire (out_row_count) since (unlike the per-column correction)
    # the physical row varies at emit time, not at generation time.
    # Both nonzero: also add A_ZERO_POINT * B_ZERO_POINT * K once per element
    # (folded into row_zp_corr below, so every (row, col) sees it exactly once).
    _zp_col_decl = ""
    _zp_col_reset = ""
    _zp_col_acc = ""
    if _azp:
        _zp_col_decl = f"    reg signed [15:0] colsum_b [0:{grid_cols * 8 - 1}];\n"
        _zp_col_reset = "".join(
            f"                        colsum_b[{j}] <= 16'sd0;\n" for j in range(grid_cols * 8))
        _zp_col_acc = "".join(
            f"                        if (beat_count < 16'd8 && current_k_mask[{lane}]) colsum_b[{c * 8 + lane}] <= "
            f"colsum_b[{c * 8 + lane}] + $signed(b_data_0_{c}[{lane}*8 +: 8]);\n"
            for c in range(grid_cols) for lane in range(8))
    _zp_row_decl = ""
    _zp_row_reset = ""
    _zp_row_acc = ""
    if _bzp:
        _zp_row_decl = f"    reg signed [15:0] rowsum_a [0:{grid_rows * 8 - 1}];\n"
        _zp_row_reset = "".join(
            f"                        rowsum_a[{i}] <= 16'sd0;\n" for i in range(grid_rows * 8))
        _zp_row_acc = "".join(
            f"                        if (beat_count < 16'd8 && current_k_mask[{lane}]) rowsum_a[{r * 8 + lane}] <= "
            f"rowsum_a[{r * 8 + lane}] + $signed(a_data_{r}_0[{lane}*8 +: 8]);\n"
            for r in range(grid_rows) for lane in range(8))
    _row_corr_terms = []
    if _bzp:
        _row_corr_terms.append(f"{_bzp} * rowsum_a[out_row_count]")
    if _azp and _bzp:
        _row_corr_terms.append(str(_azp * _bzp * k))
    row_zp_corr_decl = (
        f"    wire signed [31:0] row_zp_corr = {' + '.join(_row_corr_terms)};\n"
        if _row_corr_terms else ""
    )
    row_zp_corr_term = " + row_zp_corr" if _row_corr_terms else ""

    def _col_zp_corr_term(c, lane):
        return f" + ({_azp} * colsum_b[{c * 8 + lane}])" if _azp else ""

    row_avail_decl = []
    row_data_decl = []
    for r in range(grid_rows):
        avail_terms = " & ".join(f"c_avail_{r}_{c}" for c in range(grid_cols))
        tiles = []
        for c in range(grid_cols):
            lanes = ", ".join(
                f"stage2($signed(c_data_{r}_{c}[{lane}*16 +: 16]), "
                f"{_bias_expr_synth(c, lane)}{_col_zp_corr_term(c, lane)}{row_zp_corr_term})"
                for lane in range(7, -1, -1)
            )
            tiles.append("{" + lanes + "}")
        concat = "{" + ", ".join(reversed(tiles)) + "}"
        row_avail_decl.append(f"    wire row_avail_{r} = {avail_terms};")
        row_data_decl.append(f"    wire [{c_width-1}:0] row_data_{r} = {concat};")
    mux_lines = []
    for r in range(grid_rows):
        cond = (f"(cur_row_tile == 16'd{r}) && row_avail_{r}"
                if grid_rows > 1 else f"row_avail_{r}")
        mux_lines.append(f"        if ({cond}) begin")
        mux_lines.append(f"            row_mux = row_data_{r};")
        mux_lines.append("            row_take = 1'b1;")
        mux_lines.append("        end")

    debug_block = ""
    if debug:
        debug_block = """
    always @(posedge clk) begin
        if (!slice_reset && (transaction_active || row_take || |done_mat_mul)) begin
            $display("DBG t=%0t st=%0d beat=%0d done=%b take=%0b out_rows=%0d",
                     $time, state, beat_count, done_mat_mul, row_take, out_row_count);
        end
    end
"""

    # ── Weight-stationary (const-weight) variant ────────────────────────────
    # weight_rom = per-beat b_cols values (grid_cols*64 bits each), already in the
    # pack_b_chunk feed order (see gemm_ip/weights.py). When present, drop the
    # external b_cols port and source B from an internal ROM: one beat per presented
    # input cycle. beat_ctr/rom_base mirror in_valid exactly as the external b_cols
    # port did (the TB free-runs one beat/clock while in_valid is high), so
    # timing/systolic feed are byte-identical to the streamed path. Bias stays
    # external.
    ws = weight_rom is not None
    if ws:
        b_cols_port = ""
        b_cols_q_src = "w_rom_out"
        # emit_rom=False when the combined core provides the shared ROM above `ifndef.
        w_rom_block = _weight_rom_block(b_width, weight_rom, n, input_beats, n_passes=n_passes) if emit_rom else ""
    else:
        b_cols_port = f"    input  wire [{b_width-1}:0]   b_cols,\n"
        b_cols_q_src = "b_cols"
        w_rom_block = ""

    stage2_fn = _stage2_function(s2, out_width)

    return f"""\
// Auto-generated by rtl.py
// Chunked structural tensor-slice synth wrapper
// Dimensions: M={m}, K={k}, N={n}  |  Grid: {grid_rows}x{grid_cols} slices
`timescale 1ns/1ps

module {module_name}(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,
    input  wire [{a_width-1}:0]   a_rows,
{b_cols_port}    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);
{_zero_point_params_block(a_zero_point, b_zero_point)}
{w_rom_block}
{bias_rom_block}
{stage2_fn}

    localparam integer INPUT_BEATS = {input_beats};
    localparam integer K_CHUNKS = {k_chunks};
    localparam integer LAST_K_SIZE = {last_k_size};
    localparam integer LOGICAL_OUT_ROWS = {logical_output_rows};
    localparam integer PHYSICAL_OUT_ROWS = {physical_output_rows};
    localparam [1:0] S_IDLE=2'd0, S_PRELOAD=2'd1, S_RUN=2'd2, S_WAIT=2'd3;

    reg [1:0] state;
    reg [15:0] beat_count;
    reg [15:0] chunk_idx;
    reg [15:0] out_row_count;
    reg preload_d;
    reg transaction_active;
    reg [{c_width-1}:0] row_mux;
    reg row_take;
    reg emit_phase_d;
{_bias_grp_decl}
{"    reg [15:0] out_grp;" if _fold_n_bias else ""}
{_zp_col_decl}{_zp_row_decl}{row_zp_corr_decl}

    // ── Input pipeline stage ────────────────────────────────────────────────
    // Register the whole input bundle (en-gated, so the core stays self-timed),
    // keeping the beat decode and data gating muxes off the path into the
    // tensor_slice input pins.
    reg [{a_width-1}:0] a_rows_q;
    reg [{b_width-1}:0] b_cols_q;
    reg preload_valid_q;
    reg in_valid_q;

    always @(posedge clk) begin
        if (rst) begin
            a_rows_q        <= {a_width}'d0;
            b_cols_q        <= {b_width}'d0;
            preload_valid_q <= 1'b0;
            in_valid_q      <= 1'b0;
        end else if (en) begin
            // Flip is gated by current_k_mask (this cycle's chunk validity,
            // computed from the SAME pre-edge chunk_idx that decides which
            // chunk the a_rows/b_cols beat on the wire right now belongs
            // to): padding K lanes (mask bit clear -- always zero-code on
            // the wire) are left unflipped, so they stay raw 0 instead of
            // becoming -128. See _lane_flip_mask_masked.
            a_rows_q        <= a_rows ^ {_lane_flip_mask_masked(a_width, int(a_zero_point), 'current_k_mask')};
            b_cols_q        <= {b_cols_q_src} ^ {_lane_flip_mask_masked(b_width, int(b_zero_point), 'current_k_mask')};
            preload_valid_q <= preload_valid;
            in_valid_q      <= in_valid;
        end
    end

    wire slice_reset = rst;
    wire in_beat_active = (state == S_RUN) && in_valid_q && (beat_count < INPUT_BEATS);
    wire slice_start = in_beat_active && (beat_count == 16'd0);
    wire final_chunk = (chunk_idx == K_CHUNKS - 1);
    wire [7:0] current_k_size = final_chunk ? 8'd{last_k_size} : 8'd8;
    wire [7:0] current_k_mask = final_chunk ? {vm(last_k_mask)} : 8'hFF;

    wire [{grid_rows*grid_cols-1}:0] done_mat_mul;
    wire all_slices_done = &done_mat_mul;

    // Output collector control: hold every tile-row (out_ctrl=1) until all
    // tiles have finished the final chunk, then release one tile-row at a
    // time for its 8-row burst, in row-major order.
    wire emit_phase = transaction_active && final_chunk && all_slices_done;
    wire [15:0] cur_row_tile = out_row_count >> 3;
    // op[1] drain_stop source: the logical M'th row of the CURRENT tile-row's
    // burst has just been taken -- the rest of that physical 8-row burst is
    // padding tail, terminated by op1 (see op1_decl below), not by pe_reset.
    wire logical_output_complete = emit_phase && row_take &&
                                   (out_row_count + 16'd1 == LOGICAL_OUT_ROWS);
    // pe_reset reverts to accumulator-clear-only (cold start / error
    // recovery): asserted on chunk 0 of a new operation, never during drain.
    wire slice_pe_reset = slice_start && (chunk_idx == 16'd0);
    // op[2] shadow_swap: one-cycle pulse on the rising edge of emit_phase
    // (grid-wide compute-complete, after the last K chunk finishes).
    wire op2 = emit_phase && !emit_phase_d;

{chr(10).join(op0_decl)}
{chr(10).join(op1_decl)}

{chr(10).join(chain_wires)}

{chr(10).join(data_wires)}

{chr(10).join(boundary)}

{chr(10).join(inst_lines)}

{chr(10).join(row_avail_decl)}
{chr(10).join(row_data_decl)}

{debug_block}

    always @(*) begin
        row_mux  = {c_width}'d0;
        row_take = 1'b0;
{chr(10).join(mux_lines)}
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            beat_count <= 16'd0;
            chunk_idx <= 16'd0;
            out_row_count <= 16'd0;
            preload_d <= 1'b0;
            transaction_active <= 1'b0;
            emit_phase_d <= 1'b0;
            c_row <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
{"            out_grp <= 16'd0;" if _fold_n_bias else ""}
        end else if (en) begin
            preload_d <= 1'b0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
            emit_phase_d <= emit_phase;
{_bias_grp_body}

            case (state)
                S_IDLE: begin
                    beat_count <= 16'd0;
                    chunk_idx <= 16'd0;
                    out_row_count <= 16'd0;
                    transaction_active <= 1'b0;
                    if (preload_valid_q) begin
                        preload_d <= 1'b1;
                        transaction_active <= 1'b1;
{"                        out_grp <= bias_grp_ctr;" if _fold_n_bias else ""}
{_zp_col_reset}{_zp_row_reset}                        state <= S_PRELOAD;
                    end
                end

                S_PRELOAD: begin
                    state <= S_RUN;
                end

                S_RUN: begin
                    if (in_beat_active) begin
                        beat_count <= beat_count + 16'd1;
                        if (beat_count + 16'd1 == INPUT_BEATS)
                            state <= S_WAIT;
{_zp_col_acc}{_zp_row_acc}                    end

                    if (emit_phase && row_take) begin
                        c_row <= row_mux;
                        out_valid <= 1'b1;
                        out_last <= (out_row_count + 16'd1 == LOGICAL_OUT_ROWS);
                        out_row_count <= out_row_count + 16'd1;
                        if (out_row_count + 16'd1 == LOGICAL_OUT_ROWS) begin
                            transaction_active <= 1'b0;
                            state <= S_IDLE;
                        end
                    end
                end

                S_WAIT: begin
                    if (emit_phase && row_take) begin
                        c_row <= row_mux;
                        out_valid <= 1'b1;
                        out_last <= (out_row_count + 16'd1 == LOGICAL_OUT_ROWS);
                        out_row_count <= out_row_count + 16'd1;
                        if (out_row_count + 16'd1 == LOGICAL_OUT_ROWS) begin
                            transaction_active <= 1'b0;
                            state <= S_IDLE;
                        end
                    end else if (!final_chunk && all_slices_done) begin
                        chunk_idx <= chunk_idx + 16'd1;
                        beat_count <= 16'd0;
                        state <= S_RUN;
                    end
                end
            endcase
        end
    end

endmodule
"""


def generate_synth_verilog(m, k, n, module_name="gemm_grid_wrapper", feed_mode="chained", debug=False,
                           weight_rom=None, emit_rom=True, n_passes=1,
                           s1=0, s2=0, out_width=16, bias_codes=None, emit_bias_rom=True,
                           bias_rom_name="bias_rom", a_zero_point=0, b_zero_point=0,
                           a_zero_point_correct=None, b_zero_point_correct=None,
                           m_passes=1, logical_m=None, logical_n=None):
    """Public entry point: k_spatial=1 (K-in-time) branch of the unified
    ``_generate_general_synth_verilog``. See that function's docstring for
    the unification contract. Thin wrapper -- signature unchanged (m_passes/
    logical_m/logical_n are new, additive, default-1/None-preserving
    parameters for sub-phase 2b's combined-fold dispatch) so callers (and the
    golden generator) keep working verbatim.
    """
    return _generate_general_synth_verilog(
        m, k, n, module_name, k_spatial=1, feed_mode=feed_mode, debug=debug,
        weight_rom=weight_rom, emit_rom=emit_rom, n_passes=n_passes,
        s1=s1, s2=s2, out_width=out_width, bias_codes=bias_codes, emit_bias_rom=emit_bias_rom,
        bias_rom_name=bias_rom_name, a_zero_point=a_zero_point, b_zero_point=b_zero_point,
        a_zero_point_correct=a_zero_point_correct, b_zero_point_correct=b_zero_point_correct,
        m_passes=m_passes, logical_m=logical_m, logical_n=logical_n,
    )


def generate_grid_verilog(m, k, n, module_name="gemm_grid_wrapper", feed_mode="chained", debug=False):
    return generate_synth_verilog(m, k, n, module_name, feed_mode, debug)


def _split_module(text, name):
    """Split a single-module Verilog string into (header, body).

    header = 'module {name}(' … '\\n);'  (port list, inclusive).
    body   = everything after the port-list close up to (excluding) the final endmodule.
    """
    start = text.index(f"module {name}(")
    pclose = text.index("\n);", start) + len("\n);")
    end = text.rindex("endmodule")
    return text[start:pclose], text[pclose:end]


def _split_after_first_endmodule(text):
    """Split at the first endmodule → (first_module_incl_endmodule, remainder)."""
    idx = text.index("endmodule") + len("endmodule")
    rest = text[idx:].lstrip("\n")
    if rest and not rest.endswith("\n"):
        rest += "\n"
    return text[:idx], rest


def generate_combined_core_verilog(m, k, n, module_name="gemm_grid_wrapper", out_bits=None,
                                   s1=0, s2=0, out_width=None, weight_rom=None, n_passes=1,
                                   bias_codes=None, a_zero_point=0, b_zero_point=0,
                                   a_zero_point_correct=None, b_zero_point_correct=None):
    """Generate a single {module_name}.v with ifndef SYNTHESIS guard.

    ``ifndef SYNTHESIS`` — behavioral simulation model (wrapper + behav_grid).
    Used by Catapult SCVerify for RTL vs C++ co-simulation.

    ``else`` — structural synth wrapper with tensor_slice_int8 black-box slices.
    Used by Catapult HLS → downstream synthesis (Design Compiler).

    Both share the same port list so the ac_blackbox() binding is identical, and
    -- critically -- the SAME (s1, s2, out_width) so the two branches' stage-2
    text (and lane width) never disagree. ``bias_codes`` (or None -- the add
    folds away, decision 4) is baked as ONE compile-time bias ROM hoisted above
    the `ifndef, exactly like the weight ROM, so sim and synth read the
    identical declaration.

    ``out_bits`` is accepted as a legacy alias for ``out_width`` (some callers
    still pass it); ``out_width`` wins when both are given. Defaults to 8
    (today's legacy int8-lane default) when neither is given.
    """
    if out_width is None:
        out_width = out_bits if out_bits is not None else 8
    has_bias = bias_codes is not None
    # Weight-stationary and/or a real bias: hoist shared ROM(s) above the `ifndef
    # so both branches read the identical declaration (one source of truth). The
    # combined core becomes ONE module with the `ifndef INSIDE it; the sim's
    # behav_grid helper stays a trailing module. No-weight-rom/no-bias keeps the
    # old simple full-text ifndef/else path untouched (byte-identical output).
    #
    # Bias hoisting has one wrinkle weight-rom hoisting doesn't: the sim
    # branch's actual per-column bias use lives inside `behav_grid`, a
    # SEPARATE Verilog module (not the top wrapper), so a `bias_rom` declared
    # in the wrapper's scope (above `ifndef) is not visible there -- unlike
    # `w_rom_out`, which the wrapper passes into `grid` as a plain port. So:
    # the sim branch self-declares its OWN bias_rom inside behav_grid
    # (emit_bias_rom=True); the synth branch (single module, no submodule)
    # reads the ONE hoisted declaration below. Both are built from the same
    # `bias_codes` list, so the two declarations are byte-identical text even
    # though physically duplicated -- the single Python source of truth the
    # plan asks for, same as the K-spatial combined core already does for its
    # (also per-branch) weight ROM.
    if weight_rom is not None or has_bias:
        sim_top = generate_sim_verilog(m, k, n, module_name, s1=s1, s2=s2, out_width=out_width,
                                       weight_rom=weight_rom, emit_rom=False,
                                       bias_codes=bias_codes, emit_bias_rom=True,
                                       a_zero_point=a_zero_point, b_zero_point=b_zero_point,
                                       a_zero_point_correct=a_zero_point_correct,
                                       b_zero_point_correct=b_zero_point_correct)
        synth_top = generate_synth_verilog(m, k, n, module_name, weight_rom=weight_rom, emit_rom=False,
                                           s1=s1, s2=s2, out_width=out_width,
                                           bias_codes=bias_codes, emit_bias_rom=False,
                                           a_zero_point=a_zero_point, b_zero_point=b_zero_point,
                                           a_zero_point_correct=a_zero_point_correct,
                                           b_zero_point_correct=b_zero_point_correct)
        b_width = ((n + 7) // 8) * 64
        input_beats = max(m, n)
        header, syn_body = _split_module(synth_top, module_name)   # header incl. 'module..);'
        sim_wrapper, sim_behav = _split_after_first_endmodule(sim_top)
        _, sim_body = _split_module(sim_wrapper, module_name)
        rom_block = _weight_rom_block(b_width, weight_rom, n, input_beats, n_passes=n_passes) \
            if weight_rom is not None else ""
        bias_rom = _bias_rom_block(bias_codes) if has_bias else ""
        tag = []
        if weight_rom is not None:
            tag.append("shared const-weight ROM")
        if has_bias:
            tag.append("bias ROM (synth branch; sim's behav_grid self-declares the identical ROM)")
        out = (
            f"// Auto-generated by rtl.py\n"
            f"// Combined core: M={m}, K={k}, N={n}\n"
            f"//   {' + '.join(tag)} above `ifndef feeds both branches\n"
            f"{header}\n{rom_block}{bias_rom}\n"
            f"`ifndef SYNTHESIS\n{sim_body}`else\n{syn_body}`endif\n"
            f"endmodule\n\n"
            f"`ifndef SYNTHESIS\n{sim_behav}`endif\n"
        )
        return out

    sim_top = generate_sim_verilog(m, k, n, module_name, s1=s1, s2=s2, out_width=out_width,
                                   a_zero_point=a_zero_point, b_zero_point=b_zero_point,
                                   a_zero_point_correct=a_zero_point_correct,
                                   b_zero_point_correct=b_zero_point_correct)
    synth_top = generate_synth_verilog(m, k, n, module_name, s1=s1, s2=s2, out_width=out_width,
                                       a_zero_point=a_zero_point, b_zero_point=b_zero_point,
                                       a_zero_point_correct=a_zero_point_correct,
                                       b_zero_point_correct=b_zero_point_correct)

    lines = []
    lines.append("// Auto-generated by rtl.py")
    lines.append(f"// Combined core: M={m}, K={k}, N={n}")
    lines.append("//   ifndef SYNTHESIS → behavioral simulation model (SCVerify)")
    lines.append("//   else            → structural synth wrapper  (HLS synthesis)")
    lines.append("")
    lines.append("`ifndef SYNTHESIS")
    lines.append("")
    for l in sim_top.splitlines():
        lines.append(l)
    lines.append("")
    lines.append("`else")
    lines.append("")
    for l in synth_top.splitlines():
        lines.append(l)
    lines.append("")
    lines.append("`endif")
    return "\n".join(lines) + "\n"


def _k_spatial_partitions(k, k_spatial):
    k_chunks = (k + 7) // 8
    if k_spatial < 1:
        raise ValueError("k_spatial must be >= 1")
    if k_spatial > k_chunks:
        raise ValueError(
            f"k_spatial={k_spatial} exceeds K_CHUNKS={k_chunks}; "
            "v1 requires at most one spatial grid per K chunk"
        )
    base = k_chunks // k_spatial
    extra = k_chunks % k_spatial
    out = []
    start = 0
    for p in range(k_spatial):
        count = base + (1 if p < extra else 0)
        out.append((start, start + count - 1))
        start += count
    return out


def _general_synth_kspatial_branch(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1, n_passes=1,
                                   s1=0, s2=0, out_width=16,
                                   weight_rom=None, emit_rom=True,
                                   bias_codes=None, emit_bias_rom=True, bias_rom_name="bias_rom"):
    """k_spatial > 1 branch of ``_generate_general_synth_verilog``: K-in-space,
    ``k_spatial`` INDEPENDENT ``Ms x Ns`` grids whose 16-bit partials are
    reduced EXTERNALLY in the wrapper (see the unification docstring on
    ``_generate_general_synth_verilog``). Callers never call this directly;
    ``k_spatial == 1`` is dispatched to the other branch by the caller.

    Tensor-slice outputs (and therefore the K-chunk partials) are 16-bit, as in
    the current architecture. Stage 2 (shared with every other emitter) sums the
    partials in 16-bit wrapping arithmetic, adds the bias, and rounds/wraps to
    out_width -- no saturation anywhere.

    The structural body instantiates multiple tensor-slice grids and exposes the
    intended partition/control topology. Functional RTL simulation for this
    experimental mode is provided by the combined core's behavioral branch.
    """
    has_bias = bias_codes is not None
    bias_rom_block = _bias_rom_block(bias_codes, bias_rom_name) if (has_bias and emit_bias_rom) else ""
    _fold_n_bias = bool(has_bias and n_passes and int(n_passes) > 1)
    _bias_grp_decl, _bias_grp_body = (
        _fold_n_bias_group_decl(n_passes) if _fold_n_bias else ("", "")
    )
    stage2_fn = _stage2_function(s2, out_width)

    # Passes over K: ``k_spatial`` partitions cover K_CHUNKS chunks in
    # ``passes = ceil(K_CHUNKS/k_spatial)`` passes; pass q, beat t carries
    # chunk ``q*k_spatial + p`` on partition p. ``passes == 1`` is today's
    # full-K endpoint (k_spatial == k_chunks) and reproduces its Verilog
    # byte-for-byte -- the pass counter is otherwise unused at that endpoint.
    k_chunks = (k + 7) // 8
    passes = -(-k_chunks // k_spatial)
    full_k_spatial = passes == 1
    # Weight-stationary: same contract as the chunked emitter — drop the external
    # b_cols port and register B from the shared ROM instead. emit_rom=False when the
    # combined core hoists one ROM above the `ifndef so both branches read it.
    ksp_ws = weight_rom is not None
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    # Narrow per-beat word: partition p carries one 64-bit tile (a K chunk each
    # pass); the wrapper routes it to the row/col tile by beat index, so no
    # grid padding -- true for both the single-pass (full-K) and multi-pass
    # general K-spatial layouts.
    a_width = 64 * k_spatial
    b_width = 64 * k_spatial
    a_chunk_width = 64
    b_chunk_width = 64
    c_width = grid_cols * 8 * out_width
    input_beats = max(m, n)
    # K-spatial uses the same logical-row retirement contract as the chunked
    # wrapper; physical 8-row bursts are aborted after logical M.
    logical_output_rows = m
    physical_output_rows = grid_rows * 8

    if ksp_ws:
        ksp_b_cols_port = ""
        ksp_b_cols_src = "w_rom_out"
        ksp_rom_block = _weight_rom_block(b_width, weight_rom, n, input_beats, n_passes=n_passes) if emit_rom else ""
    else:
        ksp_b_cols_port = f"    input  wire [{b_width-1}:0]   b_cols,\n"
        ksp_b_cols_src = "b_cols"
        ksp_rom_block = ""

    row_mask_vals = [tail_mask_hex(m, r) for r in range(grid_rows)]
    col_mask_vals = [tail_mask_hex(n, c) for c in range(grid_cols)]

    decls = []
    insts = []
    if full_k_spatial:
        # ── passes == 1 (k_spatial == k_chunks): today's full-K endpoint. ──
        # Kept byte-identical to the pre-fold-K generator: partition p always
        # carries chunk p (there is only one pass), so every k_size/k_mask is
        # a Python-computed literal, not a runtime expression.
        partitions = _k_spatial_partitions(k, k_spatial)
        part_comments = "\n".join(
            f"// K_SPATIAL_PARTITION {p}: chunks {lo}..{hi}" for p, (lo, hi) in enumerate(partitions)
        )
        for p, (lo, hi) in enumerate(partitions):
            decls.append(f"    // Spatial grid {p}: contiguous K chunks {lo}..{hi}")
            decls.append(f"    wire part{p}_active = 1'b1;")
            decls.append(f"    wire part{p}_first_chunk = 1'b1;")
            decls.append(f"    wire part{p}_last_chunk = 1'b1;")
            part_k_size = k - lo * 8 if hi == k_chunks - 1 else 8
            part_k_mask = tail_mask_hex(k, hi) if hi == k_chunks - 1 else 0xFF
            decls.append(
                f"    wire [7:0] part{p}_k_size = part{p}_last_chunk ? "
                f"((16'd{hi} == K_CHUNKS - 1) ? 8'd{part_k_size} : 8'd8) : 8'd8;"
            )
            decls.append(
                f"    wire [7:0] part{p}_k_mask = part{p}_last_chunk ? "
                f"((16'd{hi} == K_CHUNKS - 1) ? {vm(part_k_mask)} : 8'hFF) : 8'hFF;"
            )
            for r in range(grid_rows):
                for c in range(grid_cols):
                    idx = p * grid_rows * grid_cols + r * grid_cols + c
                    a_expr = f"a_rows_q[{p}*{a_chunk_width} + 63:{p}*{a_chunk_width}]"
                    b_expr = f"b_cols_q[{p}*{b_chunk_width} + 63:{p}*{b_chunk_width}]"
                    a_route = f" && (beat_count >> 3 == {r})"
                    b_route = f" && (beat_count >> 3 == {c})"
                    insts.append(f"""\
        // S1 = {s1}: in-slice stage-1 round-half-up shift (IP parameter, set out of band;
        // not passed as a Verilog override -- the VTR hard-block model has no parameters)
        (* black_box = "true" *) (* keep = "true" *) tensor_slice_int8 slice_p{p}_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(slice_start && part{p}_first_chunk),
            .start_mat_mul(slice_start && part{p}_active),
            .done_mat_mul(done_mat_mul[{idx}]),
            .a_data((part{p}_active && ({c} == 0){a_route}) ? {a_expr} : 64'b0),
            .b_data((part{p}_active && ({r} == 0){b_route}) ? {b_expr} : 64'b0),
            .a_data_in(a_chain_p{p}_r{r}_c{c}),
            .b_data_in(b_chain_p{p}_r{r}_c{c}),
            .a_data_out(a_chain_p{p}_r{r}_c{c+1}),
            .b_data_out(b_chain_p{p}_r{r+1}_c{c}),
            .c_data_out(partial_c_p{p}_r{r}_c{c}),
            .c_data_available(partial_avail_p{p}_r{r}_c{c}),
            .validity_mask_a_rows({vm(row_mask_vals[r])}),
            .validity_mask_a_cols_b_rows(part{p}_k_mask),
            .validity_mask_b_cols({vm(col_mask_vals[c])}),
            .slice_dtype(2'd0), .slice_mode(1'b0), .op({{op2, op1_{r}, op0_{r}}}),
            .preload(1'b0), .no_rounding(1'b0),
            .final_mat_mul_size(part{p}_k_size),
            .a_loc(5'd{r}),
            .b_loc(5'd{c})
        );""")
    else:
        # ── passes > 1: general pass-sequenced K-spatial fold. ──
        # All k_spatial partitions are active every pass; partition p carries
        # chunk `chunk_idx*k_spatial + p` (chunk_idx is the pass counter,
        # renamed from the old placeholder's per-chunk index -- see the
        # S_WAIT case below). Chunks beyond K_CHUNKS (padding to fill the
        # last pass) are fully masked so every partition runs every pass.
        tail_chunk = k_chunks - 1
        tail_k_size = k - tail_chunk * 8
        tail_k_mask = tail_mask_hex(k, tail_chunk)
        part_comments = "\n".join(
            f"// K_SPATIAL_PARTITION {p}: chunk = pass*{k_spatial} + {p}" for p in range(k_spatial)
        )
        for p in range(k_spatial):
            decls.append(f"    // Spatial grid {p}: chunk = pass*{k_spatial} + {p}")
            decls.append(f"    wire part{p}_active = 1'b1;")
            decls.append(f"    wire part{p}_first_chunk = (chunk_idx == 16'd0);")
            decls.append(f"    wire [15:0] part{p}_chunk = chunk_idx * 16'd{k_spatial} + 16'd{p};")
            decls.append(f"    wire part{p}_chunk_pad = (part{p}_chunk >= K_CHUNKS);")
            decls.append(f"    wire part{p}_chunk_tail = (part{p}_chunk == K_CHUNKS - 16'd1);")
            decls.append(
                f"    wire [7:0] part{p}_k_size = part{p}_chunk_pad ? 8'd8 : "
                f"(part{p}_chunk_tail ? 8'd{tail_k_size} : 8'd8);"
            )
            decls.append(
                f"    wire [7:0] part{p}_k_mask = part{p}_chunk_pad ? 8'h00 : "
                f"(part{p}_chunk_tail ? {vm(tail_k_mask)} : 8'hFF);"
            )
            for r in range(grid_rows):
                for c in range(grid_cols):
                    idx = p * grid_rows * grid_cols + r * grid_cols + c
                    a_expr = f"a_rows_q[{p}*{a_chunk_width} + 63:{p}*{a_chunk_width}]"
                    b_expr = f"b_cols_q[{p}*{b_chunk_width} + 63:{p}*{b_chunk_width}]"
                    a_route = f" && (beat_count >> 3 == {r})"
                    b_route = f" && (beat_count >> 3 == {c})"
                    insts.append(f"""\
        // S1 = {s1}: in-slice stage-1 round-half-up shift (IP parameter, set out of band;
        // not passed as a Verilog override -- the VTR hard-block model has no parameters)
        (* black_box = "true" *) (* keep = "true" *) tensor_slice_int8 slice_p{p}_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(slice_start && part{p}_first_chunk),
            .start_mat_mul(slice_start && part{p}_active),
            .done_mat_mul(done_mat_mul[{idx}]),
            .a_data((part{p}_active && ({c} == 0){a_route}) ? {a_expr} : 64'b0),
            .b_data((part{p}_active && ({r} == 0){b_route}) ? {b_expr} : 64'b0),
            .a_data_in(a_chain_p{p}_r{r}_c{c}),
            .b_data_in(b_chain_p{p}_r{r}_c{c}),
            .a_data_out(a_chain_p{p}_r{r}_c{c+1}),
            .b_data_out(b_chain_p{p}_r{r+1}_c{c}),
            .c_data_out(partial_c_p{p}_r{r}_c{c}),
            .c_data_available(partial_avail_p{p}_r{r}_c{c}),
            .validity_mask_a_rows({vm(row_mask_vals[r])}),
            .validity_mask_a_cols_b_rows(part{p}_k_mask),
            .validity_mask_b_cols({vm(col_mask_vals[c])}),
            .slice_dtype(2'd0), .slice_mode(1'b0), .op({{op2, op1_{r}, op0_{r}}}),
            .preload(1'b0), .no_rounding(1'b0),
            .final_mat_mul_size(part{p}_k_size),
            .a_loc(5'd{r}),
            .b_loc(5'd{c})
        );""")

    partial_wires = []
    for p in range(k_spatial):
        for r in range(grid_rows):
            for c in range(grid_cols):
                partial_wires.append(f"    wire [127:0] partial_c_p{p}_r{r}_c{c};")
                partial_wires.append(f"    wire         partial_avail_p{p}_r{r}_c{c};")
    # Per-partition systolic chain: each K-spatial partition p is its own
    # grid_rows x grid_cols grid (A flows left->right, B flows top->bottom
    # within the partition; the k_spatial partials are reduced OUTSIDE the
    # grids by the Sum_p row_mux below -- no chain crosses partitions). Mirrors
    # the chunked emitter's a_chain/b_chain wiring, replicated per partition.
    # Boundary column (c==0) and row (r==0) chain inputs are zero; interior
    # tiles are fed only through the chain (their a_data/b_data are 0).
    for p in range(k_spatial):
        for r in range(grid_rows):
            for c in range(grid_cols + 1):
                partial_wires.append(f"    wire [63:0] a_chain_p{p}_r{r}_c{c};")
        for r in range(grid_rows + 1):
            for c in range(grid_cols):
                partial_wires.append(f"    wire [63:0] b_chain_p{p}_r{r}_c{c};")
        for r in range(grid_rows):
            partial_wires.append(f"    assign a_chain_p{p}_r{r}_c0 = 64'b0;")
        for c in range(grid_cols):
            partial_wires.append(f"    assign b_chain_p{p}_r0_c{c} = 64'b0;")

    row_avail = []
    row_mux_cases = []
    for r in range(grid_rows):
        avail_terms = " & ".join(
            f"partial_avail_p{p}_r{r}_c{c}" for p in range(k_spatial) for c in range(grid_cols)
        )
        row_avail.append(f"    wire row_avail_{r} = {avail_terms};")
        row_mux_cases.append(f"        if (row_avail_{r}) begin")
        for c in range(grid_cols):
            for lane in range(8):
                # Stage 2 (shared with every other emitter -- see
                # _stage2_function): sum the K-partition 16-bit partials in
                # 16-bit WRAPPING arithmetic (the `accum16` reg truncates the
                # sum to 16 bits on assignment), then let stage2() add the
                # bias and round/wrap to out_width. Slice outputs (and hence
                # the partials) stay 16-bit; no saturation anywhere.
                terms = " + ".join(
                    f"$signed(partial_c_p{p}_r{r}_c{c}[{lane}*16 +: 16])" for p in range(k_spatial)
                )
                row_mux_cases.append(f"            accum16 = {terms};")
                if _fold_n_bias:
                    # Frozen per-frame group (see _fold_n_bias_group_decl),
                    # not the live counter -- it may already have advanced.
                    _bias_e = f"{_bias_lane(bias_rom_name, f'out_grp * {n} + {c * 8 + lane}')}"
                elif has_bias:
                    _bias_e = f"{_bias_lane(bias_rom_name, str(c * 8 + lane))}"
                else:
                    _bias_e = "16'sd0"
                row_mux_cases.append(
                    f"            row_mux[{c}*{8*out_width} + {lane}*{out_width} +: {out_width}] = "
                    f"stage2(accum16, {_bias_e});"
                )
        row_mux_cases.append("        end")
    any_avail_expr = " | ".join(f"row_avail_{r}" for r in range(grid_rows))
    op0_lines = "\n".join(
        f"    wire op0_{r} = !((state == S_OUTPUT) && (cur_row_tile == 16'd{r}));"
        for r in range(grid_rows)
    )
    op1_lines = "\n".join(
        (f"    wire op1_{r} = logical_output_complete && (cur_row_tile == 16'd{r});"
         if grid_rows > 1 else f"    wire op1_{r} = logical_output_complete;")
        for r in range(grid_rows)
    )

    if full_k_spatial:
        wait_body = """\
                    if (all_slices_done) begin
                        state <= S_OUTPUT;
                    end"""
    else:
        # chunk_idx doubles as the pass counter here: it advances once per
        # pass (not once per chunk), and the loop exits after `passes` passes.
        wait_body = f"""\
                    if (all_slices_done) begin
                        if (chunk_idx + 16'd1 == 16'd{passes}) begin
                            state <= S_OUTPUT;
                        end else begin
                            chunk_idx <= chunk_idx + 16'd1;
                            beat_count <= 16'd0;
                            state <= S_RUN;
                        end
                    end"""

    return f"""\
// Auto-generated by rtl.py
// Experimental K-spatial structural tensor-slice synth wrapper
// Dimensions: M={m}, K={k}, N={n}  |  Grid: {grid_rows}x{grid_cols} slices, K_SPATIAL={k_spatial}
// WARNING: K-spatial partial outputs are INT16; correctness requires every partition partial sum to fit INT16.
{part_comments}
`timescale 1ns/1ps

module {module_name}(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,
    input  wire [{a_width-1}:0]   a_rows,
{ksp_b_cols_port}    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);
{ksp_rom_block}
{bias_rom_block}
    localparam integer INPUT_BEATS = {input_beats};
    localparam integer K_CHUNKS = {k_chunks};
    localparam integer K_SPATIAL = {k_spatial};
    localparam integer LOGICAL_OUT_ROWS = {logical_output_rows};
    localparam integer PHYSICAL_OUT_ROWS = {physical_output_rows};
    localparam [1:0] S_IDLE=2'd0, S_RUN=2'd1, S_WAIT=2'd2, S_OUTPUT=2'd3;

    reg [1:0] state;
    reg [15:0] beat_count;
    reg [15:0] chunk_idx;
    reg [15:0] out_row_count;
    reg signed [15:0] accum16;
    reg [{c_width-1}:0] row_mux;
    reg state_output_d;
{_bias_grp_decl}
{"    reg [15:0] out_grp;" if _fold_n_bias else ""}

    // Input pipeline stage (same rationale as the chunked emitter): register
    // the whole bundle en-gated so the beat decode + partition gating muxes
    // start from local registers instead of chaining from the Catapult
    // wrapper into the tensor_slice input pins.
    reg [{a_width-1}:0] a_rows_q;
    reg [{b_width-1}:0] b_cols_q;
    reg preload_valid_q;
    reg in_valid_q;

    always @(posedge clk) begin
        if (rst) begin
            a_rows_q        <= {a_width}'d0;
            b_cols_q        <= {b_width}'d0;
            preload_valid_q <= 1'b0;
            in_valid_q      <= 1'b0;
        end else if (en) begin
            a_rows_q        <= a_rows;
            b_cols_q        <= {ksp_b_cols_src};
            preload_valid_q <= preload_valid;
            in_valid_q      <= in_valid;
        end
    end

    wire slice_reset = rst;
    wire in_beat_active = (state == S_RUN) && in_valid_q && (beat_count < INPUT_BEATS);
    wire slice_start = in_beat_active && (beat_count == 16'd0);
    wire [{k_spatial * grid_rows * grid_cols - 1}:0] done_mat_mul;
    wire all_slices_done = &done_mat_mul;
    // op[1] drain_stop source: the logical M'th row of the current tile-row's
    // burst has just been taken -- the remaining physical rows are padding
    // tail, terminated by op1 (op1_lines below), not by pe_reset.
    wire logical_output_complete = (state == S_OUTPUT) && any_avail &&
                                   (out_row_count + 16'd1 == LOGICAL_OUT_ROWS);
    // op[2] shadow_swap: one-cycle pulse on entry into S_OUTPUT (grid-wide
    // compute-complete, after the last K pass finishes).
    wire op2 = (state == S_OUTPUT) && !state_output_d;

    // op[0] readout gate (op[0] == out_ctrl on the tensor slice): hold every
    // tile-row's completed result inside the tiles until S_OUTPUT, then
    // release one tile-row at a time, in row-major order. The released row's
    // partition partials are summed combinationally below; no parking
    // storage is needed and intermediate-chunk bursts never occur. Readout
    // sources each tile's shadow bank, snapshotted by the op[2] pulse above.
    wire [15:0] cur_row_tile = out_row_count >> 3;
{op0_lines}
{op1_lines}

{chr(10).join(decls)}

{chr(10).join(partial_wires)}

{chr(10).join(insts)}

{chr(10).join(row_avail)}

    wire any_avail = {any_avail_expr};

{stage2_fn}
    always @(*) begin
        row_mux = {c_width}'d0;
        accum16 = 16'sd0;
{chr(10).join(row_mux_cases)}
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            beat_count <= 16'd0;
            chunk_idx <= 16'd0;
            out_row_count <= 16'd0;
            c_row <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
            state_output_d <= 1'b0;
{"            out_grp <= 16'd0;" if _fold_n_bias else ""}
        end else if (en) begin
            out_valid <= 1'b0;
            out_last <= 1'b0;
            state_output_d <= (state == S_OUTPUT);
{_bias_grp_body}
            case (state)
                S_IDLE: begin
                    beat_count <= 16'd0;
                    chunk_idx <= 16'd0;
                    out_row_count <= 16'd0;
                    if (preload_valid_q) begin
{"                        out_grp <= bias_grp_ctr;" if _fold_n_bias else ""}
                        state <= S_RUN;
                    end
                end
                S_RUN: begin
                    if (in_beat_active) begin
                        beat_count <= beat_count + 16'd1;
                        if (beat_count + 16'd1 == INPUT_BEATS)
                            state <= S_WAIT;
                    end
                end
                S_WAIT: begin
{wait_body}
                end
                S_OUTPUT: begin
                    if (any_avail) begin
                        c_row <= row_mux;
                        out_valid <= 1'b1;
                        out_last <= (out_row_count + 16'd1 == LOGICAL_OUT_ROWS);
                        out_row_count <= out_row_count + 16'd1;
                        if (out_row_count + 16'd1 == LOGICAL_OUT_ROWS)
                            state <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule
"""


def _general_synth_combined_fold(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1,
                                 m_passes=1, n_passes=1, logical_m=None, logical_n=None,
                                 s1=0, s2=0, out_width=16,
                                 weight_rom=None, emit_rom=True,
                                 bias_codes=None, emit_bias_rom=True, bias_rom_name="bias_rom"):
    """ADDITIVE combined-fold general synth emitter (jojo-track/open/
    tensor-slice-general-synth-grid, sub-phase 2b).

    Used ONLY when 2+ of (m_passes, k time-passes, n_passes) fold -- see the
    dispatch in ``_generate_general_synth_verilog``. The verified single-axis
    branches (``k_spatial == 1`` body above and ``_general_synth_kspatial_branch``)
    are left byte-identical; this is a NEW, separate structural body modeled on
    ``_general_synth_kspatial_branch`` (k_spatial copies of a core_m x core_n
    flowing grid + external K-space reduction), generalized with an internal
    sequential m_group/n_group/k_pass wrapper FSM.

    ``m``/``n`` here are the per-group CORE dims (``core_m``/``core_n`` --
    ``m_spatial*8``/``n_spatial*8``), i.e. the physical grid size reused every
    group. ``logical_m``/``logical_n`` are the TRUE M/N (default to ``m``/``n``
    when a caller leaves an axis unfolded) and only matter for the ragged
    final group's row/column validity masks.

    Per the confirmed contract and the csim-verified simplification: weight
    and bias ROM contents depend ONLY on ``(k_pass, n_group)``, never on
    ``m_group`` -- ``m_group`` (``mg``) purely gates which logical A-rows are
    fed/captured. Groups are decoded from a single ``frame`` counter,
    mg-major/ng-minor (``mg = frame // n_passes``, ``ng = frame % n_passes``),
    matching the existing csim/package.py merged-frame driver exactly, and run
    STRICTLY SEQUENTIALLY (drain group g fully before computing g+1) -- no
    ping-pong banks, no overlap (Step 3).

    No preload: weights are either streamed (``b_cols`` beats) or baked into
    ``weight_rom`` (weight-stationary); the ``preload`` tensor_slice_int8 pin
    is tied low either way.
    """
    logical_m = m if logical_m is None else int(logical_m)
    logical_n = n if logical_n is None else int(logical_n)
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    # a_loc/b_loc are 5-bit ports (0..31): hard-error rather than silently
    # truncate a spatial factor the owner has not extended the port for yet.
    if grid_rows > 32 or grid_cols > 32:
        raise ValueError(
            f"{module_name}: combined-fold grid is {grid_rows}x{grid_cols} slices "
            f"(m_spatial={grid_rows}, n_spatial={grid_cols}); a_loc/b_loc are 5-bit "
            "ports capping m_spatial/n_spatial at 32. Reduce the fold so the core grid "
            "fits, or ask the tensor_slice_int8 owner to widen a_loc/b_loc."
        )

    has_bias = bias_codes is not None
    bias_rom_block = _bias_rom_block(bias_codes, bias_rom_name) if (has_bias and emit_bias_rom) else ""
    stage2_fn = _stage2_function(s2, out_width)

    k_chunks = (k + 7) // 8
    if k_spatial < 1 or k_spatial > k_chunks:
        raise ValueError(f"k_spatial={k_spatial} must be in [1, K_CHUNKS={k_chunks}]")
    passes = -(-k_chunks // k_spatial)
    tail_chunk = k_chunks - 1
    tail_k_size = k - tail_chunk * 8
    tail_k_mask = tail_mask_hex(k, tail_chunk)

    total_frames = int(m_passes) * int(n_passes)

    # weight-stationary: same contract as every other emitter -- no external
    # b_cols port when a ROM is supplied.
    ws = weight_rom is not None
    a_width = 64 * k_spatial
    b_width = 64 * k_spatial
    c_width = grid_cols * 8 * out_width
    input_beats = max(m, n)

    if ws:
        b_cols_port = ""
        b_cols_src = "w_rom_out"
        # #RUTHWIK: this combined-fold ROM block is a NEW, simpler address
        # generator (base = (k_pass*n_passes + n_group)*n + beat), unlike
        # _weight_rom_block's frame-boundary-detecting grp_ctr (built for
        # EXTERNAL multi-frame replay). Unify the two once 2c generalizes
        # weight ROM addressing for real. The read address (``rom_addr``) is
        # now a REGISTERED mirror of that same expression -- updated in
        # lockstep with the FSM's own beat_count/chunk_idx/frame transitions
        # below (same trick _weight_rom_block uses: the register always holds
        # the address to use THIS cycle, computed from next-state values at
        # the prior edge) -- so VTR's parmys infers w_rom as a clocked
        # single_port_ram (needs a register-fed address, not a combinational
        # chunk_idx/ng/beat_count expression). See the FSM always block for
        # the actual rom_addr updates.
        rom_beats = int(passes) * int(n_passes) * n
        if weight_rom is not None and len(weight_rom) != rom_beats:
            raise ValueError(
                f"{module_name}: combined-fold weight_rom has {len(weight_rom)} beats, "
                f"expected passes*n_passes*n = {rom_beats}"
            )
        if emit_rom:
            hexw = (b_width + 3) // 4
            mask = (1 << b_width) - 1
            rom_init = "\n".join(
                f"        w_rom[{i}] = {b_width}'h{(int(v) & mask):0{hexw}x};"
                for i, v in enumerate(weight_rom)
            )
            rom_block = f"""
    // Combined-fold weight-stationary ROM: {rom_beats} beats = K_PASSES({passes}) x
    // N_PASSES({n_passes}) x n({n}); weight depends only on (k_pass, n_group), never
    // m_group -- base = (chunk_idx*n_passes + ng)*n, replayed identically for every mg.
    // rom_addr is a plain register (declared with the FSM state below, updated
    // in the FSM always block) so parmys infers a clocked single_port_ram.
    reg [{b_width - 1}:0] w_rom [0:{rom_beats - 1}];
    initial begin
{rom_init}
    end
    wire [{b_width - 1}:0] w_rom_out = (beat_count < 16'd{n}) ? w_rom[rom_addr] : {b_width}'d0;
"""
        else:
            rom_block = ""
    else:
        b_cols_port = f"    input  wire [{b_width-1}:0]   b_cols,\n"
        b_cols_src = "b_cols"
        rom_block = ""

    part_comments = "\n".join(
        f"// K_SPATIAL_PARTITION {p}: chunk = k_pass*{k_spatial} + {p}" for p in range(k_spatial)
    )
    decls = []
    insts = []
    for p in range(k_spatial):
        decls.append(f"    // Spatial grid {p}: chunk = chunk_idx*{k_spatial} + {p}")
        decls.append(f"    wire part{p}_first_chunk = (chunk_idx == 16'd0);")
        decls.append(f"    wire [15:0] part{p}_chunk = chunk_idx * 16'd{k_spatial} + 16'd{p};")
        decls.append(f"    wire part{p}_chunk_pad = (part{p}_chunk >= K_CHUNKS);")
        decls.append(f"    wire part{p}_chunk_tail = (part{p}_chunk == K_CHUNKS - 16'd1);")
        decls.append(
            f"    wire [7:0] part{p}_k_size = part{p}_chunk_pad ? 8'd8 : "
            f"(part{p}_chunk_tail ? 8'd{tail_k_size} : 8'd8);"
        )
        decls.append(
            f"    wire [7:0] part{p}_k_mask = part{p}_chunk_pad ? 8'h00 : "
            f"(part{p}_chunk_tail ? {vm(tail_k_mask)} : 8'hFF);"
        )
        for r in range(grid_rows):
            for c in range(grid_cols):
                idx = p * grid_rows * grid_cols + r * grid_cols + c
                a_expr = f"a_rows_q[{p}*64 + 63:{p}*64]"
                b_expr = f"b_cols_q[{p}*64 + 63:{p}*64]"
                a_route = f" && (beat_count >> 3 == {r})"
                b_route = f" && (beat_count >> 3 == {c})"
                insts.append(f"""\
        // S1 = {s1}: in-slice stage-1 round-half-up shift (IP parameter, set out of band)
        (* black_box = "true" *) (* keep = "true" *) tensor_slice_int8 slice_p{p}_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(slice_start && part{p}_first_chunk),
            .start_mat_mul(slice_start),
            .done_mat_mul(done_mat_mul[{idx}]),
            .a_data(({c} == 0){a_route} ? {a_expr} : 64'b0),
            .b_data(({r} == 0){b_route} ? {b_expr} : 64'b0),
            .a_data_in(a_chain_p{p}_r{r}_c{c}),
            .b_data_in(b_chain_p{p}_r{r}_c{c}),
            .a_data_out(a_chain_p{p}_r{r}_c{c+1}),
            .b_data_out(b_chain_p{p}_r{r+1}_c{c}),
            .c_data_out(partial_c_p{p}_r{r}_c{c}),
            .c_data_available(partial_avail_p{p}_r{r}_c{c}),
            .validity_mask_a_rows(row_mask_{r}),
            .validity_mask_a_cols_b_rows(part{p}_k_mask),
            .validity_mask_b_cols(col_mask_{c}),
            .slice_dtype(2'd0), .slice_mode(1'b0), .op({{op2, op1_{r}, op0_{r}}}),
            .preload(1'b0), .no_rounding(1'b0),
            .final_mat_mul_size(part{p}_k_size),
            .a_loc(5'd{r}),
            .b_loc(5'd{c})
        );""")

    partial_wires = []
    for p in range(k_spatial):
        for r in range(grid_rows):
            for c in range(grid_cols):
                partial_wires.append(f"    wire [127:0] partial_c_p{p}_r{r}_c{c};")
                partial_wires.append(f"    wire         partial_avail_p{p}_r{r}_c{c};")
    # Per-partition systolic chain: each K-spatial partition p is its own
    # grid_rows x grid_cols grid (A flows left->right, B flows top->bottom
    # within the partition; the k_spatial partials are reduced OUTSIDE the
    # grids by the Sum_p row_mux below -- no chain crosses partitions). Mirrors
    # the chunked emitter's a_chain/b_chain wiring, replicated per partition.
    # Boundary column (c==0) and row (r==0) chain inputs are zero; interior
    # tiles are fed only through the chain (their a_data/b_data are 0).
    for p in range(k_spatial):
        for r in range(grid_rows):
            for c in range(grid_cols + 1):
                partial_wires.append(f"    wire [63:0] a_chain_p{p}_r{r}_c{c};")
        for r in range(grid_rows + 1):
            for c in range(grid_cols):
                partial_wires.append(f"    wire [63:0] b_chain_p{p}_r{r}_c{c};")
        for r in range(grid_rows):
            partial_wires.append(f"    assign a_chain_p{p}_r{r}_c0 = 64'b0;")
        for c in range(grid_cols):
            partial_wires.append(f"    assign b_chain_p{p}_r0_c{c} = 64'b0;")

    # Per-group ragged-tail masks (#RUTHWIK: same tail_mask_hex algorithm the
    # single-axis branches evaluate at PYTHON compile time -- here it must be
    # a RUNTIME wire because only the LAST m_group/n_group is ragged, and mg/ng
    # vary at runtime; unify into one mask helper once the single-axis branches
    # also need a runtime form).
    row_avail = []
    row_mux_cases = []
    for r in range(grid_rows):
        avail_terms = " & ".join(
            f"partial_avail_p{p}_r{r}_c{c}" for p in range(k_spatial) for c in range(grid_cols)
        )
        row_avail.append(f"    wire row_avail_{r} = {avail_terms};")
        row_mux_cases.append(f"        if (row_avail_{r}) begin")
        for c in range(grid_cols):
            for lane in range(8):
                terms = " + ".join(
                    f"$signed(partial_c_p{p}_r{r}_c{c}[{lane}*16 +: 16])" for p in range(k_spatial)
                )
                row_mux_cases.append(f"            accum16 = {terms};")
                if has_bias:
                    # Bias depends only on n_group (ng), never on mg or k_pass.
                    # #RUTHWIK: with the Step-3 compute/drain overlap, this must
                    # read the DRAIN engine's latched cap_ng (the n_group of the
                    # group currently being drained), NOT the live compute
                    # engine's ng -- compute may already be several beats into
                    # the NEXT group by the time this group's rows drain. This
                    # duplicates the single-axis fold-N branches' out_grp
                    # freeze-on-drain pattern; unify once that generalizes.
                    _bias_e = f"{_bias_lane(bias_rom_name, f'cap_ng * {n} + {c * 8 + lane}')}"
                else:
                    _bias_e = "16'sd0"
                row_mux_cases.append(
                    f"            row_mux[{c}*{8*out_width} + {lane}*{out_width} +: {out_width}] = "
                    f"stage2(accum16, {_bias_e});"
                )
        row_mux_cases.append("        end")
    any_avail_expr = " | ".join(f"row_avail_{r}" for r in range(grid_rows))

    # Registered weight-ROM read address (VTR-safe): rom_addr is a plain
    # register whose only driver is the main FSM always block below and whose
    # only use is indexing w_rom (see the rom_block comment above) -- parmys
    # only infers a clocked single_port_ram when the address comes straight
    # from a register. It mirrors ``(chunk_idx*n_passes + ng)*n + beat_count``
    # exactly, but computed ONE STEP AHEAD at each FSM transition (using the
    # NEXT chunk_idx/ng/beat_count the transition is about to commit to), the
    # same "held value already equals this cycle's address" trick
    # ``_weight_rom_block`` uses -- so it lands on the same value the old
    # combinational expression had, cycle for cycle.
    emit_rom_addr = ws and emit_rom
    rom_addr_decl = "    reg [15:0] rom_addr;" if emit_rom_addr else ""
    rom_addr_reset = "            rom_addr <= 16'd0;" if emit_rom_addr else ""
    rom_addr_idle = "                    rom_addr <= 16'd0;" if emit_rom_addr else ""
    rom_addr_run = (
        f"                        if (beat_count + 16'd1 < 16'd{n}) "
        f"rom_addr <= (chunk_idx * 16'd{n_passes} + ng) * 16'd{n} + (beat_count + 16'd1);"
    ) if emit_rom_addr else ""
    rom_addr_wait = (
        f"                            rom_addr <= ((chunk_idx + 16'd1) * 16'd{n_passes} + ng) * 16'd{n};"
    ) if emit_rom_addr else ""
    rom_addr_output_next_group = (
        f"                        rom_addr <= (((frame_c + 16'd1) % 16'd{n_passes})) * 16'd{n};"
    ) if emit_rom_addr else ""
    rom_addr_output_done = "                        rom_addr <= 16'd0;" if emit_rom_addr else ""

    # #RUTHWIK: op0/op1 (shadow readout control) are now driven by the DRAIN
    # engine's own state (dstate/cur_row_tile), independent of the compute
    # engine's state -- this is the crux of the Step-3 overlap: op0/op1 read
    # out group g's shadow bank while cstate/chunk_idx/beat_count advance
    # through group g+1's live compute. Duplicates the single-axis emitters'
    # per-row op0/op1 shape; unify once that generalizes.
    op0_lines = "\n".join(
        f"    wire op0_{r} = !((dstate == D_OUTPUT) && (cur_row_tile == 16'd{r}));"
        for r in range(grid_rows)
    )
    op1_lines = "\n".join(
        (f"    wire op1_{r} = logical_output_complete && (cur_row_tile == 16'd{r});"
         if grid_rows > 1 else f"    wire op1_{r} = logical_output_complete;")
        for r in range(grid_rows)
    )

    row_mask_lines = []
    for r in range(grid_rows):
        row_mask_lines.append(
            f"    wire [15:0] row_remain_{r} = (group_logical_rows > 16'd{r*8}) ? "
            f"(group_logical_rows - 16'd{r*8}) : 16'd0;"
        )
        row_mask_lines.append(
            f"    wire [15:0] row_remain_{r}_c = (row_remain_{r} > 16'd8) ? 16'd8 : row_remain_{r};"
        )
        row_mask_lines.append(f"    wire [7:0] row_mask_{r} = 8'hFF >> (4'd8 - row_remain_{r}_c[3:0]);")
    col_mask_lines = []
    for c in range(grid_cols):
        col_mask_lines.append(
            f"    wire [15:0] col_remain_{c} = (group_logical_cols > 16'd{c*8}) ? "
            f"(group_logical_cols - 16'd{c*8}) : 16'd0;"
        )
        col_mask_lines.append(
            f"    wire [15:0] col_remain_{c}_c = (col_remain_{c} > 16'd8) ? 16'd8 : col_remain_{c};"
        )
        col_mask_lines.append(f"    wire [7:0] col_mask_{c} = 8'hFF >> (4'd8 - col_remain_{c}_c[3:0]);")

    return f"""\
// Auto-generated by rtl.py
// ADDITIVE combined-fold structural tensor-slice synth wrapper (2b)
// Per-group core: M={m}, K={k}, N={n}  |  Grid: {grid_rows}x{grid_cols} slices, K_SPATIAL={k_spatial}
// Groups: M_PASSES={m_passes} x N_PASSES={n_passes} (mg-major, ng-minor), LOGICAL_M={logical_m}, LOGICAL_N={logical_n}
// WARNING: K-spatial partial outputs are INT16; correctness requires every partition partial sum to fit INT16.
// Groups run STRICTLY SEQUENTIALLY (drain g before computing g+1) -- overlap is Step 3.
{part_comments}
`timescale 1ns/1ps

module {module_name}(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,
    input  wire [{a_width-1}:0]   a_rows,
{b_cols_port}    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);
{rom_block}
{bias_rom_block}
    localparam integer INPUT_BEATS = {input_beats};
    localparam integer K_CHUNKS = {k_chunks};
    localparam integer K_PASSES = {passes};
    localparam integer K_SPATIAL = {k_spatial};
    localparam integer M_PASSES = {m_passes};
    localparam integer N_PASSES = {n_passes};
    localparam integer TOTAL_FRAMES = {total_frames};
    localparam integer CORE_M = {m};
    localparam integer CORE_N = {n};
    localparam integer LOGICAL_M = {logical_m};
    localparam integer LOGICAL_N = {logical_n};
    // Step 3 (tensor-slice-group-overlap): TWO cooperating engines instead of
    // one sequential FSM. COMPUTE (cstate) feeds A/B, drives start_mat_mul and
    // the K passes, and advances the compute-group index (frame_c). DRAIN
    // (dstate) walks the shadow-bank readout (op[0]/op[1]) for a captured
    // group and advances an INDEPENDENT drain-group index. op[2] (capture_pulse,
    // below) hands a finished compute group to the drain engine and, in the
    // same cycle, lets the compute engine start the NEXT group -- so drain(g)
    // overlaps compute(g+1). The drain-group index trails the compute-group
    // index by at most one: compute stalls in C_HANDOFF if the previous
    // capture hasn't finished draining yet (no queue depth > 1). No inter-
    // group gap and no preload, per the op contract.
    localparam [2:0] C_IDLE=3'd0, C_RUN=3'd1, C_WAIT=3'd2, C_HANDOFF=3'd3, C_DONE=3'd4;
    localparam D_IDLE=1'd0, D_OUTPUT=1'd1;

    reg [2:0] cstate;
    reg       dstate;
    reg [15:0] beat_count;
    reg [15:0] chunk_idx;
    reg [15:0] frame_c;        // compute-group index
    reg [15:0] out_row_count;  // drain-side row cursor within the draining group
    reg        drain_active;   // 1 while dstate is draining a captured group (owned by drain engine)
    reg [15:0] cap_ng;         // #RUTHWIK: latched ng of the group currently draining
    reg [15:0] cap_rows;       // latched group_logical_rows of the group currently draining
    reg        cap_last;       // latched: this is the final logical group overall
    reg signed [15:0] accum16;
    reg [{c_width-1}:0] row_mux;
{rom_addr_decl}

    // mg/ng decode (contract: mg = frame // n_passes, ng = frame % n_passes)
    // for the COMPUTE engine -- stable for a whole group's compute, i.e. until
    // the C_HANDOFF -> C_RUN transition into the next group.
    wire [15:0] mg = frame_c / 16'd{n_passes};
    wire [15:0] ng = frame_c % 16'd{n_passes};
    // Ragged final-group logical extents (only the LAST m_group/n_group is
    // ragged; every other group is a full CORE_M x CORE_N tile).
    wire [15:0] group_logical_rows = (mg == 16'd{m_passes - 1}) ?
        (16'd{logical_m} - mg * 16'd{m}) : 16'd{m};
    wire [15:0] group_logical_cols = (ng == 16'd{n_passes - 1}) ?
        (16'd{logical_n} - ng * 16'd{n}) : 16'd{n};

{chr(10).join(row_mask_lines)}
{chr(10).join(col_mask_lines)}

    reg [{a_width-1}:0] a_rows_q;
    reg [{b_width-1}:0] b_cols_q;
    reg preload_valid_q;
    reg in_valid_q;

    always @(posedge clk) begin
        if (rst) begin
            a_rows_q        <= {a_width}'d0;
            b_cols_q        <= {b_width}'d0;
            preload_valid_q <= 1'b0;
            in_valid_q      <= 1'b0;
        end else if (en) begin
            a_rows_q        <= a_rows;
            b_cols_q        <= {b_cols_src};
            preload_valid_q <= preload_valid;
            in_valid_q      <= in_valid;
        end
    end

    wire slice_reset = rst;
    wire in_beat_active = (cstate == C_RUN) && in_valid_q && (beat_count < INPUT_BEATS);
    wire slice_start = in_beat_active && (beat_count == 16'd0);
    wire [{k_spatial * grid_rows * grid_cols - 1}:0] done_mat_mul;
    wire all_slices_done = &done_mat_mul;
    // Compute-group handoff: once cstate reaches C_HANDOFF (this group's
    // compute is fully done -- chunk_idx reached the last K pass), fire the
    // moment the drain engine has freed the previous capture (or never had
    // one) -- capture NOW (op2 pulse) and let compute proceed into the next
    // group's C_RUN this same edge.
    wire capture_pulse = (cstate == C_HANDOFF) && !drain_active;

    // op[1] drain_stop source: the logical row count of the group currently
    // DRAINING (cap_rows, not the live compute group's group_logical_rows)
    // has just been taken.
    wire logical_output_complete = (dstate == D_OUTPUT) && any_avail &&
                                   (out_row_count + 16'd1 == cap_rows);
    // op[2] shadow_swap: exactly the capture handoff pulse.
    wire op2 = capture_pulse;
    wire [15:0] cur_row_tile = out_row_count >> 3;
{op0_lines}
{op1_lines}

{chr(10).join(decls)}

{chr(10).join(partial_wires)}

{chr(10).join(insts)}

{chr(10).join(row_avail)}

    wire any_avail = {any_avail_expr};

{stage2_fn}
    always @(*) begin
        row_mux = {c_width}'d0;
        accum16 = 16'sd0;
{chr(10).join(row_mux_cases)}
    end

    // COMPUTE engine: feed + start_mat_mul + K passes, advancing frame_c.
    // Owns cstate, beat_count, chunk_idx, frame_c and rom_addr. Reads
    // drain_active only (never writes drain-engine regs).
    always @(posedge clk) begin
        if (rst) begin
            cstate <= C_IDLE;
            beat_count <= 16'd0;
            chunk_idx <= 16'd0;
            frame_c <= 16'd0;
            cap_ng <= 16'd0;
            cap_rows <= 16'd0;
            cap_last <= 1'b0;
{rom_addr_reset}
        end else if (en) begin
            case (cstate)
                C_IDLE: begin
                    beat_count <= 16'd0;
                    chunk_idx <= 16'd0;
                    frame_c <= 16'd0;
{rom_addr_idle}
                    if (preload_valid_q) begin
                        cstate <= C_RUN;
                    end
                end
                C_RUN: begin
                    if (in_beat_active) begin
                        beat_count <= beat_count + 16'd1;
{rom_addr_run}
                        if (beat_count + 16'd1 == INPUT_BEATS)
                            cstate <= C_WAIT;
                    end
                end
                C_WAIT: begin
                    if (all_slices_done) begin
                        if (chunk_idx + 16'd1 == 16'd{passes}) begin
                            cstate <= C_HANDOFF;
                        end else begin
                            chunk_idx <= chunk_idx + 16'd1;
                            beat_count <= 16'd0;
{rom_addr_wait}
                            cstate <= C_RUN;
                        end
                    end
                end
                C_HANDOFF: begin
                    // Stall here (accumulators held, undisturbed) until the
                    // drain engine has freed the previous capture -- at most
                    // one captured-but-undrained group in flight.
                    if (capture_pulse) begin
                        cap_ng <= ng;
                        cap_rows <= group_logical_rows;
                        cap_last <= (frame_c + 16'd1 == 16'd{total_frames});
                        if (frame_c + 16'd1 == 16'd{total_frames}) begin
                            cstate <= C_DONE;
{rom_addr_output_done}
                        end else begin
                            frame_c <= frame_c + 16'd1;
                            chunk_idx <= 16'd0;
                            beat_count <= 16'd0;
{rom_addr_output_next_group}
                            cstate <= C_RUN;
                        end
                    end
                end
                C_DONE: begin
                    // All compute issued; wait for the final drain to finish
                    // before the module is ready to accept a new preload.
                    if (dstate == D_IDLE && !drain_active)
                        cstate <= C_IDLE;
                end
                default: cstate <= C_IDLE;
            endcase
        end
    end

    // DRAIN engine: shadow readout via op[0]/op[1], advancing an independent
    // drain-group index (cap_ng/cap_rows/cap_last, latched by the compute
    // engine at each capture_pulse). Owns dstate, out_row_count,
    // drain_active and the module's c_row/out_valid/out_last outputs.
    // #RUTHWIK: this duplicates the single-axis emitters' per-row S_OUTPUT
    // drain shape (row_mux capture + out_row_count walk); unify once that
    // generalizes to a shared drain-engine helper.
    always @(posedge clk) begin
        if (rst) begin
            dstate <= D_IDLE;
            out_row_count <= 16'd0;
            drain_active <= 1'b0;
            c_row <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
        end else if (en) begin
            out_valid <= 1'b0;
            out_last <= 1'b0;
            case (dstate)
                D_IDLE: begin
                    out_row_count <= 16'd0;
                    if (capture_pulse) begin
                        drain_active <= 1'b1;
                        dstate <= D_OUTPUT;
                    end
                end
                D_OUTPUT: begin
                    if (any_avail) begin
                        c_row <= row_mux;
                        out_valid <= 1'b1;
                        out_last <= cap_last && (out_row_count + 16'd1 == cap_rows);
                        out_row_count <= out_row_count + 16'd1;
                        if (out_row_count + 16'd1 == cap_rows) begin
                            out_row_count <= 16'd0;
                            drain_active <= 1'b0;
                            dstate <= D_IDLE;
                        end
                    end
                end
                default: dstate <= D_IDLE;
            endcase
        end
    end

endmodule
"""


def generate_k_spatial_synth_verilog(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1, n_passes=1,
                                     s1=0, s2=0, out_width=16,
                                     weight_rom=None, emit_rom=True,
                                     bias_codes=None, emit_bias_rom=True, bias_rom_name="bias_rom",
                                     m_passes=1, logical_m=None, logical_n=None):
    """Public entry point: k_spatial>1 (K-in-space) branch of the unified
    ``_generate_general_synth_verilog`` -- k_spatial==1 delegates to the
    other branch (byte-identical to plain ``generate_synth_verilog``). Thin
    wrapper -- signature unchanged (m_passes/logical_m/logical_n are new,
    additive, default-1/None-preserving parameters for sub-phase 2b's
    combined-fold dispatch) so callers (and the golden generator) keep
    working verbatim.
    """
    return _generate_general_synth_verilog(
        m, k, n, module_name, k_spatial=k_spatial,
        weight_rom=weight_rom, emit_rom=emit_rom, n_passes=n_passes,
        s1=s1, s2=s2, out_width=out_width, bias_codes=bias_codes, emit_bias_rom=emit_bias_rom,
        bias_rom_name=bias_rom_name, m_passes=m_passes, logical_m=logical_m, logical_n=logical_n,
    )


def generate_k_spatial_sim_verilog(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1,
                                   s1=0, s2=0, out_width=16,
                                   weight_rom=None, emit_rom=True, n_passes=1,
                                   bias_codes=None, emit_bias_rom=True):
    sim = generate_sim_verilog(m, k, n, module_name, k_spatial=k_spatial,
                               s1=s1, s2=s2, out_width=out_width,
                               weight_rom=weight_rom, emit_rom=emit_rom, n_passes=n_passes,
                               bias_codes=bias_codes, emit_bias_rom=emit_bias_rom)
    if k_spatial == 1:
        return sim
    banner = (
        f"// Experimental K-spatial behavioral simulation model, K_SPATIAL={k_spatial}\n"
        "// WARNING: structural K-spatial partial outputs are INT16; simulation uses exact INT32 reference math.\n"
    )
    return banner + sim


def generate_k_spatial_combined_core_verilog(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1, out_bits=None,
                                            s1=0, s2=0, out_width=None, weight_rom=None, n_passes=1,
                                            bias_codes=None, a_zero_point=0, b_zero_point=0):
    if out_width is None:
        out_width = out_bits if out_bits is not None else 8
    if k_spatial == 1:
        return generate_combined_core_verilog(m, k, n, module_name, out_width=out_width,
                                              s1=s1, s2=s2,
                                              weight_rom=weight_rom, n_passes=n_passes,
                                              bias_codes=bias_codes,
                                              a_zero_point=a_zero_point, b_zero_point=b_zero_point)
    if a_zero_point or b_zero_point:
        raise NotImplementedError(
            f"{module_name}: k_spatial={k_spatial} structural K-spatial RTL "
            "does not implement the zero-point running-sum correction across "
            "partitions; only k_spatial==1 (chunked) supports a zero-pointed "
            "operand today."
        )
    _k_spatial_partitions(k, k_spatial)
    # Weight-stationary and/or bias: each branch emits its OWN ROM(s). Unlike
    # the chunked combined core -- which splits the two modules apart to hoist
    # a single shared ROM above the `ifndef -- the K-spatial core keeps sim
    # and synth as whole modules, so hoisting would mean the same module
    # surgery on an experimental structural body. Both ROMs are built from the
    # same weight_rom/bias_codes, so sim and synth stay identical by
    # construction (byte-for-byte the same declaration in each branch); only
    # one branch is ever compiled.
    sim_top = generate_k_spatial_sim_verilog(m, k, n, module_name, k_spatial,
                                            s1=s1, s2=s2, out_width=out_width,
                                            weight_rom=weight_rom, n_passes=n_passes,
                                            bias_codes=bias_codes)
    synth_top = generate_k_spatial_synth_verilog(m, k, n, module_name, k_spatial,
                                                s1=s1, s2=s2, out_width=out_width,
                                                weight_rom=weight_rom, n_passes=n_passes,
                                                bias_codes=bias_codes)

    lines = []
    lines.append("// Auto-generated by rtl.py")
    lines.append(f"// Combined K-spatial core: M={m}, K={k}, N={n}, K_SPATIAL={k_spatial}")
    lines.append("//   ifndef SYNTHESIS -> behavioral simulation model")
    lines.append("//   else             -> experimental structural K-spatial wrapper")
    lines.append("// WARNING: INT16 partial overflow is possible in K-spatial structural mode.")
    lines.append("")
    lines.append("`ifndef SYNTHESIS")
    lines.append("")
    lines.extend(sim_top.splitlines())
    lines.append("")
    lines.append("`else")
    lines.append("")
    lines.extend(synth_top.splitlines())
    lines.append("")
    lines.append("`endif")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Verilog grid wrapper for tensor_slice_int8 modules")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--name", type=str, default="gemm_grid_wrapper")
    parser.add_argument("--output", type=str, default="gemm_grid.v")
    parser.add_argument("--sim-output", type=str, default=None)
    parser.add_argument("--synth-output", type=str, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Emit Verilog $display debug traces")
    args = parser.parse_args()

    if args.sim_output or args.synth_output:
        sim_output = args.sim_output or args.output.replace(".v", "_sim.v")
        synth_output = args.synth_output or args.output.replace(".v", "_synth.v")
        Path(sim_output).write_text(generate_sim_verilog(args.m, args.k, args.n, args.name))
        Path(synth_output).write_text(generate_synth_verilog(args.m, args.k, args.n, args.name, debug=args.debug))
        print(f"Generated {sim_output} and {synth_output}  (M={args.m}, K={args.k}, N={args.n})")
    else:
        content = generate_synth_verilog(args.m, args.k, args.n, args.name, debug=args.debug)
        Path(args.output).write_text(content)
        print(f"Generated {args.output}  (M={args.m}, K={args.k}, N={args.n})")
