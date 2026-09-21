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
#     Verilog module PARAMETER on the tensor_slice_int8_atlas black box, not a port;
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


def _require_symmetric_quant(a_zero_point=0, b_zero_point=0,
                             a_zero_point_correct=None, b_zero_point_correct=None,
                             who="tensor_slice"):
    """Fail fast on any nonzero zero point.

    Scope decision: every quantizer is symmetric-style fixed-point, so the
    zero-point machinery (feed-side bit-7 flip + post-accumulation
    correction) has been removed from all emitters. A nonzero value here is
    a caller error and is never silently ignored. Unsigned operands narrower
    than 8 bits need no offset (bit 7 is never set) and remain supported.
    """
    bad = []
    for label, value in (("a_zero_point", a_zero_point),
                         ("b_zero_point", b_zero_point),
                         ("a_zero_point_correct", a_zero_point_correct),
                         ("b_zero_point_correct", b_zero_point_correct)):
        if value is not None and int(value) != 0:
            bad.append(f"{label}={value!r}")
    if bad:
        raise ValueError(
            f"{who}: symmetric-only quantization scope -- nonzero zero "
            f"point not supported ({', '.join(bad)}). Use symmetric "
            "(signed, or <8-bit unsigned) operands."
        )


def _weight_rom_block(b_width, weight_rom, n, input_beats, n_passes=1, passes=1, gapless=False):
    """Shared const-weight ROM: declaration + inline init + beat/addr counters + w_rom_out.

    Emitted once (hoisted above the single structural module in the combined
    core) so a single ROM feeds the wrapper. The ROM holds only
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

    ``gapless``: frames follow each other with no in_valid=0 beat between them
    (combined-fold feed), so the wrap on a frame's LAST beat does what the idle
    beat otherwise does -- rewind to pass 0 and step the N-group.
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
        kn_frame_end = f"""
                if (pass_ctr + 16'd1 >= 16'd{passes}) begin
                    // Frame's last beat: next beat is pass 0 of the next N-group.
                    pass_ctr <= 16'd0;
                    grp_was_feeding <= 1'b0;
                    if (grp_ctr + 16'd1 >= 16'd{n_passes}) begin
                        rom_addr <= 16'd0;
                        grp_ctr <= 16'd0;
                    end else begin
                        rom_addr <= (grp_ctr + 16'd1) * 16'd{n};
                        grp_ctr <= grp_ctr + 16'd1;
                    end
                end""" if gapless else ""
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
                grp_was_feeding <= 1'b1;{kn_frame_end}
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
    # Gapless: entries are laid out in feed order (pass-major, or group-major
    # when only N folds), so the wrap past the last entry is the rewind.
    rom_wrap = (f"(rom_addr + 16'd1 >= 16'd{nbeats}) ? 16'd0 : rom_addr + 16'd1"
                if gapless else "rom_addr + 16'd1")
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
                rom_addr <= {rom_wrap};{grp_feeding}
            end
        end
    end
    wire [{b_width - 1}:0] w_rom_out = (beat_ctr < 16'd{n}) ? w_rom[rom_addr] : {b_width}'d0;
"""
    return text


def _a_replay_block(a_width, passes, n_passes, m, input_beats, k=None, gapless=False):
    """A-row replay held in the core: port width, replay logic, a_rows_q source.

    The wrapper reads each A row off its stream once and presents every K-pass
    slice of it on the pass-0 beat (slice p at ``a_rows[p*a_width +: a_width]``);
    every beat that re-uses the row -- a later K pass, or any pass of a later
    N-group frame of the same M-group -- carries zeros on the port and is fed
    from here instead. Keeping the cache in RTL lets it map to a RAM block; as a
    C array in the HLS wrapper it becomes a register file plus an m:1 mux.

    Stored slices: passes 1.. when ``n_passes == 1`` (pass 0 always comes from
    the port), every pass when N folds (frames ng > 0 replay pass 0 too).
    Rows t >= m of a pass are feed padding and read back zero. A single-row
    cache (m == 1) is a plain register; anything deeper is a memory addressed
    PORT-side like w_rom/rom_addr -- the feed FSM's beat_count/chunk_idx lag
    the port by a cycle and must not be used.

    Only live K lanes are stored: with ``k`` given (narrow-word layouts, where
    pass p carries columns [p*a_width/8, ...) from bit 0), the lanes past K in
    the last pass are constant zero and are left out of the memory word -- on
    block-RAM fabrics a memory costs by its width, not its depth.

    ``gapless``: frames follow each other with no in_valid=0 beat between them
    (combined-fold feed), so the pass and N-group counters wrap at the frame's
    last beat instead of being cleared by an idle beat.
    """
    passes, n_passes, m = int(passes), int(n_passes), int(m)
    cols = a_width // 8
    live = [a_width if k is None else 8 * max(0, min(cols, int(k) - p_ * cols))
            for p_ in range(passes)]
    a_port_width = passes * a_width
    fold_n = n_passes > 1
    if passes < 2 and not fold_n:
        return a_port_width, "", "a_rows"
    lo = 0 if fold_n else 1
    # Memory word = the stored passes' live lanes packed back to back; a read
    # zero-extends the pass's lanes to the a_width word the core consumes.
    stored = [p_ for p_ in range(lo, passes) if live[p_] > 0]
    offs, replay_width = {}, 0
    for p_ in stored:
        offs[p_] = replay_width
        replay_width += live[p_]

    def _slice(p_):
        q = "replay_q[%d:%d]" % (offs[p_] + live[p_] - 1, offs[p_])
        pad = a_width - live[p_]
        return q if pad == 0 else "{%d'd0, %s}" % (pad, q)

    sel = _slice(stored[0])
    for p_ in stored[1:]:
        sel = "(replay_pass == 16'd%d) ? %s : %s" % (p_, _slice(p_), sel)
    store_data = "{" + ", ".join(
        "a_rows[%d:%d]" % (p_ * a_width + live[p_] - 1, p_ * a_width) for p_ in reversed(stored)) + "}"
    fresh = "(replay_pass == 16'd0) && (replay_grp == 16'd0)" if fold_n else "(replay_pass == 16'd0)"
    if fold_n:
        grp_decl = """
    // N-group of the frame on the port, stepped at real frame boundaries only
    // (same cadence as the weight ROM's grp_ctr).
    reg [15:0] replay_grp;
    reg replay_was_feeding;"""
        grp_reset = """
            replay_grp <= 16'd0;
            replay_was_feeding <= 1'b0;"""
        grp_idle = f"""
                if (replay_was_feeding)
                    replay_grp <= (replay_grp + 16'd1 >= 16'd{n_passes}) ? 16'd0 : replay_grp + 16'd1;
                replay_was_feeding <= 1'b0;"""
        grp_feeding = """
                replay_was_feeding <= 1'b1;"""
        grp_frame_end = f"""
                    replay_grp <= (replay_grp + 16'd1 >= 16'd{n_passes}) ? 16'd0 : replay_grp + 16'd1;
                    replay_was_feeding <= 1'b0;"""
    else:
        grp_decl = grp_reset = grp_idle = grp_feeding = grp_frame_end = ""
    if gapless:
        pass_wrap = f"""
                if (replay_pass + 16'd1 >= 16'd{passes}) begin
                    replay_pass <= 16'd0;{grp_frame_end}
                end else begin
                    replay_pass <= replay_pass + 16'd1;
                end"""
    else:
        pass_wrap = """
                replay_pass <= replay_pass + 16'd1;"""
    if m == 1:
        store_decl = f"    reg [{replay_width-1}:0] replay_q;"
        addr_decl = addr_reset = addr_hold = addr_step = ""
        store_write = f"replay_q <= {store_data};"
    else:
        store_decl = (f"    reg [{replay_width-1}:0] replay_mem [0:{m-1}];\n"
                      f"    wire [{replay_width-1}:0] replay_q = replay_mem[replay_addr];")
        addr_decl = """
    // replay_addr is a plain single-driver register used only as the memory
    // index so VTR's parmys infers a RAM block; it holds at the last row over
    // the padding beats.
    reg [15:0] replay_addr;"""
        addr_reset = """
            replay_addr <= 16'd0;"""
        addr_hold = """
                replay_addr <= 16'd0;"""
        addr_step = f"""
                if (replay_beat + 16'd1 < 16'd{m}) replay_addr <= replay_addr + 16'd1;"""
        store_write = f"replay_mem[replay_addr] <= {store_data};"
    # No padding beats when the pass is exactly m rows long (M >= N layers):
    # the row-valid compare and the zero mux drop out of the a_rows_q path.
    padded = int(input_beats) > m
    row_valid = f" && (replay_beat < 16'd{m})" if padded else ""
    replay_row = f"(replay_beat < 16'd{m}) ? ({sel}) : {a_width}'d0" if padded else sel
    block = f"""
    // A-row replay: the row on the port this cycle is replay_beat of K pass
    // replay_pass, both cleared by the frame's in_valid=0 preload beat. Fresh
    // rows are stored as they pass; the read lands in a_rows_q on the same edge
    // port data would, so latency is unchanged. Contents need no reset.
    reg [15:0] replay_beat;
    reg [15:0] replay_pass;{addr_decl}{grp_decl}
    always @(posedge clk) begin
        if (rst) begin
            replay_beat <= 16'd0;
            replay_pass <= 16'd0;{addr_reset}{grp_reset}
        end else if (en) begin
            if (!in_valid) begin
                replay_beat <= 16'd0;
                replay_pass <= 16'd0;{addr_hold}{grp_idle}
            end else if (replay_beat < 16'd{input_beats - 1}) begin
                replay_beat <= replay_beat + 16'd1;{addr_step}{grp_feeding}
            end else begin
                replay_beat <= 16'd0;{addr_hold}{grp_feeding}{pass_wrap}
            end
        end
    end
{store_decl}
    wire replay_fresh = {fresh};
    always @(posedge clk) begin
        if (en && in_valid && replay_fresh{row_valid})
            {store_write}
    end
    wire [{a_width-1}:0] replay_row = {replay_row};
"""
    return a_port_width, block, f"replay_fresh ? a_rows[{a_width-1}:0] : replay_row"


def _generate_general_synth_verilog(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1,
                                    feed_mode="chained", debug=False,
                                    weight_rom=None, emit_rom=True, n_passes=1,
                                    s1=0, s2=0, out_width=16, bias_codes=None, emit_bias_rom=True,
                                     bias_rom_name="bias_rom", a_zero_point=0, b_zero_point=0,
                                     a_zero_point_correct=None, b_zero_point_correct=None,
                                     m_passes=1, logical_m=None, logical_n=None,
                                     a_replay_n_passes=None):
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
    _require_symmetric_quant(a_zero_point, b_zero_point,
                             a_zero_point_correct, b_zero_point_correct,
                             who=f"{module_name} (_generate_general_synth_verilog)")
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
    # Symmetric-only quantization scope: the zero-point parameters are
    # fail-fast checked on entry; operands reach the core unflipped and no
    # running-sum correction exists.
    has_bias = bias_codes is not None
    bias_rom_block = _bias_rom_block(bias_codes, bias_rom_name) if (has_bias and emit_bias_rom) else ""
    _fold_n_bias = bool(has_bias and n_passes and int(n_passes) > 1)
    # Multi-frame overlap: the emitted frame is always the OLDEST un-emitted
    # frame, so its fold-N bias group is tracked by a pop-advanced counter
    # (out_grp) instead of the old "latch the live fed-group at frame start"
    # scheme, which a pipelined feed would clobber (see _fold_n_bias_group_decl
    # docstring: the live counter tracks the group being FED).
    _bias_grp_decl, _bias_grp_body = "", ""
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8

    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 8 * out_width
    # One pass per K chunk (k_spatial == 1 on this branch); pad short shapes
    # so consecutive pass starts stay >= 8 cycles apart.
    input_beats = _geometry.feed_beats(m, n, (k + 7) // 8)
    # The combined core calls this emitter with n_passes left at 1 (its ROMs
    # are hoisted), so the A replay's N-group count arrives separately.
    a_port_width, a_replay_block, a_rows_src = _a_replay_block(
        a_width, (k + 7) // 8,
        n_passes if a_replay_n_passes is None else a_replay_n_passes, m, input_beats,
        k=(k if grid_rows == 1 else None))
    # The slice emits 8 physical rows per tile.  The wrapper retires after M
    # logical rows and uses op[1] to truncate the masked remainder.
    logical_output_rows = m
    physical_output_rows = grid_rows * 8
    # op[1] terminates a masked/padded tail; 8-aligned M has none, and a
    # gratuitous op1 can suppress the next overlapped frame (see the
    # logical_output_complete declaration).
    _op1_tail_cond = "" if (logical_output_rows % 8) else " && 1'b0"
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
        // S1 = {s1}: in-slice stage-1 round-half-up shift, driven into the slice's
        // shift_amount pin (= S1 value; that pin
        // carries no other function -- the VTR hard-block model has no parameters)
        (* black_box = "true" *) (* keep = "true" *) tensor_slice_int8_atlas slice_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(1'b0),
            .en(en),
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
            .op({{op2, op1_{r}, op0_{r}}}),
            .shift_amount(4'd{s1}),
            .final_mat_mul_size(8'd0),
            .a_loc(5'd{r}),
            .b_loc(5'd{c})
        );""")

    # ── Zero-storage output collector (op[0]/op[1]/op[2] contract) ───────────
    # tensor_slice_int8_atlas contract (jojo-track/open/tensor-slice-op-shadow-drain):
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
    #     this job (it is tied low on every slice).
    # Legacy free-run behaviour is op tied to 3'b000. All tiles finish
    # together, so column tiles of one tile-row concatenate as pure wiring;
    # the wrapper holds every tile-row and releases them one at a time, in
    # row-major order, for their 8-row bursts. No parking storage and no
    # delay pyramid (the old alignment shift-lines cost sum-of-delays x 129
    # FFs — ~6.2k on a 2x2 grid, ~29k on 4x2).
    op0_decl = []
    op1_decl = []
    for r in range(grid_rows):
        # Per-row-tile release: a row tile may only be drained once ITS OWN
        # slices have completed the head frame. Waiting for every row tile
        # would delay row 0's take past the next overlapped frame's captures
        # on the early slices (loc grows by 8 per tile row/col).
        op0_decl.append(
            f"    wire row_emit_{r} = row_done[{r}] && (frames_fed > frames_emitted);")
        release = (f"row_emit_{r} && (cur_row_tile == 16'd{r})"
                   if grid_rows > 1 else f"row_emit_{r}")
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
    # Symmetric-only quantization scope: no zero-point running-sum
    # correction exists. All correction fragments below render empty; the
    # stage-2 call sites keep their shape (bias term only).
    _zp_col_decl = ""
    _zp_col_reset = ""
    _zp_col_acc = ""
    _zp_row_decl = ""
    _zp_row_reset = ""
    _zp_row_acc = ""
    row_zp_corr_decl = ""
    row_zp_corr_term = ""

    def _col_zp_corr_term(c, lane):
        return ""

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
        if (!slice_reset && (emit_active || row_take || |done_mat_mul)) begin
            $display("DBG t=%0t st=%0d beat=%0d done=%b take=%0b out_rows=%0d fed=%0d emit=%0d head=%0b rav=%b cav=%b op0=%b op1=%b trunc=%b",
                     $time, state, beat_count, done_mat_mul, row_take, out_row_count,
                     frames_fed, frames_emitted, head_done,
                     row_avail_0, c_avail_0_0, op0_0, op1_0, slice_r0_c0.trunc);
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
    input  wire [{a_port_width-1}:0]   a_rows,
{b_cols_port}    input  wire                   preload_valid,
    input  wire                   in_valid,
    output reg  [{c_width-1}:0]   c_row,
    output reg                    out_valid,
    output reg                    out_last
);
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
    reg [{c_width-1}:0] row_mux;
    reg row_take;
{_bias_grp_decl}
{"    reg [15:0] out_grp;" if _fold_n_bias else ""}
{_zp_col_decl}{_zp_row_decl}{row_zp_corr_decl}
    // Continuous multi-frame bookkeeping (rev-3 overlapped frames).
    // frames_fed counts preload pulses seen; frames_emitted counts frames
    // whose LOGICAL_OUT_ROWS rows have been taken. done_count[s] counts the
    // committed-wave done pulses per slice; the head (oldest un-emitted)
    // frame's wave is complete when every slice has counted past
    // frames_emitted, so emission never waits on the feed FSM.
    reg [15:0] frames_fed;
    reg [15:0] frames_emitted;
    reg [15:0] done_count [0:{grid_rows*grid_cols-1}];
    reg        row_done [0:{grid_rows-1}];
    reg        any_row_done;
    integer    hd, hc;
    always @* begin
        any_row_done = 1'b0;
        for (hd = 0; hd < {grid_rows}; hd = hd + 1) begin
            row_done[hd] = 1'b1;
            for (hc = 0; hc < {grid_cols}; hc = hc + 1)
                if (done_count[hd*{grid_cols} + hc] <= frames_emitted)
                    row_done[hd] = 1'b0;
            if (row_done[hd]) any_row_done = 1'b1;
        end
    end

    // ── Input pipeline stage ────────────────────────────────────────────────
    // Register the whole input bundle (en-gated, so the core stays self-timed),
    // keeping the beat decode and data gating muxes off the path into the
    // tensor_slice input pins.
{a_replay_block}
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
            // Symmetric-only quantization scope: operands reach the slices
            // unflipped (signed codes straight through; padding K lanes are
            // raw 0 on the wire).
            a_rows_q        <= {a_rows_src};
            b_cols_q        <= {b_cols_q_src};
            preload_valid_q <= preload_valid;
            in_valid_q      <= in_valid;
        end
    end

    wire slice_reset = rst;
    wire in_beat_active = ((state == S_PRELOAD) || (state == S_RUN)) && in_valid_q && (beat_count < INPUT_BEATS);
    wire slice_start = in_beat_active && (beat_count == 16'd0);
    wire final_chunk = (chunk_idx == K_CHUNKS - 1);
    wire [7:0] current_k_mask = final_chunk ? {vm(last_k_mask)} : 8'hFF;

    wire [{grid_rows*grid_cols-1}:0] done_mat_mul;
    // Rev-3: done_mat_mul is a per-committed-wave 1-cycle pulse per slice.
    // done_count[s] tracks them; row_done[r] says every slice of tile-row r
    // has counted a done past frames_emitted, so each tile-row is released
    // as soon as ITS wave is complete (not when all rows are). Emission is
    // independent of the feed FSM, so frame t+1 feeds while frame t drains.
    wire emit_active = any_row_done && (frames_fed > frames_emitted);
    wire [15:0] cur_row_tile = out_row_count >> 3;
    // op[1] drain_stop source: the logical M'th row of the CURRENT tile-row's
    // burst has just been taken -- the rest of that physical 8-row burst is
    // padding tail, terminated by op1 (see op1_decl below), not by pe_reset.
    // Gate on a real tail: for 8-aligned M there is nothing to truncate, and
    // an unnecessary op1 at a frame end can tie with the NEXT frame's row-0
    // capture (cap[7] re-arm) and would then suppress that frame's whole
    // drain under overlap.
    wire logical_output_complete = emit_active && row_take &&
                                   (out_row_count + 16'd1 == LOGICAL_OUT_ROWS){_op1_tail_cond};
    // pe_reset is tied low on every slice: the slice masks it on any start
    // edge and cells self-clear at capture, so `rst` is the only recovery
    // path. final_mat_mul_size is unread by the slice and tied to 0.
    // op[2] commit tag (rev A2): asserted at the committed (final) pass's
    // start edge; the slice latches it into its wave slot and captures only
    // for committed waves. Intermediate K passes see op2 == 0.
    wire op2 = slice_start && final_chunk;

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
            frames_fed <= 16'd0;
            frames_emitted <= 16'd0;
            for (hd = 0; hd < {grid_rows*grid_cols}; hd = hd + 1)
                done_count[hd] <= 16'd0;
            c_row <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
{"            out_grp <= 16'd0;" if _fold_n_bias else ""}
        end else if (en) begin
            out_valid <= 1'b0;
            out_last <= 1'b0;
            for (hd = 0; hd < {grid_rows*grid_cols}; hd = hd + 1)
                if (done_mat_mul[hd])
                    done_count[hd] <= done_count[hd] + 16'd1;

            case (state)
                S_IDLE: begin
                    beat_count <= 16'd0;
                    chunk_idx <= 16'd0;
                    if (preload_valid_q) begin
                        frames_fed <= frames_fed + 16'd1;
{_zp_col_reset}{_zp_row_reset}                        state <= S_PRELOAD;
                    end
                end

                S_PRELOAD: begin
                    // Feed the first data beat HERE: the input register already
                    // holds beat 0 during this cycle, and consuming only in
                    // S_RUN would advance the register once more (dropping beat
                    // 0). Launching in S_PRELOAD aligns the slice window for
                    // both supported drive cadences (preload with in_valid held
                    // off, or preload immediately followed by data).
                    if (in_beat_active) begin
                        beat_count <= beat_count + 16'd1;
                        if (beat_count + 16'd1 == INPUT_BEATS) begin
                            if (final_chunk) begin
                                state <= S_IDLE;
                            end else begin
                                chunk_idx <= chunk_idx + 16'd1;
                                beat_count <= 16'd0;
                                state <= S_RUN;
                            end
                        end else begin
                            state <= S_RUN;
                        end
{_zp_col_acc}{_zp_row_acc}                    end else begin
                        state <= S_RUN;
                    end
                end

                S_RUN: begin
                    if (in_beat_active) begin
                        if (beat_count + 16'd1 == INPUT_BEATS) begin
                            if (final_chunk) begin
                                state <= S_IDLE;
                            end else begin
                                // Chunks are fed back-to-back by the driver
                                // (total_beats = k_chunks * input_beats with no
                                // gaps), so advance to the next chunk IN PLACE:
                                // the next cycle's beat_count==0 re-pulses
                                // slice_start for the new chunk.
                                chunk_idx <= chunk_idx + 16'd1;
                                beat_count <= 16'd0;
                            end
                        end else begin
                            beat_count <= beat_count + 16'd1;
                        end
{_zp_col_acc}{_zp_row_acc}                    end
                end
            endcase

            // Emission runs independently of the feed FSM: the head frame's
            // rows are taken one per edge as the slices present them. Feed
            // and drain of different frames therefore overlap, while the
            // module keeps rows in frame order.
            if (emit_active && row_take) begin
                c_row <= row_mux;
                out_valid <= 1'b1;
                out_last <= (out_row_count + 16'd1 == LOGICAL_OUT_ROWS);
                out_row_count <= out_row_count + 16'd1;
                if (out_row_count + 16'd1 == LOGICAL_OUT_ROWS) begin
                    out_row_count <= 16'd0;
                    frames_emitted <= frames_emitted + 16'd1;
{(f"                    if (out_grp + 16'd1 >= 16'd{n_passes}) out_grp <= 16'd0;") if _fold_n_bias else ""}
{"                    else out_grp <= out_grp + 16'd1;" if _fold_n_bias else ""}
                end
            end
        end
    end

endmodule
"""


def generate_synth_verilog(m, k, n, module_name="gemm_grid_wrapper", feed_mode="chained", debug=False,
                           weight_rom=None, emit_rom=True, n_passes=1,
                           s1=0, s2=0, out_width=16, bias_codes=None, emit_bias_rom=True,
                           bias_rom_name="bias_rom", a_zero_point=0, b_zero_point=0,
                           a_zero_point_correct=None, b_zero_point_correct=None,
                           m_passes=1, logical_m=None, logical_n=None, a_replay_n_passes=None):
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
        a_replay_n_passes=a_replay_n_passes,
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
    """Generate a single structural {module_name}.v (single-branch RTL).

    Structural synth wrapper with tensor_slice_int8_atlas black-box slices. Used
    both for local simulation (iverilog) and by Catapult HLS / SCVerify /
    downstream synthesis (Design Compiler) -- one source of truth, no
    behavioral/synth branch split.

    ``bias_codes`` (or None -- the add folds away, decision 4) is baked as
    ONE compile-time bias ROM hoisted above the single module, exactly like
    the weight ROM.

    ``out_bits`` is accepted as a legacy alias for ``out_width`` (some callers
    still pass it); ``out_width`` wins when both are given. Defaults to 8
    (today's legacy int8-lane default) when neither is given.

    Symmetric-only quantization scope: nonzero ``a_zero_point``/
    ``b_zero_point`` (or ``*_correct``) raise here via
    ``_require_symmetric_quant``.
    """
    _require_symmetric_quant(a_zero_point, b_zero_point,
                             a_zero_point_correct, b_zero_point_correct,
                             who=f"{module_name} (generate_combined_core_verilog)")
    if out_width is None:
        out_width = out_bits if out_bits is not None else 8
    has_bias = bias_codes is not None
    # Weight-stationary and/or a real bias: hoist shared ROM(s) above the
    # single module (one source of truth), which reads them directly.
    if weight_rom is not None or has_bias:
        synth_top = generate_synth_verilog(m, k, n, module_name, weight_rom=weight_rom, emit_rom=False,
                                           s1=s1, s2=s2, out_width=out_width,
                                           bias_codes=bias_codes, emit_bias_rom=False,
                                           a_zero_point=a_zero_point, b_zero_point=b_zero_point,
                                           a_zero_point_correct=a_zero_point_correct,
                                           b_zero_point_correct=b_zero_point_correct,
                                           a_replay_n_passes=n_passes)
        b_width = ((n + 7) // 8) * 64
        input_beats = _geometry.feed_beats(m, n, (k + 7) // 8)
        header, syn_body = _split_module(synth_top, module_name)   # header incl. 'module..);'
        rom_block = _weight_rom_block(b_width, weight_rom, n, input_beats, n_passes=n_passes) \
            if weight_rom is not None else ""
        bias_rom = _bias_rom_block(bias_codes) if has_bias else ""
        tag = []
        if weight_rom is not None:
            tag.append("shared const-weight ROM")
        if has_bias:
            tag.append("bias ROM")
        out = (
            f"// Auto-generated by rtl.py\n"
            f"// Combined core: M={m}, K={k}, N={n}\n"
            f"//   {' + '.join(tag)} hoisted above the single structural module\n"
            f"{header}\n{rom_block}{bias_rom}\n"
            f"{syn_body}"
            f"endmodule\n"
        )
        return out

    synth_top = generate_synth_verilog(m, k, n, module_name, s1=s1, s2=s2, out_width=out_width,
                                       a_zero_point=a_zero_point, b_zero_point=b_zero_point,
                                       a_zero_point_correct=a_zero_point_correct,
                                       b_zero_point_correct=b_zero_point_correct,
                                       a_replay_n_passes=n_passes)

    lines = []
    lines.append("// Auto-generated by rtl.py")
    lines.append(f"// Combined core: M={m}, K={k}, N={n}")
    lines.append("//   structural synth wrapper (single-branch RTL: simulation + HLS synthesis)")
    lines.append("")
    for l in synth_top.splitlines():
        lines.append(l)
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
    # Multi-frame overlap: emit-side group tracked by the pop-advanced out_grp
    # (see the chunked emitter's note) instead of the fed-group live counter.
    _bias_grp_decl, _bias_grp_body = "", ""
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
    input_beats = _geometry.feed_beats(m, n, passes)
    a_port_width, a_replay_block, a_rows_src = _a_replay_block(
        a_width, passes, n_passes, m, input_beats, k=k)
    # K-spatial uses the same logical-row retirement contract as the chunked
    # wrapper; physical 8-row bursts are aborted after logical M.
    logical_output_rows = m
    physical_output_rows = grid_rows * 8
    # op[1] only for a real masked tail (see the chunked emitter).
    _ksp_tail_cond = "" if (logical_output_rows % 8) else " && 1'b0"

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
        # carries chunk p (there is only one pass), so every k_mask is
        # a Python-computed literal, not a runtime expression.
        partitions = _k_spatial_partitions(k, k_spatial)
        part_comments = "\n".join(
            f"// K_SPATIAL_PARTITION {p}: chunks {lo}..{hi}" for p, (lo, hi) in enumerate(partitions)
        )
        for p, (lo, hi) in enumerate(partitions):
            decls.append(f"    // Spatial grid {p}: contiguous K chunks {lo}..{hi}")
            decls.append(f"    wire part{p}_active = 1'b1;")
            decls.append(f"    wire part{p}_last_chunk = 1'b1;")
            part_k_mask = tail_mask_hex(k, hi) if hi == k_chunks - 1 else 0xFF
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
        // S1 = {s1}: in-slice stage-1 round-half-up shift, driven into the slice's
        // shift_amount pin (= S1 value; that pin
        // carries no other function -- the VTR hard-block model has no parameters)
        (* black_box = "true" *) (* keep = "true" *) tensor_slice_int8_atlas slice_p{p}_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(1'b0),
            .en(en),
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
            .op({{op2, op1_{r}, op0_{r}}}),
            .shift_amount(4'd{s1}),
            .final_mat_mul_size(8'd0),
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
        tail_k_mask = tail_mask_hex(k, tail_chunk)
        part_comments = "\n".join(
            f"// K_SPATIAL_PARTITION {p}: chunk = pass*{k_spatial} + {p}" for p in range(k_spatial)
        )
        for p in range(k_spatial):
            decls.append(f"    // Spatial grid {p}: chunk = pass*{k_spatial} + {p}")
            decls.append(f"    wire part{p}_active = 1'b1;")
            decls.append(f"    wire [15:0] part{p}_chunk = chunk_idx * 16'd{k_spatial} + 16'd{p};")
            decls.append(f"    wire part{p}_chunk_pad = (part{p}_chunk >= K_CHUNKS);")
            decls.append(f"    wire part{p}_chunk_tail = (part{p}_chunk == K_CHUNKS - 16'd1);")
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
        // S1 = {s1}: in-slice stage-1 round-half-up shift, driven into the slice's
        // shift_amount pin (= S1 value; that pin
        // carries no other function -- the VTR hard-block model has no parameters)
        (* black_box = "true" *) (* keep = "true" *) tensor_slice_int8_atlas slice_p{p}_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(1'b0),
            .en(en),
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
            .op({{op2, op1_{r}, op0_{r}}}),
            .shift_amount(4'd{s1}),
            .final_mat_mul_size(8'd0),
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
        f"    wire row_emit_{r} = row_done[{r}] && (frames_fed > frames_emitted);\n"
        f"    wire op0_{r} = !(row_emit_{r} && (cur_row_tile == 16'd{r}));"
        for r in range(grid_rows)
    )
    op1_lines = "\n".join(
        (f"    wire op1_{r} = logical_output_complete && (cur_row_tile == 16'd{r});"
         if grid_rows > 1 else f"    wire op1_{r} = logical_output_complete;")
        for r in range(grid_rows)
    )

    if full_k_spatial:
        # Single pass: end of window -> feed FSM returns to S_IDLE while the
        # independent emission block drains the committed wave.
        run_end_body = """\
                            state <= S_IDLE;"""
        wait_body = ""
        op2_expr = "slice_start"
    else:
        # chunk_idx doubles as the pass counter here: it advances once per
        # pass (not once per chunk), and the loop exits after `passes` passes.
        # Non-final passes advance IN PLACE at the end of their beat window:
        # the host run loop (docs/wrapper_run_loop.md) feeds every pass's
        # beats contiguously -- any non-in_valid call ends the frame -- so
        # waiting for all_slices_done between passes would strand the later
        # passes' beats (the iverilog golden TB feeds contiguously too).
        # The feed FSM never waits for the grid; the emission block drains the
        # committed (final-pass) wave.
        run_end_body = f"""\
                        if (chunk_idx + 16'd1 == 16'd{passes}) begin
                            state <= S_IDLE;
                        end else begin
                            chunk_idx <= chunk_idx + 16'd1;
                            beat_count <= 16'd0;
                        end"""
        wait_body = ""
        op2_expr = f"slice_start && (chunk_idx + 16'd1 == 16'd{passes})"

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
    input  wire [{a_port_width-1}:0]   a_rows,
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
{_bias_grp_decl}
{"    reg [15:0] out_grp;" if _fold_n_bias else ""}
    // Continuous multi-frame bookkeeping (rev-3 overlapped frames); same
    // contract as the chunked emitter: frames_fed counts preload pulses,
    // frames_emitted counts drained frames, and the head (oldest un-emitted)
    // frame's wave is complete when every slice has counted a done past
    // frames_emitted.
    reg [15:0] frames_fed;
    reg [15:0] frames_emitted;
    reg [15:0] done_count [0:{k_spatial * grid_rows * grid_cols - 1}];
    reg        row_done [0:{grid_rows-1}];
    reg        any_row_done;
    integer    hp, hd, hc;
    always @* begin
        any_row_done = 1'b0;
        for (hd = 0; hd < {grid_rows}; hd = hd + 1) begin
            row_done[hd] = 1'b1;
            for (hp = 0; hp < {k_spatial}; hp = hp + 1)
                for (hc = 0; hc < {grid_cols}; hc = hc + 1)
                    if (done_count[hp*{grid_rows*grid_cols} + hd*{grid_cols} + hc] <= frames_emitted)
                        row_done[hd] = 1'b0;
            if (row_done[hd]) any_row_done = 1'b1;
        end
    end

    // Input pipeline stage (same rationale as the chunked emitter): register
    // the whole bundle en-gated so the beat decode + partition gating muxes
    // start from local registers instead of chaining from the Catapult
    // wrapper into the tensor_slice input pins.
{a_replay_block}
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
            a_rows_q        <= {a_rows_src};
            b_cols_q        <= {ksp_b_cols_src};
            preload_valid_q <= preload_valid;
            in_valid_q      <= in_valid;
        end
    end

    wire slice_reset = rst;
    wire in_beat_active = (state == S_RUN) && in_valid_q && (beat_count < INPUT_BEATS);
    wire slice_start = in_beat_active && (beat_count == 16'd0);
    wire [{k_spatial * grid_rows * grid_cols - 1}:0] done_mat_mul;
    // Rev-3: done_mat_mul is a per-committed-wave 1-cycle pulse per slice.
    // done_count[s] tracks them; row_done[r] releases each tile-row as soon
    // as its own slices complete the head frame (not when all rows finish),
    // so emission is independent of the feed FSM and frames overlap.
    wire emit_active = any_row_done && (frames_fed > frames_emitted);

{chr(10).join(partial_wires)}

{chr(10).join(row_avail)}

    wire any_avail = {any_avail_expr};

    // op[1] drain_stop source: the logical M'th row of the current tile-row's
    // burst has just been taken -- the remaining physical rows are padding
    // tail, terminated by op1 (op1_lines below), not by pe_reset. Gate on a
    // real tail; see the chunked emitter (a gratuitous op1 can tie with the
    // next overlapped frame's cap[7] re-arm and suppress its whole drain).
    wire logical_output_complete = emit_active && any_avail &&
                                   (out_row_count + 16'd1 == LOGICAL_OUT_ROWS){_ksp_tail_cond};
    // op[2] commit tag (rev A2): asserted at the committed (final) pass's
    // start edge; the slices latch it into their wave slot and capture only
    // for committed waves. Intermediate K passes see op2 == 0.
    wire op2 = {op2_expr};

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

{chr(10).join(insts)}

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
            frames_fed <= 16'd0;
            frames_emitted <= 16'd0;
            for (hd = 0; hd < {k_spatial * grid_rows * grid_cols}; hd = hd + 1)
                done_count[hd] <= 16'd0;
{"            out_grp <= 16'd0;" if _fold_n_bias else ""}
        end else if (en) begin
            out_valid <= 1'b0;
            out_last <= 1'b0;
            for (hd = 0; hd < {k_spatial * grid_rows * grid_cols}; hd = hd + 1)
                if (done_mat_mul[hd])
                    done_count[hd] <= done_count[hd] + 16'd1;
{_bias_grp_body}
            case (state)
                S_IDLE: begin
                    beat_count <= 16'd0;
                    chunk_idx <= 16'd0;
                    if (preload_valid_q) begin
                        frames_fed <= frames_fed + 16'd1;
                        state <= S_RUN;
                    end
                end
                S_RUN: begin
                    if (in_beat_active) begin
                        beat_count <= beat_count + 16'd1;
                        if (beat_count + 16'd1 == INPUT_BEATS)
{run_end_body}
                    end
                end
{wait_body}
            endcase

            // Emission runs independently of the feed FSM: the head frame
            // drains while later frames feed.
            if (emit_active && any_avail) begin
                c_row <= row_mux;
                out_valid <= 1'b1;
                out_last <= (out_row_count + 16'd1 == LOGICAL_OUT_ROWS);
                out_row_count <= out_row_count + 16'd1;
                if (out_row_count + 16'd1 == LOGICAL_OUT_ROWS) begin
                    out_row_count <= 16'd0;
                    frames_emitted <= frames_emitted + 16'd1;
{(f"                    if (out_grp + 16'd1 >= 16'd{n_passes}) out_grp <= 16'd0;") if _fold_n_bias else ""}
{"                    else out_grp <= out_grp + 16'd1;" if _fold_n_bias else ""}
                end
            end
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
    ``weight_rom`` (weight-stationary).
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
            "fits, or ask the tensor_slice_int8_atlas owner to widen a_loc/b_loc."
        )

    has_bias = bias_codes is not None
    bias_rom_block = _bias_rom_block(bias_codes, bias_rom_name) if (has_bias and emit_bias_rom) else ""
    stage2_fn = _stage2_function(s2, out_width)

    k_chunks = (k + 7) // 8
    if k_spatial < 1 or k_spatial > k_chunks:
        raise ValueError(f"k_spatial={k_spatial} must be in [1, K_CHUNKS={k_chunks}]")
    passes = -(-k_chunks // k_spatial)
    tail_chunk = k_chunks - 1
    tail_k_mask = tail_mask_hex(k, tail_chunk)

    total_frames = int(m_passes) * int(n_passes)

    # weight-stationary: same contract as every other emitter -- no external
    # b_cols port when a ROM is supplied.
    ws = weight_rom is not None
    a_width = 64 * k_spatial
    b_width = 64 * k_spatial
    c_width = grid_cols * 8 * out_width
    input_beats = _geometry.feed_beats(m, n, passes)

    a_port_width, a_replay_block, a_rows_src = _a_replay_block(
        a_width, passes, n_passes, m, input_beats, k=k, gapless=True)

    if ws:
        b_cols_port = ""
        b_cols_src = "w_rom_out"
        # Reuse the shared const-weight ROM block: its passes>1 & n_passes>1
        # branch already implements the combined chunk-major/ng-minor layout
        # (addr = (k_pass*n_passes + ng)*n + beat) with the registered address
        # VTR's parmys needs -- driven by in_valid/beat_ctr/pass_ctr/grp_ctr,
        # exactly like the proven single-axis weight-stationary path. This
        # replaces the old hand-rolled rom_addr that had to be kept in lockstep
        # with the FSM (and was off by a cycle under the overlapped feed).
        rom_beats = int(passes) * int(n_passes) * n
        if weight_rom is not None and len(weight_rom) != rom_beats:
            raise ValueError(
                f"{module_name}: combined-fold weight_rom has {len(weight_rom)} beats, "
                f"expected passes*n_passes*n = {rom_beats}"
            )
        rom_block = (
            _weight_rom_block(b_width, weight_rom, n, input_beats,
                              n_passes=n_passes, passes=passes, gapless=True)
            if emit_rom else ""
        )
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
        decls.append(f"    wire [15:0] part{p}_chunk = chunk_idx * 16'd{k_spatial} + 16'd{p};")
        decls.append(f"    wire part{p}_chunk_pad = (part{p}_chunk >= K_CHUNKS);")
        decls.append(f"    wire part{p}_chunk_tail = (part{p}_chunk == K_CHUNKS - 16'd1);")
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
        // S1 = {s1}: in-slice stage-1 round-half-up shift, driven into the slice's
        // shift_amount pin (= S1 value)
        (* black_box = "true" *) (* keep = "true" *) tensor_slice_int8_atlas slice_p{p}_r{r}_c{c} (
            .clk(clk), .reset(slice_reset), .pe_reset(1'b0),
            .en(en),
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
            .op({{op2, op1_{r}, op0_{r}}}),
            .shift_amount(4'd{s1}),
            .final_mat_mul_size(8'd0),
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
        _row_cond = (f"(cur_row_tile == 16'd{r}) && row_avail_{r}"
                     if grid_rows > 1 else f"row_avail_{r}")
        row_mux_cases.append(f"        if ({_row_cond}) begin")
        for c in range(grid_cols):
            for lane in range(8):
                terms = " + ".join(
                    f"$signed(partial_c_p{p}_r{r}_c{c}[{lane}*16 +: 16])" for p in range(k_spatial)
                )
                row_mux_cases.append(f"            accum16 = {terms};")
                if has_bias:
                    # Bias depends only on n_group (ng), never on mg or k_pass.
                    # Read the DRAIN engine's own n_group (drain_ng, from the
                    # independent drain_frame counter) -- the feed engine may
                    # already be several groups ahead by the time this group's
                    # rows drain.
                    _bias_e = f"{_bias_lane(bias_rom_name, f'drain_ng * {n} + {c * 8 + lane}')}"
                else:
                    _bias_e = "16'sd0"
                row_mux_cases.append(
                    f"            row_mux[{c}*{8*out_width} + {lane}*{out_width} +: {out_width}] = "
                    f"stage2(accum16, {_bias_e});"
                )
        row_mux_cases.append("            row_take = 1'b1;")
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
    emit_rom_addr = False  # ROM addressing now lives inside _weight_rom_block
    rom_addr_decl = ""
    rom_addr_reset = ""
    rom_addr_idle = ""
    rom_addr_run = ""
    rom_addr_wait = ""
    rom_addr_final = ""

    # op0/op1 (shadow readout control) are driven by the independent emission
    # engine (row_done/frames_emitted/cur_row_tile), never by the feed FSM --
    # the crux of the overlapped-frame model: op0/op1 read out group g's shadow
    # bank while feed advances through group g+1's live compute. Duplicates the
    # single-axis emitters' per-row op0/op1 shape; unify once that generalizes.
    op0_lines = "\n".join(
        (f"    wire op0_{r} = !(row_done[{r}] && (frames_fed > frames_emitted) && (cur_row_tile == 16'd{r}));"
         if grid_rows > 1 else
         f"    wire op0_{r} = !(row_done[{r}] && (frames_fed > frames_emitted));")
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
// Overlapped gapless multi-frame feed: each group's beats are consumed
// without waiting on compute, and the head frame's rows drain independently.
{part_comments}
`timescale 1ns/1ps

module {module_name}(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   en,
    input  wire [{a_port_width-1}:0]   a_rows,
{b_cols_port}    input  wire                   preload_valid,   // unused: this core has no preload stage
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
    // Overlapped multi-frame model (mirrors the chunked emitter): a FEED FSM
    // consumes each group's contiguous beats WITHOUT waiting on
    // compute, and an INDEPENDENT emission path walks the head frame's shadow
    // rows as its slices report committed-wave done pulses. feed_frame drives
    // the per-group validity masks and the weight-ROM address; drain_frame
    // drives the emitted frame's ragged row count and bias group. A feed-side
    // counter is never read at emit time.

    reg [15:0] beat_count;
    reg [15:0] chunk_idx;
    reg [15:0] feed_frame;     // frame currently being fed (masks + ROM)
    reg [15:0] drain_frame;    // frame currently being drained (rows + bias)
    reg [15:0] out_row_count;  // row cursor within the draining frame
    reg [15:0] frames_fed;     // frames whose first beat has been taken
    reg [15:0] frames_emitted; // frames whose rows have fully drained
    reg        row_take;
    reg signed [15:0] accum16;
    reg [{c_width-1}:0] row_mux;
{rom_addr_decl}

    // Per-frame mg/ng decode (contract: mg = frame // n_passes, ng = frame %
    // n_passes). Split into feed/drain so feed may already be in a later group
    // while drain is still reading an earlier one.
    wire [15:0] feed_mg = feed_frame / 16'd{n_passes};
    wire [15:0] feed_ng = feed_frame % 16'd{n_passes};
    wire [15:0] drain_mg = drain_frame / 16'd{n_passes};
    wire [15:0] drain_ng = drain_frame % 16'd{n_passes};
    // Combined-fold computes full CORE_M x CORE_N tiles every group; ragged
    // logical M/N is reassembled downstream (the C++ capture stores only
    // rowOut < LOGICAL_M and the final assembly keeps only LOGICAL_N columns).
    // Keeping the full extents here means every physical row/column of each
    // frame's burst is valid, so the shadow drain progresses and the framing
    // is one out_last per CORE_M rows -- matching package.py's `captured % m`
    // and the regression TB's full-tile golden.
    wire [15:0] group_logical_rows = 16'd{m};
    wire [15:0] group_logical_cols = 16'd{n};
    wire [15:0] drain_rows = 16'd{m};

{chr(10).join(row_mask_lines)}
{chr(10).join(col_mask_lines)}
{a_replay_block}
    reg [{a_width-1}:0] a_rows_q;
    reg [{b_width-1}:0] b_cols_q;
    reg in_valid_q;

    always @(posedge clk) begin
        if (rst) begin
            a_rows_q        <= {a_width}'d0;
            b_cols_q        <= {b_width}'d0;
            in_valid_q      <= 1'b0;
        end else if (en) begin
            a_rows_q        <= {a_rows_src};
            b_cols_q        <= {b_cols_src};
            in_valid_q      <= in_valid;
        end
    end

    wire slice_reset = rst;
    wire in_beat_active = in_valid_q && (beat_count < INPUT_BEATS);
    wire slice_start = in_beat_active && (beat_count == 16'd0);
    wire final_chunk = (chunk_idx == 16'd{passes - 1});
    wire [{k_spatial * grid_rows * grid_cols - 1}:0] done_mat_mul;
    // Rev-3: done_mat_mul is a per-committed-wave 1-cycle pulse per slice.
    // done_count[s] counts them; row_done[r] says every slice of tile-row r
    // (across every K-spatial partition) has counted a done past
    // frames_emitted, so each tile-row is released as soon as ITS wave is
    // complete -- emission never waits on the feed FSM. Same shape as the
    // chunked emitter.
    reg [15:0] done_count [0:{k_spatial * grid_rows * grid_cols - 1}];
    reg        row_done [0:{grid_rows - 1}];
    reg        any_row_done;
    integer    hp, hd, hc;
    always @* begin
        any_row_done = 1'b0;
        for (hd = 0; hd < {grid_rows}; hd = hd + 1) begin
            row_done[hd] = 1'b1;
            for (hp = 0; hp < {k_spatial}; hp = hp + 1)
                for (hc = 0; hc < {grid_cols}; hc = hc + 1)
                    if (done_count[(hp*{grid_rows} + hd)*{grid_cols} + hc] <= frames_emitted)
                        row_done[hd] = 1'b0;
            if (row_done[hd]) any_row_done = 1'b1;
        end
    end

{chr(10).join(partial_wires)}

{chr(10).join(row_avail)}

    wire any_avail = {any_avail_expr};
    wire emit_active = any_row_done && (frames_fed > frames_emitted);

    // op[1] drain_stop source: the emitting frame's last logical row has just
    // been taken AND that frame has a ragged tail to truncate (an 8-aligned
    // row count needs no op1; firing one would tie with the next frame's row-0
    // capture under overlap -- see the chunked emitter's tail condition).
    wire logical_output_complete = emit_active && row_take &&
                                   (out_row_count + 16'd1 == drain_rows) &&
                                   (|drain_rows[2:0]);
    // op[2] commit tag (rev A2): asserted at the committed (final) K pass's
    // start edge; the slices latch it into their wave slot and capture only
    // for that wave. Intermediate K passes see op2 == 0.
    wire op2 = slice_start && final_chunk;
    wire [15:0] cur_row_tile = out_row_count >> 3;
{op0_lines}
{op1_lines}

{chr(10).join(decls)}

{chr(10).join(insts)}

{stage2_fn}
    always @(*) begin
        row_mux = {c_width}'d0;
        accum16 = 16'sd0;
        row_take = 1'b0;
{chr(10).join(row_mux_cases)}
    end

    // FEED engine: consumes each group's contiguous beats, advancing
    // feed_frame after the final K pass -- with NO wait on compute, so the next
    // group's beats are accepted while the current group is still computing
    // (the module's wave slots pipeline them). Owns beat_count/chunk_idx/
    // feed_frame/frames_fed/done_count/rom_addr and the independent emission of
    // the head frame's rows (row_done/frames_emitted/drain_frame/out_row_count).
    always @(posedge clk) begin
        if (rst) begin
            beat_count <= 16'd0;
            chunk_idx <= 16'd0;
            feed_frame <= 16'd0;
            drain_frame <= 16'd0;
            out_row_count <= 16'd0;
            frames_fed <= 16'd0;
            frames_emitted <= 16'd0;
            // row_take is combinational only (driven in the row_mux always @(*)
            // block); it must not be written from this clocked block or the two
            // drivers race and the drain desyncs from the slices' own ptr.
            for (hd = 0; hd < {k_spatial * grid_rows * grid_cols}; hd = hd + 1)
                done_count[hd] <= 16'd0;
            c_row <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
{rom_addr_reset}
        end else if (en) begin
            out_valid <= 1'b0;
            out_last <= 1'b0;
            for (hd = 0; hd < {k_spatial * grid_rows * grid_cols}; hd = hd + 1)
                if (done_mat_mul[hd])
                    done_count[hd] <= done_count[hd] + 16'd1;

            // A frame is K_PASSES * INPUT_BEATS consecutive in_valid beats and the
            // next frame's first beat may follow its last directly: no preload
            // stage (weights sit in the ROM or stream alongside A), so the frame
            // boundary is the beat/chunk wrap itself.
            if (in_beat_active) begin
                if ((beat_count == 16'd0) && (chunk_idx == 16'd0)) begin
                    feed_frame <= frames_fed;
                    frames_fed <= frames_fed + 16'd1;
                end
                if (beat_count + 16'd1 == INPUT_BEATS) begin
                    beat_count <= 16'd0;
                    chunk_idx <= final_chunk ? 16'd0 : chunk_idx + 16'd1;
                end else begin
                    beat_count <= beat_count + 16'd1;
                end
            end

            // Independent emission of the head frame's rows; frame t+1 feeds
            // while frame t drains.
            if (emit_active && row_take) begin
                c_row <= row_mux;
                out_valid <= 1'b1;
                out_last <= (out_row_count + 16'd1 == drain_rows);
                out_row_count <= out_row_count + 16'd1;
                if (out_row_count + 16'd1 == drain_rows) begin
                    out_row_count <= 16'd0;
                    frames_emitted <= frames_emitted + 16'd1;
                    // Wrap to the first group: the next inference drains from
                    // group 0 again (holding the last group would give every
                    // later inference the last N-group's bias).
                    drain_frame <= (drain_frame + 16'd1 == 16'd{total_frames}) ?
                        16'd0 : drain_frame + 16'd1;
                end
            end
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


def generate_k_spatial_combined_core_verilog(m, k, n, module_name="gemm_grid_wrapper", k_spatial=1, out_bits=None,
                                            s1=0, s2=0, out_width=None, weight_rom=None, n_passes=1,
                                            bias_codes=None, a_zero_point=0, b_zero_point=0,
                                            m_passes=1, logical_m=None, logical_n=None):
    """Structural-only K-spatial combined core (single-branch RTL)."""
    if out_width is None:
        out_width = out_bits if out_bits is not None else 8
    _require_symmetric_quant(a_zero_point, b_zero_point,
                             who=f"{module_name} (generate_k_spatial_combined_core_verilog)")
    if k_spatial == 1 and int(m_passes) <= 1:
        return generate_combined_core_verilog(m, k, n, module_name, out_width=out_width,
                                              s1=s1, s2=s2,
                                              weight_rom=weight_rom, n_passes=n_passes,
                                              bias_codes=bias_codes,
                                              a_zero_point=a_zero_point, b_zero_point=b_zero_point)
    if k_spatial != 1:
        _k_spatial_partitions(k, k_spatial)
    # Weight-stationary and/or bias: the structural body emits its OWN ROM(s)
    # as a whole module. Both ROMs are built from the same
    # weight_rom/bias_codes single source of truth.
    synth_top = generate_k_spatial_synth_verilog(m, k, n, module_name, k_spatial,
                                                s1=s1, s2=s2, out_width=out_width,
                                                weight_rom=weight_rom, n_passes=n_passes,
                                                bias_codes=bias_codes, m_passes=m_passes,
                                                logical_m=logical_m, logical_n=logical_n)

    lines = []
    lines.append("// Auto-generated by rtl.py")
    lines.append(f"// Combined K-spatial core: M={m}, K={k}, N={n}, K_SPATIAL={k_spatial}")
    lines.append("//   structural K-spatial wrapper (single-branch RTL: simulation + HLS synthesis)")
    lines.append("// WARNING: INT16 partial overflow is possible in K-spatial structural mode.")
    lines.append("")
    lines.extend(synth_top.splitlines())
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Verilog grid wrapper for tensor_slice_int8_atlas modules")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--name", type=str, default="gemm_grid_wrapper")
    parser.add_argument("--output", type=str, default="gemm_grid.v")
    parser.add_argument("--synth-output", type=str, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Emit Verilog $display debug traces")
    args = parser.parse_args()

    # Single-branch RTL: --output (or --synth-output) always gets the
    # structural wrapper. The former --sim-output behavioral option is gone.
    out_path = args.synth_output or args.output
    content = generate_synth_verilog(args.m, args.k, args.n, args.name, debug=args.debug)
    Path(out_path).write_text(content)
    print(f"Generated {out_path}  (M={args.m}, K={args.k}, N={args.n})")
