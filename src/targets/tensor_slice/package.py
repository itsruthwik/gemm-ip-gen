"""tensor_slice HLS package generator (tool: Catapult).

Assembles a per-layer blackbox package for the tensor_slice hardblock:
ac_channel-based C++ wrappers, the grid RTL core, the Catapult synthesis Tcl,
and a combined dispatch header for hls4ml integration. This is the ``package``
half of the target — the ``rtl`` / ``golden`` / ``geometry`` siblings supply the
RTL, testbenches, and tile geometry; the core supplies quant math (``gemm_ip.quant``)
and common helpers (``gemm_ip.common``).
"""

import json
import re
import sys
from pathlib import Path

from gemm_ip.common import _is_ac_integer_type
from gemm_ip.quant import _frac_bits, _operand_bits, _output_bits, _accum_shift_bits, _truncates

from . import geometry as _geometry

LANE_WIDTH = _geometry.LANE_WIDTH
_ceil_div = _geometry._ceil_div
_geom_k_chunks = _geometry.k_chunks
_geom_k_passes = _geometry.k_passes
resolve_reuse_factor = _geometry.resolve_reuse_factor
resolve_mkn_geometry = _geometry.resolve_mkn_geometry
latency_first_out = _geometry.latency_first_out
a_stream_width = _geometry.a_stream_width
b_stream_width = _geometry.b_stream_width

# Combinational delay (ns) Catapult must budget for any cycle that touches the
# blackbox boundary. The structural grid registers its inputs and outputs, but
# the residual paths (input-reg setup muxing on Catapult's side; registered
# c_row through the wait_dp live mux into the drain logic) are real. The old
# 0.5 claim let Catapult chain its own feed/drain logic into the same cycle as
# the hard block's pin timing, and post-route Fmax collapsed (~179 MHz vs the
# 306 MHz baseline on fc_large at a 5 ns target).
_BLACKBOX_DELAY_FALLBACK_NS = 3.5


def _blackbox_delay_ns(clock_period_ns):
    """Blackbox delay budget derived from the project clock.

    70% of the period, but always leaving Catapult >= 1.5 ns of glue headroom
    (the scheduler rejects the component outright below that: 3.3/2.3 and
    3.3/2.0 fail, 3.6/2.0 and 5.0/3.5 schedule). At the 5 ns project standard
    this reproduces the validated .delay(3.5). NOTE: tighter clock targets do
    NOT improve post-route Fmax on the tensor_slice arch — a 3.6 ns build
    routed at 186 MHz vs 222.7 MHz for the 5 ns build (routing-dominated).
    """
    if not clock_period_ns:
        return _BLACKBOX_DELAY_FALLBACK_NS
    period = float(clock_period_ns)
    return round(min(0.7 * period, period - 1.5), 2)


def latency_cycles(m, k, n, grid_rows, grid_cols, k_spatial=1):
    """First-output cycle offset for the C++ simulation model.

    Delegates to ``geometry.latency_first_out`` for a given ``k_spatial``
    (number of K partitions fed per pass; ``k_spatial == 1`` is the chunked
    endpoint, ``k_spatial == k_chunks`` is the full-K endpoint).
    """
    return latency_first_out(m, k, n, k_spatial)


def dead_cycles_raw(m, k, n, grid_cols, k_spatial=1):
    """Drain dead cycles before the C++ wrapper starts capturing output."""
    return latency_cycles(m, k, n, grid_rows=1, grid_cols=grid_cols,
                          k_spatial=k_spatial) + 1


def dead_cycles(m, k, n, grid_cols, k_spatial=1):
    """Drain dead cycles + 1-cycle padding."""
    return dead_cycles_raw(m, k, n, grid_cols, k_spatial=k_spatial) + 1


def gen_public_header(name, m, k, n, grid_rows, grid_cols, result_type=None, k_spatial=1,
                      input_precision=None, weight_precision=None, clock_period_ns=None,
                      n_frames=1, weight_rom=None, m_passes=1, logical_m=None,
                       n_passes=1, logical_n=None, out_width=16, bias_codes=None, s1=0,
                       a_zero_point=0, b_zero_point=0):
    # Symmetric-only quantization scope: nonzero zero points are rejected
    # (see generate_catapult_pkg, which derives these as 0 after the
    # precision guard).
    for _label, _value in (("a_zero_point", a_zero_point),
                           ("b_zero_point", b_zero_point)):
        if _value is not None and int(_value) != 0:
            raise ValueError(
                f"{name}: symmetric-only quantization scope -- nonzero "
                f"zero point not supported ({_label}={_value!r}). Use "
                "symmetric (signed, or <8-bit unsigned) operands."
            )
    # Weight-stationary (const-weight) mode: weights live in the RTL wrapper ROM,
    # so the ccore run() drops the b_cols port (matching the ROM wrapper), and the
    # csim-only behavioral branch bakes the same per-beat words into an internal
    # B_ROM. Single source of truth = weight_rom (built once from the .dat).
    weights_in_core = weight_rom is not None
    bb_delay_ns = _blackbox_delay_ns(clock_period_ns)
    row_chunk_bits = grid_rows * 64
    col_chunk_bits = grid_cols * 64
    # Output lane narrows to out_width (decision 8/scope item 5): the core
    # itself now emits the fully-requantised, bias-added result -- the drain
    # below is a pure unpack, no rescale/bias/cast.
    c_bits = grid_cols * 8 * out_width
    physical_rows = grid_rows * 8
    ks = int(k_spatial)
    k_chunks = _geom_k_chunks(k)
    passes = _geom_k_passes(k, ks)
    # 2+ folded axes route to the general combined-fold structural emitter
    # (`_general_synth_combined_fold`), which uses the narrow 64*k_spatial word
    # at EVERY k_spatial -- so a k_spatial==1 combined core still needs the
    # narrow (not grid_rows*64) word. Single-axis folds (fold-M only, fold-N
    # only, K only) keep the wide grid-padded word of their emitters.
    _folded_axes = int(int(m_passes) > 1) + int(int(n_passes) > 1) + int(passes > 1)
    _combined_fold = _folded_axes >= 2
    # ``k_spatial == 1`` keeps today's grid-padded word width (chunked
    # endpoint). ``k_spatial > 1`` is the narrow K-spatial word: 64*k_spatial
    # bits per beat, independent of the row/col tile count, replayed across
    # ``passes`` sweeps of K. This is a single general layout: the chunked
    # (k_spatial == 1, passes == k_chunks) and full-K (k_spatial == k_chunks,
    # passes == 1) cases are just its two endpoints.
    if ks > 1 or _combined_fold:
        a_bits = 64 * ks
        b_bits = 64 * ks
    else:
        a_bits = a_stream_width(m, ks)
        b_bits = b_stream_width(n, ks)
    # A-row replay lives in the core (rtl.py's _a_replay_block), never in a C
    # array here: the a_rows port carries every K-pass slice of a row on the
    # beat that reads it (slice p at bit p*a_bits) and zeros on every beat that
    # re-uses it -- later K passes, and all of a later N-group's frame.
    a_port_bits = passes * a_bits
    # Bias is a COMPILE-TIME constant now (decision 4): no bias_cols port at
    # all. ``bias_codes`` (or None -- the add folds away) is the SAME codes
    # list baked as the Verilog bias ROM (see rtl.py's _bias_rom_block); here
    # it is baked as a C twin static array, one 16-bit-equivalent signed
    # value per column, read at the stage-2 intermediate scale.
    has_bias = bias_codes is not None
    from gemm_ip.biasrom import bias_c_decl as _bias_c_decl
    bias_c_array_name = f"{name}_bias_codes"
    bias_c_decl_block = (
        "        " + _bias_c_decl(bias_c_array_name, list(bias_codes)).replace("\n", "\n        ").rstrip() + "\n"
        if has_bias else ""
    )
    input_beats = _geometry.feed_beats(m, n, passes)
    total_beats = passes * input_beats
    first_out = latency_cycles(m, k, n, grid_rows, grid_cols, k_spatial=ks)
    # Frame slots for the pipelined sim core: a frame may start while the
    # previous one is still computing/draining (min frame period = total_beats calls).
    #
    # op-contract note: this C
    # core has no op/pe_reset/shadow state -- it is a grid-level `gemm.run()`
    # frame-period abstraction, not a port-level model. The op[0..2] pins live
    # only in the synth branch's tensor_slice_int8_atlas black-box instantiation
    # (see rtl.py). This frame-period overlap (feed of t+1 while t is still
    # draining) is the same compute/drain decoupling the op contract expresses
    # at the pin level; it is realized here without modeling the pins.
    slots = -(-(first_out + m) // total_beats) + 1

    # Merged feed+drain call budget. The RUN loop polls out_valid on every
    # call, so it absorbs the core's port lag. The frame's last row sits at
    # run()-call index first_out + (m-1) - 1 (the core has no preload call, so
    # a frame's first in_valid beat is call 0, and its first row appears
    # first_out - 1 calls later), and the worst port lag is 3 calls (sim
    # branch: 2 registered stages; structural branch adds an input register)
    # — plus 2 calls spare.
    run_calls = first_out + m + 4
    if run_calls < total_beats + 2:
        raise RuntimeError(
            f"{name}: GEMM wrapper RUN budget shorter than the feed itself "
            f"(run_calls={run_calls}, total_beats={total_beats}; m={m} k={k} n={n})."
        )

    # Back-to-back multi-frame schedule. No BIAS preload step: bias is a
    # compile-time constant baked into the core (decision 4), not a runtime
    # port. The cores have no preload stage: a frame starts on its first
    # in_valid beat and frames are gapless (one frame's last beat is followed
    # directly by the next frame's first), so each frame is total_beats
    # in_valid beats and preload_valid is tied to 0.
    # Steady-state frame period = total_beats; the last frame's outputs drain
    # in the trailing first_out + m + slack tail. With n_frames == 1 this reduces
    # to a single frame (real hls4ml flow: one frame per wrapper call).
    period = total_beats
    # in_valid is asserted only inside the feed region; feed_total covers every
    # frame's total_beats feed cycles.
    feed_total = n_frames * period
    # Loop length: the LAST frame starts feeding at (n_frames-1)*period and its
    # final output lands first_out + m later (the behavioral core emits m rows per
    # frame). The structural core aborts masked physical rows after logical M,
    # so the tail is sized on m. Sizing on feed_total
    # here would over-run by a full period per frame (a serialized-latency
    # regression for n_frames == 1).
    total_steps = (n_frames - 1) * period + first_out + m + 4
    total_rows = n_frames * m

    # Fold-M: ``m`` here is the CORE row count (M_g = 8*mg); ``logical_m`` is
    # the true M the caller's CONFIG_T::gemm_m carries. ``m_passes`` frames of
    # M_g core rows are issued back-to-back (n_frames == m_passes drives the
    # period/feed_total/total_steps schedule above); only the trailing rows of
    # the LAST frame that fall past logical_m are padding. fold_m is False
    # (m_passes == 1) for every phase 1 (fold_axis="k") package, in which case
    # logical_m == m and nothing below changes any generated text.
    fold_m = int(m_passes) > 1
    logical_m = m if logical_m is None else int(logical_m)

    # Fold-M remains conservative. Its core M is tile-aligned today, so this
    # is normally a zero-trip loop; retain the legacy flush text if that ever
    # changes rather than assuming a cross-frame abort protocol here.
    array_padding_drain = f"""
    #pragma hls_pipeline_init_interval 1
    DRAIN_ARRAY_WL_PADDED_ROWS: for (int i = 0; i < {physical_rows - m}; i++) {{
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> drain_valid = 0;
        ac_int<1, false> drain_preload_valid = 0;
        gemm.run(last_a_rows, drain_preload_valid, drain_valid, c_row, v, l);
    }}
""" if fold_m else ""
    array_padding_drain_two_operand = f"""
    #pragma hls_pipeline_init_interval 1
    DRAIN_ARRAY_PADDED_ROWS: for (int i = 0; i < {physical_rows - m}; i++) {{
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> drain_valid = 0;
        ac_int<1, false> drain_preload_valid = 0;
        gemm.run(last_a_rows, last_b_cols, drain_preload_valid, drain_valid, c_row, v, l);
    }}
""" if fold_m else ""

    # Fold-N: ``n`` here is the CORE column count (N_g == 8*cg); ``logical_n``
    # is the true N the caller's CONFIG_T::gemm_n and res_T carry. ``n_passes``
    # frames of N_g core columns are issued back-to-back (n_frames == n_passes
    # drives the period/feed_total/total_steps schedule below, same as
    # fold-M). Unlike fold-M (which pads spare ROWS in the LAST frame), every
    # frame here emits M real rows -- only the tail COLUMN tiles of the LAST
    # group may be padding (silently zero, dropped by the emission loop's
    # ``col < logical_n`` bound). fold_n is False (n_passes == 1) for every
    # phase 1/2 (fold_axis in {{"k", "m"}}) package, in which case logical_n
    # == n and nothing below changes any generated text.
    fold_n = int(n_passes) > 1
    # C-model group counter for the baked-ROM read under fold-N (see bbuf_src).
    if fold_n and weights_in_core:
        grp_static_decl = f"\n        static int _grp = {int(n_passes) - 1};"
        grp_static_step = f"\n                _grp = (_grp + 1) % {int(n_passes)};"
    else:
        grp_static_decl = ""
        grp_static_step = ""
    # Fold-N + bias: an INDEPENDENT group counter (not gated on
    # weights_in_core -- fold-N + bias must work with an external b_cols
    # port too). Unlike the weight ROM's `_grp` (read during FEED, so the
    # live value at capture time is always correct), bias is read at EMIT
    # time, many cycles after a frame's own feed ends and (with several
    # frames in flight) possibly after `_grp`-equivalent tracking has already
    # advanced to a LATER frame's group -- so each slot freezes its OWN group
    # in `slot_bias_grp[]` at frame-allocation time (mirrors the old
    # `bias_buf[wr_slot] = bias_cols` capture), read back at emit time
    # instead of any live counter.
    fold_n_bias = bool(fold_n and bias_codes is not None)
    if fold_n_bias:
        bias_grp_static_decl = (
            f"\n        static int _bias_grp = {int(n_passes) - 1};"
            f"\n        static int slot_bias_grp[{slots}] = {{0}};"
        )
        bias_grp_static_step = (
            f"\n                _bias_grp = (_bias_grp + 1) % {int(n_passes)};"
            f"\n                slot_bias_grp[wr_slot] = _bias_grp;"
        )
    else:
        bias_grp_static_decl = ""
        bias_grp_static_step = ""
    logical_n = n if logical_n is None else int(logical_n)

    # General M/K/N fold: the frame schedule (period/feed_total/total_steps/
    # total_rows, computed above from the caller-supplied ``n_frames``) must
    # cover the PRODUCT of the independent M- and N-group pass counts once
    # either one folds (m_passes*n_passes frames total: mg-major, ng-minor --
    # see the fold_any RUN loop below). This reduces to the caller-supplied
    # ``n_frames`` exactly at every existing single-axis call site (which
    # already passes n_frames == m_passes for fold-M-only and n_frames ==
    # n_passes for fold-N-only, since n_passes/m_passes respectively == 1
    # there), and only changes behavior for the new combined case where a
    # caller has NOT already folded that product into n_frames itself. Pure
    # multi-frame batching (n_frames > 1 with m_passes == n_passes == 1) is
    # untouched -- it is an orthogonal feature (repeated activation frames
    # against the same weights), not geometry folding.
    if fold_m or fold_n:
        n_frames = int(m_passes) * int(n_passes)
        period = total_beats
        feed_total = n_frames * period
        # Structural-core tail: the last frame's rows are produced through the
        # tensor_slice_int8_atlas blackbox, whose registered inputs + handshake add
        # latency beyond the behavioral `first_out`. With the bare `+ 6` the
        # capture RUN loop could end before the final frame's drain (seen on
        # mkn_2x2: k_spatial==1 combined, 4 frames -> only 59/64 rows emitted
        # inside the window, second N-group columns captured as 0). One extra
        # `period` of slack guarantees the loop outlives the last drain for
        # every combined shape; harmless idle cycles otherwise.
        _struct_tail = period if _combined_fold else 0
        total_steps = (n_frames - 1) * period + first_out + m + 4 + _struct_tail
        total_rows = n_frames * m

    # Choose the RHS expression for the final output assignment based on the
    # configured result type.  Integer types (ac_int / ac_uint) need an explicit
    # ``.to_int()`` call because directly casting an ``ac_fixed`` accumulator to
    # an ``ac_int`` may fail to compile or produce unexpected truncation.
    # Fixed-point types should preserve the normal AC-datatype conversion
    # (rounding / saturation) by omitting ``.to_int()``.
    #
    # ``result_type is None`` is the legacy/default package, whose result lane
    # typedef is the INTEGER ``ac_int<16, true>`` — so it needs ``.to_int()`` too.
    # Emit the raw fixed-point conversion only for an explicit fixed-point result
    # type; every other case (default None or an explicit integer type) uses
    # ``.to_int()``. Casting an ``ac_fixed`` accumulator straight to an ``ac_int``
    # lane does not compile under the AC datatypes (no such constructor).
    _fixed_result = isinstance(result_type, str) and not _is_ac_integer_type(result_type)
    assign_expr = "value" if _fixed_result else "value.to_int()"

    # The core is a pure INTEGER matmul: it multiplies operand int8 *codes*
    # (the fixed-point mantissa = value · 2^frac). The raw integer dot-product
    # therefore carries 2^(frac_a + frac_b); the wrapper drain shifts it back to
    # the real value, then adds the (full-precision) bias and quantizes to the
    # result type. gemm_shift == 0 collapses to the legacy integer-coded path.
    gemm_shift = _frac_bits(input_precision) + _frac_bits(weight_precision)
    # Total gemm->result shift: the accumulator carries frac_a + frac_b, the
    # result lane carries frac(out). A zero result fraction is an ordinary
    # value here (the shift is then the whole product fraction) -- there is no
    # legacy "drain rescales" path any more, the drain is a pure unpack.
    _out_frac = _frac_bits(result_type)
    requant_shift = gemm_shift - _out_frac
    if requant_shift < 0:
        raise ValueError(
            f"{name}: result type {result_type!r} carries {_out_frac} fraction bits, "
            f"more than the {gemm_shift} the product carries; a left shift is not "
            "supported by the tensor_slice requant.")
    # Two-stage requant: this ccore mirrors the Verilog sim branch's folded model exactly. Stage 1
    # (round-half-up shift by S1, wrap to 16) is applied PER K-spatial partition
    # in-slice -- the RTL computes each of the ``ks`` partitions in its own
    # 16-bit slice partial and then sums those 16-bit partials (accum16), so the
    # csim must partition K the same way and wrap each partial to 16 before
    # summing (see the accumulate loop below). For ``ks == 1`` this reduces to a
    # single partition == the exact-full-sum-then-S1 behavior it had before.
    # Stage 2 (wrap-add the bias at the 16-bit intermediate scale, round-half-up
    # shift by S2, wrap to the physical lane) then runs once on the summed
    # 16-bit accumulator. No saturation anywhere -- decision 5. `requant_shift`
    # here is the TOTAL shift (S1 + S2); S2 is the remainder after S1.
    _s1 = int(s1) if s1 else 0
    _s2 = max(0, requant_shift - _s1)
    _half1 = (1 << (_s1 - 1)) if _s1 > 0 else 0
    _half2 = (1 << (_s2 - 1)) if _s2 > 0 else 0
    core_requant_emit = (
        f"                        ac_int<{out_width}, true> sat_val;\n"
        "                        {\n"
        "                            // Stage 2: wrap-add the bias, round-half-up shift by S2, wrap.\n"
        "                            ac_int<16, true> _biased = _p1 + bias_el;\n"
        f"                            ac_int<32, true> _r2 = (ac_int<32, true>) _biased + {_half2};\n"
        f"                            sat_val = (ac_int<{out_width}, true>) (_r2 >> {_s2});\n"
        "                        }"
    )
    if fold_n_bias:
        # Frozen per-slot group (slot_bias_grp[s], captured at that frame's
        # allocation) -- NOT the live _bias_grp counter, which may already
        # have advanced to a later frame's group by emit time.
        bias_el_expr = f"(ac_int<16, true>) {bias_c_array_name}_bias[slot_bias_grp[s] * {n} + actual_col]"
    elif has_bias:
        bias_el_expr = f"(ac_int<16, true>) {bias_c_array_name}_bias[actual_col]"
    else:
        bias_el_expr = "(ac_int<16, true>) 0"
    a_el_expr = (
        f"a_buf[s][(k_chunk / {ks}) * {input_beats} + actual_row]"
        f".slc<8>((k_chunk % {ks}) * 64 + k_lane * 8)"
        if ks > 1 or _combined_fold
        else f"a_buf[s][k_chunk * {input_beats} + actual_row].slc<8>(row_tile * 64 + k_lane * 8)"
    )
    # b_buf is sized passes*n (only real columns t < n are ever captured — see the
    # capture block below), so its index scales by n, not input_beats, unlike a_buf
    # (which keeps the full input_beats stride: every beat t < m carries a real row).
    b_el_expr = (
        f"b_buf[s][(k_chunk / {ks}) * {n} + actual_col]"
        f".slc<8>((k_chunk % {ks}) * 64 + k_lane * 8)"
        if ks > 1 or _combined_fold
        else f"b_buf[s][k_chunk * {n} + actual_col].slc<8>(ct * 64 + k_lane * 8)"
    )

    # ---- Weight-stationary vs. two-stream: b_cols plumbing inserts -------------
    # Weight-stationary drops b_cols everywhere (run() port, blackbox stub xor,
    # feed-loop decl/pack/arg) and sources the csim b_buf from an internal B_ROM.
    # Two-stream keeps today's external b_cols beat. Weight-stationary mode
    # works for every k_spatial (the B ROM is built by the general
    # build_weight_rom_k_spatial for whatever k_spatial/passes this package
    # resolved to), so both the narrow and grid-padded feed loops need the
    # conditional inserts.
    # Under fold-N the external weight beats of frame g are group ng's columns.
    # Index by the N-group ``ng`` (declared by ``_stream_g_decl`` whenever
    # fold_any) rather than the frame index ``g``: for combined M+N fold
    # ng != g once mg > 0, and indexing by g reads a later M-group's columns
    # (the stream path mirrored the already-fixed array path here). For
    # fold-N alone ng == g, so this is byte-identical to the old text.
    b_col_idx = f"ng * {n} + t" if fold_n else "t"
    if weights_in_core:
        bcols_run_param = ""
        bcols_bb_xor = ""
        bcols_run_arg = ""
        # Raw feed beat cc_slot[wr_slot] runs 0..total_beats-1 (kc*input_beats+t), but
        # only beat t < n of each pass carries a real column, so B_ROM (like the RTL
        # ROM) holds just passes*n entries; the capture block below only reads this
        # for _t < n (see _bidx), matching rtl.py's beat_ctr/rom_base mux exactly.
        # Under fold-N frame g reads group g's columns: the ROM holds n_passes
        # groups of core_n entries per pass, so offset the read by the frame's
        # group (a static counter stepping at each frame start, as the RTL's
        # group counter does). Single-group packages keep the plain index.
        #
        # ``_bidx + _grp*n`` (== kc*n + t + grp*n) only matches
        # build_weight_rom_fold_n's N-group-major/chunk-minor layout
        # (grp*passes*n + kc*n + t) when passes == 1 -- true for every
        # single-axis fold-N package (resolve_mkn_geometry defaults k_spatial
        # to K_CHUNKS, i.e. passes == 1, unless the caller ALSO explicitly
        # folds K), so this branch is byte-identical there. Once K also folds
        # in time (passes > 1) -- only reachable via a combined M/K/N fold --
        # the ROM is built by build_weight_rom_combined_fold instead, whose layout is
        # chunk-major/ng-minor ((kc*n_passes + grp)*n + t); address it to match.
        bbuf_src = (
            f"B_ROM[((cc_slot[wr_slot] / {input_beats}) * {n_passes} + _grp) * {n} + _t]"
            if (fold_n and passes > 1)
            else ("B_ROM[_bidx + _grp * %d]" % n if fold_n else "B_ROM[_bidx]")
        )
        brom_decl = _const_weights_brom_cpp(b_bits, grid_cols, weight_rom)
        stream_bcols_decl = ""
        # Weight-stationary: the IP holds B, so the feed packs no B beat at all.
        stream_bcols_pack = ""
        fr_bcols_pack = ""
        # Array feed loop, const_weights: no b_cols decl / pack / run-arg (weights in ROM).
        array_bcols_decl = ""
        array_bcols_run_arg = ""
        array_bcols_pack = ""
    else:
        bcols_run_param = f"        ac_int<{b_bits}, false>  b_cols,\n"
        bcols_bb_xor = " ^ b_cols[0]"
        bcols_run_arg = "b_cols, "
        bbuf_src = "b_cols"
        brom_decl = ""
        stream_bcols_decl = f"ac_int<{b_bits}, false> b_cols = 0;"
        # ``weight_cols`` is a plain array (random access, not a single-read
        # stream), so B never needs a replay buffer: every pass simply
        # re-slices the same full-K row/column it already has in hand. ``kc``
        # is the pass index (== the K chunk index when k_spatial == 1).
        def _stream_bcols_pack_text(guard, load, elem):
            if ks > 1 or _combined_fold:
                return f"""if ({guard}) {{
            {load}
            #pragma hls_unroll
            COL_PACK_KC: for (int kc_local = 0; kc_local < {ks}; kc_local++) {{
                #pragma hls_unroll
                COL_PACK_KL: for (int kl = 0; kl < 8; kl++) {{
                    int kk = (kc * {ks} + kc_local) * 8 + kl;
                    if (kk < {k}) {{
                        b_cols.set_slc(kc_local * 64 + kl * 8,
                                       {elem});
                    }}
                }}
            }}
        }}"""
            return f"""if ({guard}) {{
            {load}
            #pragma hls_unroll
            COL_PACK: for (int kl = 0; kl < 8; kl++) {{
                int kk = kc * 8 + kl;
                int col_tile = t / 8;
                #pragma hls_unroll
                COL_TILE_CHUNK: for (int ct = 0; ct < {grid_cols}; ct++) {{
                    if (col_tile == ct && kk < {k}) {{
                        b_cols.set_slc(ct * 64 + kl * 8,
                                       {elem});
                    }}
                }}
            }}
        }}"""
        stream_bcols_pack = _stream_bcols_pack_text(
            f"feeding_now && t < {n}", f"b_beat_T b_beat = weight_cols[{b_col_idx}];",
            f"{name}_to_gemm_int8(b_beat[kk])")
        # Free-running runtime-B entry: the same pack, sourced from the latched raw column.
        # Under fold-N the latch holds every group's columns; columns past logical_n are zero.
        fr_bcols_pack = _stream_bcols_pack_text(
            f"active && t < {n}" + (f" && ng * {n} + t < {logical_n}" if int(n_passes) > 1 else ""),
            f"ac_int<{8 * k}, false> b_raw = b_lat[{'ng * %d + t' % n if int(n_passes) > 1 else 't'}];",
            "static_cast<ac_int<8, true> >(b_raw.template slc<8>(kk * 8))")
        # Array feed loop, two-stream: pack the external weight_cols beat into b_cols_packed.
        array_bcols_decl = f"\n        ac_int<{b_bits}, false> b_cols_packed = 0;"
        array_bcols_run_arg = "b_cols_packed, "
        if ks > 1 or _combined_fold:
            array_bcols_pack = f"""
        if (step < {total_beats} && t < {n}) {{
            b_beat_T b_beat = weight_cols[{b_col_idx}];
            #pragma hls_unroll
            for (int kc_local = 0; kc_local < {ks}; kc_local++) {{
                #pragma hls_unroll
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = (kc * {ks} + kc_local) * 8 + kl;
                    if (kk < {k}) {{
                        b_cols_packed.set_slc(kc_local * 64 + kl * 8,
                                              {name}_to_gemm_int8(b_beat[kk]));
                    }}
                }}
            }}
        }}"""
        else:
            array_bcols_pack = f"""
        if (step < {total_beats} && t < {n}) {{
            b_beat_T b_beat = weight_cols[{b_col_idx}];
            #pragma hls_unroll
            for (int kl = 0; kl < 8; kl++) {{
                int kk = kc * 8 + kl;
                int col_tile = t / 8;
                #pragma hls_unroll
                for (int ct = 0; ct < {grid_cols}; ct++) {{
                    if (col_tile == ct && kk < {k}) {{
                        b_cols_packed.set_slc(ct * 64 + kl * 8,
                                              {name}_to_gemm_int8(b_beat[kk]));
                    }}
                }}
            }}
        }}"""

    # Per-call output capture, embedded in the merged RUN loop so rows are
    # collected as they emerge while the frame is still feeding/computing.
    _capture_body = f"""\
        if (v) {{
            if (captured < {logical_m}) {{
                res_T out_pack;
                #pragma hls_unroll
                for (int col = 0; col < {n}; col++) {{
                    int col_tile = col / 8;
                    int col_local = col % 8;
                    // Pure unpack (decision 8): the core already requantised
                    // (stage 1 + stage 2, bias baked in) to out_width bits --
                    // no rescale, no bias add, no accum_t. Reinterpret the
                    // out_width-bit code as the result type's raw bits.
                    ac_int<{out_width}, true> raw_val =
                        c_row.template slc<{out_width}>(col_tile * {8 * out_width} + col_local * {out_width});
                    typename res_T::value_type out_val;
                    out_val.set_slc(0, raw_val);
                    out_pack[col] = out_val;
                }}
                %SINK%
            }}
            captured++;
        }}"""
    stream_capture = _capture_body.replace("%SINK%", "res_stream.write(out_pack);")
    # Element-wise, unrolled: `results[captured] = out_pack` would call
    # nnet::array::operator=, whose copy loop is NOT unrolled in Catapult's
    # nnet_types.h. A rolled loop inside the II=1 run loop is an unschedulable
    # feedback path (SCHD-3 "feedback path too long") for every io_parallel build.
    array_capture = _capture_body.replace("%SINK%", f"""#pragma hls_unroll
                for (int col = 0; col < {n}; col++) {{
                    results[captured][col] = out_pack[col];
                }}""")

    # Back-to-back capture: identical body but the row budget spans all frames
    # (n_frames * m rows emerge across the run, retiring in frame order).
    _capture_body_b2b = f"""\
        if (v) {{
            // The behavioral core emits exactly {m} out_valid pulses per frame
            // (TOTAL_ROWS = m, no padding pulses), retiring in frame order, so
            // every pulse is a real result row: write the first {total_rows}.
            if (written < {total_rows}) {{
                res_T out_pack;
                #pragma hls_unroll
                for (int col = 0; col < {n}; col++) {{
                    int col_tile = col / 8;
                    int col_local = col % 8;
                    // Pure unpack (decision 8) -- see _capture_body's comment.
                    ac_int<{out_width}, true> raw_val =
                        c_row.template slc<{out_width}>(col_tile * {8 * out_width} + col_local * {out_width});
                    typename res_T::value_type out_val;
                    out_val.set_slc(0, raw_val);
                    out_pack[col] = out_val;
                }}
                res_stream.write(out_pack);
                written++;
            }}
            captured++;
        }}"""
    stream_capture_b2b = _capture_body_b2b

    # A-side row pack for the current pass ``kc``, unrolled and written into
    # ``dest`` (either the live ``a_rows`` word for pass 0, or a ``replay_rows``
    # scratch word for a later pass prepacked from the same beat).  ks == 1
    # keeps today's single-chunk, row-tile-addressed pack (offset by row_tile);
    # ks > 1 packs ``ks`` K chunks into one narrow word (offset by kc_local),
    # generalizing the old full-K-only pack across every pass.
    def _a_pack_block(dest, pass_expr, label, base="", elem=None):
        # ``elem`` overrides the per-element source (default: the A beat's int8
        # cast); the free-running entries pack from a raw latched row instead.
        elem = elem or f"{name}_to_gemm_int8(a_beat[kk])"
        if ks > 1 or _combined_fold:
            return f"""
                #pragma hls_unroll
                {label}_KC: for (int kc_local = 0; kc_local < {ks}; kc_local++) {{
                    #pragma hls_unroll
                    {label}_KL: for (int kl = 0; kl < 8; kl++) {{
                        int kk = (({pass_expr}) * {ks} + kc_local) * 8 + kl;
                        if (kk < {k}) {{
                            {dest}.set_slc({base}kc_local * 64 + kl * 8,
                                           {elem});
                        }}
                    }}
                }}"""
        # The row lands in one row tile, picked at run time. Build its 64-bit chunk
        # once, then write every tile slot unconditionally (the chunk or zero): a
        # conditional set_slc per byte and tile made Catapult chain the writes onto
        # one wide word, a feedback path longer than a 2 ns cycle at 13 row tiles
        # x 7 K passes. Unconditional writes to constant slices are plain wiring.
        return f"""
                {{
                    ac_int<64, false> {label}_chunk = 0;
                    #pragma hls_unroll
                    {label}_KL: for (int kl = 0; kl < 8; kl++) {{
                        int kk = ({pass_expr}) * 8 + kl;
                        if (kk < {k}) {{
                            {label}_chunk.set_slc(kl * 8, {elem});
                        }}
                    }}
                    int row_tile = t / 8;
                    #pragma hls_unroll
                    {label}_RT: for (int rt = 0; rt < {grid_rows}; rt++) {{
                        {dest}.set_slc({base}rt * 64,
                                       (row_tile == rt) ? {label}_chunk : ac_int<64, false>(0));
                    }}
                }}"""

    # Every K-pass slice of the row just read goes out on this beat; the core
    # keeps the later ones.
    def _a_pack_later_passes(dest, label, elem=None):
        if passes < 2:
            return ""
        return f"""
                #pragma hls_unroll
                {label}: for (int pack_kc = 1; pack_kc < {passes}; pack_kc++) {{{_a_pack_block(dest, "pack_kc", label + "_ROW", base=f"pack_kc * {a_bits} + ", elem=elem)}
                }}"""

    a_replay_decl = ""
    a_prepack_replay = _a_pack_later_passes("a_rows", "PACK_REPLAY")
    a_replay_else = """
            }"""

    # General M/N-group raw output capture: shared by the stream and array RUN
    # loops whenever n_passes > 1 (the group-scoped weight/bias/ROM machinery
    # is keyed purely on n_passes, unaffected by whether M also folds). Every
    # frame g decomposes into mg = g // n_passes, ng = g % n_passes and emits
    # M REAL rows for M-group mg's N-group ng; only that group's N_g columns
    # are stored, at the group's row/column offset -- the emission loop after
    # RUN assembles/rescales/biases the full logical_m x logical_n result once
    # every group has landed. Reduces to the old fold-N-only body exactly when
    # m_passes == 1 (mg always 0, rowOut == captured % m).
    capture_general = f"""\
        if (v) {{
            if (captured < {m_passes * n_passes * m}) {{
                int gOut = captured / {m};
                int rowIn = captured % {m};
                int mgOut = gOut / {n_passes};
                int ngOut = gOut % {n_passes};
                int rowOut = mgOut * {m} + rowIn;
                if (rowOut < {logical_m}) {{
                    #pragma hls_unroll
                    for (int col = 0; col < {n}; col++) {{
                        int col_tile = col / 8;
                        int col_local = col % 8;
                        c_buf[rowOut][ngOut * {n} + col] =
                            c_row.template slc<{out_width}>(col_tile * {8 * out_width} + col_local * {out_width});
                    }}
                }}
            }}
            captured++;
        }}"""

    # General M/K/N fold: every frame g in [0, m_passes*n_passes) decomposes
    # into an M-group mg = g // n_passes and an N-group ng = g % n_passes
    # (mg-major, ng-minor -- the weight/bias group counters inside the ccore
    # step by 1 every ccore-detected frame, so `_grp mod n_passes` already
    # equals ng regardless of mg; no change needed there). This subsumes the
    # old fold-M-only (n_passes == 1, ng always 0) and fold-N-only
    # (m_passes == 1, mg always 0) branches as degenerate points, and adds
    # the new case where both are simultaneously > 1.
    fold_any = fold_m or fold_n
    _a_read_guard = "kc == 0"
    _stream_feed_cond = f"feeding_now && t < {m}"
    _stream_g_decl = ""
    _stream_capture_use = stream_capture_b2b
    if fold_any:
        # A rows repeat identically across every N-group of the same M-group
        # (only B's group / output group changes across ng); a fresh stream
        # read happens only at the first K-pass of the first N-group of each
        # M-group (kc == 0 && ng == 0). The core caches the CURRENT M-group's
        # rows, so later K-passes within the same frame (kc > 0) and later
        # N-groups of the same M-group (ng > 0, any kc) both replay from it.
        _a_read_guard = "kc == 0 && ng == 0"
        _stream_g_decl = (
            "\n        int g = step / %d;"
            "\n        int mg = g / %d;"
            "\n        int ng = g %% %d;" % (period, n_passes, n_passes)
        )
        _stream_feed_cond = f"feeding_now && t < {m} && (mg * {m} + t) < {logical_m}"
        # n_passes == 1 (pure fold-M, ng always 0): results emerge in row
        # order exactly as today's fold-M -- stream them live (no deferred
        # buffer, no latency regression for the M-only case).
        # n_passes > 1: defer through the group-scoped c_buf (today's fold-N
        # behavior), now offset by the M-group so it generalizes to
        # simultaneous M+N folding.
        _stream_capture_use = stream_capture if n_passes == 1 else capture_general

    # csim twin of the core's A replay: one a_bits slice per (pass, beat) slot.
    # A fresh row fills every pass's slot on its pass-0 beat; under fold-N the
    # N-group-0 frame's slots are kept and re-used by the group's later frames.
    if fold_n:
        a_grp_static_decl = (f"\n        static int _a_grp = {int(n_passes) - 1};"
                             f"\n        static ac_int<{a_bits}, false> a_keep[{total_beats}];")
        a_grp_static_step = f"\n                _a_grp = (_a_grp + 1) % {int(n_passes)};"
        _a_buf_capture = f"""\
                if (_a_grp == 0 && cc_slot[wr_slot] < {input_beats}) {{
                    for (int _p = 0; _p < {passes}; _p++) {{
                        a_keep[_p * {input_beats} + cc_slot[wr_slot]] = a_rows.slc<{a_bits}>(_p * {a_bits});
                    }}
                }}
                a_buf[wr_slot][cc_slot[wr_slot]] = a_keep[cc_slot[wr_slot]];"""
    else:
        a_grp_static_decl = a_grp_static_step = ""
        _a_buf_capture = f"""\
                if (cc_slot[wr_slot] < {input_beats}) {{
                    for (int _p = 0; _p < {passes}; _p++) {{
                        a_buf[wr_slot][_p * {input_beats} + cc_slot[wr_slot]] = a_rows.slc<{a_bits}>(_p * {a_bits});
                    }}
                }}"""


    stream_feed_loop = f"""
{a_replay_decl}
    // Back-to-back feed of {n_frames} frame(s): M A rows + N B columns, each
    // pass carrying k_spatial={ks} K chunks (passes={passes} sweeps of K).
    // Each frame is {total_beats} in_valid beats (period {period}, gapless:
    // the core has no preload stage); bias is a
    // compile-time constant baked into the core (decision 4). Every step polls
    // out_valid, so rows are captured as they emerge, including while later
    // beats of the same call are still being fed.  (Under synthesis the stream
    // entries use the free-running body instead, where frames overlap across calls.)
    #pragma hls_pipeline_init_interval 1
    RUN: for (int step = 0; step < {total_steps}; step++) {{
        bool in_feed = (step < {feed_total});
        int p = in_feed ? (step % {period}) : {period};{_stream_g_decl}
        bool feeding_now = in_feed && (p < {total_beats});
        int pf = p;
        int kc = pf / {input_beats};
        int t = pf % {input_beats};
        ac_int<{a_port_bits}, false> a_rows = 0;
        {stream_bcols_decl}

        if ({_stream_feed_cond}) {{
            if ({_a_read_guard}) {{
                a_beat_T a_beat = a_stream.read();{_a_pack_block("a_rows", "0", "ROW_PACK_DIRECT")}{a_prepack_replay}{a_replay_else}
        }}
        {stream_bcols_pack}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = feeding_now ? 1 : 0;
        // preload_valid is an unused port (the core has no preload stage).
        ac_int<1, false> frame_preload = 0;
        gemm.run(a_rows, {bcols_run_arg}frame_preload, feed_valid, c_row, v, l);
{_stream_capture_use}
    }}
"""

    # ---- Free-running stream entries (synthesis only) -------------------------------
    # The IO stream entries become single-cycle `hls_design block`s: one wrapper clock
    # per call, so frames overlap and the interval is the feed length (groups x
    # total_beats) instead of feed + drain. The core pauses on an idle beat inside a group
    # (in_valid low mid-frame freezes it), so a group may start before its A rows arrive:
    # each call makes one non-blocking read of the A stream when the current beat carries
    # a fresh row, feeds the beat if the row arrived, and otherwise presents an idle beat
    # and retries next call. A blocking read or available() on the A stream is not an
    # option: Catapult stalls the whole block (core drain included) whenever the stream is
    # empty, which delays the previous frame's outputs until the next frame's first row
    # arrives. Beats that need no fresh row (later K passes, which the core replays, and
    # padding beats t >= m) are fed without reading.
    # Folded axes: one hls4ml frame is m_passes x n_passes groups (mg-major, ng-minor).
    # A is popped only on the ng == 0 group (the core replays it for later N-groups),
    # rows past logical_m are never fed. Runtime B columns are latched in an indexed
    # array: column j is gathered once its last-pass beat is fed. A runtime B latch holds all logical_n columns for the
    # whole frame (every M-group re-reads them) and is refilled during the last M-group.
    # Output rows are written one per out_valid pulse, never deferred: fold-N slices from
    # the earlier N-groups of an M-group wait in c_buf and the last N-group's pulse
    # completes and writes each full row. out_valid is a one-cycle pulse. The csim
    # (`#else`) path is unchanged: one frame per call.
    if input_beats < max(m, n):
        raise RuntimeError(f"{name}: feed pass ({input_beats} beats) shorter than m={m}/n={n}.")

    def _bw(v):
        return max(1, int(v).bit_length())

    def _fr_stream_body(runtime_b):
        fold_ng = int(n_passes) > 1
        fold_mg = int(m_passes) > 1
        last_mg = int(m_passes) - 1
        last_rows = logical_m - last_mg * m   # core rows fed in the last M-group
        cnt_w = _bw(max(m, logical_n))
        ng_e = "ng" if fold_ng else "0"
        mg_e = "mg" if fold_mg else "0"
        at_first = f"{mg_e} == 0 && {ng_e} == 0"
        b_decl = (f"static ac_int<{8 * k}, false> b_lat[{logical_n}];"
                  f"\n    static ac_int<{cnt_w}, false> b_cnt = 0;"
                  f"\n    static bool b_full = false;"
                  f"\n    static bool b_rst = false;"
                  f"\n    static bool bq_act = false;"
                  f"\n    static bool bq_fed = false;"
                  f"\n    static ac_int<{_bw(input_beats)}, false> bq_t = 0;"
                  f"\n    static ac_int<{_bw(passes)}, false> bq_kc = 0;"
                  f"\n    static ac_int<{_bw(max(int(n_passes) - 1, 1))}, false> bq_ng = 0;"
                  f"\n    static ac_int<{_bw(max(last_mg, 1))}, false> bq_mg = 0;\n    ") if runtime_b else ""
        b_word_decl = f"\n    ac_int<{b_bits}, false> b_cols = 0;" if runtime_b else ""
        b_feed = f"\n    {fr_bcols_pack}" if runtime_b else ""
        b_arg = "b_cols, " if runtime_b else ""
        if runtime_b and (fold_mg or fold_ng):
            # The frame's first group needs every column; later groups reuse them.
            b_ready = f" && (!({at_first}) || b_full)"
        else:
            b_ready = " && b_full" if runtime_b else ""
        # The next frame's columns are gathered during the last M-group, which is each
        # column's final use. The start test sees only the registered b_full flag (the
        # read that completes the latch sets it), never a same-call count compare, so the
        # start decision does not sit behind the B read. When that group's first N-group
        # starts, the latch is consumed: b_full drops and b_rst is raised; the next call
        # restarts b_cnt (and blocks the B read for that call) instead of the activation
        # call doing it, which keeps the activation path off the stream-read results.
        b_reset = (f"\n        if ({mg_e} == {last_mg} && {ng_e} == 0) {{\n            b_full = false;\n            b_rst = true;\n        }}"
                   if runtime_b else "")
        # B reads are gated by the release derived from the previous call's feed
        # progress. A gate that used this call's `fed` would
        # make the B read depend on the A read's result in the same call, and an I/O op
        # predicated on another I/O op's result forces an extra cycle (breaking II=1).
        # The price is that a column is released one call after its last use.
        # The release is computed from registered copies of the previous call's beat
        # (bq_*: fed, t, kc, ng, mg), not from this call's `fed`, so the whole B read
        # predicate is a function of registers only. Doing the derivation here, rather
        # than through a further release register, keeps the release at one call of lag.
        b_gate = f"!b_rst && b_cnt < {logical_n if (fold_mg or fold_ng) else n} && b_cnt < b_avail"
        bq_mg_e = "bq_mg.to_int()" if fold_mg else "0"
        bq_ng_e = "bq_ng.to_int()" if fold_ng else "0"
        bq_first = f"{bq_mg_e} == 0 && {bq_ng_e} == 0"
        if runtime_b and (fold_mg or fold_ng):
            b_avail_expr = f"""
    // Column j is free once its last-pass beat in the last M-group has been fed, or
    // while the next frame waits. An idle call feeds no beat, so beat t itself is not
    // yet consumed then.
    int b_avail = 0;
    if (!bq_act && {bq_first}) {{
        b_avail = {logical_n};
    }} else if ({bq_mg_e} == {last_mg}) {{
        int t_used = bq_fed ? bq_t.to_int() + 1 : bq_t.to_int();
        b_avail = {bq_ng_e} * {n} + ((bq_act && bq_kc == {passes - 1}) ? (t_used < {n} ? t_used : {n}) : 0);
    }}"""
        elif runtime_b:
            b_avail_expr = f"""
    // B column j is free once its last-pass beat (t >= j) has been fed. An idle call
    // feeds no beat, so beat t itself is not yet consumed then.
    int b_avail = 0;
    if (!bq_act) {{
        b_avail = {n};
    }} else if (bq_kc == {passes - 1}) {{
        b_avail = bq_fed ? bq_t.to_int() + 1 : bq_t.to_int();
    }}"""
        if runtime_b:
            b_gather = f"""{b_avail_expr}
    bool b_got = false;
    if ({b_gate}) {{
        b_beat_T b_beat;
        if (b_stream.nb_read(b_beat)) {{
            b_got = true;
            ac_int<{8 * k}, false> raw = 0;
            #pragma hls_unroll
            B_LATCH: for (int i = 0; i < {k}; i++) {{
                raw.set_slc(i * 8, {name}_to_gemm_int8(b_beat[i]));
            }}
            b_lat[b_cnt] = raw;
        }}
    }}
    if (b_rst) {{
        b_cnt = 0;
        b_rst = false;
    }} else if (b_got) {{
        if (b_cnt == {(logical_n if (fold_mg or fold_ng) else n) - 1}) {{
            b_full = true;
        }}
        b_cnt++;
    }}
    bq_act = active;
    bq_fed = fed;
    bq_t = t;
    bq_kc = kc;{"" if not fold_ng else chr(10) + "    bq_ng = ng;"}{"" if not fold_mg else chr(10) + "    bq_mg = mg;"}"""
        else:
            b_gather = ""
        grp_decl = ((f"\n    static ac_int<{_bw(n_passes - 1)}, false> fr_ng = 0;" if fold_ng else "")
                    + (f"\n    static ac_int<{_bw(last_mg)}, false> fr_mg = 0;" if fold_mg else ""))
        grp_local = (f"\n    int mg = {'fr_mg.to_int()' if fold_mg else '0'};"
                     f"\n    int ng = {'fr_ng.to_int()' if fold_ng else '0'};")
        # A is read on the first N-group of each M-group only.
        a_rows_cond = f"active && kc == 0 && t < {m}"
        if fold_ng:
            a_rows_cond += " && ng == 0"
        if fold_mg and last_rows != m:
            a_rows_cond += f" && (mg != {last_mg} || t < {last_rows})"
        grp_advance = ""
        if fold_mg:
            grp_advance = f"fr_mg = (mg == {last_mg}) ? 0 : mg + 1;"
        if fold_ng:
            adv_m = f"\n                    {grp_advance}" if grp_advance else ""
            grp_advance = f"""if (ng == {n_passes - 1}) {{
                    fr_ng = 0;{adv_m}
                }} else {{
                    fr_ng = ng + 1;
                }}"""
        grp_advance = f"\n                {grp_advance}" if grp_advance else ""
        # Output: one write per out_valid pulse. Per-pulse position counters (row within
        # the group, N-group, M-group) run on their own since outputs lag the feed.
        ow = out_width
        need_omg = fold_mg and last_rows != m
        o_decl = ""
        if fold_ng or fold_mg:
            o_decl += f"\n    static ac_int<{_bw(m - 1)}, false> o_row = 0;"
        if fold_ng:
            o_decl += f"\n    static ac_int<{_bw(n_passes - 1)}, false> o_ng = 0;"
            o_decl += f"\n    static ac_int<{(n_passes - 1) * n * ow}, false> c_buf[{m}];"
        if need_omg:
            o_decl += f"\n    static ac_int<{_bw(last_mg)}, false> o_mg = 0;"
        keep = f"(o_mg.to_int() != {last_mg} || o_row.to_int() < {last_rows})" if need_omg else "true"
        omg_adv = (f"if (o_mg == {last_mg}) {{ o_mg = 0; }} else {{ o_mg = o_mg + 1; }}"
                   if need_omg else "")
        if fold_ng:
            o_adv = f"""if (o_row == {m - 1}) {{
            o_row = 0;
            if (o_ng == {n_passes - 1}) {{
                o_ng = 0;
                {omg_adv}
            }} else {{
                o_ng = o_ng + 1;
            }}
        }} else {{
            o_row = o_row + 1;
        }}"""
        elif fold_mg:
            o_adv = f"""if (o_row == {m - 1}) {{
            o_row = 0;
            {omg_adv}
        }} else {{
            o_row = o_row + 1;
        }}"""
        else:
            o_adv = ""
        if fold_ng:
            o_store = f"""
        if (o_ng != {n_passes - 1}) {{
            // Earlier N-group slice of this row: keep it until the last N-group's pulse.
            #pragma hls_unroll
            C_ROW: for (int r = 0; r < {m}; r++) {{
                #pragma hls_unroll
                C_GRP: for (int gc = 0; gc < {n_passes - 1}; gc++) {{
                    if (o_row == r && o_ng == gc) {{
                        c_buf[r].set_slc(gc * {n * ow}, c_row.template slc<{n * ow}>(0));
                    }}
                }}
            }}
        }} else if ({keep}) {{
            ac_int<{(n_passes - 1) * n * ow}, false> held = c_buf[o_row.to_int()];
            res_T out_pack;
            #pragma hls_unroll
            for (int col = 0; col < {logical_n}; col++) {{
                ac_int<{ow}, true> raw_val = (col < {(n_passes - 1) * n})
                    ? ac_int<{ow}, true>(held.template slc<{ow}>(col * {ow}))
                    : ac_int<{ow}, true>(c_row.template slc<{ow}>((col - {(n_passes - 1) * n}) * {ow}));
                typename res_T::value_type out_val;
                out_val.set_slc(0, raw_val);
                out_pack[col] = out_val;
            }}
            res_stream.write(out_pack);
        }}"""
        else:
            o_store = f"""
        if ({keep}) {{
            res_T out_pack;
            #pragma hls_unroll
            for (int col = 0; col < {n}; col++) {{
                int col_tile = col / 8;
                int col_local = col % 8;
                // Pure unpack (decision 8) -- see _capture_body's comment.
                ac_int<{ow}, true> raw_val =
                    c_row.template slc<{ow}>(col_tile * {8 * ow} + col_local * {ow});
                typename res_T::value_type out_val;
                out_val.set_slc(0, raw_val);
                out_pack[col] = out_val;
            }}
            res_stream.write(out_pack);
        }}"""
        o_adv = f"\n        {o_adv}" if o_adv else ""
        return f"""\
    // One wrapper clock per call; state persists across calls. A group is
    // {total_beats} contiguous beats ({passes} pass(es) x {input_beats} beats); the
    // next group may start on the following call. A beat that needs a fresh A row and
    // finds the stream empty is presented as an idle beat (the core pauses) and retried.
    {b_decl}static ac_int<{_bw(passes)}, false> fr_kc = 0;
    static ac_int<{_bw(input_beats)}, false> fr_t = 0;{grp_decl}
    static bool active = false;{o_decl}{grp_local}
    if (!active{b_ready}) {{
        active = true;
        fr_kc = 0;
        fr_t = 0;{b_reset}
    }}
    int kc = fr_kc.to_int();
    int t = fr_t.to_int();
    ac_int<{a_port_bits}, false> a_rows = 0;{b_word_decl}
    // Every K-pass slice of A row t goes out on its pass-0 beat; the core keeps the later ones.
    bool a_need = {a_rows_cond};
    ac_int<{8 * k}, false> a_raw = 0;
    bool a_got = false;
    if (a_need) {{
        a_beat_T a_beat;
        if (a_stream.nb_read(a_beat)) {{
            #pragma hls_unroll
            A_LATCH: for (int i = 0; i < {k}; i++) {{
                a_raw.set_slc(i * 8, {name}_to_gemm_int8(a_beat[i]));
            }}
            a_got = true;
        }}
    }}
    bool fed = active && (!a_need || a_got);
    if (a_got) {{{_a_pack_block("a_rows", "0", "ROW_PACK_DIRECT", elem="static_cast<ac_int<8, true> >(a_raw.template slc<8>(kk * 8))")}{_a_pack_later_passes("a_rows", "PACK_REPLAY", elem="static_cast<ac_int<8, true> >(a_raw.template slc<8>(kk * 8))")}
    }}{b_feed}
    ac_int<{c_bits}, false> c_row;
    ac_int<1, false> v, l;
    ac_int<1, false> feed_valid = fed ? 1 : 0;
    // preload_valid is an unused port (the core has no preload stage).
    ac_int<1, false> frame_preload = 0;
    gemm.run(a_rows, {b_arg}frame_preload, feed_valid, c_row, v, l);{b_gather}
    if (fed) {{
        if (t == {input_beats - 1}) {{
            fr_t = 0;
            if (kc == {passes - 1}) {{
                active = false;
                fr_kc = 0;{grp_advance}
            }} else {{
                fr_kc = kc + 1;
            }}
        }} else {{
            fr_t = t + 1;
        }}
    }}
    // Core has no backpressure and out_valid is a pulse: handle it on every call.
    if (v) {{{o_store}{o_adv}
    }}
"""

    fr_block_pragma = """\
#if defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW)
// Free-running entry: its own single-cycle block, one wrapper clock per call, frames overlap.
#pragma hls_design block
#pragma hls_pipeline_init_interval 1
#endif
"""

    # Merged feed+drain array loop, parameterised over weight-stationarity the same way
    # the stream feed loop is: two-stream packs weight_cols into b_cols_packed and passes
    # it to run(); const_weights drops all three b_cols inserts (decl/pack/run-arg) because
    # the ccore holds B in its ROM. One loop serves both the _gemm_ip_array (two-stream)
    # and _gemm_ip_array_const_weights entries.
    # a_rows[] is a plain array (random access), so — like weight_cols — the
    # array feed loop needs no replay buffer: every pass re-slices the row it
    # already has in hand.
    array_feed_loop = f"""
    // Merged feed+drain: one run() call per cycle. Steps 0..{total_beats - 1}
    // feed M A rows (and, two-stream, N B columns), each pass carrying
    // k_spatial={ks} K chunks (passes={passes} sweeps of K). Every step polls
    // out_valid, so the frame's rows are captured as they emerge, not in a drain loop.
    #pragma hls_pipeline_init_interval 1
    RUN_ARRAY: for (int step = 0; step < {run_calls}; step++) {{
        int eff_step = (step >= {total_beats}) ? 0 : step;
        int kc = eff_step / {input_beats};
        int t = eff_step % {input_beats};
        ac_int<{a_port_bits}, false> a_rows_packed = 0;{array_bcols_decl}

        if (step < {total_beats} && t < {m} && kc == 0) {{
            a_beat_T a_beat = a_rows[t];{_a_pack_block("a_rows_packed", "0", "ROW_PACK_ARRAY")}{_a_pack_later_passes("a_rows_packed", "PACK_ARRAY")}
        }}{array_bcols_pack}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = (step < {total_beats}) ? 1 : 0;
        ac_int<1, false> feed_preload_valid = 0;
        gemm.run(a_rows_packed, {array_bcols_run_arg}feed_preload_valid, feed_valid, c_row, v, l);
{array_capture}
    }}
"""

    if fold_any:
        # General array feed: B-side (two-stream) weight beats are indexed by
        # the N-group `ng` (the ROM/const-weight path addresses by the same
        # `ng` inside the ccore's `_grp` counter -- see above). Reduces to the
        # old fold-array_bcols_pack (indexed by `g`) exactly when n_passes==1
        # (ng always 0, weight_cols[0*n+t] == weight_cols[t] -- fine, that
        # branch is only emitted when n_passes>1 needs an explicit group
        # anyway) and to the old fold_n_array_bcols_pack when m_passes==1
        # (mg always 0, ng == g).
        if weights_in_core:
            fold_general_bcols_pack = ""
        elif ks > 1 or _combined_fold:
            fold_general_bcols_pack = f"""
        if (feeding_now && t < {n}) {{
            b_beat_T b_beat = weight_cols[ng * {n} + t];
            #pragma hls_unroll
            for (int kc_local = 0; kc_local < {ks}; kc_local++) {{
                #pragma hls_unroll
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = (kc * {ks} + kc_local) * 8 + kl;
                    if (kk < {k}) {{
                        b_cols_packed.set_slc(kc_local * 64 + kl * 8,
                                              {name}_to_gemm_int8(b_beat[kk]));
                    }}
                }}
            }}
        }}"""
        else:
            fold_general_bcols_pack = f"""
        if (feeding_now && t < {n}) {{
            b_beat_T b_beat = weight_cols[ng * {n} + t];
            #pragma hls_unroll
            for (int kl = 0; kl < 8; kl++) {{
                int kk = kc * 8 + kl;
                int col_tile = t / 8;
                #pragma hls_unroll
                for (int ct = 0; ct < {grid_cols}; ct++) {{
                    if (col_tile == ct && kk < {k}) {{
                        b_cols_packed.set_slc(ct * 64 + kl * 8,
                                              {name}_to_gemm_int8(b_beat[kk]));
                    }}
                }}
            }}
        }}"""
        _array_capture_use = array_capture if n_passes == 1 else capture_general
        # General array loop: one nested (implicit) mg x ng x kc schedule --
        # frame g = mg * n_passes + ng, decoded the same way as the stream RUN
        # loop above. A rows come straight from a_rows[] (random access -- no
        # replay buffer needed, unlike the channel-fed stream entry: every
        # (kc, ng) combination for the same M-group just re-slices the row it
        # already has in hand at a_rows[mg*m + t]). Reduces to the old
        # fold-M-only array loop when n_passes == 1 (ng always 0, g == mg) and
        # to the old fold-N-only array loop when m_passes == 1 (mg always 0,
        # g == ng, a_rows[mg*m+t] == a_rows[t]).
        _array_a_cond = f"feeding_now && t < {m} && (mg * {m} + t) < {logical_m} && kc == 0 && ng == 0"
        _array_a_pack = (_a_pack_block("a_rows_packed", "0", "ROW_PACK_ARRAY_FOLD")
                         + _a_pack_later_passes("a_rows_packed", "PACK_ARRAY_FOLD"))
        array_feed_loop = f"""
    // General M/K/N-fold multi-frame feed: {m_passes} M-group(s) x {n_passes}
    // N-group(s) of {m} core rows / {n} core columns each frame (K sweeps
    // k_spatial={ks} chunks in {passes} passes within each frame), issued
    // back-to-back. Frame g = mg * {n_passes} + ng reads logical rows
    // [mg*{m}, (mg+1)*{m}) directly from a_rows; rows at/after the logical M
    // bound are the last M-group's padding (fed zero A, dropped by the
    // capture body's logical bound).
    #pragma hls_pipeline_init_interval 1
    RUN_ARRAY: for (int step = 0; step < {total_steps}; step++) {{
        bool in_feed = (step < {feed_total});
        int p = in_feed ? (step % {period}) : {period};
        int g = step / {period};
        int mg = g / {n_passes};
        int ng = g % {n_passes};
        bool feeding_now = in_feed && (p < {total_beats});
        int pf = p;
        int kc = pf / {input_beats};
        int t = pf % {input_beats};
        ac_int<{a_port_bits}, false> a_rows_packed = 0;{array_bcols_decl}

        if ({_array_a_cond}) {{
            a_beat_T a_beat = a_rows[mg * {m} + t];{_array_a_pack}
        }}
        {fold_general_bcols_pack}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = feeding_now ? 1 : 0;
        ac_int<1, false> feed_preload_valid = 0;
        gemm.run(a_rows_packed, {array_bcols_run_arg}feed_preload_valid, feed_valid, c_row, v, l);
{_array_capture_use}
    }}
"""

    # General M/N-group C assembly: raw group-scoped lanes land in c_buf
    # during the RUN loop above (capture_general) whenever n_passes > 1; once
    # every group's frame has landed, this separate row-iteration loop
    # pure-unpacks the full logical_n-wide rows (decision 8: no rescale/bias/
    # accum_t -- the core already requantised, bias baked in) across every
    # M-group's logical rows -- exactly today's per-row drain, just deferred
    # past the last frame instead of interleaved with the feed.
    _c_buf_decl = (
        f"    ac_int<{out_width}, true> c_buf[{logical_m}][{n_passes * n}];\n" if fold_n else ""
    )
    _fold_n_emit_body = f"""\
    #pragma hls_pipeline_init_interval 1
    EMIT_FOLD_N: for (int row = 0; row < {logical_m}; row++) {{
        %SINK_DECL%
        #pragma hls_unroll
        for (int col = 0; col < {logical_n}; col++) {{
            ac_int<{out_width}, true> raw_val = c_buf[row][col];
            %SINK_COL%
        }}
        %SINK_ROW%
    }}
"""
    _emit_stream = (
        _fold_n_emit_body
        .replace("        %SINK_DECL%\n", "        res_T out_pack;\n")
        .replace("            %SINK_COL%",
                 "            typename res_T::value_type out_val; out_val.set_slc(0, raw_val); out_pack[col] = out_val;")
        .replace("        %SINK_ROW%\n", "        res_stream.write(out_pack);\n")
    ) if fold_n else ""
    _emit_array = (
        _fold_n_emit_body
        .replace("        %SINK_DECL%\n", "")
        .replace("            %SINK_COL%",
                 "            typename res_T::value_type out_val; out_val.set_slc(0, raw_val); results[row][col] = out_val;")
        .replace("        %SINK_ROW%\n", "")
    ) if fold_n else ""

    stream_body = f"""\
    int captured = 0;   // total out_valid pulses seen (incl. padding rows)
    int written = 0;    // real result rows written to res_stream
{_c_buf_decl}
{stream_feed_loop}
{_emit_stream}"""
    rb_fr_prefix = f"""\
    static_assert(CONFIG_T::gemm_m == {logical_m}, "Generated GEMM wrapper requires matching gemm_m.");
    static_assert(CONFIG_T::gemm_n == {logical_n}, "Generated GEMM wrapper requires matching gemm_n.");
    static_assert(a_beat_T::size == CONFIG_T::gemm_k,
                  "a_beat_T must carry one A-row K-width beat.");
    static_assert(b_beat_T::size == CONFIG_T::gemm_k,
                  "b_beat_T must carry one B-col K-width beat.");
    static_assert(res_T::size == CONFIG_T::gemm_n,
                  "res_T must carry one full GEMM result row.");

    static {name}_ccore gemm;
"""
    rb_stream_body = f"""\
    b_beat_T weight_cols[{logical_n}];
    #pragma hls_pipeline_init_interval 1
    READ_B_COLS: for (int col = 0; col < {logical_n}; col++) {{
        weight_cols[col] = b_stream.read();
    }}
    {name}_gemm_ip_stream_buffered_b<a_beat_T, b_beat_T, typename CONFIG_T::bias_t, res_T, CONFIG_T, false>(
        a_stream, weight_cols, nullptr, res_stream);
"""

    def _fr_wrap(fr_body, csim_body):
        # Free-running body under synthesis; the csim body (one frame per call) otherwise.
        if not fr_body:
            return csim_body
        return f"""\
#if defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW)
{fr_body}#else
{csim_body}#endif
"""

    if weights_in_core:
        # Weight-stationary: the self-contained const_weights STREAM entry plus the
        # const_weights ARRAY entry (io_parallel). Neither references an external b_cols;
        # weights live in the csim B_ROM (baked above) and the RTL wrapper ROM.
        entries_block = f"""\
// Weight-stationary (const-weight) entry: A only, no weight argument. Synthesis
// binds gemm.run to the const_weights RTL core (weights in the wrapper ROM); csim
// uses the ccore's internal B_ROM (same .dat-sourced beats). No frontend weight
// accessor is involved.
{fr_block_pragma}template <class a_beat_T, class bias_T, class res_T, typename CONFIG_T, bool HAS_BIAS = true>
void {name}_gemm_ip_stream_const_weights(
    ac_channel<a_beat_T> &a_stream,
    bias_T *biases,
    ac_channel<res_T> &res_stream
) {{
    static_assert(CONFIG_T::gemm_m == {logical_m}, "Generated GEMM wrapper requires matching gemm_m.");
    static_assert(CONFIG_T::gemm_n == {logical_n}, "Generated GEMM wrapper requires matching gemm_n.");
    static_assert(a_beat_T::size == CONFIG_T::gemm_k,
                  "a_beat_T must carry one A-row K-width beat.");
    static_assert(res_T::size == CONFIG_T::gemm_n,
                  "res_T must carry one full GEMM result row.");

    static {name}_ccore gemm;
{_fr_wrap(_fr_stream_body(False), stream_body)}}}

// Weight-stationary ARRAY entry (io_parallel): array in / array out, weights in
// the core. Direct feed — A rows go straight into gemm.run() (no b_cols, weights
// from the ROM), mirroring the two-stream _gemm_ip_array but const_weights. Channel-
// free, so it synthesises (a channel bridge to the stream entry hits HIER-11).
template <class a_beat_T, class bias_T, class res_T, typename CONFIG_T, bool HAS_BIAS = true>
void {name}_gemm_ip_array_const_weights(
    a_beat_T a_rows[CONFIG_T::gemm_m],
    bias_T *biases,
    res_T results[CONFIG_T::gemm_m]
) {{
    static_assert(CONFIG_T::gemm_m == {logical_m}, "Generated GEMM wrapper requires matching gemm_m.");
    static_assert(CONFIG_T::gemm_n == {logical_n}, "Generated GEMM wrapper requires matching gemm_n.");
    static_assert(a_beat_T::size == CONFIG_T::gemm_k,
                  "a_beat_T must carry one A-row K-width beat.");
    static_assert(res_T::size == CONFIG_T::gemm_n,
                  "res_T must carry one full GEMM result row.");

    static {name}_ccore gemm;
    int captured = 0;
    ac_int<{a_port_bits}, false> last_a_rows = 0;
{_c_buf_decl}
{array_feed_loop}
{array_padding_drain}
{_emit_array}}}
"""
    else:
        # Two-stream: buffered-B worker + external-weight stream/array entries
        # (today's behavior). The ccore run() keeps its b_cols port.
        entries_block = f"""\
template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T, bool HAS_BIAS = true>
void {name}_gemm_ip_stream_buffered_b(
    ac_channel<a_beat_T> &a_stream,
    b_beat_T weight_cols[CONFIG_T::gemm_n],
    bias_T *biases,
    ac_channel<res_T> &res_stream
) {{
    static_assert(CONFIG_T::gemm_m == {logical_m}, "Generated GEMM wrapper requires matching gemm_m.");
    static_assert(CONFIG_T::gemm_n == {logical_n}, "Generated GEMM wrapper requires matching gemm_n.");
    static_assert(a_beat_T::size == CONFIG_T::gemm_k,
                  "a_beat_T must carry one A-row K-width beat.");
    static_assert(CONFIG_T::transpose_weights,
                  "Generated GEMM IP wrapper expects transposed weights.");
    static_assert(b_beat_T::size == CONFIG_T::gemm_k,
                  "b_beat_T must carry one B-col K-width beat.");
    static_assert(res_T::size == CONFIG_T::gemm_n,
                  "res_T must carry one full GEMM result row.");


    static {name}_ccore gemm;
    int captured = 0;   // total out_valid pulses seen (incl. padding rows)
    int written = 0;    // real result rows written to res_stream
{_c_buf_decl}
{stream_feed_loop}
{_emit_stream}}}

// Two-operand entry: no bias port -- a two-operand GEMM never owns one, so the
// shared buffered-B worker above is instantiated with HAS_BIAS=false (biases=nullptr),
// folding away the drain add and leaving no bias array anywhere in this instantiation.
{fr_block_pragma}template <class a_beat_T, class b_beat_T, class res_T, typename CONFIG_T>
void {name}_gemm_ip_stream(
    ac_channel<a_beat_T> &a_stream,
    ac_channel<b_beat_T> &b_stream,
    ac_channel<res_T> &res_stream
) {{
{_fr_wrap(rb_fr_prefix + _fr_stream_body(True), rb_stream_body)}}}

// Two-operand entry: no bias port -- a two-operand GEMM never owns one. HAS_BIAS is
// a local compile-time false (not a template parameter -- nothing else instantiates
// this entry with a real bias), so the shared drain text below folds away the add.
template <class a_beat_T, class b_beat_T, class res_T, typename CONFIG_T>
void {name}_gemm_ip_array(
    a_beat_T a_rows[CONFIG_T::gemm_m],
    b_beat_T weight_cols[CONFIG_T::gemm_n],
    res_T results[CONFIG_T::gemm_m]
) {{
    static_assert(CONFIG_T::gemm_m == {logical_m}, "Generated GEMM wrapper requires matching gemm_m.");
    static_assert(CONFIG_T::gemm_n == {logical_n}, "Generated GEMM wrapper requires matching gemm_n.");

    static {name}_ccore gemm;
    int captured = 0;
    ac_int<{a_port_bits}, false> last_a_rows = 0;
    ac_int<{b_bits}, false> last_b_cols = 0;
    constexpr bool HAS_BIAS = false;
    typename CONFIG_T::bias_t *biases = nullptr;
{_c_buf_decl}

{array_feed_loop}
{array_padding_drain_two_operand}
{_emit_array}}}
"""

    return f"""\
#ifndef {name.upper()}_GEMM_IP_H
#define {name.upper()}_GEMM_IP_H

#include "ac_int.h"
#include "ac_fixed.h"
#include "ac_channel.h"

#if defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW)
#include "ac_blackbox.h"
#endif

namespace nnet {{

struct {name}_raw_result_t {{
    ac_int<{c_bits}, false> c_row;
    ac_int<1, false> out_last;
}};

class {name}_ccore {{
  public:
    {name}_ccore() {{}}

    #pragma hls_design interface ccore blackbox
    void run(
        ac_int<{a_port_bits}, false>  a_rows,
{bcols_run_param}        ac_int<1, false>         preload_valid,
        ac_int<1, false>         in_valid,
        ac_int<{c_bits}, false>& c_row,
        ac_int<1, false>&        out_valid,
        ac_int<1, false>&        out_last
    ) {{
#if defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW)
        ac_blackbox()
            .entity("{name}_core")
            .verilog_files("{name}_core.v")
            .outputs("c_row out_valid out_last")
            .area(2048.0)
            .delay({bb_delay_ns})
            .latency(1)
            .init_delay(1)
            .clock_name("clk")
            .posedge_clock(true)
            .sync_reset_name("rst")
            .active_high_sync_reset(true)
            .start_name("en")
            .has_state(true)
            .end();
        c_row = 0;
        c_row[0] = a_rows[0]{bcols_bb_xor} ^ preload_valid[0] ^ in_valid[0];
        out_valid = in_valid;
        out_last = in_valid;
#else{brom_decl}
{bias_c_decl_block}        // Frame-slot scheduler (mirrors the structural wrapper's overlapped-frame schedule): up to
        // {slots} frames in flight. A frame starts at the first in_valid call
        // after a non-in_valid call, or right after the previous frame's last
        // beat when frames are gapless (no preload stage); each frame keeps a private operand
        // buffer and cycle counter and emits its {m} rows at
        // [first_out, first_out+{m}) of its own clock. Back-to-back
        // frames sustain a frame II of {total_beats} calls.
        static ac_int<{a_bits}, false> a_buf[{slots}][{total_beats}];
        static ac_int<{b_bits}, false> b_buf[{slots}][{passes * n}];
        static int cc_slot[{slots}] = {{0}};
        static bool slot_run[{slots}] = {{false}};
        static int wr_slot = {slots - 1};
        static bool feeding = false;{grp_static_decl}{bias_grp_static_decl}{a_grp_static_decl}

        c_row = 0;
        out_valid = 0;
        out_last = 0;

        if (in_valid) {{
            if (!feeding || cc_slot[wr_slot] >= {total_beats}) {{
                wr_slot = (wr_slot + 1) % {slots};
                feeding = true;
                slot_run[wr_slot] = true;
                cc_slot[wr_slot] = 0;{grp_static_step}{bias_grp_static_step}{a_grp_static_step}
            }}
            if (cc_slot[wr_slot] < {total_beats}) {{
{_a_buf_capture}
                // Only beat t < n of each pass carries a real column (see b_el_expr /
                // B_ROM above); beats t >= n are never captured (and never read back).
                int _t = cc_slot[wr_slot] % {input_beats};
                if (_t < {n}) {{
                    int _bidx = (cc_slot[wr_slot] / {input_beats}) * {n} + _t;
                    (void) _bidx;
                    b_buf[wr_slot][_bidx] = {bbuf_src};
                }}
            }}
        }} else {{
            feeding = false;
        }}

        for (int s = 0; s < {slots}; s++) {{
            if (!slot_run[s]) continue;
            int scc = cc_slot[s];
            if (scc >= {first_out} && scc < {first_out} + {m}) {{
                int out_idx = scc - ({first_out});
                int actual_row = out_idx;
                int row_tile = actual_row / 8;
                (void) row_tile;
                ac_int<{c_bits}, false> row_out = 0;
                for (int ct = 0; ct < {grid_cols}; ct++) {{
                    for (int cl = 0; cl < 8; cl++) {{
                        int actual_col = ct * 8 + cl;
                        // Per-K-spatial-partition INT16 accumulation, mirroring
                        // the RTL: each of the {ks} partitions (partition pp owns
                        // K chunks with chunk % {ks} == pp) accumulates its own
                        // exact product sum, gets stage-1 requantised and wrapped
                        // to 16 bits in-slice, and only then are the {ks} 16-bit
                        // partials summed (16-bit wrap, == the RTL's accum16). Bias
                        // is added post-stage-2 (core_requant_emit below). For
                        // {ks} == 1 this is one partition over the full K == the
                        // former exact-full-sum-then-S1 model.
                        ac_int<16, true> _p1 = 0;
                        for (int _pp = 0; _pp < {ks}; _pp++) {{
                            ac_int<32, true> acc = 0;
                            if (actual_row < {m} && actual_col < {n}) {{
                                for (int kk = 0; kk < {k}; kk++) {{
                                    int k_chunk = kk / 8;
                                    if (k_chunk % {ks} != _pp) continue;
                                    int k_lane = kk % 8;
                                    ac_int<8, true> a_el = {a_el_expr};
                                    ac_int<8, true> b_el = {b_el_expr};
                                    acc += a_el * b_el;
                                }}
                            }}
                            // Stage 1 (per partition): round-half-up shift the
                            // partition sum by S1, wrap to 16, accumulate into the
                            // 16-bit partial sum (wraps -- RTL accum16 is 16-bit).
                            ac_int<33, true> _r1 = (ac_int<33, true>) acc + {_half1};
                            _p1 += (ac_int<16, true>) (_r1 >> {_s1});
                        }}
                        ac_int<16, true> bias_el = {bias_el_expr};
{core_requant_emit}
                        row_out.set_slc(ct * {8 * out_width} + cl * {out_width}, sat_val);
                    }}
                }}
                c_row = row_out;
                out_valid = 1;
                out_last = (out_idx == {m} - 1) ? 1 : 0;
            }}
            cc_slot[s] = scc + 1;
            if (scc + 1 >= {first_out} + {m}) {{
                slot_run[s] = false;
                cc_slot[s] = 0;
            }}
        }}
#endif
    }}
}};

template <class src_T>
ac_int<8, true> {name}_to_gemm_int8(const src_T &value) {{
    // The int8 code is the fixed-point MANTISSA (value · 2^frac), i.e. the raw
    // stored bits -- NOT value.to_int(), which would truncate the fractional part
    // of an ac_fixed operand and destroy it. slc<8>(0) reinterprets the low 8
    // mantissa bits as a signed int8 code; the drain rescales by 2^-(fa+fb).
    // Symmetric-only quantization scope: operands reach the core unflipped
    // (signed codes straight through). Unsigned 8-bit operands are rejected
    // at package time (see _check_operand_fits_int8_core); narrower unsigned
    // codes pass through unchanged (bit 7 is never set).
    static_assert(src_T::width <= 8, "tensor_slice int8 core: operand wider than 8 bits");
    ac_int<8, false> raw = value.template slc<8>(0);
    return static_cast<ac_int<8, true> >(raw);
}}

{entries_block}
}} // namespace nnet

#endif // {name.upper()}_GEMM_IP_H
"""


def gen_nnet_types_header():
    return """\
#ifndef NNET_TYPES_H_
#define NNET_TYPES_H_

#include <assert.h>
#include <cstddef>

namespace nnet {

template <typename T, unsigned N> struct array {
    typedef T value_type;
    static const unsigned size = N;
    T data[N];

    T &operator[](size_t pos) { return data[pos]; }
    const T &operator[](size_t pos) const { return data[pos]; }

    array &operator=(const array &other) {
#ifndef __SYNTHESIS__
        // Software-only self-assignment guard. Under HLS the pointer compare
        // becomes a SELECT; with a dynamic destination index (the einsum GEMM
        // drain's results[captured]) Catapult cannot prove non-aliasing and
        // schedules a dynamic conditional whole-array copy, which is an
        // unschedulable recurrence. Value-semantics arrays never need it.
        if (&other == this)
            return *this;
#endif
        #pragma hls_unroll
        for (unsigned i = 0; i < N; i++) {
            data[i] = other[i];
        }
        return *this;
    }
};

} // namespace nnet

#endif
"""


def gen_tb(name, m, k, n, interface="stream", n_frames=1,
           requant_shift=0, weight_matrix=None,
           out_width=16, bias_codes=None, s1=0, s2=0, k_spatial=1):
    weights_in_core = weight_matrix is not None
    if weights_in_core:
        # Golden uses the SAME baked weights as the core ROM.
        import numpy as _np
        _B = _np.asarray(weight_matrix)
        weights_init = "\n".join(
            f"    weights[{j}][{kk}] = {int(_B[kk, j])};" for j in range(n) for kk in range(k)
        )
    else:
        weights_init = (
            f"    for (int j = 0; j < {n}; j++) {{\n"
            f"        for (int kk = 0; kk < {k}; kk++) {{\n"
            f"            weights[j][kk] = ((j * 5 - kk + 1) & 0x7) - 3;\n"
            f"        }}\n"
            f"    }}"
        )
    if weights_in_core and interface == "array":
        # Weight-stationary ARRAY tb (io_parallel): array in/out, no weight port
        # (weights baked in core). Single-frame smoke test. Golden uses the same
        # baked weights (see weights init above).
        call_setup = f"""\
    a_beat_t a_rows[{m}];
    res_t results[{m}];

    for (int i = 0; i < {m}; i++) {{
        for (int kk = 0; kk < {k}; kk++) {{
            a_rows[i][kk] = activations[0][i][kk];
        }}
    }}

#ifdef CCS_SCVERIFY
    CCS_DESIGN({name}_inst)(a_rows, biases, results);
#else
    nnet::{name}_gemm_ip_array_const_weights<a_beat_t, int, res_t, {name}_config>(
        a_rows, biases, results);
#endif

    for (int i = 0; i < {m}; i++) {{
        res_t out = results[i];
        check_row(out, activations[0][i], weights, biases, 0, i, failed);
    }}
"""
    elif weights_in_core:
        # Weight-stationary stream tb: no b_stream (weights baked in core); call the
        # const_weights entry. Golden uses the same baked weights (see weights init below).
        call_setup = f"""\
    ac_channel<a_beat_t> a_stream;
    ac_channel<res_t> res_stream;

    for (int f = 0; f < NFRAMES; f++) {{
        for (int i = 0; i < {m}; i++) {{
            a_beat_t a_beat;
            for (int kk = 0; kk < {k}; kk++) {{
                a_beat[kk] = activations[f][i][kk];
            }}
            a_stream.write(a_beat);
        }}
    }}

#ifdef CCS_SCVERIFY
    CCS_DESIGN({name}_inst)(a_stream, biases, res_stream);
#else
    nnet::{name}_gemm_ip_stream_const_weights<a_beat_t, int, res_t, {name}_config>(
        a_stream, biases, res_stream);
#endif

    for (int f = 0; f < NFRAMES; f++) {{
        for (int i = 0; i < {m}; i++) {{
            res_t out = res_stream.read();
            check_row(out, activations[f][i], weights, biases, f, i, failed);
        }}
    }}
"""
    elif interface == "array":
        # Array interface is a single-frame smoke test (n_frames ignored).
        call_setup = f"""\
    a_beat_t a_rows[{m}];
    b_beat_t weight_cols[{n}];
    res_t results[{m}];

    for (int i = 0; i < {m}; i++) {{
        for (int kk = 0; kk < {k}; kk++) {{
            a_rows[i][kk] = activations[0][i][kk];
        }}
    }}
    for (int j = 0; j < {n}; j++) {{
        for (int kk = 0; kk < {k}; kk++) {{
            weight_cols[j][kk] = weights[j][kk];
        }}
    }}

#ifdef CCS_SCVERIFY
    CCS_DESIGN({name}_inst)(a_rows, weight_cols, biases, results);
#else
    // Two-operand entry: no bias port (a two-operand GEMM never owns one) --
    // 3 call args, 4 template params, matching {name}_gemm_ip_array's actual
    // signature exactly (no bias_T, no biases pointer).
    nnet::{name}_gemm_ip_array<a_beat_t, b_beat_t, res_t, {name}_config>(
        a_rows, weight_cols, results);
#endif

    for (int i = 0; i < {m}; i++) {{
        res_t out = results[i];
        check_row(out, activations[0][i], weights, biases, 0, i, failed);
    }}
"""
    else:
        # Stream interface: feed NFRAMES distinct activation frames back-to-back
        # (weights are constant across frames), then read and verify every frame's
        # rows. Cycle/II timing is observed from the core's BEH_START/BEH_II/BEH_DONE
        # $display lines in the QuestaSim transcript, not from the testbench.
        call_setup = f"""\
    ac_channel<a_beat_t> a_stream;
    ac_channel<b_beat_t> b_stream;
    ac_channel<res_t> res_stream;

    for (int j = 0; j < {n}; j++) {{
        b_beat_t b_beat;
        for (int kk = 0; kk < {k}; kk++) {{
            b_beat[kk] = weights[j][kk];
        }}
        b_stream.write(b_beat);
    }}
    for (int f = 0; f < NFRAMES; f++) {{
        for (int i = 0; i < {m}; i++) {{
            a_beat_t a_beat;
            for (int kk = 0; kk < {k}; kk++) {{
                a_beat[kk] = activations[f][i][kk];
            }}
            a_stream.write(a_beat);
        }}
    }}

#ifdef CCS_SCVERIFY
    CCS_DESIGN({name}_inst)(a_stream, b_stream, biases, res_stream);
#else
    // Two-operand entry: no bias port (a two-operand GEMM never owns one) --
    // 3 call args, 4 template params, matching {name}_gemm_ip_stream's actual
    // signature exactly (no bias_T, no biases channel).
    nnet::{name}_gemm_ip_stream<a_beat_t, b_beat_t, res_t, {name}_config>(
        a_stream, b_stream, res_stream);
#endif

    for (int f = 0; f < NFRAMES; f++) {{
        for (int i = 0; i < {m}; i++) {{
            res_t out = res_stream.read();
            check_row(out, activations[f][i], weights, biases, f, i, failed);
        }}
    }}
"""

    # Two-stage requant, no saturation: the core now ALWAYS applies stage 1 (S1, unknown to this
    # standalone smoke harness -- treated as 0, a single round) + stage 2
    # (round-half-up shift by the TOTAL `requant_shift`, wrap to out_width),
    # bias baked in at generation time (this harness never forwards a real
    # bias to the packager, so its own `biases[]` fixture is zeroed and
    # unused -- kept only for entry-point signature compatibility). The
    # drain is a pure unpack: this reference reinterprets the requantised
    # code as the result type directly, matching the core bit-for-bit.
    res_typedef = f"typedef nnet::array<ac_int<{out_width}, true>, {n}> res_t;"
    # Bias codes (decision 4): the SAME integer codes baked into the core's
    # bias ROM at generation time (gemm_ip.biasrom.bias_acc_codes, at the
    # intermediate scale gemm_frac - S1). This harness assumes S1 == 0 (the
    # caller is responsible for choosing precisions that make that true --
    # see generate_catapult_pkg's own S1 derivation), so the intermediate
    # scale is the full gemm_shift and this add lands at the exact point
    # two_stage_reference's stage-2 add does when s1 == 0. No bias -> zeros.
    _bias_vals = list(bias_codes) if bias_codes is not None else [0] * n
    _bias_lit = ", ".join(str(int(b)) for b in _bias_vals)
    _ks = int(k_spatial)
    _s1 = int(s1)
    _s2 = int(s2)
    _half1 = (1 << (_s1 - 1)) if _s1 else 0
    _half2 = (1 << (_s2 - 1)) if _s2 else 0
    _st1 = (f"(ac_int<16, true>)((part[p] + {_half1}) >> {_s1})" if _s1
            else "(ac_int<16, true>)(part[p])")
    _st2 = (f"(ac_int<{out_width}, true>)((biased.to_int() + {_half2}) >> {_s2})"
            if _s2 else f"(ac_int<{out_width}, true>)(biased.to_int())")
    # Per-K-partition two-stage reference, matching the structural RTL bit-for-bit:
    # chunk c = kk/8 belongs to partition c % k_spatial (chunk = pass*k_spatial + p),
    # each partition is stage-1 round-half-up shifted and wrapped to 16, the 16-bit
    # partials are summed (wrap 16), the bias code is added (wrap 16), then stage 2
    # round-half-up shifts and wraps to out_width. Reduces to the exact full-K sum +
    # stage1 when k_spatial == 1. (The old single-round-by-total-shift form was only
    # correct at S1 == 0; it mis-scored every S1>0 package and made the smoke TB's
    # own check_row fail csim/cosim even though the IP was correct.)
    check_row = f"""\
// Per-row golden check: per-K-partition stage 1 (round-half-up shift by S1, wrap
// to 16), sum the 16-bit partials, add the SAME bias code baked into the core's
// bias ROM, stage 2 (round-half-up shift by S2, wrap to out_width). No saturation.
static const int _golden_bias_codes[{n}] = {{{_bias_lit}}};
static void check_row(const res_t &out, ac_int<8, true> a_row[{k}],
                      ac_int<8, true> weights[{n}][{k}], int biases[{n}],
                      int f, int i, int &failed) {{
    for (int j = 0; j < {n}; j++) {{
        int part[{_ks}] = {{0}};
        for (int kk = 0; kk < {k}; kk++) {{
            part[(kk / 8) % {_ks}] += a_row[kk].to_int() * weights[j][kk].to_int();
        }}
        int psum = 0;
        for (int p = 0; p < {_ks}; p++) {{
            ac_int<16, true> st = {_st1};
            psum += st.to_int();
        }}
        ac_int<16, true> biased = (ac_int<16, true>)(psum + _golden_bias_codes[j]);
        ac_int<{out_width}, true> expect_code = {_st2};
        if (out[j].to_int() != expect_code.to_int()) {{
            printf("Mismatch frame %d row %d col %d: got %d expected %d\\n",
                   f, i, j, out[j].to_int(), expect_code.to_int());
            failed = 1;
        }}
    }}
}}"""

    return f"""\
#ifdef CCS_SCVERIFY
#include "mc_testbench.h"
#include "mc_scverify.h"
#else
#include <stdio.h>
#endif

#include "nnet_types.h"
#include "{name}_gemm_ip.h"

#define NFRAMES {n_frames}

struct {name}_config {{
    static const unsigned gemm_m = {m};
    static const unsigned gemm_k = {k};
    static const unsigned gemm_n = {n};
    static const unsigned n_in = {k};
    static const unsigned n_out = {n};
    static const bool transpose_weights = true;
    typedef int bias_t;
    typedef ac_fixed<48, 24, true> accum_t;
}};

typedef nnet::array<ac_int<8, true>, {k}> a_beat_t;
typedef nnet::array<ac_int<8, true>, {k}> b_beat_t;
{res_typedef}

{check_row}

#ifdef CCS_SCVERIFY
CCS_MAIN(int argc, char *argv[]) {{
#else
int main() {{
#endif
    ac_int<8, true> activations[NFRAMES][{m}][{k}];
    ac_int<8, true> weights[{n}][{k}];
    int biases[{n}];
    int failed = 0;

    for (int f = 0; f < NFRAMES; f++) {{
        for (int i = 0; i < {m}; i++) {{
            for (int kk = 0; kk < {k}; kk++) {{
                // Frame-dependent stimulus so consecutive frames differ.
                activations[f][i][kk] = ((i * 3 + kk - 4 + f * 2) & 0x7) - 4;
            }}
        }}
    }}

    // Bias is compile-time now (decision 4): baked into the core at
    // generation time, if any. This smoke harness never bakes one, so its
    // own biases[] fixture is zeroed (kept only for entry-point signature
    // compatibility; check_row ignores it).
    for (int j = 0; j < {n}; j++) {{
        biases[j] = 0;
    }}
{weights_init}

{call_setup}

#ifdef CCS_SCVERIFY
    CCS_RETURN(failed);
#else
    if (failed) {{
        printf("Test FAILED\\n");
        return 1;
    }}
    printf("Test passed\\n");
    return 0;
#endif
}}
"""


def _const_weights_brom_cpp(b_bits, grid_cols, weight_rom):
    """csim-only internal weight ROM for the const_weights ccore #else branch.

    ``weight_rom`` is the list of per-pass, per-real-column b_cols words (each
    ``b_bits`` wide) from ``build_weight_rom`` — the SAME ``passes*n`` beats the
    RTL wrapper ROM holds (only beat ``t < n`` of each pass carries a real
    column; the ccore, like the RTL, zero-fills the read for ``t >= n`` rather
    than storing a wasted entry). Big words don't fit a single C++ integer
    literal, so each word is split into ``grid_cols`` 64-bit chunks stored as
    ``unsigned long long`` and reassembled into an ``ac_int`` via ``set_slc``
    in a one-time init. Emitted inside the ``#else`` (behavioral) branch, so
    the synth/cosim translation unit never contains the table.
    """
    nbeats = len(weight_rom)
    chunks = b_bits // 64
    mask64 = (1 << 64) - 1
    rows = []
    for w in weight_rom:
        w = int(w) & ((1 << b_bits) - 1)
        rows.append("{" + ", ".join(f"{(w >> (g * 64)) & mask64}ULL" for g in range(chunks)) + "}")
    init = ",\n            ".join(rows)
    return f"""
        // csim-only weight ROM: per-pass, per-real-column b_cols words baked from
        // the same weight_matrix that fills the RTL wrapper ROM (single .dat
        // source; only n of every input_beats beats per pass are stored, mirroring
        // the RTL's rom_base/beat_ctr addressing). Split into 64-bit chunks (a full
        // word may exceed a C++ literal) and reassembled.
        static const unsigned long long _brom_w[{nbeats}][{chunks}] = {{
            {init}
        }};
        static ac_int<{b_bits}, false> B_ROM[{nbeats}];
        {{
            static bool _brom_init = false;
            if (!_brom_init) {{
                for (int _i = 0; _i < {nbeats}; _i++)
                    for (int _g = 0; _g < {chunks}; _g++)
                        B_ROM[_i].set_slc(_g * 64, ac_int<64, false>(_brom_w[_i][_g]));
                _brom_init = true;
            }}
        }}"""


def gen_inst_cpp(name, m, k, n, interface="stream",
                 requant_shift=0, weight_matrix=None,
                 out_width=16):
    grid_rows, grid_cols = _geometry.grid_rows, _geometry.grid_cols
    weights_in_core = weight_matrix is not None
    if weights_in_core and interface == "array":
        # Weight-stationary array (io_parallel): array ports, no external weight port.
        top_signature = f"""\
    a_beat_t a_rows[{m}],
    int biases[{n}],
    res_t results[{m}]
) {{
    nnet::{name}_gemm_ip_array_const_weights<a_beat_t, int, res_t, {name}_config>(
        a_rows, biases, results);
}}"""
    elif weights_in_core:
        top_signature = f"""\
    ac_channel<a_beat_t> &a_stream,
    int biases[{n}],
    ac_channel<res_t> &res_stream
) {{
    nnet::{name}_gemm_ip_stream_const_weights<a_beat_t, int, res_t, {name}_config>(
        a_stream, biases, res_stream);
}}"""
    elif interface == "array":
        top_signature = f"""\
    a_beat_t a_rows[{m}],
    b_beat_t weight_cols[{n}],
    int biases[{n}],
    res_t results[{m}]
) {{
    // Two-operand entry has no bias port; biases[] stays part of this
    // smoke top's own signature but is not forwarded.
    nnet::{name}_gemm_ip_array<a_beat_t, b_beat_t, res_t, {name}_config>(
        a_rows, weight_cols, results);
}}"""
    else:
        top_signature = f"""\
    ac_channel<a_beat_t> &a_stream,
    ac_channel<b_beat_t> &b_stream,
    int biases[{n}],
    ac_channel<res_t> &res_stream
) {{
    // Two-operand entry has no bias port; biases[] stays part of this
    // smoke top's own signature but is not forwarded.
    nnet::{name}_gemm_ip_stream<a_beat_t, b_beat_t, res_t, {name}_config>(
        a_stream, b_stream, res_stream);
}}"""
    a_w = grid_rows(m) * 8
    b_w = grid_cols(n) * 8
    # Pure unpack (decision 8): the core already requantised (bias baked in,
    # if any); this standalone smoke top just carries the out_width-bit code.
    res_typedef = f"typedef nnet::array<ac_int<{out_width}, true>, {n}> res_t;"
    return f"""\
#include "nnet_types.h"
#include "{name}_gemm_ip.h"

struct {name}_config {{
    static const unsigned gemm_m = {m};
    static const unsigned gemm_k = {k};
    static const unsigned gemm_n = {n};
    static const unsigned n_in = {k};
    static const unsigned n_out = {n};
    static const bool transpose_weights = true;
    typedef int bias_t;
    typedef ac_fixed<48, 24, true> accum_t;
}};

typedef nnet::array<ac_int<8, true>, {k}> a_beat_t;
typedef nnet::array<ac_int<8, true>, {k}> b_beat_t;
{res_typedef}

#pragma hls_design top
void {name}_inst(
{top_signature}
"""


def gen_tcl(name, m, k, n, interface="stream", weights_in_core=False, clock_period_ns=None):
    # Map each top-level port to a ccs_ioport resource. The resource name is the
    # top argument's variable name, so the map MUST track the four signatures'
    # actual ports: the const_weights tops carry NO B operand (weights live in the
    # core ROM), and the array top's operands are a_rows / weight_cols. A directive
    # on a non-existent port fails the Catapult run, so we emit exactly the ports
    # the standalone top declares.
    if interface == "array":
        # The array top's operands/result are arrays of nnet::array structs, which
        # Catapult synthesizes as MEMORY interfaces (not ccs_ioport streaming
        # resources). Forcing them onto ccs_ioport.ccs_in_wait fails ("Unknown path
        # .../a_rows:rsc"); the default memory mapping schedules and cosims cleanly,
        # so the array top adds no explicit port directives.
        map_lines = f"""\
# Array top-level package smoke synthesis. hls4ml integration instantiates the
# ccore through the generated combined header rather than this standalone top.
# Array ports default-map as memories — no explicit ccs_ioport directives."""
    else:
        b_map = "" if weights_in_core else (
            f"directive set /{name}_inst/b_stream:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait\n")
        map_lines = f"""\
# Phase 5: Map streams to real streaming resources
directive set /{name}_inst/a_stream:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
{b_map}directive set /{name}_inst/biases:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
directive set /{name}_inst/res_stream:rsc -MAP_TO_MODULE ccs_ioport.ccs_out_wait"""

    # Follow the design's clock when the caller has one (matches the hls4ml/build_prj.tcl
    # side of the same package); fall back to Catapult's Agilex default of 3 ns otherwise.
    clock_period = float(clock_period_ns) if clock_period_ns else 3.0
    clock_high = clock_period / 2.0

    return f"""\
set project_name "{name}_proj"
set solution_name "{name}_sol"

project new -name $project_name
solution new $solution_name
solution options defaults
solution options set /Output/OutputVerilog true
solution options set /Output/GenerateCycleNetlist false

# Turn on SCVerify (C++ TB vs RTL cosim) before go analyze. The generated tcl
# previously called /SCVerify/launch_make without requiring the package, which
# dies with "Flow '/SCVerify/launch_make' not found".
flow package require /SCVerify

options set Input/CompilerFlags {{-DBLACKBOX_FLOW}}

solution file add ./{name}_inst.cpp -type C++
solution file add ./{name}_tb.cpp -type C++
# Structural core RTL (single-branch) for SCVerify RTL cosim.
solution file add ./{name}_core.v -type Verilog -exclude true
# hard-block model: one copy per package root, shared by every layer
solution file add ../tensor_slice_int8_atlas.v -type Verilog -exclude true

directive set -DESIGN_GOAL area
directive set -SPECULATE true
directive set -MERGEABLE true
# Sim-only unit package (back-to-back demonstration): map the small internal
# operand/replay arrays to registers so the back-to-back modulo feed indexing
# never contends for limited RAM read ports (not an area-optimised build).
directive set -REGISTER_THRESHOLD 4096
directive set -MEM_MAP_THRESHOLD 4096
directive set -LOGIC_OPT false
directive set -FSM_ENCODING none
directive set -UNROLL no
directive set -IO_MODE super
directive set -CHAN_IO_PROTOCOL use_library
directive set -TIMING_CHECKS true

go new
solution design set {name}_inst -top
go analyze
go compile

solution library add mgc_Altera-Agilex-2_beh -- -rtlsyntool Quartus -manufacturer Altera -family Agilex -speed 2 -part AGFB014R24B2E2V
solution library add Altera_M20K
solution library add Altera_MLAB
solution library add Altera_DIST
solution library add Altera_ROMS
go libraries

directive set -CLOCKS {{clk {{-CLOCK_PERIOD {clock_period} -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME {clock_high} -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND sync -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low}}}}

{map_lines}

go assembly
go architect
go allocate
go schedule
go extract

# RTL co-simulation (QuestaSim/msim) on the generated Verilog. Single-branch
# RTL: the structural wrapper (+ tensor_slice_int8_atlas.v black-box slices) is
# exercised — the same core large-bench cosim uses. Parse T:first_output /
# T:last_output from the transcript for back-to-back frame timing.
flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {{}} SIMTOOL=msim sim

project save
puts "{name} Catapult run complete."
"""


def _dispatch_condition(item):
    shape_condition = (
        f"CONFIG_T::gemm_m == {item['m']} &&\n"
        f"                  CONFIG_T::gemm_k == {item['k']} &&\n"
        f"                  CONFIG_T::gemm_n == {item['n']}"
    )
    if item.get("gemm_ip_index") is not None:
        return (
            f"CONFIG_T::gemm_ip_id == {item['gemm_ip_index']} &&\n"
            f"                  {shape_condition}"
        )
    return shape_condition


def gen_combined_header(items):
    includes = "\n".join(f'#include "{item["name"]}/{item["name"]}_gemm_ip.h"' for item in items)
    stream_branches = []
    buffered_b_stream_branches = []
    array_branches = []
    const_weights_branches = []
    array_const_weights_branches = []
    for item in items:
        target = item.get("interface", "stream")
        # Weight-stationary items emit ONLY the const_weights entry (A-only; weights in
        # the core ROM). Their package has no _gemm_ip_stream / _buffered_b /
        # _array functions, so referencing those in the other dispatchers would be an
        # undeclared-identifier error (non-dependent name, checked even in a discarded
        # `if constexpr` branch). Route each item to exactly the dispatchers whose
        # per-core function its package actually emits.
        if item.get("weights_in_core"):
            # Weight-stationary packages emit BOTH the stream and array const_weights
            # entries (shared RTL core), so route each item's shape to both
            # dispatchers — io_stream layers reach the stream one, io_parallel the array.
            _has_bias_lit = "true" if item.get("has_bias", True) else "false"
            const_weights_branches.append(f"""\
    if constexpr ({_dispatch_condition(item)}) {{
        {item["name"]}_gemm_ip_stream_const_weights<a_beat_T, typename CONFIG_T::bias_t, res_T, CONFIG_T, {_has_bias_lit}>(
            a_stream, biases, res_stream);
    }}""")
            array_const_weights_branches.append(f"""\
    if constexpr ({_dispatch_condition(item)}) {{
        {item["name"]}_gemm_ip_array_const_weights<a_beat_T, typename CONFIG_T::bias_t, res_T, CONFIG_T, {_has_bias_lit}>(
            a_rows, biases, results);
    }}""")
            continue
        branch = f"""\
    if constexpr ({_dispatch_condition(item)}) {{
        {item["name"]}_gemm_ip_{target}<a_beat_T, b_beat_T, res_T, CONFIG_T>(
            TARGET_ARGS);
    }}"""
        # The buffered-B stream wrapper is emitted by every two-stream package
        # (regardless of its declared interface), so route every such item's shape to
        # it — the einsum GEMM uses the buffered-B stream path for QKt/A.V even
        # though its package item is registered with interface="array".
        buffered_b_stream_branches.append(f"""\
    if constexpr ({_dispatch_condition(item)}) {{
        {item["name"]}_gemm_ip_stream_buffered_b<a_beat_T, b_beat_T, bias_T, res_T, CONFIG_T>(
            a_stream, weight_cols, biases, res_stream);
    }}""")
        # Every two-stream package also emits _gemm_ip_array; the io_stream einsum's
        # buffered fallback (gemm_ip_array_wrapper) is always compiled, so route every
        # such item's shape to the array dispatcher too (regardless of interface).
        array_branches.append(f"""\
    if constexpr ({_dispatch_condition(item)}) {{
        {item["name"]}_gemm_ip_array<a_beat_T, b_beat_T, res_T, CONFIG_T>(
            a_rows, weight_cols, results);
    }}""")
        if target != "array":
            stream_branches.append(branch.replace("TARGET_ARGS", "a_stream, b_stream, res_stream"))
    stream_branches_text = " else ".join(stream_branches)
    buffered_b_stream_branches_text = " else ".join(buffered_b_stream_branches)
    array_branches_text = " else ".join(array_branches)
    if not stream_branches_text:
        stream_branches_text = """\
    static_assert(CONFIG_T::gemm_m == 0,
                  "No generated stream GEMM IP implementation is present in this package.");"""
    else:
        stream_branches_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated stream GEMM IP implementation matches this CONFIG_T.");
    }"""
    if not buffered_b_stream_branches_text:
        buffered_b_stream_branches_text = """\
    static_assert(CONFIG_T::gemm_m == 0,
                  "No generated buffered-B stream GEMM IP implementation is present in this package.");"""
    else:
        buffered_b_stream_branches_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated buffered-B stream GEMM IP implementation matches this CONFIG_T.");
    }"""
    if not array_branches_text:
        array_branches_text = """\
    static_assert(CONFIG_T::gemm_m == 0,
                  "No generated array GEMM IP implementation is present in this package.");"""
    else:
        array_branches_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated array GEMM IP implementation matches this CONFIG_T.");
    }"""
    const_weights_branches_text = " else ".join(const_weights_branches)
    if not const_weights_branches_text:
        const_weights_branches_text = """\
    static_assert(CONFIG_T::gemm_m == 0,
                  "No generated weight-stationary GEMM IP implementation is present in this package.");"""
    else:
        const_weights_branches_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated weight-stationary GEMM IP implementation matches this CONFIG_T.");
    }"""
    array_const_weights_branches_text = " else ".join(array_const_weights_branches)
    if not array_const_weights_branches_text:
        array_const_weights_branches_text = """\
    static_assert(CONFIG_T::gemm_m == 0,
                  "No generated weight-stationary array GEMM IP implementation is present in this package.");"""
    else:
        array_const_weights_branches_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated weight-stationary array GEMM IP implementation matches this CONFIG_T.");
    }"""
    return f"""\
#ifndef GEMM_IP_COMBINED_H_
#define GEMM_IP_COMBINED_H_

#include "ac_channel.h"
{includes}

namespace nnet {{

// Two-operand entry (hls4ml call site: nnet::gemm_stream<...>(a, b, res), no bias --
// a two-operand GEMM never owns one). Routes to each item's own no-bias entry
// (HAS_BIAS=false baked in at the per-item definition), so no bias array is
// declared and no bias add is synthesized anywhere in this instantiation.
template <class a_beat_T, class b_beat_T, class res_T, typename CONFIG_T>
void gemm_stream(
    ac_channel<a_beat_T> &a_stream,
    ac_channel<b_beat_T> &b_stream,
    ac_channel<res_T> &res_stream
) {{
{stream_branches_text}
}}
template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_stream_buffered_b(
    ac_channel<a_beat_T> &a_stream,
    b_beat_T weight_cols[CONFIG_T::gemm_n],
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
) {{
{buffered_b_stream_branches_text}
}}
// Two-operand entry (hls4ml call site: nnet::gemm_array<...>(a_rows, b_cols, results),
// no bias -- see gemm_stream above; same no-bias routing, no zero array declared).
template <class a_beat_T, class b_beat_T, class res_T, typename CONFIG_T>
void gemm_array(
    a_beat_T a_rows[CONFIG_T::gemm_m],
    b_beat_T weight_cols[CONFIG_T::gemm_n],
    res_T results[CONFIG_T::gemm_m]
) {{
{array_branches_text}
}}

template <class a_beat_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(
    ac_channel<a_beat_T> &a_stream,
    ac_channel<res_T> &res_stream
) {{
    typename CONFIG_T::bias_t *biases = CONFIG_T::gemm_bias();
{const_weights_branches_text}
}}

// Weight-stationary entry (hls4ml call site: nnet::gemm_array_const_weights<a_row_t,
// res_row_t, config>(a_rows, results), no bias -- it is read through the config
// (CONFIG_T::gemm_bias(), injected alongside the weight ROM) rather than a function
// argument, so it never becomes a port.
template <class a_beat_T, class res_T, typename CONFIG_T>
void gemm_array_const_weights(
    a_beat_T a_rows[CONFIG_T::gemm_m],
    res_T results[CONFIG_T::gemm_m]
) {{
    typename CONFIG_T::bias_t *biases = CONFIG_T::gemm_bias();
{array_const_weights_branches_text}
}}

template <class data_T, class weight_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_stream_sim(
    ac_channel<data_T> &data_stream,
    weight_T weights[CONFIG_T::n_in * CONFIG_T::n_out],
    bias_T biases[CONFIG_T::n_out],
    ac_channel<res_T> &res_stream
) {{
    typedef nnet::array<typename data_T::value_type, CONFIG_T::gemm_k> a_beat_t;
    typedef nnet::array<weight_T, CONFIG_T::gemm_k> b_beat_t;
    ac_channel<a_beat_t> a_beat_stream;
    data_T activation_rows[CONFIG_T::gemm_m];
    b_beat_t weight_cols[CONFIG_T::gemm_n];

    static_assert(CONFIG_T::transpose_weights,
                  "GEMM IP simulation helper expects transposed weight storage.");

    for (unsigned int row = 0; row < CONFIG_T::gemm_m; row++) {{
        activation_rows[row] = data_stream.read();
    }}

    for (unsigned int row = 0; row < CONFIG_T::gemm_m; row++) {{
        a_beat_t a_beat;
        for (unsigned int kk = 0; kk < CONFIG_T::gemm_k; kk++) {{
            a_beat[kk] = activation_rows[row][kk];
        }}
        a_beat_stream.write(a_beat);
    }}

    for (unsigned int col = 0; col < CONFIG_T::gemm_n; col++) {{
        for (unsigned int kk = 0; kk < CONFIG_T::gemm_k; kk++) {{
            weight_cols[col][kk] = weights[col * CONFIG_T::gemm_k + kk];
        }}
    }}

    gemm_ip_stream_buffered_b<a_beat_t, b_beat_t, bias_T, res_T, CONFIG_T>(
        a_beat_stream, weight_cols, biases, res_stream);
}}

}} // namespace nnet

#endif
"""


def gen_integration_manifest(items):
    """Items must have been through ``flow.normalize_config``: every fold
    field below is read as resolved there, never recomputed here."""
    cores = []
    for item in items:
        fold_axis = item["fold_axis"]
        mg = item["m_groups"]
        mp = item["m_passes"]
        rf_req = item["reuse_factor_requested"]
        rf = item["reuse_factor"]
        eff_rf = item["effective_reuse"]
        ks = item["k_spatial"]
        kp = item["k_passes"]
        kc_pad = item["k_chunks_pad"]
        mult = item["multipliers"]
        gr_pad = item["grid_rows_pad"]
        core_rows = mg * 8 if mp > 1 else item["m"]
        ng = item["n_groups"]
        np_ = item["n_passes"]
        gc_pad = item["grid_cols_pad"]
        core_cols = item["core_cols"]
        # Closed-form overlapped cycle model: predicted
        # latency/interval/frame-count for the DSE, valid for every fold_axis
        # (mp == np_ == 1 for fold_axis "k"/single-axis reduces to the
        # existing single-group model -- see geometry.combined_fold_cycles).
        cycles = _geometry.combined_fold_cycles(core_rows, item["k"], core_cols, ks,
                                                 m_passes=mp, n_passes=np_)
        cores.append({
            "name": item["name"],
            "interface": item.get("interface", "stream"),
            "protocol": item.get("protocol", {}),
            "entity": f"{item['name']}_core",
            "rtl": f"{item['name']}/{item['name']}_core.v",
            "m": item["m"],
            "k": item["k"],
            "n": item["n"],
            "reuse_factor_requested": rf_req,
            "reuse_factor": rf,
            "effective_reuse": eff_rf,
            "k_spatial": ks,
            "k_passes": kp,
            "k_chunks_pad": kc_pad,
            "multipliers": mult,
            "fold_axis": fold_axis,
            "m_groups": mg,
            "m_passes": mp,
            "grid_rows_pad": gr_pad,
            "core_rows": core_rows,
            "n_groups": ng,
            "n_passes": np_,
            "grid_cols_pad": gc_pad,
            "core_cols": core_cols,
            "predicted_cycles": {
                "frames": cycles["frames"],
                "first_out": cycles["first_out"],
                "interval": cycles["interval"],
                "latency": cycles["latency"],
                "total_cycles": cycles["total_cycles"],
            },
            "reset": {
                "name": "rst",
                "sync_active": "high"
            }
        })
    manifest = {
        "package_format": "single_top_catapult_blackboxes",
        "header": "gemm_ip_combined.h",
        "cores": cores
    }
    return json.dumps(manifest, indent=2)


def gen_blackbox_tcl(items):
    lines = ["# Generated by gemm-ip-gen for single-top blackbox integration"]
    for item in items:
        lines.append(f'solution file add [file join $script_dir {item["name"]} {item["name"]}_core.v] -type Verilog -exclude true')
    return "\n".join(lines) + "\n"


_CORE_PORTS = ("a_rows", "b_cols", "c_row")


def _assert_core_port_widths(name, header_text, grid_v, weights_in_core=False):
    """Cross-check the blackbox word widths between the generated C++ header and
    the generated grid RTL. Catapult wires a width-mismatched blackbox port
    without erroring and the simulator X-pads the missing bits (silent cosim
    corruption), so any disagreement must fail generation instead."""
    hdr = {}
    for w, port in re.findall(
            r"ac_int<(\d+), false>\s*&?\s*(%s)\b" % "|".join(_CORE_PORTS), header_text):
        hdr.setdefault(port, set()).add(int(w))
    rtl = {}
    for w, port in re.findall(
            r"\[(\d+):0\]\s*(%s)\b" % "|".join(_CORE_PORTS), grid_v):
        rtl.setdefault(port, set()).add(int(w) + 1)
    for port in _CORE_PORTS:
        # Weight-stationary drops b_cols from the ccore run() AND the top wrapper
        # (weights come from the ROM) — b_cols simply isn't a blackbox port in
        # this mode, so skip it.
        if weights_in_core and port == "b_cols":
            continue
        hw, rw = hdr.get(port, set()), rtl.get(port, set())
        if len(hw) != 1 or len(rw) != 1 or hw != rw:
            raise RuntimeError(
                f"{name}: blackbox port width mismatch for '{port}': C++ header "
                f"declares {sorted(hw)} bits, grid RTL declares {sorted(rw)} bits. "
                "The wrapper and the core were generated with inconsistent word "
                "layouts (full-K-spatial vs chunked)."
            )


def _check_operand_fits_int8_core(name, operand_label, precision):
    """Raise ValueError if `precision` cannot be losslessly read as the
    tensor_slice int8 core's ac_int<8,true> operand.

    Symmetric-only quantization scope: signed operands fit with width <= 8,
    and unsigned operands fit only when narrower than 8 bits (bit 7 is never
    set, so the code reads back unchanged). An unsigned exactly-8-bit
    operand would need a zero-point offset to keep bit 7 from being misread
    as the sign bit -- zero points are out of scope, so it is rejected.
    """
    bits = _operand_bits(precision)
    if bits is None:
        return
    width, signed = bits
    fits = width <= 8 and (signed or width < 8)
    if not fits:
        raise ValueError(
            f"{name}: {operand_label} precision '{precision}' does not fit the "
            "tensor_slice int8 core under the symmetric-only quantization "
            "scope: signed operands need width <= 8, unsigned operands need "
            "width < 8 (an unsigned 8-bit operand would need a zero-point "
            "offset, which is not supported)"
        )


def _resolve_axis_reuse_factors(name, fold_axis, reuse_factor,
                                 m_reuse_factor, k_reuse_factor, n_reuse_factor):
    """Merge the legacy single-axis ``fold_axis``/``reuse_factor`` knob with the
    independent per-axis ``m_reuse_factor``/``k_reuse_factor``/``n_reuse_factor``
    knobs into the ``(m_rf, k_rf, n_rf)`` triple :func:`resolve_mkn_geometry`
    expects.

    Precedence: any explicit per-axis knob (not ``None``) wins outright over
    the legacy pair. Legacy ``fold_axis``/``reuse_factor`` is only used when
    NONE of the per-axis knobs are given. Supplying both an explicit per-axis
    knob and a non-default legacy pair is not an error -- it prints a
    precedence warning naming which value wins -- so this stays permissive of
    ATLASConfig items that carry stale legacy fields.
    """
    explicit = {ax: v for ax, v in
                (("m", m_reuse_factor), ("k", k_reuse_factor), ("n", n_reuse_factor))
                if v is not None}
    fold_axis = str(fold_axis or "k").lower()
    legacy_active = fold_axis != "k" or int(reuse_factor) != 1
    if explicit and legacy_active:
        print(
            f"WARNING: {name}: both explicit per-axis fold knob(s) "
            f"({', '.join(f'{ax.upper()}Fold={v}' for ax, v in explicit.items())}) "
            f"and the legacy FoldAxis={fold_axis!r}/ReuseFactor={reuse_factor} were "
            "given; the explicit per-axis knob(s) win and the legacy pair is ignored "
            "for any axis not named above.",
            file=sys.stderr,
        )
    if explicit:
        return (explicit.get("m", 1), explicit.get("k", 1), explicit.get("n", 1))
    # Legacy path: fold_axis picks which single axis reuse_factor legalizes;
    # this degenerates to resolve_mkn_geometry's single-axis form exactly.
    if fold_axis == "m":
        return (reuse_factor, 1, 1)
    if fold_axis == "n":
        return (1, 1, reuse_factor)
    return (1, reuse_factor, 1)


def generate_catapult_pkg(m, k, n, name, output_dir, interface="stream", output_precision=None,
                          reuse_factor=1, input_precision=None, weight_precision=None,
                          clock_period_ns=None, n_frames=1, weight_matrix=None, fold_axis="k",
                          m_reuse_factor=None, k_reuse_factor=None, n_reuse_factor=None,
                          accum_precision=None, bias_precision=None, has_bias=None, bias=None,
                          **_ignored):
    if interface not in ("stream", "array"):
        raise ValueError(f"Unsupported GEMM interface '{interface}' for {name}; expected stream or array")
    _check_operand_fits_int8_core(name, "input_precision", input_precision)
    _check_operand_fits_int8_core(name, "weight_precision", weight_precision)
    # Symmetric-only quantization scope: zero points are rejected outright
    # (unsigned exactly-8-bit precisions already raise in the width guard
    # above, so no zero-point offset can arise here).
    a_zero_point = 0
    b_zero_point = 0
    fold_axis = str(fold_axis).lower()
    m_rf, k_rf, n_rf = _resolve_axis_reuse_factors(
        name, fold_axis, reuse_factor, m_reuse_factor, k_reuse_factor, n_reuse_factor)
    resolved = resolve_mkn_geometry(m, k, n, m_reuse_factor=m_rf, k_reuse_factor=k_rf,
                                     n_reuse_factor=n_rf, name=name)
    for w in resolved["warnings"]:
        print(w, file=sys.stderr)
    k_spatial = resolved["k_spatial"]
    passes = resolved["passes"]
    m_passes = resolved["m_passes"]
    n_passes = resolved["n_passes"]
    fold_m = m_passes > 1
    fold_n = n_passes > 1
    fold_k = passes > 1
    # Combined M/K/N folding: 2+ folded axes now route to the general structural synth
    # emitter (`_general_synth_combined_fold`, dispatched from
    # `_generate_general_synth_verilog`/`generate_k_spatial_synth_verilog`
    # whenever 2+ of {m_passes, k time-passes, n_passes} > 1) instead of the
    # single-axis emitters below. What remains genuinely unsupported is a
    # folded spatial factor (m_spatial/n_spatial) that overflows the 5-bit
    # a_loc/b_loc ports (cap 32) -- the combined emitter itself raises a clear
    # ValueError for that; it is allowed to propagate as-is (not caught or
    # rewrapped here) rather than let it surface as some obscurer downstream
    # failure.
    folded_axes = [ax for ax, folded in (("M", fold_m), ("K", fold_k), ("N", fold_n)) if folded]
    combined_fold = len(folded_axes) >= 2
    rf_legalized = resolved["reuse_factor"]
    # ``core_m`` is the RTL/csim-core row count: M_g = 8*m_spatial when fold-M
    # issues more than one frame (m_passes frames of core_m rows assemble the
    # logical M rows), otherwise the logical M itself, so a single-frame
    # package is today's shape whichever axis was named. fold_m/fold_n/fold_k
    # are independent knobs -- 2+ can be true at once (combined_fold above).
    core_m = resolved["m_spatial"] * 8 if m_passes > 1 else m
    # ``core_n`` is the RTL/csim-core column count: N_g = 8*n_spatial when
    # fold-N issues more than one frame (n_passes frames of core_n columns
    # assemble the logical N columns), otherwise the logical N itself.
    core_n = resolved["n_spatial"] * 8 if n_passes > 1 else n
    # Weight-stationary (const-weight) variant: weights (B, shape [K, N]) baked into
    # the core ROM AND the csim header; the wrapper takes no external weight port and
    # the header entry takes A only. Single source of truth = weight_matrix. Works at
    # every k_spatial/passes combination -- the ROM builder packs the same beats the
    # generalized narrow/chunked feed loops expect.
    weights_in_core = weight_matrix is not None
    weight_rom = None
    if weights_in_core:
        from gemm_ip.weights import (
            build_weight_rom_k_spatial, build_weight_rom_fold_n, build_weight_rom_combined_fold,
        )
        if combined_fold:
            # Combined fold (2+ axes): weight contents depend only on
            # (k_pass, n_group), never m_group (confirmed contract + csim
            # simplification -- see _general_synth_combined_fold's docstring),
            # so the same n_passes*core_n padded-B layout as fold-N feeds the
            # combined builder; it re-orders into the chunk-major/ng-minor
            # layout the combined emitter's ROM address expects.
            import numpy as _np
            n_full = n_passes * core_n
            b_full = _np.zeros((k, n_full), dtype=_np.asarray(weight_matrix).dtype)
            b_full[:, :n] = weight_matrix
            weight_rom = build_weight_rom_combined_fold(b_full, core_m, core_n, k, k_spatial, n_passes)
            expected_beats = passes * n_passes * core_n
        elif n_passes > 1:
            # Fold-N: the ROM holds every group's columns back to back (group
            # g's block at base g*core_n, padded tail columns zero -- weight_
            # matrix's own N may be smaller than n_passes*core_n). Built per
            # group so each word is core_n wide (see build_weight_rom_fold_n).
            import numpy as _np
            n_full = n_passes * core_n
            b_full = _np.zeros((k, n_full), dtype=_np.asarray(weight_matrix).dtype)
            b_full[:, :n] = weight_matrix
            weight_rom = build_weight_rom_fold_n(b_full, core_m, core_n, k, k_spatial, n_passes)
            expected_beats = n_full
        else:
            weight_rom = build_weight_rom_k_spatial(weight_matrix, core_m, core_n, k, k_spatial)
            expected_beats = passes * core_n
        if len(weight_rom) != expected_beats:
            raise RuntimeError(
                f"{name}: weight ROM has {len(weight_rom)} beats, expected "
                f"{expected_beats}"
            )
    # Two-stage requant. S1 (in-slice pre-round) is derived from accum_precision when given; S2
    # is the remainder of the total gemm->result shift. The output lane
    # narrows to out_width = _output_bits(output_precision) -- 8 bits when
    # output_precision is unset (the legacy/no-result-type case: this matches
    # _output_bits()'s own existing fallback and rtl.py's own out_width
    # default, so an un-annotated package keeps today's already-established
    # 8-bit lane rather than reverting to a 16-bit one).
    out_bits = _output_bits(output_precision)
    _gemm_shift = _frac_bits(input_precision) + _frac_bits(weight_precision)
    _out_frac = _frac_bits(output_precision)
    # No zero-fraction guard: a result with no fraction bits still needs the
    # whole product fraction shifted out (the old guard fed a legacy path in
    # which the drain rescaled; the drain is a pure unpack now).
    _total_shift = _gemm_shift - _out_frac
    if _total_shift < 0:
        raise ValueError(
            f"{name}: output_precision {output_precision!r} carries {_out_frac} "
            f"fraction bits, more than the {_gemm_shift} the product carries; a "
            "left shift is not supported by the tensor_slice requant.")
    _accum_w = _accum_shift_bits(accum_precision, _gemm_shift)
    s1 = max(0, _accum_w - 16) if _accum_w else 0
    if s1 > _total_shift:
        raise RuntimeError(
            f"{name}: accum_precision needs S1={s1} bits of in-slice pre-rounding, "
            f"more than the total gemm->result shift ({_total_shift}); the output "
            "needs more than 16 bits of range at the gemm scale.")
    # The slice rounds half-up for any non-zero shift_amount and passes the raw
    # low 16 bits through at 0, so a single exact requant is reachable two ways:
    #   - S1 == 0: the wrapper (stage 2) does the whole shift in the result type's
    #     own mode, round-half-up for RND, floor for TRN;
    #   - S1 > 0 on an RND result: the whole shift moves into the slice (S1 = T,
    #     S2 = 0) so it rounds once. The bias is then added at the output scale, so
    #     this needs a bias-free layer or a bias no finer than the result.
    # Anything else keeps the two-stage path, which can differ by one output LSB.
    _trn = _truncates(output_precision)
    _bias_fine = bool(has_bias) and _frac_bits(bias_precision) > _out_frac
    if s1 > 0 and not _trn and not _bias_fine:
        s1 = _total_shift
    elif s1 > 0:
        _why = ("a truncating (TRN) result" if _trn
                else "a bias finer than the result scale")
        print(f"WARNING: {name}: accum_t forces S1={s1} (in-slice round-half-up) on "
              f"{_why}; the requant cannot be a single exact step and may differ "
              "from the reference by one output LSB.", file=sys.stderr)
    if s1 > 0 and int(k_spatial) > 1:
        print(f"WARNING: {name}: S1={s1} with k_spatial={k_spatial}: every K partition "
              "is rounded on its own before the 16-bit partials are summed, which may "
              "differ from the reference by one output LSB. Fold K fully in time "
              "(k_spatial 1) for an exact result.", file=sys.stderr)
    s2 = _total_shift - s1
    # gen_inst_cpp/gen_tb's own self-check reference is a single-round
    # formula (gemm_acc rounded once by the TOTAL shift). Identical to the
    # two-stage DUT when S1 == 0; expected to disagree by <= 1 output LSB when
    # S1 > 0 (decision 7) -- a loud measured mismatch, not a silent bug.
    requant_shift = _total_shift

    # Bake the bias (decision 4): baked at the intermediate scale (gemm_frac
    # - S1, the scale stage 2 adds it at), one 16-bit signed code per column,
    # via the SAME shared helper mvau uses (gemm_ip.biasrom), so the C twin
    # and the Verilog ROM render from one codes list.
    from gemm_ip.biasrom import bias_acc_codes as _bias_acc_codes
    _has_bias = bool(has_bias)
    _intermediate_frac = max(0, _gemm_shift - s1)
    bias_codes = None
    if _has_bias:
        if fold_n:
            # Fold-N replays the SAME physical core_n-wide core across
            # n_passes column groups; the bias ROM is baked n_passes*core_n
            # wide (like the weight ROM's n_full), group g's REAL columns at
            # base g*core_n, zero-padded tail -- the RTL's out_grp-indexed
            # lookup (frozen per-frame, not the live/already-advanced group
            # counter) selects the right slice at emit time.
            _real_codes = _bias_acc_codes(bias, _intermediate_frac, n, True)
            bias_codes = [0] * (n_passes * core_n)
            bias_codes[:n] = _real_codes
        else:
            bias_codes = _bias_acc_codes(bias, _intermediate_frac, core_n, True)
        _bad = [c for c in bias_codes if not (-32768 <= c <= 32767)]
        if _bad:
            raise RuntimeError(
                f"{name}: bias code(s) {_bad} do not fit the 16-bit stage-2 "
                f"intermediate scale (2^{_intermediate_frac} fractional bits) -- "
                "reduce the bias magnitude or accum_precision's fractional bits.")
    if _trn and s2 > 0:
        # Truncating result: floor(x / 2^S2) == round_half_up(x - 2^(S2-1), S2), and
        # stage 2 already adds a baked 16-bit wrapping constant before its round, so
        # the half goes into the bias codes (one list feeds the ROM and the C twins).
        _half2 = 1 << (s2 - 1)
        _codes = bias_codes if bias_codes is not None else [0] * (n_passes * core_n if fold_n else core_n)
        bias_codes = [((int(c) - _half2 + 32768) % 65536) - 32768 for c in _codes]

    pkg_dir = Path(output_dir) / name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # ``grid_rows``/``grid_cols``/RTL-core generation are sized on the CORE
    # row/column counts (core_m == m, core_n == n under fold_axis="k"; core_m
    # == M_g under fold_axis="m"; core_n == N_g under fold_axis="n").
    grid_rows = (core_m + 7) // 8
    grid_cols = (core_n + 7) // 8

    # The combined core lives in the rtl sibling module (structural-only,
    # single-branch RTL).
    from . import rtl as _rtl
    generate_combined_core_verilog = _rtl.generate_combined_core_verilog
    generate_k_spatial_combined_core_verilog = _rtl.generate_k_spatial_combined_core_verilog
    if combined_fold:
        # Combined M/K/N fold (2+ axes): route to the general structural synth
        # emitter (`_general_synth_combined_fold`) via the public
        # `generate_k_spatial_synth_verilog` wrapper -- it already threads
        # m_passes/logical_m/logical_n (added additively in sub-phase 2b) and
        # works at k_spatial==1 too. Unlike
        # `generate_combined_core_verilog`'s ROM-hoisting split-module trick,
        # each combined-fold branch here emits its OWN ROM(s) as a whole
        # module -- the same simpler pattern
        # `generate_k_spatial_combined_core_verilog` already uses for its
        # (also experimental/structural) k_spatial>1 body.
        core_module = f"{name}_core"
        synth_top = _rtl.generate_k_spatial_synth_verilog(
            core_m, k, core_n, module_name=core_module, k_spatial=k_spatial,
            n_passes=n_passes, s1=s1, s2=s2, out_width=out_bits,
            weight_rom=weight_rom, emit_rom=True,
            bias_codes=bias_codes, emit_bias_rom=True,
            m_passes=m_passes, logical_m=m, logical_n=n,
        )
        grid_v = (
            f"// Auto-generated by package.py (combined M/K/N fold)\n"
            f"// Combined-fold core: M={m}, K={k}, N={n} (core_m={core_m}, core_n={core_n}, "
            f"m_passes={m_passes}, k_spatial={k_spatial}, n_passes={n_passes}, "
            f"folded_axes={','.join(folded_axes)})\n"
            "//   structural synth wrapper (_general_synth_combined_fold), "
            "single-branch RTL\n"
            "\n" + synth_top + "\n"
        )
    elif k_spatial == 1:
        grid_v = generate_combined_core_verilog(core_m, k, core_n, module_name=f"{name}_core",
                                                out_width=out_bits, s1=s1, s2=s2,
                                                weight_rom=weight_rom, n_passes=n_passes,
                                                bias_codes=bias_codes)
    else:
        print(
            f"WARNING: {name}: ReuseFactor={rf_legalized} partitions K into "
            f"k_spatial={k_spatial} parallel chunks; tensor-slice partial outputs "
            "are INT16 and partial overflow is possible. Correctness depends on "
            "quantized operand ranges and partition size.",
            file=sys.stderr,
        )
        grid_v = generate_k_spatial_combined_core_verilog(
            core_m, k, core_n, module_name=f"{name}_core", k_spatial=k_spatial,
            out_width=out_bits, s1=s1, s2=s2, weight_rom=weight_rom,
            n_passes=n_passes, bias_codes=bias_codes,
        )
    header_text = gen_public_header(
        name, core_m, k, core_n, grid_rows, grid_cols,
        result_type=output_precision,
        k_spatial=k_spatial,
        input_precision=input_precision,
        weight_precision=weight_precision,
        clock_period_ns=clock_period_ns,
        n_frames=(m_passes if fold_m else (n_passes if fold_n else n_frames)),
        weight_rom=weight_rom,
        m_passes=m_passes,
        logical_m=m,
        n_passes=n_passes,
        logical_n=n,
        out_width=out_bits,
        s1=s1,
        bias_codes=bias_codes,
        a_zero_point=a_zero_point,
        b_zero_point=b_zero_point,
    )
    _assert_core_port_widths(name, header_text, grid_v, weights_in_core=weights_in_core)
    (pkg_dir / f"{name}_core.v").write_text(grid_v)
    (pkg_dir / "nnet_types.h").write_text(gen_nnet_types_header())
    (pkg_dir / f"{name}_gemm_ip.h").write_text(header_text)
    # The inst wrapper, testbench and tcl see the logical shape: the header
    # asserts CONFIG_T::gemm_m == m and the fold-M frames are internal to one
    # call, so they never see core_m or m_passes.
    (pkg_dir / f"{name}_inst.cpp").write_text(gen_inst_cpp(
        name, m, k, n, interface,
        requant_shift=requant_shift,
        weight_matrix=weight_matrix, out_width=out_bits,
    ))
    (pkg_dir / f"{name}_tb.cpp").write_text(gen_tb(
        name, m, k, n, interface, n_frames=n_frames,
        requant_shift=requant_shift,
        weight_matrix=weight_matrix, out_width=out_bits,
        bias_codes=(bias_codes[:n] if bias_codes is not None else None),
        s1=s1, s2=s2, k_spatial=k_spatial,
    ))
    (pkg_dir / "run_catapult.tcl").write_text(
        gen_tcl(name, m, k, n, interface, weights_in_core=weight_matrix is not None,
                clock_period_ns=clock_period_ns))
    print(f"Generated {pkg_dir}  (M={m}, K={k}, N={n}, interface={interface}, "
          f"reuse_factor={rf_legalized}, k_spatial={k_spatial}, fold_axis={fold_axis}, "
          f"m_passes={m_passes}, n_passes={n_passes})")
