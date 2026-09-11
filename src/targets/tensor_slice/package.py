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
from gemm_ip.quant import _frac_bits, _operand_bits, _output_bits, _accum_shift_bits

# geometry is a sibling target module; put this dir on the path and import by
# name (the same idiom the RTL loaders below use).
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)
from geometry import (  # noqa: E402
    LANE_WIDTH, _ceil_div, k_chunks as _geom_k_chunks, k_passes as _geom_k_passes,
    resolve_reuse_factor, latency_first_out, a_stream_width, b_stream_width,
)

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

    Delegates to ``geometry.latency_first_out`` so the sim-Verilog FIRST_OUT
    constant and this C++-header-facing count always agree on the same
    number for a given ``k_spatial`` (number of K partitions fed per pass;
    ``k_spatial == 1`` is the chunked endpoint, ``k_spatial == k_chunks`` is
    the full-K endpoint).
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
                      n_passes=1, logical_n=None, out_width=16, bias_codes=None, s1=0):
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
    mr = grid_rows * 8
    ks = int(k_spatial)
    k_chunks = _geom_k_chunks(k)
    passes = _geom_k_passes(k, ks)
    # ``k_spatial == 1`` keeps today's grid-padded word width (chunked
    # endpoint). ``k_spatial > 1`` is the narrow K-spatial word: 64*k_spatial
    # bits per beat, independent of the row/col tile count, replayed across
    # ``passes`` sweeps of K. This is a single general layout: the chunked
    # (k_spatial == 1, passes == k_chunks) and full-K (k_spatial == k_chunks,
    # passes == 1) cases are just its two endpoints.
    a_bits = a_stream_width(m, ks)
    b_bits = b_stream_width(n, ks)
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
    input_beats = max(m, n)
    total_beats = passes * input_beats
    first_out = latency_cycles(m, k, n, grid_rows, grid_cols, k_spatial=ks)
    # Frame slots for the pipelined sim core: feed of frame t+1 may overlap
    # compute/drain of frame t (min frame period = total_beats + 1 calls).
    slots = -(-(first_out + 1 + m) // (total_beats + 1)) + 1

    # Merged feed+drain call budget. The RUN loop polls out_valid on every
    # call, so it absorbs the core's port lag. The frame's last row sits at
    # run()-call index first_out + (m-1) + 2 (preload call + clk_cnt->call
    # offset), and the worst port lag is 3 calls (sim branch: 2 registered
    # stages; structural branch adds an input register) — plus 2 calls spare.
    run_calls = first_out + m + 6
    if run_calls < total_beats + 2:
        raise RuntimeError(
            f"{name}: GEMM wrapper RUN budget shorter than the feed itself "
            f"(run_calls={run_calls}, total_beats={total_beats}; m={m} k={k} n={n})."
        )

    # Back-to-back multi-frame schedule. No BIAS preload step: bias is a
    # compile-time constant baked into the core (decision 4), not a runtime
    # port. Each frame is ONE in_valid=0
    # beat (p == 0, carrying a live preload_valid pulse - see the RUN loop comment)
    # followed by total_beats in_valid beats; the idle beat drops the core's
    # `feeding` flag so the next frame allocates a fresh slot.
    # Steady-state frame period = total_beats + 1; the last frame's outputs drain
    # in the trailing first_out + mr + slack tail. With n_frames == 1 this reduces
    # to a single frame (real hls4ml flow: one frame per wrapper call).
    period = total_beats + 1
    # in_valid is asserted only inside the feed region; feed_total covers every
    # frame's total_beats feed cycles plus its trailing 1-cycle separator.
    feed_total = n_frames * period
    # Loop length: the LAST frame starts feeding at (n_frames-1)*period and its
    # final output lands first_out + m later (the behavioral core emits m rows per
    # frame). Use mr (>= m) + slack for the port-lag tail. Sizing on feed_total
    # here would over-run by a full period per frame (a serialized-latency
    # regression for n_frames == 1).
    total_steps = (n_frames - 1) * period + first_out + mr + 6
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
    # Requantise ONCE, after the whole contraction is accumulated (in-slice and
    # cross-chunk folded together in the behavioural model). The accumulator
    # carries frac = frac_a + frac_b; the result lane carries frac(out), so the
    # single shift is their difference. requant_shift == 0 -> legacy path (core
    # emits the raw accumulator and the drain does the whole rescale).
    _out_frac = _frac_bits(result_type)
    requant_shift = max(0, gemm_shift - _out_frac) if _out_frac else 0
    requant_bits = _output_bits(result_type) if requant_shift else None
    drain_shift = _out_frac if requant_shift else gemm_shift
    # Two-stage requant (jojo-track/open/tensor-slice-bias-in-rtl, phase 1):
    # this ccore mirrors the Verilog sim branch's folded model exactly: stage
    # 1 (round-half-up shift by S1, wrap to 16) applied once to the exact
    # full sum, then stage 2 (wrap-add the bias at the 16-bit intermediate
    # scale, round-half-up shift by S2, wrap to the physical lane). No
    # saturation anywhere -- decision 5. `requant_shift` here is the TOTAL
    # shift (S1 + S2); S2 is the remainder after S1.
    _s1 = int(s1) if s1 else 0
    _s2 = max(0, requant_shift - _s1)
    _half1 = (1 << (_s1 - 1)) if _s1 > 0 else 0
    _half2 = (1 << (_s2 - 1)) if _s2 > 0 else 0
    core_requant_emit = (
        f"                        ac_int<{out_width}, true> sat_val;\n"
        "                        {\n"
        "                            // Stage 1: round-half-up shift the exact sum by S1, wrap to 16.\n"
        f"                            ac_int<33, true> _r1 = (ac_int<33, true>) acc + {_half1};\n"
        f"                            ac_int<16, true> _p1 = (ac_int<16, true>) (_r1 >> {_s1});\n"
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
        if ks > 1
        else f"a_buf[s][k_chunk * {input_beats} + actual_row].slc<8>(row_tile * 64 + k_lane * 8)"
    )
    # b_buf is sized passes*n (only real columns t < n are ever captured — see the
    # capture block below), so its index scales by n, not input_beats, unlike a_buf
    # (which keeps the full input_beats stride: every beat t < m carries a real row).
    b_el_expr = (
        f"b_buf[s][(k_chunk / {ks}) * {n} + actual_col]"
        f".slc<8>((k_chunk % {ks}) * 64 + k_lane * 8)"
        if ks > 1
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
    # Under fold-N the external weight beats of frame g are group g's columns
    # (the array feed under fold-N has its own override further down).
    b_col_idx = f"g * {n} + t" if fold_n else "t"
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
        bbuf_src = "B_ROM[_bidx + _grp * %d]" % n if fold_n else "B_ROM[_bidx]"
        brom_decl = _const_weights_brom_cpp(b_bits, grid_cols, weight_rom)
        stream_bcols_decl = ""
        # Weight-stationary: the IP holds B, so the feed packs no B beat at all.
        stream_bcols_pack = ""
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
        if ks > 1:
            stream_bcols_pack = f"""if (feeding_now && t < {n}) {{
            b_beat_T b_beat = weight_cols[{b_col_idx}];
            #pragma hls_unroll
            COL_PACK_KC: for (int kc_local = 0; kc_local < {ks}; kc_local++) {{
                #pragma hls_unroll
                COL_PACK_KL: for (int kl = 0; kl < 8; kl++) {{
                    int kk = (kc * {ks} + kc_local) * 8 + kl;
                    if (kk < {k}) {{
                        b_cols.set_slc(kc_local * 64 + kl * 8,
                                       {name}_to_gemm_int8(b_beat[kk]));
                    }}
                }}
            }}
        }}"""
        else:
            stream_bcols_pack = f"""if (feeding_now && t < {n}) {{
            b_beat_T b_beat = weight_cols[{b_col_idx}];
            #pragma hls_unroll
            COL_PACK: for (int kl = 0; kl < 8; kl++) {{
                int kk = kc * 8 + kl;
                int col_tile = t / 8;
                #pragma hls_unroll
                COL_TILE_CHUNK: for (int ct = 0; ct < {grid_cols}; ct++) {{
                    if (col_tile == ct && kk < {k}) {{
                        b_cols.set_slc(ct * 64 + kl * 8,
                                       {name}_to_gemm_int8(b_beat[kk]));
                    }}
                }}
            }}
        }}"""
        # Array feed loop, two-stream: pack the external weight_cols beat into b_cols_packed.
        array_bcols_decl = f"\n        ac_int<{b_bits}, false> b_cols_packed = 0;"
        array_bcols_run_arg = "b_cols_packed, "
        if ks > 1:
            array_bcols_pack = f"""
        if (step > 0 && step <= {total_beats} && t < {n}) {{
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
        if (step > 0 && step <= {total_beats} && t < {n}) {{
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
    def _a_pack_block(dest, pass_expr, label):
        if ks > 1:
            return f"""
                #pragma hls_unroll
                {label}_KC: for (int kc_local = 0; kc_local < {ks}; kc_local++) {{
                    #pragma hls_unroll
                    {label}_KL: for (int kl = 0; kl < 8; kl++) {{
                        int kk = (({pass_expr}) * {ks} + kc_local) * 8 + kl;
                        if (kk < {k}) {{
                            {dest}.set_slc(kc_local * 64 + kl * 8,
                                           {name}_to_gemm_int8(a_beat[kk]));
                        }}
                    }}
                }}"""
        return f"""
                #pragma hls_unroll
                {label}_KL: for (int kl = 0; kl < 8; kl++) {{
                    int kk = ({pass_expr}) * 8 + kl;
                    int row_tile = t / 8;
                    #pragma hls_unroll
                    {label}_RT: for (int rt = 0; rt < {grid_rows}; rt++) {{
                        if (row_tile == rt && kk < {k}) {{
                            {dest}.set_slc(rt * 64 + kl * 8,
                                           {name}_to_gemm_int8(a_beat[kk]));
                        }}
                    }}
                }}"""

    if passes >= 2:
        a_replay_decl = f"""
    // Replay storage is packed to the blackbox protocol. HLS4ML still emits
    // each logical K-wide A row once; later K passes replay packed slices.
    // Reused per frame (written at each frame's kc==0 beats, read within the
    // same frame's later passes — the feed is sequential in step order).
    // Slot 0 (pass 0, fed directly from the stream) is never stored, so the
    // buffer holds only the {passes - 1} replayed passes, each M rows wide.
    ac_int<{a_bits}, false> a_replay[{passes - 1}][{m}];
"""
        a_prepack_replay = f"""
                #pragma hls_unroll
                PREPACK_REPLAY: for (int replay_kc = 1; replay_kc < {passes}; replay_kc++) {{
                    ac_int<{a_bits}, false> replay_rows = 0;{_a_pack_block("replay_rows", "replay_kc", "ROW_PACK_REPLAY")}
                    a_replay[replay_kc - 1][t] = replay_rows;
                }}"""
        a_replay_else = f"""
            }} else {{
                a_rows = a_replay[kc - 1][t];
            }}"""
    else:
        a_replay_decl = ""
        a_prepack_replay = ""
        a_replay_else = """
            }"""

    # Fold-N raw output capture: shared by the stream and array RUN loops.
    # Every frame emits M REAL rows (M is fully spatial under fold_axis="n"),
    # so `captured` needs no padding guard beyond the plain n_passes*m total;
    # only the group's N_g columns are stored, at that group's column offset
    # -- the emission loop after RUN assembles/rescales/biases the full
    # logical_n-wide rows once every group has landed.
    capture_fold_n = f"""\
        if (v) {{
            if (captured < {n_passes * m}) {{
                int gOut = captured / {m};
                int rowOut = captured % {m};
                #pragma hls_unroll
                for (int col = 0; col < {n}; col++) {{
                    int col_tile = col / 8;
                    int col_local = col % 8;
                    c_buf[rowOut][gOut * {n} + col] =
                        c_row.template slc<{out_width}>(col_tile * {8 * out_width} + col_local * {out_width});
                }}
            }}
            captured++;
        }}"""

    _a_read_guard = "kc == 0"
    _stream_feed_cond = f"feeding_now && t < {m}"
    _stream_g_decl = ""
    _stream_capture_use = stream_capture_b2b
    if fold_m:
        # Fold-M: gate the per-frame read against the logical row bound too
        # (frame g's beat t is global row g*M_g + t; only real rows are read
        # off a_stream -- there are exactly `logical_m` of them, not
        # m_passes*M_g). Padding rows feed zero A. Capture uses the
        # single-frame-style body (bound `logical_m`, a plain monotonic
        # counter across every frame), which already drops the last frame's
        # trailing padding pulses in emission order.
        _stream_g_decl = "\n        int g = step / %d;" % period
        _stream_feed_cond = f"feeding_now && t < {m} && (g * {m} + t) < {logical_m}"
        _stream_capture_use = stream_capture
    elif fold_n:
        # Fold-N: A rows are IDENTICAL across every frame (same M rows, only
        # the B columns / output group change), but the stream is single-read
        # -- frame 0 stores the rows as they are read (one shared slot: every
        # later frame replays the exact same value, so a_replay does not need
        # a per-frame copy the way the K-pass replay does). Capture is the
        # raw group-scoped c_buf write above.
        a_replay_decl = f"""
    // Fold-N: A rows repeat identically across every frame (only B's group
    // and the output group change); frame 0 stores them as they are read off
    // the stream (single-read ac_channel), frames >= 1 replay the same slot.
    ac_int<{a_bits}, false> a_replay_n[{m}];
"""
        a_prepack_replay = f"""
                a_replay_n[t] = a_rows;"""
        a_replay_else = f"""
            }} else {{
                a_rows = a_replay_n[t];
            }}"""
        _a_read_guard = "g == 0"
        _stream_g_decl = "\n        int g = step / %d;" % period
        _stream_capture_use = capture_fold_n

    stream_feed_loop = f"""
{a_replay_decl}
    // Back-to-back feed of {n_frames} frame(s): M A rows + N B columns, each
    // pass carrying k_spatial={ks} K chunks (passes={passes} sweeps of K).
    // Each frame is ONE in_valid=0 beat (p == 0, carrying the preload pulse)
    // + {total_beats} in_valid beats (period {period}); bias is a
    // compile-time constant baked into the core (decision 4). Every step polls
    // out_valid, so rows are captured as they emerge — frame t+1 feeds while
    // frame t drains in the core's FRAME_SLOTS.
    #pragma hls_pipeline_init_interval 1
    RUN: for (int step = 0; step < {total_steps}; step++) {{
        bool in_feed = (step < {feed_total});
        int p = in_feed ? (step % {period}) : {period};{_stream_g_decl}
        bool feeding_now = in_feed && (p >= 1) && (p <= {total_beats});
        int pf = p - 1;
        int kc = pf / {input_beats};
        int t = pf % {input_beats};
        ac_int<{a_bits}, false> a_rows = 0;
        {stream_bcols_decl}

        if ({_stream_feed_cond}) {{
            if ({_a_read_guard}) {{
                a_beat_T a_beat = a_stream.read();{_a_pack_block("a_rows", "0", "ROW_PACK_DIRECT")}{a_prepack_replay}{a_replay_else}
        }}
        {stream_bcols_pack}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = feeding_now ? 1 : 0;
        // preload_valid pulses on each frame's leading beat (p==0) so the
        // structural core's S_IDLE->S_PRELOAD->S_RUN arm is a live, non-constant
        // signal. Without it (literal 0) VTR synthesis proves transaction_active,
        // hence the tensor_slice result path, dead and prunes every slice. This
        // reuses the per-frame idle beat (formerly a trailing separator -> now a
        // leading preload, same period); bias is baked into the core, not fed here.
        ac_int<1, false> frame_preload = (in_feed && p == 0) ? 1 : 0;
        gemm.run(a_rows, {bcols_run_arg}frame_preload, feed_valid, c_row, v, l);
{_stream_capture_use}
    }}
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
    // Merged feed+drain: one run() call per cycle. Steps 0..{total_beats} preload then
    // feed M A rows (and, two-stream, N B columns), each pass carrying
    // k_spatial={ks} K chunks (passes={passes} sweeps of K). Every step polls
    // out_valid, so the frame's rows are captured as they emerge, not in a drain loop.
    #pragma hls_pipeline_init_interval 1
    RUN_ARRAY: for (int step = 0; step < {run_calls}; step++) {{
        int eff_step = (step == 0 || step > {total_beats}) ? 0 : step - 1;
        int kc = eff_step / {input_beats};
        int t = eff_step % {input_beats};
        ac_int<{a_bits}, false> a_rows_packed = 0;{array_bcols_decl}

        if (step > 0 && step <= {total_beats} && t < {m}) {{
            a_beat_T a_beat = a_rows[t];{_a_pack_block("a_rows_packed", "kc", "ROW_PACK_ARRAY")}
        }}{array_bcols_pack}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = (step >= 1 && step <= {total_beats}) ? 1 : 0;
        ac_int<1, false> feed_preload_valid = (step == 0) ? 1 : 0;
        gemm.run(a_rows_packed, {array_bcols_run_arg}feed_preload_valid, feed_valid, c_row, v, l);
{array_capture}
    }}
"""

    if fold_m:
        if weights_in_core:
            fold_array_bcols_pack = ""
        elif ks > 1:
            fold_array_bcols_pack = f"""
        if (feeding_now && t < {n}) {{
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
            fold_array_bcols_pack = f"""
        if (feeding_now && t < {n}) {{
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
        # Fold-M array loop: structurally the same frame schedule as the stream
        # RUN loop above (period/feed_total/total_steps), sourcing A directly
        # from a_rows[] instead of a channel read, gated the same way (frame
        # g's beat t is global row g*M_g + t; only rows < logical_m are real).
        array_feed_loop = f"""
    // Fold-M multi-frame feed: {m_passes} frames of {m} core rows each (K and N
    // fully spatial, k_spatial={ks}), issued back-to-back like the stream RUN
    // loop. Frame g reads logical rows [g*{m}, (g+1)*{m}) directly from
    // a_rows; rows at/after the logical M bound are the last frame's padding
    // (fed zero A, dropped by the captured < {logical_m} guard in the capture body).
    #pragma hls_pipeline_init_interval 1
    RUN_ARRAY: for (int step = 0; step < {total_steps}; step++) {{
        bool in_feed = (step < {feed_total});
        int p = in_feed ? (step % {period}) : {period};
        int g = step / {period};
        bool feeding_now = in_feed && (p >= 1) && (p <= {total_beats});
        int pf = p - 1;
        int kc = pf / {input_beats};
        int t = pf % {input_beats};
        ac_int<{a_bits}, false> a_rows_packed = 0;{array_bcols_decl}

        if (feeding_now && t < {m} && (g * {m} + t) < {logical_m}) {{
            a_beat_T a_beat = a_rows[g * {m} + t];{_a_pack_block("a_rows_packed", "kc", "ROW_PACK_ARRAY_FOLD")}
        }}
        {fold_array_bcols_pack}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = feeding_now ? 1 : 0;
        ac_int<1, false> feed_preload_valid = (in_feed && p == 0) ? 1 : 0;
        gemm.run(a_rows_packed, {array_bcols_run_arg}feed_preload_valid, feed_valid, c_row, v, l);
{array_capture}
    }}
"""
    elif fold_n:
        if weights_in_core:
            fold_n_array_bcols_pack = ""
        elif ks > 1:
            fold_n_array_bcols_pack = f"""
        if (feeding_now && t < {n}) {{
            b_beat_T b_beat = weight_cols[g * {n} + t];
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
            fold_n_array_bcols_pack = f"""
        if (feeding_now && t < {n}) {{
            b_beat_T b_beat = weight_cols[g * {n} + t];
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
        # Fold-N array loop: same multi-frame schedule as the fold-N stream RUN
        # loop above (period/feed_total/total_steps). A rows come straight from
        # a_rows[] (random access -- no replay buffer needed, unlike the
        # channel-fed stream entry); every frame re-reads the SAME M rows, and
        # group g's B columns are restricted by beat index (two-stream) or by
        # the group-scoped ROM (const weights). Capture is the raw group-scoped
        # c_buf write; the drain/rescale/bias/cast is a separate loop after RUN.
        array_feed_loop = f"""
    // Fold-N multi-frame feed: {n_passes} frames of {m} rows each (core
    // N_g={n} columns per frame, K and M fully spatial, k_spatial={ks}),
    // issued back-to-back. Every frame re-reads the same M rows off a_rows[].
    #pragma hls_pipeline_init_interval 1
    RUN_ARRAY: for (int step = 0; step < {total_steps}; step++) {{
        bool in_feed = (step < {feed_total});
        int p = in_feed ? (step % {period}) : {period};
        int g = step / {period};
        bool feeding_now = in_feed && (p >= 1) && (p <= {total_beats});
        int pf = p - 1;
        int kc = pf / {input_beats};
        int t = pf % {input_beats};
        ac_int<{a_bits}, false> a_rows_packed = 0;{array_bcols_decl}

        if (feeding_now && t < {m}) {{
            a_beat_T a_beat = a_rows[t];{_a_pack_block("a_rows_packed", "kc", "ROW_PACK_ARRAY_FOLD_N")}
        }}
        {fold_n_array_bcols_pack}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = feeding_now ? 1 : 0;
        ac_int<1, false> feed_preload_valid = (in_feed && p == 0) ? 1 : 0;
        gemm.run(a_rows_packed, {array_bcols_run_arg}feed_preload_valid, feed_valid, c_row, v, l);
{capture_fold_n}
    }}
"""

    # Fold-N C assembly: raw group-scoped lanes land in c_buf during the RUN
    # loop above (capture_fold_n); once every group's frame has landed, this
    # separate M-iteration loop pure-unpacks the full logical_n-wide rows
    # (decision 8: no rescale/bias/accum_t -- the core already requantised,
    # bias baked in) -- exactly today's per-row drain, just deferred past the
    # last frame instead of interleaved with the feed.
    _c_buf_decl = (
        f"    ac_int<{out_width}, true> c_buf[{m}][{n_passes * n}];\n" if fold_n else ""
    )
    _fold_n_emit_body = f"""\
    #pragma hls_pipeline_init_interval 1
    EMIT_FOLD_N: for (int row = 0; row < {m}; row++) {{
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

    if weights_in_core:
        # Weight-stationary: the self-contained const_weights STREAM entry plus the
        # const_weights ARRAY entry (io_parallel). Neither references an external b_cols;
        # weights live in the csim B_ROM (baked above) and the RTL wrapper ROM.
        entries_block = f"""\
// Weight-stationary (const-weight) entry: A only, no weight argument. Synthesis
// binds gemm.run to the const_weights RTL core (weights in the wrapper ROM); csim
// uses the ccore's internal B_ROM (same .dat-sourced beats). No frontend weight
// accessor is involved.
template <class a_beat_T, class bias_T, class res_T, typename CONFIG_T, bool HAS_BIAS = true>
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
    int captured = 0;   // total out_valid pulses seen (incl. padding rows)
    int written = 0;    // real result rows written to res_stream
{_c_buf_decl}
{stream_feed_loop}
{_emit_stream}}}

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
    ac_int<{a_bits}, false> last_a_rows = 0;
{_c_buf_decl}
{array_feed_loop}

    #pragma hls_pipeline_init_interval 1
    DRAIN_ARRAY_WL_PADDED_ROWS: for (int i = 0; i < {mr - m}; i++) {{
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> drain_valid = 0;
        ac_int<1, false> drain_preload_valid = 0;
        gemm.run(last_a_rows, drain_preload_valid, drain_valid, c_row, v, l);
    }}
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
template <class a_beat_T, class b_beat_T, class res_T, typename CONFIG_T>
void {name}_gemm_ip_stream(
    ac_channel<a_beat_T> &a_stream,
    ac_channel<b_beat_T> &b_stream,
    ac_channel<res_T> &res_stream
) {{
    b_beat_T weight_cols[{logical_n}];
    #pragma hls_pipeline_init_interval 1
    READ_B_COLS: for (int col = 0; col < {logical_n}; col++) {{
        weight_cols[col] = b_stream.read();
    }}
    {name}_gemm_ip_stream_buffered_b<a_beat_T, b_beat_T, typename CONFIG_T::bias_t, res_T, CONFIG_T, false>(
        a_stream, weight_cols, nullptr, res_stream);
}}

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
    ac_int<{a_bits}, false> last_a_rows = 0;
    ac_int<{b_bits}, false> last_b_cols = 0;
    constexpr bool HAS_BIAS = false;
    typename CONFIG_T::bias_t *biases = nullptr;
{_c_buf_decl}

{array_feed_loop}

    #pragma hls_pipeline_init_interval 1
    DRAIN_ARRAY_PADDED_ROWS: for (int i = 0; i < {mr - m}; i++) {{
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> drain_valid = 0;
        ac_int<1, false> drain_preload_valid = 0;
        gemm.run(last_a_rows, last_b_cols, drain_preload_valid, drain_valid, c_row, v, l);
    }}
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
        ac_int<{a_bits}, false>  a_rows,
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
{bias_c_decl_block}        // Frame-slot behavioral scheduler (mirrors the RTL sim model): up to
        // {slots} frames in flight. A frame starts at the first in_valid call
        // after a non-in_valid call (the FEED protocol always inserts the
        // preload step between frames); each frame keeps a private operand
        // buffer and cycle counter and emits its {m} rows at
        // [first_out+1, first_out+1+{m}) of its own clock. Back-to-back
        // frames sustain a frame II of {total_beats + 1} calls.
        static ac_int<{a_bits}, false> a_buf[{slots}][{total_beats}];
        static ac_int<{b_bits}, false> b_buf[{slots}][{passes * n}];
        static int cc_slot[{slots}] = {{0}};
        static bool slot_run[{slots}] = {{false}};
        static int wr_slot = {slots - 1};
        static bool feeding = false;{grp_static_decl}{bias_grp_static_decl}

        c_row = 0;
        out_valid = 0;
        out_last = 0;

        if (in_valid) {{
            if (!feeding) {{
                wr_slot = (wr_slot + 1) % {slots};
                feeding = true;
                slot_run[wr_slot] = true;
                cc_slot[wr_slot] = 0;{grp_static_step}{bias_grp_static_step}
            }}
            if (cc_slot[wr_slot] < {total_beats}) {{
                a_buf[wr_slot][cc_slot[wr_slot]] = a_rows;
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
            if (scc >= {first_out + 1} && scc < {first_out + 1} + {m}) {{
                int out_idx = scc - ({first_out + 1});
                int actual_row = out_idx;
                int row_tile = actual_row / 8;
                (void) row_tile;
                ac_int<{c_bits}, false> row_out = 0;
                for (int ct = 0; ct < {grid_cols}; ct++) {{
                    for (int cl = 0; cl < 8; cl++) {{
                        int actual_col = ct * 8 + cl;
                        ac_int<32, true> acc = 0;
                        // Bias no longer pre-adds into the raw accumulation --
                        // it lands post-stage-1 (core_requant_emit below), read
                        // from the compile-time bias_c_array (or 0 -- folds the
                        // add away when has_bias is False), mirroring the
                        // Verilog sim branch's bias_rom.
                        if (actual_row < {m} && actual_col < {n}) {{
                            for (int kk = 0; kk < {k}; kk++) {{
                                int k_chunk = kk / 8;
                                int k_lane = kk % 8;
                                ac_int<8, true> a_el = {a_el_expr};
                                ac_int<8, true> b_el = {b_el_expr};
                                acc += a_el * b_el;
                            }}
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
            if (scc + 1 >= {first_out + 1} + {m}) {{
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
    // stored bits — NOT value.to_int(), which would truncate the fractional part
    // of an ac_fixed operand and destroy it. slc<8>(0) reinterprets the low 8
    // mantissa bits as a signed int8 code; the drain rescales by 2^-(fa+fb).
    static_assert(src_T::width <= 8, "tensor_slice int8 core: operand wider than 8 bits");
    static_assert(src_T::sign || src_T::width <= 7, "tensor_slice int8 core: unsigned operand needs width <= 7 (bit 7 would be read as the sign)");
    return static_cast<ac_int<8, true> >(value.template slc<8>(0));
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
           requant_shift=0, requant_bits=None, drain_shift=None, weight_matrix=None,
           out_width=16):
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
    nnet::{name}_gemm_ip_array<a_beat_t, b_beat_t, int, res_t, {name}_config>(
        a_rows, weight_cols, biases, results);
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
    nnet::{name}_gemm_ip_stream<a_beat_t, b_beat_t, int, res_t, {name}_config>(
        a_stream, b_stream, biases, res_stream);
#endif

    for (int f = 0; f < NFRAMES; f++) {{
        for (int i = 0; i < {m}; i++) {{
            res_t out = res_stream.read();
            check_row(out, activations[f][i], weights, biases, f, i, failed);
        }}
    }}
"""

    # Two-stage requant, no saturation (jojo-track/open/tensor-slice-bias-in-
    # rtl, phase 1): the core now ALWAYS applies stage 1 (S1, unknown to this
    # standalone smoke harness -- treated as 0, a single round) + stage 2
    # (round-half-up shift by the TOTAL `requant_shift`, wrap to out_width),
    # bias baked in at generation time (this harness never forwards a real
    # bias to the packager, so its own `biases[]` fixture is zeroed and
    # unused -- kept only for entry-point signature compatibility). The
    # drain is a pure unpack: this reference reinterprets the requantised
    # code as the result type directly, matching the core bit-for-bit
    # whenever the packager's own S1 is 0 (the common case).
    _half = (1 << (requant_shift - 1)) if requant_shift else 0
    res_typedef = f"typedef nnet::array<ac_int<{out_width}, true>, {n}> res_t;"
    check_row = f"""\
// Per-row golden check: accumulate the whole dot product exactly, then
// round-half-up shift by the total gemm->result shift and wrap to
// out_width -- no saturation, no bias (baked into the core already, if any;
// this standalone smoke harness never bakes one).
static void check_row(const res_t &out, ac_int<8, true> a_row[{k}],
                      ac_int<8, true> weights[{n}][{k}], int biases[{n}],
                      int f, int i, int &failed) {{
    for (int j = 0; j < {n}; j++) {{
        int gemm_acc = 0;
        for (int kk = 0; kk < {k}; kk++) {{
            gemm_acc += a_row[kk].to_int() * weights[j][kk].to_int();
        }}
        ac_int<32, true> rounded = (ac_int<32, true>)(gemm_acc + {_half}) >> {requant_shift};
        ac_int<{out_width}, true> expect_code = (ac_int<{out_width}, true>) rounded;
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
                 requant_shift=0, requant_bits=None, drain_shift=None, weight_matrix=None,
                 out_width=16):
    from geometry import grid_rows, grid_cols
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
    nnet::{name}_gemm_ip_array<a_beat_t, b_beat_t, int, res_t, {name}_config>(
        a_rows, weight_cols, biases, results);
}}"""
    else:
        top_signature = f"""\
    ac_channel<a_beat_t> &a_stream,
    ac_channel<b_beat_t> &b_stream,
    int biases[{n}],
    ac_channel<res_t> &res_stream
) {{
    nnet::{name}_gemm_ip_stream<a_beat_t, b_beat_t, int, res_t, {name}_config>(
        a_stream, b_stream, biases, res_stream);
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


def gen_tcl(name, m, k, n, interface="stream", weights_in_core=False):
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

    return f"""\
set project_name "{name}_proj"
set solution_name "{name}_sol"

project new -name $project_name
solution new $solution_name
solution options defaults
solution options set /Output/OutputVerilog true
solution options set /Output/GenerateCycleNetlist false
options set Input/CompilerFlags {{-DBLACKBOX_FLOW}}

solution file add ./{name}_inst.cpp -type C++
solution file add ./{name}_tb.cpp -type C++
# Behavioral blackbox core RTL (ifndef SYNTHESIS branch) for SCVerify RTL cosim.
solution file add ./{name}_core.v -type Verilog -exclude true

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

solution library add nangate-45nm_beh -- -rtlsyntool DesignCompiler -vendor Nangate -technology 045nm
solution library add ccs_sample_mem
solution library add ccs_sample_rom
go libraries

directive set -CLOCKS {{clk {{-CLOCK_PERIOD 10.0 -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME 5.0 -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND both -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low}}}}

{map_lines}

go assembly
go architect
go allocate
go schedule
go extract

# RTL co-simulation (QuestaSim/msim) on the generated Verilog. Compiled WITHOUT
# -DSYNTHESIS, so the core's ifndef SYNTHESIS behavioral (frame-slot) branch is
# exercised — the same path large-bench cosim uses. Parse BEH_START/BEH_II/
# BEH_DONE from the transcript for back-to-back frame timing.
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
        # (weights come from the ROM). The inner behavioral grid submodule still
        # has a b_cols port fed by the ROM, so a whole-file scan would false-flag
        # it — b_cols simply isn't a blackbox port in this mode, so skip it.
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


def _assert_core_first_out(name, m, k, n, k_spatial, grid_v):
    """Cross-check the behavioral grid's FIRST_OUT localparam against
    latency_cycles. The C++ sim core, the wrapper's DRAIN capture window, and
    the behavioral Verilog model must agree on the first-output cycle at every
    k_spatial/passes combination; silent drift would desynchronize cosim
    capture."""
    expected = latency_cycles(m, k, n, grid_rows=(m + 7) // 8,
                              grid_cols=(n + 7) // 8, k_spatial=k_spatial)
    found = re.findall(r"localparam integer FIRST_OUT\s*=\s*(\d+);", grid_v)
    if len(found) != 1 or int(found[0]) != expected:
        raise RuntimeError(
            f"{name}: behavioral grid FIRST_OUT {found} does not match "
            f"latency_cycles()={expected} (m={m} k={k} n={n}, "
            f"k_spatial={k_spatial}). The sim model and the wrapper "
            "were generated with inconsistent drain timing."
        )


def _check_operand_fits_int8_core(name, operand_label, precision):
    """Raise ValueError if `precision` cannot be losslessly read as the
    tensor_slice int8 core's ac_int<8,true> operand.

    A signed operand fits with width <= 8 (sign bit is bit 7). An unsigned
    operand only fits with width <= 7, because the {name}_to_gemm_int8
    conversion reinterprets bit 7 as the sign bit; an unsigned value with bit
    7 set (e.g. 2.0 in ufixed<8,2>) would silently become negative.
    """
    bits = _operand_bits(precision)
    if bits is None:
        return
    width, signed = bits
    fits = width <= 8 if signed else width <= 7
    if not fits:
        raise ValueError(
            f"{name}: {operand_label} precision '{precision}' does not fit the "
            "tensor_slice int8 core: signed operands need width <= 8, unsigned "
            "<= 7; bit 7 of an unsigned 8-bit code would be read as the sign"
        )


def generate_catapult_pkg(m, k, n, name, output_dir, interface="stream", output_precision=None,
                          reuse_factor=1, input_precision=None, weight_precision=None,
                          clock_period_ns=None, n_frames=1, weight_matrix=None, fold_axis="k",
                          accum_precision=None, bias_precision=None, has_bias=None, bias=None,
                          **_ignored):
    if interface not in ("stream", "array"):
        raise ValueError(f"Unsupported GEMM interface '{interface}' for {name}; expected stream or array")
    _check_operand_fits_int8_core(name, "input_precision", input_precision)
    _check_operand_fits_int8_core(name, "weight_precision", weight_precision)
    fold_axis = str(fold_axis).lower()
    resolved = resolve_reuse_factor(k, reuse_factor, name, fold_axis=fold_axis, m=m, n=n)
    for w in resolved["warnings"]:
        print(w, file=sys.stderr)
    k_spatial = resolved["k_spatial"]
    passes = resolved["passes"]
    rf_legalized = resolved["reuse_factor"]
    fold_m = fold_axis == "m"
    fold_n = fold_axis == "n"
    m_passes = resolved.get("m_passes", 1) if fold_m else 1
    n_passes = resolved.get("n_passes", 1) if fold_n else 1
    # ``core_m`` is the RTL/csim-core row count: M_g = 8*mg when fold-M issues
    # more than one frame (m_passes frames of core_m rows assemble the logical
    # M rows), otherwise the logical M itself, so a single-frame package is
    # today's shape whichever axis was named.
    core_m = resolved["mg"] * 8 if m_passes > 1 else m
    # ``core_n`` is the RTL/csim-core column count: N_g = 8*cg when fold-N
    # issues more than one frame (n_passes frames of core_n columns assemble
    # the logical N columns), otherwise the logical N itself.
    core_n = resolved["cg"] * 8 if n_passes > 1 else n
    # Weight-stationary (const-weight) variant: weights (B, shape [K, N]) baked into
    # the core ROM AND the csim header; the wrapper takes no external weight port and
    # the header entry takes A only. Single source of truth = weight_matrix. Works at
    # every k_spatial/passes combination -- the ROM builder packs the same beats the
    # generalized narrow/chunked feed loops expect.
    weights_in_core = weight_matrix is not None
    weight_rom = None
    if weights_in_core:
        from gemm_ip.weights import build_weight_rom_k_spatial
        if n_passes > 1:
            # Fold-N: the ROM holds every group's columns back to back (group
            # g's block at base g*core_n, padded tail columns zero -- weight_
            # matrix's own N may be smaller than n_passes*core_n).
            import numpy as _np
            n_full = n_passes * core_n
            b_full = _np.zeros((k, n_full), dtype=_np.asarray(weight_matrix).dtype)
            b_full[:, :n] = weight_matrix
            weight_rom = build_weight_rom_k_spatial(b_full, core_m, n_full, k, k_spatial)
            expected_beats = n_full
        else:
            weight_rom = build_weight_rom_k_spatial(weight_matrix, core_m, core_n, k, k_spatial)
            expected_beats = passes * core_n
        if len(weight_rom) != expected_beats:
            raise RuntimeError(
                f"{name}: weight ROM has {len(weight_rom)} beats, expected "
                f"{expected_beats}"
            )
    # Two-stage requant (jojo-track/open/tensor-slice-bias-in-rtl, phase 1).
    # S1 (in-slice pre-round) is derived from accum_precision when given; S2
    # is the remainder of the total gemm->result shift. The output lane
    # narrows to out_width = _output_bits(output_precision) -- 8 bits when
    # output_precision is unset (the legacy/no-result-type case: this matches
    # _output_bits()'s own existing fallback and rtl.py's own out_width
    # default, so an un-annotated package keeps today's already-established
    # 8-bit lane rather than reverting to a 16-bit one).
    out_bits = _output_bits(output_precision)
    _gemm_shift = _frac_bits(input_precision) + _frac_bits(weight_precision)
    _out_frac = _frac_bits(output_precision)
    _total_shift = max(0, _gemm_shift - _out_frac) if _out_frac else 0
    _accum_w = _accum_shift_bits(accum_precision, _gemm_shift)
    s1 = max(0, _accum_w - 16) if _accum_w else 0
    if s1 > _total_shift:
        raise RuntimeError(
            f"{name}: accum_precision needs S1={s1} bits of in-slice pre-rounding, "
            f"more than the total gemm->result shift ({_total_shift}); the output "
            "needs more than 16 bits of range at the gemm scale.")
    if s1 > 0:
        print(f"WARNING: {name}: accum_t forces S1={s1} (in-slice pre-round); "
              "the phase-1 sim model folds the whole contraction before rounding, "
              "so this double-rounds against the per-partition synth behavior -- "
              "see jojo-track/open/tensor-slice-bias-in-rtl.", file=sys.stderr)
    s2 = _total_shift - s1
    # gen_inst_cpp/gen_tb's own self-check reference is still a single-round
    # formula (gemm_acc rounded once by `requant_shift`, bias added at
    # `drain_shift`) -- it must use the TOTAL shift, not S2 alone, or it
    # silently under-shifts whenever S1 > 0. This makes the C++ testbench's
    # self-check a single-round reference (identical to the two-stage DUT
    # when S1 == 0; expected to disagree by <= 1 output LSB when S1 > 0 --
    # decision 7 -- not a silent bug, a loud measured mismatch).
    requant_shift = _total_shift
    requant_bits = _output_bits(output_precision) if requant_shift else None

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

    pkg_dir = Path(output_dir) / name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # ``grid_rows``/``grid_cols``/RTL-core generation are sized on the CORE
    # row/column counts (core_m == m, core_n == n under fold_axis="k"; core_m
    # == M_g under fold_axis="m"; core_n == N_g under fold_axis="n").
    grid_rows = (core_m + 7) // 8
    grid_cols = (core_n + 7) // 8

    # The combined core (ifndef SYNTHESIS) lives in the rtl sibling module.
    ts_dir = str(Path(__file__).resolve().parent)
    if ts_dir not in sys.path:
        sys.path.insert(0, ts_dir)
    from rtl import generate_combined_core_verilog, generate_k_spatial_combined_core_verilog
    if k_spatial == 1:
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
    )
    _assert_core_port_widths(name, header_text, grid_v, weights_in_core=weights_in_core)
    _assert_core_first_out(name, core_m, k, core_n, k_spatial, grid_v)
    (pkg_dir / f"{name}_core.v").write_text(grid_v)
    (pkg_dir / "nnet_types.h").write_text(gen_nnet_types_header())
    (pkg_dir / f"{name}_gemm_ip.h").write_text(header_text)
    # The inst wrapper, testbench and tcl see the logical shape: the header
    # asserts CONFIG_T::gemm_m == m and the fold-M frames are internal to one
    # call, so they never see core_m or m_passes.
    (pkg_dir / f"{name}_inst.cpp").write_text(gen_inst_cpp(
        name, m, k, n, interface,
        requant_shift=requant_shift, requant_bits=requant_bits,
        drain_shift=(_out_frac if requant_shift else _gemm_shift),
        weight_matrix=weight_matrix, out_width=out_bits,
    ))
    (pkg_dir / f"{name}_tb.cpp").write_text(gen_tb(
        name, m, k, n, interface, n_frames=n_frames,
        requant_shift=requant_shift, requant_bits=requant_bits,
        drain_shift=(_out_frac if requant_shift else _gemm_shift),
        weight_matrix=weight_matrix, out_width=out_bits,
    ))
    (pkg_dir / "run_catapult.tcl").write_text(
        gen_tcl(name, m, k, n, interface, weights_in_core=weight_matrix is not None))
    print(f"Generated {pkg_dir}  (M={m}, K={k}, N={n}, interface={interface}, "
          f"reuse_factor={rf_legalized}, k_spatial={k_spatial}, fold_axis={fold_axis})")
