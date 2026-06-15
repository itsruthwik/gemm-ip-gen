"""
Catapult HLS GEMM blackbox package generator.

Generates per-layer packages with ac_channel-based wrappers,
tensor-slice grid RTL, Catapult synthesis Tcl, and a combined
dispatch header for hls4ml integration.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from gemm_ip.metadata import (
    LANE_WIDTH,
    TENSOR_SLICE_SRC,
    _safe_name,
    _ceil_div,
    _is_ac_integer_type,
    load_catapult_rtl_generator,
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


def latency_cycles(m, k, n, grid_rows, grid_cols, full_k_spatial=False):
    """First-output cycle offset for the C++ simulation model.

    Matches the behavioral grid timing: feed beats + the systolic K+N wave
    remainder.
      chunked: first_out = k_chunks × max(M,N) + max(0, K+N − k_chunks×max(M,N))
      full-K:  first_out =            max(M,N) + max(0, K+N −          max(M,N))
    Full-K mode feeds every K chunk spatially in one max(M,N)-beat pass, so
    its first output arrives correspondingly earlier. By construction
    first_out >= total feed beats, so the first row always lands inside the
    DRAIN window, never inside FEED. This is the cycle where out_valid fires
    in the simulation model, offset from clk_cnt = 0 (first in_valid beat).
    """
    k_chunks = _ceil_div(k, LANE_WIDTH)
    input_beats = max(m, n)
    total_beats = input_beats if full_k_spatial else k_chunks * input_beats
    return total_beats + max(0, k + n - total_beats)


def dead_cycles_raw(m, k, n, grid_cols, full_k_spatial=False):
    """Drain dead cycles before the C++ wrapper starts capturing output."""
    return latency_cycles(m, k, n, grid_rows=1, grid_cols=grid_cols,
                          full_k_spatial=full_k_spatial) + 1


def dead_cycles(m, k, n, grid_cols, full_k_spatial=False):
    """Drain dead cycles + 1-cycle padding."""
    return dead_cycles_raw(m, k, n, grid_cols, full_k_spatial=full_k_spatial) + 1


def gen_public_header(name, m, k, n, grid_rows, grid_cols, result_type=None, gemm_k_spatial=1,
                      input_precision=None, weight_precision=None, clock_period_ns=None,
                      n_frames=1):
    bb_delay_ns = _blackbox_delay_ns(clock_period_ns)
    row_chunk_bits = grid_rows * 64
    col_chunk_bits = grid_cols * 64
    c_bits = grid_cols * 128
    mr = grid_rows * 8
    k_chunks = _ceil_div(k, LANE_WIDTH)
    # Full-K-spatial layout applies ONLY to the dedicated k-spatial grid
    # (gemm_k_spatial == k_chunks > 1). For k_chunks == 1 the package always
    # generates the CHUNKED grid (generate_combined_core_verilog), so the
    # wrapper must pack chunked words too: matching k_chunks==1 here used to
    # emit 64-bit full-mode words against the grid's grid_cols*64-bit ports,
    # X-poisoning every row/col tile beyond the first (cosim-only corruption
    # for max(m,n) > 8).
    full_k_spatial = k_chunks > 1 and gemm_k_spatial == k_chunks
    # Full-K uses the NARROW per-beat word (one tile, 64 bits per K-chunk); the
    # wrapper RTL re-inserts the grid row/col tile offset by beat index, so the
    # deep input FIFO never stores the always-zero grid padding (area saving on
    # tiled designs).  Chunked keeps the single-chunk grid-padded width.
    a_bits = 64 * k_chunks if full_k_spatial else row_chunk_bits
    b_bits = 64 * k_chunks if full_k_spatial else col_chunk_bits
    bias_bits = col_chunk_bits
    input_beats = max(m, n)
    total_beats = input_beats if full_k_spatial else k_chunks * input_beats
    first_out = latency_cycles(m, k, n, grid_rows, grid_cols, full_k_spatial=full_k_spatial)
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

    # Back-to-back multi-frame schedule. The wrapper feeds n_frames frames with NO
    # preload step (preload_valid is tied off; bias lives in the drain capture).
    # Each frame is total_beats in_valid beats + exactly ONE in_valid=0 separator
    # so the core's `feeding` flag drops and the next frame allocates a fresh slot.
    # Steady-state frame period = total_beats + 1; the last frame's outputs drain
    # in the trailing first_out + m + slack tail. With n_frames == 1 this reduces
    # to a single preload-free frame (real hls4ml flow: one frame per wrapper call).
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

    # Choose the RHS expression for the final output assignment based on the
    # configured result type.  Integer types (ac_int / ac_uint) need an explicit
    # ``.to_int()`` call because directly casting an ``ac_fixed`` accumulator to
    # an ``ac_int`` may fail to compile or produce unexpected truncation.
    # Fixed-point types should preserve the normal AC-datatype conversion
    # (rounding / saturation) by omitting ``.to_int()``.
    assign_expr = "value.to_int()" if _is_ac_integer_type(result_type) else "value"

    # The core is a pure INTEGER matmul: it multiplies operand int8 *codes*
    # (the fixed-point mantissa = value · 2^frac). The raw integer dot-product
    # therefore carries 2^(frac_a + frac_b); the wrapper drain shifts it back to
    # the real value, then adds the (full-precision) bias and quantizes to the
    # result type. gemm_shift == 0 collapses to the legacy integer-coded path.
    gemm_shift = _frac_bits(input_precision) + _frac_bits(weight_precision)
    a_el_expr = (
        f"a_buf[s][actual_row].slc<8>(k_chunk * 64 + k_lane * 8)"
        if full_k_spatial
        else f"a_buf[s][k_chunk * {input_beats} + actual_row].slc<8>(row_tile * 64 + k_lane * 8)"
    )
    b_el_expr = (
        f"b_buf[s][actual_col].slc<8>(k_chunk * 64 + k_lane * 8)"
        if full_k_spatial
        else f"b_buf[s][k_chunk * {input_beats} + actual_col].slc<8>(ct * 64 + k_lane * 8)"
    )

    # Per-call output capture, embedded in the merged RUN loop so rows are
    # collected as they emerge while the frame is still feeding/computing.
    _capture_body = f"""\
        if (v) {{
            if (captured < {m}) {{
                res_T out_pack;
                #pragma hls_unroll
                for (int col = 0; col < {n}; col++) {{
                    int col_tile = col / 8;
                    int col_local = col % 8;
                    ac_int<16, true> raw_val = c_row.template slc<16>(col_tile * 128 + col_local * 16);
                    // Rescale the raw integer dot-product by 2^-(fa+fb) to the
                    // real value, then add the full-precision bias (Keras order:
                    // matmul + bias, then quantize on the result-type cast below).
                    typename CONFIG_T::accum_t value =
                        static_cast<typename CONFIG_T::accum_t>(
                            ((ac_fixed<48, 24, true>) raw_val.to_int()) >> {gemm_shift})
                        + static_cast<typename CONFIG_T::accum_t>(biases[col]);
                    out_pack[col] = static_cast<typename res_T::value_type>({assign_expr});
                }}
                %SINK%
            }}
            captured++;
        }}"""
    stream_capture = _capture_body.replace("%SINK%", "res_stream.write(out_pack);")
    array_capture = _capture_body.replace("%SINK%", "results[captured] = out_pack;")

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
                    ac_int<16, true> raw_val = c_row.template slc<16>(col_tile * 128 + col_local * 16);
                    typename CONFIG_T::accum_t value =
                        static_cast<typename CONFIG_T::accum_t>(
                            ((ac_fixed<48, 24, true>) raw_val.to_int()) >> {gemm_shift})
                        + static_cast<typename CONFIG_T::accum_t>(biases[col]);
                    out_pack[col] = static_cast<typename res_T::value_type>({assign_expr});
                }}
                res_stream.write(out_pack);
                written++;
            }}
            captured++;
        }}"""
    stream_capture_b2b = _capture_body_b2b

    if full_k_spatial:
        stream_feed_loop = f"""
    // Back-to-back, preload-free feed of {n_frames} frame(s) (full K-spatial:
    // each logical A row once, A/B words widened to carry every 8-wide K chunk).
    // Each frame is {total_beats} in_valid beats + ONE in_valid=0 separator
    // (period {period}); preload_valid is tied off (bias added in the drain
    // capture). Every step polls out_valid, so rows are captured as they emerge
    // — frame t+1 feeds while frame t drains in the core's FRAME_SLOTS.
    #pragma hls_pipeline_init_interval 1
    RUN: for (int step = 0; step < {total_steps}; step++) {{
        bool in_feed = (step < {feed_total});
        int p = in_feed ? (step % {period}) : {period};
        bool feeding_now = in_feed && (p < {total_beats});
        int t = p;
        ac_int<{a_bits}, false> a_rows = 0;
        ac_int<{b_bits}, false> b_cols = 0;

        if (feeding_now && t < {m}) {{
            a_beat_T a_beat = a_stream.read();
            #pragma hls_unroll
            ROW_PACK_FULL_KC: for (int kc = 0; kc < {k_chunks}; kc++) {{
                #pragma hls_unroll
                ROW_PACK_FULL_KL: for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k}) {{
                        a_rows.set_slc(kc * 64 + kl * 8,
                                       {name}_to_gemm_int8(a_beat[kk]));
                    }}
                }}
            }}
        }}
        if (feeding_now && t < {n}) {{
            b_beat_T b_beat = weight_cols[t];
            #pragma hls_unroll
            COL_PACK_FULL_KC: for (int kc = 0; kc < {k_chunks}; kc++) {{
                #pragma hls_unroll
                COL_PACK_FULL_KL: for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k}) {{
                        b_cols.set_slc(kc * 64 + kl * 8,
                                       {name}_to_gemm_int8(b_beat[kk]));
                    }}
                }}
            }}
        }}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = feeding_now ? 1 : 0;
        gemm.run(a_rows, b_cols, bias_packed, 0, feed_valid, c_row, v, l);
{stream_capture_b2b}
    }}
"""
    else:
        stream_feed_loop = f"""
    // Replay storage is packed to the blackbox protocol. HLS4ML still emits
    // each logical K-wide A row once; later K chunks replay packed slices.
    // Reused per frame (written at each frame's kc==0 beats, read within the
    // same frame's later chunks — the feed is sequential in step order).
    ac_int<{a_bits}, false> a_replay[{k_chunks}][{input_beats}];

    // Back-to-back, preload-free feed of {n_frames} frame(s): M A rows + N B
    // columns as 8-lane K chunks. Each frame is {total_beats} in_valid beats +
    // ONE in_valid=0 separator (period {period}); preload_valid tied off (bias
    // added in the drain capture). Every step polls out_valid, so rows are
    // captured as they emerge — frame t+1 feeds while frame t drains in the
    // core's FRAME_SLOTS.
    #pragma hls_pipeline_init_interval 1
    RUN: for (int step = 0; step < {total_steps}; step++) {{
        bool in_feed = (step < {feed_total});
        int p = in_feed ? (step % {period}) : {period};
        bool feeding_now = in_feed && (p < {total_beats});
        int kc = p / {input_beats};
        int t = p % {input_beats};
        ac_int<{a_bits}, false> a_rows = 0;
        ac_int<{b_bits}, false> b_cols = 0;

        if (feeding_now && t < {m}) {{
            if (kc == 0) {{
                a_beat_T a_beat = a_stream.read();
                #pragma hls_unroll
                ROW_PACK_DIRECT: for (int kl = 0; kl < 8; kl++) {{
                    int kk = kl;
                    int row_tile = t / 8;
                    #pragma hls_unroll
                    ROW_TILE_DIRECT: for (int rt = 0; rt < {grid_rows}; rt++) {{
                        if (row_tile == rt && kk < {k}) {{
                            a_rows.set_slc(rt * 64 + kl * 8,
                                           {name}_to_gemm_int8(a_beat[kk]));
                        }}
                    }}
                }}
                #pragma hls_unroll
                PREPACK_REPLAY: for (int replay_kc = 1; replay_kc < {k_chunks}; replay_kc++) {{
                    ac_int<{a_bits}, false> replay_rows = 0;
                    #pragma hls_unroll
                    ROW_PACK_REPLAY: for (int kl = 0; kl < 8; kl++) {{
                        int kk = replay_kc * 8 + kl;
                        int row_tile = t / 8;
                        #pragma hls_unroll
                        ROW_TILE_REPLAY: for (int rt = 0; rt < {grid_rows}; rt++) {{
                            if (row_tile == rt && kk < {k}) {{
                                replay_rows.set_slc(rt * 64 + kl * 8,
                                                    {name}_to_gemm_int8(a_beat[kk]));
                            }}
                        }}
                    }}
                    a_replay[replay_kc][t] = replay_rows;
                }}
            }} else {{
                a_rows = a_replay[kc][t];
            }}
        }}
        if (feeding_now && t < {n}) {{
            b_beat_T b_beat = weight_cols[t];
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
        }}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = feeding_now ? 1 : 0;
        gemm.run(a_rows, b_cols, bias_packed, 0, feed_valid, c_row, v, l);
{stream_capture_b2b}
    }}
"""

    if full_k_spatial:
        array_feed_loop = f"""
    // Merged feed+drain: one run() call per cycle. Steps 0..{total_beats}
    // preload then feed each logical A row and B column once (full K-spatial
    // mode: A/B words widened to carry every 8-wide K chunk); every step
    // polls out_valid, so the frame's rows are captured as they emerge
    // instead of in a separate drain loop after the feed.
    #pragma hls_pipeline_init_interval 1
    RUN_ARRAY: for (int step = 0; step < {run_calls}; step++) {{
        int t = (step == 0) ? 0 : step - 1;
        ac_int<{a_bits}, false> a_rows_packed = 0;
        ac_int<{b_bits}, false> b_cols_packed = 0;

        if (step > 0 && step <= {total_beats} && t < {m}) {{
            a_beat_T a_beat = a_rows[t];
            #pragma hls_unroll
            ROW_PACK_ARRAY_FULL_KC: for (int kc = 0; kc < {k_chunks}; kc++) {{
                #pragma hls_unroll
                ROW_PACK_ARRAY_FULL_KL: for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k}) {{
                        a_rows_packed.set_slc(kc * 64 + kl * 8,
                                              {name}_to_gemm_int8(a_beat[kk]));
                    }}
                }}
            }}
        }}
        if (step > 0 && step <= {total_beats} && t < {n}) {{
            b_beat_T b_beat = weight_cols[t];
            #pragma hls_unroll
            COL_PACK_ARRAY_FULL_KC: for (int kc = 0; kc < {k_chunks}; kc++) {{
                #pragma hls_unroll
                COL_PACK_ARRAY_FULL_KL: for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k}) {{
                        b_cols_packed.set_slc(kc * 64 + kl * 8,
                                              {name}_to_gemm_int8(b_beat[kk]));
                    }}
                }}
            }}
        }}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = (step >= 1 && step <= {total_beats}) ? 1 : 0;
        ac_int<1, false> feed_preload_valid = (step == 0) ? 1 : 0;
        gemm.run(a_rows_packed, b_cols_packed, bias_packed, feed_preload_valid, feed_valid, c_row, v, l);
{array_capture}
    }}
"""
    else:
        array_feed_loop = f"""
    // Merged feed+drain: one run() call per cycle. Steps 0..{total_beats}
    // preload then feed M A rows and N B columns as 8-lane K chunks; every
    // step polls out_valid, so the frame's rows are captured as they emerge
    // instead of in a separate drain loop after the feed.
    #pragma hls_pipeline_init_interval 1
    RUN_ARRAY: for (int step = 0; step < {run_calls}; step++) {{
        int eff_step = (step == 0 || step > {total_beats}) ? 0 : step - 1;
        int kc = eff_step / {input_beats};
        int t = eff_step % {input_beats};
        ac_int<{a_bits}, false> a_rows_packed = 0;
        ac_int<{b_bits}, false> b_cols_packed = 0;

        if (step > 0 && step <= {total_beats} && t < {m}) {{
            a_beat_T a_beat = a_rows[t];
            #pragma hls_unroll
            for (int kl = 0; kl < 8; kl++) {{
                int kk = kc * 8 + kl;
                int row_tile = t / 8;
                #pragma hls_unroll
                for (int rt = 0; rt < {grid_rows}; rt++) {{
                    if (row_tile == rt && kk < {k}) {{
                        a_rows_packed.set_slc(rt * 64 + kl * 8,
                                              {name}_to_gemm_int8(a_beat[kk]));
                    }}
                }}
            }}
        }}
        if (step > 0 && step <= {total_beats} && t < {n}) {{
            b_beat_T b_beat = weight_cols[t];
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
        }}

        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = (step >= 1 && step <= {total_beats}) ? 1 : 0;
        ac_int<1, false> feed_preload_valid = (step == 0) ? 1 : 0;
        gemm.run(a_rows_packed, b_cols_packed, bias_packed, feed_preload_valid, feed_valid, c_row, v, l);
{array_capture}
    }}
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
        ac_int<{b_bits}, false>  b_cols,
        ac_int<{bias_bits}, false>  bias_cols,
        ac_int<1, false>         preload_valid,
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
        c_row[0] = a_rows[0] ^ b_cols[0] ^ bias_cols[0] ^ preload_valid[0] ^ in_valid[0];
        out_valid = in_valid;
        out_last = in_valid;
#else
        // Frame-slot behavioral scheduler (mirrors the RTL sim model): up to
        // {slots} frames in flight. A frame starts at the first in_valid call
        // after a non-in_valid call (the FEED protocol always inserts the
        // preload step between frames); each frame keeps a private operand
        // buffer and cycle counter and emits its {m} rows at
        // [first_out+1, first_out+1+{m}) of its own clock. Back-to-back
        // frames sustain a frame II of {total_beats + 1} calls.
        static ac_int<{a_bits}, false> a_buf[{slots}][{total_beats}];
        static ac_int<{b_bits}, false> b_buf[{slots}][{total_beats}];
        static ac_int<{bias_bits}, false> bias_buf[{slots}];
        static int cc_slot[{slots}] = {{0}};
        static bool slot_run[{slots}] = {{false}};
        static int wr_slot = {slots - 1};
        static bool feeding = false;

        c_row = 0;
        out_valid = 0;
        out_last = 0;

        if (in_valid) {{
            if (!feeding) {{
                wr_slot = (wr_slot + 1) % {slots};
                feeding = true;
                slot_run[wr_slot] = true;
                cc_slot[wr_slot] = 0;
                bias_buf[wr_slot] = bias_cols;
            }}
            if (cc_slot[wr_slot] < {total_beats}) {{
                a_buf[wr_slot][cc_slot[wr_slot]] = a_rows;
                b_buf[wr_slot][cc_slot[wr_slot]] = b_cols;
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
                        // Apply bias from this frame's bias buffer
                        ac_int<8, true> bias_el = bias_buf[s].slc<8>(ct * 64 + cl * 8);
                        if (actual_row < {m} && actual_col < {n}) {{
                            acc = bias_el;
                            for (int kk = 0; kk < {k}; kk++) {{
                                int k_chunk = kk / 8;
                                int k_lane = kk % 8;
                                ac_int<8, true> a_el = {a_el_expr};
                                ac_int<8, true> b_el = {b_el_expr};
                                acc += a_el * b_el;
                            }}
                        }}
                        // Saturate the raw integer dot-product to the physical
                        // 16-bit output lane (NOT int8): the core emits the integer
                        // accumulator; the wrapper drain rescales + quantizes to the
                        // result type. Matches the RTL core (out_bits=16).
                        ac_int<16, true> sat_val;
                        if (acc > 32767) sat_val = 32767;
                        else if (acc < -32768) sat_val = -32768;
                        else sat_val = acc;
                        row_out.set_slc(ct * 128 + cl * 16, sat_val);
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
    return static_cast<ac_int<8, true> >(value.template slc<8>(0));
}}

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void {name}_gemm_ip_stream_const_weights(
    ac_channel<a_beat_T> &a_stream,
    b_beat_T weight_cols[CONFIG_T::gemm_n],
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
) {{
    static_assert(CONFIG_T::gemm_m == {m}, "Generated GEMM wrapper requires matching gemm_m.");
    static_assert(CONFIG_T::gemm_n == {n}, "Generated GEMM wrapper requires matching gemm_n.");
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
    ac_int<{bias_bits}, false> bias_packed = 0;

    #pragma hls_unroll
    BIAS_PACK: for (int col = 0; col < {n}; col++) {{
        int col_tile = col / 8;
        int col_local = col % 8;
        // Bias is added in the drain (post-rescale, full precision), so the
        // pure-matmul core is fed zero bias. Keeps the core a clean integer GEMM.
        (void) col_tile; (void) col_local;
        bias_packed.set_slc(col_tile * 64 + col_local * 8, ac_int<8, true>(0));
    }}

{stream_feed_loop}
}}

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void {name}_gemm_ip_stream(
    ac_channel<a_beat_T> &a_stream,
    ac_channel<b_beat_T> &b_stream,
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
) {{
    b_beat_T weight_cols[{n}];
    #pragma hls_pipeline_init_interval 1
    READ_B_COLS: for (int col = 0; col < {n}; col++) {{
        weight_cols[col] = b_stream.read();
    }}
    {name}_gemm_ip_stream_const_weights<a_beat_T, b_beat_T, bias_T, res_T, CONFIG_T>(
        a_stream, weight_cols, biases, res_stream);
}}

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void {name}_gemm_ip_array(
    a_beat_T a_rows[CONFIG_T::gemm_m],
    b_beat_T weight_cols[CONFIG_T::gemm_n],
    bias_T biases[CONFIG_T::gemm_n],
    res_T results[CONFIG_T::gemm_m]
) {{
    static_assert(CONFIG_T::gemm_m == {m}, "Generated GEMM wrapper requires matching gemm_m.");
    static_assert(CONFIG_T::gemm_n == {n}, "Generated GEMM wrapper requires matching gemm_n.");

    static {name}_ccore gemm;
    int captured = 0;
    ac_int<{a_bits}, false> last_a_rows = 0;
    ac_int<{b_bits}, false> last_b_cols = 0;
    ac_int<{bias_bits}, false> bias_packed = 0;

    #pragma hls_unroll
    BIAS_PACK_ARRAY: for (int col = 0; col < {n}; col++) {{
        int col_tile = col / 8;
        int col_local = col % 8;
        // Bias is added in the drain (post-rescale, full precision), so the
        // pure-matmul core is fed zero bias. Keeps the core a clean integer GEMM.
        (void) col_tile; (void) col_local;
        bias_packed.set_slc(col_tile * 64 + col_local * 8, ac_int<8, true>(0));
    }}

{array_feed_loop}

    #pragma hls_pipeline_init_interval 1
    DRAIN_ARRAY_PADDED_ROWS: for (int i = 0; i < {mr - m}; i++) {{
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> drain_valid = 0;
        ac_int<1, false> drain_preload_valid = 0;
        gemm.run(last_a_rows, last_b_cols, bias_packed, drain_preload_valid, drain_valid, c_row, v, l);
    }}
}}

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


def gen_tb(name, m, k, n, interface="stream", n_frames=1):
    if interface == "array":
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
typedef nnet::array<ac_int<16, true>, {n}> res_t;

// Per-row golden check: raw integer dot-product saturated to the 16-bit output
// lane, then bias added (matches the core + wrapper-drain order).
static void check_row(const res_t &out, ac_int<8, true> a_row[{k}],
                      ac_int<8, true> weights[{n}][{k}], int biases[{n}],
                      int f, int i, int &failed) {{
    for (int j = 0; j < {n}; j++) {{
        int gemm_acc = 0;
        for (int kk = 0; kk < {k}; kk++) {{
            gemm_acc += a_row[kk].to_int() * weights[j][kk].to_int();
        }}
        if (gemm_acc > 32767) gemm_acc = 32767;
        else if (gemm_acc < -32768) gemm_acc = -32768;
        gemm_acc += biases[j];
        if (out[j].to_int() != gemm_acc) {{
            printf("Mismatch frame %d row %d col %d: got %d expected %d\\n",
                   f, i, j, out[j].to_int(), gemm_acc);
            failed = 1;
        }}
    }}
}}

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

    for (int j = 0; j < {n}; j++) {{
        biases[j] = (j % 5) - 2;
        for (int kk = 0; kk < {k}; kk++) {{
            weights[j][kk] = ((j * 5 - kk + 1) & 0x7) - 3;
        }}
    }}

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


def gen_inst_cpp(name, m, k, n, interface="stream"):
    from gemm_ip.metadata import grid_rows, grid_cols
    if interface == "array":
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
typedef nnet::array<ac_int<16, true>, {n}> res_t;

#pragma hls_design top
void {name}_inst(
{top_signature}
"""


def gen_tcl(name, m, k, n, interface="stream"):
    if interface == "array":
        map_lines = f"""\
# Array top-level package smoke synthesis. hls4ml integration instantiates the
# ccore through the generated combined header rather than this standalone top.
directive set /{name}_inst/a_beats:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
directive set /{name}_inst/b_beats:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
directive set /{name}_inst/biases:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
directive set /{name}_inst/results:rsc -MAP_TO_MODULE ccs_ioport.ccs_out_wait"""
    else:
        map_lines = f"""\
# Phase 5: Map streams to real streaming resources
directive set /{name}_inst/a_stream:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
directive set /{name}_inst/b_stream:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
directive set /{name}_inst/biases:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
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
    const_weight_stream_branches = []
    array_branches = []
    for item in items:
        target = item.get("interface", "stream")
        branch = f"""\
    if constexpr ({_dispatch_condition(item)}) {{
        {item["name"]}_gemm_ip_{target}<a_beat_T, b_beat_T, bias_T, res_T, CONFIG_T>(
            TARGET_ARGS);
    }}"""
        if target == "array":
            array_branches.append(branch.replace("TARGET_ARGS", "a_rows, weight_cols, biases, results"))
        else:
            stream_branches.append(branch.replace("TARGET_ARGS", "a_stream, b_stream, biases, res_stream"))
            const_weight_stream_branches.append(
                branch.replace("gemm_ip_stream", "gemm_ip_stream_const_weights").replace(
                    "TARGET_ARGS", "a_stream, weight_cols, biases, res_stream"
                )
            )
    stream_branches_text = " else ".join(stream_branches)
    const_weight_stream_branches_text = " else ".join(const_weight_stream_branches)
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
    if not const_weight_stream_branches_text:
        const_weight_stream_branches_text = """\
    static_assert(CONFIG_T::gemm_m == 0,
                  "No generated const-weight stream GEMM IP implementation is present in this package.");"""
    else:
        const_weight_stream_branches_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated const-weight stream GEMM IP implementation matches this CONFIG_T.");
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
    return f"""\
#ifndef GEMM_IP_COMBINED_H_
#define GEMM_IP_COMBINED_H_

#include "ac_channel.h"
{includes}

namespace nnet {{

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_stream(
    ac_channel<a_beat_T> &a_stream,
    ac_channel<b_beat_T> &b_stream,
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
) {{
{stream_branches_text}
}}
template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_stream_const_weights(
    ac_channel<a_beat_T> &a_stream,
    b_beat_T weight_cols[CONFIG_T::gemm_n],
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
) {{
{const_weight_stream_branches_text}
}}
template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_array(
    a_beat_T a_rows[CONFIG_T::gemm_m],
    b_beat_T weight_cols[CONFIG_T::gemm_n],
    bias_T biases[CONFIG_T::gemm_n],
    res_T results[CONFIG_T::gemm_m]
) {{
{array_branches_text}
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

    gemm_ip_stream_const_weights<a_beat_t, b_beat_t, bias_T, res_T, CONFIG_T>(
        a_beat_stream, weight_cols, biases, res_stream);
}}

}} // namespace nnet

#endif
"""


def gen_integration_manifest(items):
    cores = []
    for item in items:
        cores.append({
            "name": item["name"],
            "interface": item.get("interface", "stream"),
            "protocol": item.get("protocol", {}),
            "entity": f"{item['name']}_core",
            "rtl": f"{item['name']}/{item['name']}_core.v",
            "m": item["m"],
            "k": item["k"],
            "n": item["n"],
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


def _validate_gemm_k_spatial(k, gemm_k_spatial):
    k_chunks = _ceil_div(k, LANE_WIDTH)
    if gemm_k_spatial is None:
        return k_chunks
    k_spatial = int(gemm_k_spatial)
    if k_spatial < 1:
        raise ValueError("gemm_k_spatial must be >= 1")
    if k_spatial > k_chunks:
        raise ValueError(
            f"gemm_k_spatial={k_spatial} exceeds K_CHUNKS={k_chunks}; "
            "v1 requires at most one spatial grid per K chunk"
        )
    return k_spatial


def _frac_bits(precision):
    """Fractional-bit count (W - I) of a precision like 'fixed<9,5,…>'.

    The GEMM IP feeds operands as int8 *codes* = the fixed-point mantissa
    (value · 2^frac). The product of two operands therefore carries
    2^(frac_a + frac_b), which the wrapper drain divides back out. Returns 0 for
    an unset/unparseable precision (integer-coded operand ⇒ no rescale), so the
    rescale collapses to the identity for the legacy integer-input case.
    """
    if not precision:
        return 0
    m = re.search(r"u?fixed<\s*(\d+)\s*,\s*(-?\d+)", str(precision))
    if not m:
        return 0
    width, integer_bits = int(m.group(1)), int(m.group(2))
    return width - integer_bits


def _output_bits(output_precision):
    """Result-lane width (bits) from an output_precision like 'fixed<16,6,…>'.

    Drives the GEMM-IP output saturation/packing so the result honors the
    configured precision instead of the legacy hardcoded int8 clamp. Returns 8
    (legacy int8) when unset/unparseable so callers without a precision keep
    the old behavior. The first ``fixed<>`` field is the total bit width.
    """
    if not output_precision:
        return 8
    m = re.search(r"u?fixed<\s*(\d+)", str(output_precision))
    return int(m.group(1)) if m else 8


_CORE_PORTS = ("a_rows", "b_cols", "bias_cols", "c_row")


def _assert_core_port_widths(name, header_text, grid_v):
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
        hw, rw = hdr.get(port, set()), rtl.get(port, set())
        if len(hw) != 1 or len(rw) != 1 or hw != rw:
            raise RuntimeError(
                f"{name}: blackbox port width mismatch for '{port}': C++ header "
                f"declares {sorted(hw)} bits, grid RTL declares {sorted(rw)} bits. "
                "The wrapper and the core were generated with inconsistent word "
                "layouts (full-K-spatial vs chunked)."
            )


def _assert_core_first_out(name, m, k, n, gemm_k_spatial, grid_v):
    """Cross-check the behavioral grid's FIRST_OUT localparam against
    latency_cycles. The C++ sim core, the wrapper's DRAIN capture window, and
    the behavioral Verilog model must agree on the first-output cycle (chunked
    vs full-K-spatial); silent drift would desynchronize cosim capture."""
    k_chunks = _ceil_div(k, LANE_WIDTH)
    full_k_spatial = k_chunks > 1 and gemm_k_spatial == k_chunks
    expected = latency_cycles(m, k, n, grid_rows=(m + 7) // 8,
                              grid_cols=(n + 7) // 8, full_k_spatial=full_k_spatial)
    found = re.findall(r"localparam integer FIRST_OUT\s*=\s*(\d+);", grid_v)
    if len(found) != 1 or int(found[0]) != expected:
        raise RuntimeError(
            f"{name}: behavioral grid FIRST_OUT {found} does not match "
            f"latency_cycles()={expected} (m={m} k={k} n={n}, "
            f"gemm_k_spatial={gemm_k_spatial}). The sim model and the wrapper "
            "were generated with inconsistent drain timing."
        )


def generate_catapult_pkg(m, k, n, name, output_dir, interface="stream", output_precision=None,
                          gemm_k_spatial=None, input_precision=None, weight_precision=None,
                          clock_period_ns=None, n_frames=1):
    if interface not in ("stream", "array"):
        raise ValueError(f"Unsupported GEMM interface '{interface}' for {name}; expected stream or array")
    gemm_k_spatial = _validate_gemm_k_spatial(k, gemm_k_spatial)
    # The core saturates the raw integer dot-product to the physical 16-bit
    # output lane; result-precision quantization happens in the wrapper drain
    # (rescale + bias + result-type cast), not in the core.
    out_bits = 16
    pkg_dir = Path(output_dir) / name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8

    # The combined core (ifndef SYNTHESIS) is in the tensor-slice module
    ts_dir = str(Path(__file__).resolve().parent.parent / "tensor-slice")
    if ts_dir not in sys.path:
        sys.path.insert(0, ts_dir)
    from generate_catapult_rtl import generate_combined_core_verilog, generate_k_spatial_combined_core_verilog
    if gemm_k_spatial == 1:
        grid_v = generate_combined_core_verilog(m, k, n, module_name=f"{name}_core", out_bits=out_bits)
    else:
        print(
            f"WARNING: {name}: gemm_k_spatial={gemm_k_spatial} is experimental; "
            "tensor-slice partial outputs are INT16 and partial overflow is possible. "
            "Correctness depends on quantized operand ranges and partition size.",
            file=sys.stderr,
        )
        grid_v = generate_k_spatial_combined_core_verilog(
            m, k, n, module_name=f"{name}_core", k_spatial=gemm_k_spatial, out_bits=out_bits
        )
    header_text = gen_public_header(
        name, m, k, n, grid_rows, grid_cols,
        result_type=output_precision,
        gemm_k_spatial=gemm_k_spatial,
        input_precision=input_precision,
        weight_precision=weight_precision,
        clock_period_ns=clock_period_ns,
        n_frames=n_frames,
    )
    _assert_core_port_widths(name, header_text, grid_v)
    _assert_core_first_out(name, m, k, n, gemm_k_spatial, grid_v)
    (pkg_dir / f"{name}_core.v").write_text(grid_v)
    (pkg_dir / "nnet_types.h").write_text(gen_nnet_types_header())
    (pkg_dir / f"{name}_gemm_ip.h").write_text(header_text)
    (pkg_dir / f"{name}_inst.cpp").write_text(gen_inst_cpp(name, m, k, n, interface))
    (pkg_dir / f"{name}_tb.cpp").write_text(gen_tb(name, m, k, n, interface, n_frames=n_frames))
    (pkg_dir / "run_catapult.tcl").write_text(gen_tcl(name, m, k, n, interface))
    print(f"Generated {pkg_dir}  (M={m}, K={k}, N={n}, interface={interface}, k_spatial={gemm_k_spatial})")


def _normalize_config_items(cfg):
    if isinstance(cfg, list):
        for item in cfg:
            item.setdefault("interface", "stream")
            item.setdefault("protocol", {})
            item.setdefault("gemm_ip_id", item.get("name"))
            item.setdefault("gemm_ip_index", None)
            item["gemm_k_spatial"] = _validate_gemm_k_spatial(
                int(item.get("gemm_k", item.get("k", item.get("n_in", 8)))),
                item.get("gemm_k_spatial"),
            )
        return cfg
    if isinstance(cfg, dict):
        if "m" in cfg and "k" in cfg and "n" in cfg and "name" in cfg:
            cfg.setdefault("interface", "stream")
            cfg.setdefault("protocol", {})
            cfg.setdefault("gemm_ip_id", cfg.get("name"))
            cfg.setdefault("gemm_ip_index", None)
            cfg["gemm_k_spatial"] = _validate_gemm_k_spatial(
                int(cfg.get("gemm_k", cfg.get("k", cfg.get("n_in", 8)))),
                cfg.get("gemm_k_spatial"),
            )
            return [cfg]
        items = []
        for name, item in cfg.items():
            interface = item.get("interface", "stream")
            items.append({
                "name": name,
                "m": item.get("gemm_m", 1),
                "k": item.get("gemm_k", item["n_in"]),
                "n": item.get("gemm_n", item["n_out"]),
                "interface": interface,
                "protocol": item.get("protocol", {}),
                "gemm_ip_id": item.get("gemm_ip_id", name),
                "gemm_ip_index": item.get("gemm_ip_index"),
                "output_precision": item.get("output_precision"),
                "input_precision": item.get("input_precision"),
                "weight_precision": item.get("weight_precision"),
                "clock_period_ns": item.get("clock_period_ns"),
                "gemm_k_spatial": _validate_gemm_k_spatial(
                    int(item.get("gemm_k", item.get("n_in", 8))),
                    item.get("gemm_k_spatial"),
                ),
            })
        return items
    raise TypeError("Unsupported config format")
