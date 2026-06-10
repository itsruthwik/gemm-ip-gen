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


def latency_cycles(m, k, n, grid_rows, grid_cols):
    """First-output cycle offset for the C++ simulation model.

    Matches the double-buffer behavioral grid timing:
      first_out = k_chunks × max(M,N) + max(0, K+N − k_chunks×max(M,N))
    This is the cycle where out_valid fires in the simulation model,
    offset from clk_cnt = 0 (first in_valid beat).
    """
    k_chunks = _ceil_div(k, LANE_WIDTH)
    input_beats = max(m, n)
    wait = max(0, k + n - k_chunks * input_beats)
    return k_chunks * input_beats + wait


def dead_cycles_raw(m, k, n, grid_cols):
    """Drain dead cycles before the C++ wrapper starts capturing output."""
    return latency_cycles(m, k, n, grid_rows=1, grid_cols=grid_cols) + 1


def dead_cycles(m, k, n, grid_cols):
    """Drain dead cycles + 1-cycle padding."""
    return dead_cycles_raw(m, k, n, grid_cols) + 1


def gen_public_header(name, m, k, n, grid_rows, grid_cols, result_type=None, gemm_k_spatial=1):
    row_chunk_bits = grid_rows * 64
    col_chunk_bits = grid_cols * 64
    c_bits = grid_cols * 128
    mr = grid_rows * 8
    k_chunks = _ceil_div(k, LANE_WIDTH)
    full_k_spatial = gemm_k_spatial == k_chunks
    # Full-K uses the NARROW per-beat word (one tile, 64 bits per K-chunk); the
    # wrapper RTL re-inserts the grid row/col tile offset by beat index, so the
    # deep input FIFO never stores the always-zero grid padding (area saving on
    # tiled designs).  Chunked keeps the single-chunk grid-padded width.
    a_bits = 64 * k_chunks if full_k_spatial else row_chunk_bits
    b_bits = 64 * k_chunks if full_k_spatial else col_chunk_bits
    bias_bits = col_chunk_bits
    input_beats = max(m, n)
    total_beats = input_beats if full_k_spatial else k_chunks * input_beats
    first_out = latency_cycles(m, k, n, grid_rows, grid_cols)
    blind = dead_cycles(m, k, n, grid_cols)
    drain = blind + mr

    # Choose the RHS expression for the final output assignment based on the
    # configured result type.  Integer types (ac_int / ac_uint) need an explicit
    # ``.to_int()`` call because directly casting an ``ac_fixed`` accumulator to
    # an ``ac_int`` may fail to compile or produce unexpected truncation.
    # Fixed-point types should preserve the normal AC-datatype conversion
    # (rounding / saturation) by omitting ``.to_int()``.
    assign_expr = "value.to_int()" if _is_ac_integer_type(result_type) else "value"
    a_el_expr = (
        f"a_buf[actual_row].slc<8>(k_chunk * 64 + k_lane * 8)"
        if full_k_spatial
        else f"a_buf[k_chunk * {input_beats} + actual_row].slc<8>(row_tile * 64 + k_lane * 8)"
    )
    b_el_expr = (
        f"b_buf[actual_col].slc<8>(k_chunk * 64 + k_lane * 8)"
        if full_k_spatial
        else f"b_buf[k_chunk * {input_beats} + actual_col].slc<8>(ct * 64 + k_lane * 8)"
    )
    if full_k_spatial:
        stream_feed_loop = f"""
    // Full K-spatial mode: feed each logical A row once. The blackbox A/B
    // words are widened to carry every 8-wide K chunk for the current row/col.
    #pragma hls_pipeline_init_interval 1
    FEED: for (int step = 0; step < {input_beats + 1}; step++) {{
        int t = (step == 0) ? 0 : step - 1;
        ac_int<{a_bits}, false> a_rows = 0;
        ac_int<{b_bits}, false> b_cols = 0;

        if (step > 0 && t < {m}) {{
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
        if (step > 0 && t < {n}) {{
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

        last_a_rows = a_rows;
        last_b_cols = b_cols;
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = (step == 0) ? 0 : 1;
        ac_int<1, false> feed_preload_valid = (step == 0) ? 1 : 0;
        gemm.run(last_a_rows, last_b_cols, bias_packed, feed_preload_valid, feed_valid, c_row, v, l);
        feed_dependency |= v;
    }}
"""
    else:
        stream_feed_loop = f"""
    // Replay storage is packed to the blackbox protocol. HLS4ML still emits
    // each logical K-wide A row once; later K chunks replay packed slices.
    ac_int<{a_bits}, false> a_replay[{k_chunks}][{input_beats}];

    // Feed M A rows and N B columns to the grid core as 8-lane K chunks.
    // Step 0 preloads bias, steps 1+ feed data beats. During K chunk 0,
    // read the logical A rows and prepack remaining chunks for replay.
    #pragma hls_pipeline_init_interval 1
    FEED: for (int step = 0; step < {k_chunks * input_beats + 1}; step++) {{
        int eff_step = (step == 0) ? 0 : step - 1;
        int kc = eff_step / {input_beats};
        int t = eff_step % {input_beats};
        ac_int<{a_bits}, false> a_rows = 0;
        ac_int<{b_bits}, false> b_cols = 0;

        if (step > 0 && t < {m}) {{
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
        if (step > 0 && t < {n}) {{
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

        last_a_rows = a_rows;
        last_b_cols = b_cols;
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = (step == 0) ? 0 : 1;
        ac_int<1, false> feed_preload_valid = (step == 0) ? 1 : 0;
        gemm.run(last_a_rows, last_b_cols, bias_packed, feed_preload_valid, feed_valid, c_row, v, l);
        feed_dependency |= v;
    }}
"""

    if full_k_spatial:
        array_feed_loop = f"""
    // Full K-spatial mode: feed each logical A row and B column once. The
    // blackbox A/B words are widened to carry every 8-wide K chunk.
    #pragma hls_pipeline_init_interval 1
    FEED_ARRAY: for (int step = 0; step < {input_beats + 1}; step++) {{
        int t = (step == 0) ? 0 : step - 1;
        ac_int<{a_bits}, false> a_rows_packed = 0;
        ac_int<{b_bits}, false> b_cols_packed = 0;

        if (step > 0 && t < {m}) {{
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
        if (step > 0 && t < {n}) {{
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

        last_a_rows = a_rows_packed;
        last_b_cols = b_cols_packed;
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = (step == 0) ? 0 : 1;
        ac_int<1, false> feed_preload_valid = (step == 0) ? 1 : 0;
        gemm.run(last_a_rows, last_b_cols, bias_packed, feed_preload_valid, feed_valid, c_row, v, l);
        feed_dependency |= v;
    }}
"""
    else:
        array_feed_loop = f"""
    // Feed M A rows and N B columns as 8-lane K chunks.
    // Step 0 preloads bias, steps 1+ feed data beats.
    #pragma hls_pipeline_init_interval 1
    FEED_ARRAY: for (int step = 0; step < {k_chunks * input_beats + 1}; step++) {{
        int eff_step = (step == 0) ? 0 : step - 1;
        int kc = eff_step / {input_beats};
        int t = eff_step % {input_beats};
        ac_int<{a_bits}, false> a_rows_packed = 0;
        ac_int<{b_bits}, false> b_cols_packed = 0;

        if (step > 0 && t < {m}) {{
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
        if (step > 0 && t < {n}) {{
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

        last_a_rows = a_rows_packed;
        last_b_cols = b_cols_packed;
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> feed_valid = (step == 0) ? 0 : 1;
        ac_int<1, false> feed_preload_valid = (step == 0) ? 1 : 0;
        gemm.run(last_a_rows, last_b_cols, bias_packed, feed_preload_valid, feed_valid, c_row, v, l);
        feed_dependency |= v;
    }}
"""

    return f"""\
#ifndef {name.upper()}_GEMM_IP_H
#define {name.upper()}_GEMM_IP_H

#include "ac_int.h"
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
            .delay(0.5)
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
        static ac_int<{a_bits}, false> a_buf[{total_beats}];
        static ac_int<{b_bits}, false> b_buf[{total_beats}];
        static int clk_cnt = 0;
        static bool running = false;
        static ac_int<{bias_bits}, false> bias_buf = 0;

        c_row = 0;
        out_valid = 0;
        out_last = 0;

        if (in_valid) {{
            if (!running) {{
                clk_cnt = 0;
                running = true;
                bias_buf = bias_cols;
            }}
            if (clk_cnt < {total_beats}) {{
                a_buf[clk_cnt] = a_rows;
                b_buf[clk_cnt] = b_cols;
            }}
        }}

        if (running && clk_cnt >= {first_out + 1} && clk_cnt < {first_out + 1} + {mr}) {{
            int out_idx = clk_cnt - ({first_out + 1});
            int row_tile = out_idx / 8;
            int row_local = out_idx % 8;
            int actual_row = row_tile * 8 + row_local;
            ac_int<{c_bits}, false> row_out = 0;
            for (int ct = 0; ct < {grid_cols}; ct++) {{
                for (int cl = 0; cl < 8; cl++) {{
                    int actual_col = ct * 8 + cl;
                    ac_int<32, true> acc = 0;
                    // Apply bias from bias_buf
                    ac_int<8, true> bias_el = bias_buf.slc<8>(ct * 64 + cl * 8);
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
                    ac_int<16, true> sat_val;
                    if (acc > 127) sat_val = 127;
                    else if (acc < -128) sat_val = -128;
                    else sat_val = acc;
                    row_out.set_slc(ct * 128 + cl * 16, sat_val);
                }}
            }}
            c_row = row_out;
            out_valid = 1;
            out_last = (out_idx == {mr} - 1) ? 1 : 0;
        }}

        clk_cnt++;
        if (running && clk_cnt >= {first_out + 1} + {mr}) {{
            running = false;
            clk_cnt = 0;
        }}
#endif
    }}
}};

template <class src_T>
ac_int<8, true> {name}_to_gemm_int8(const src_T &value) {{
    return static_cast<ac_int<8, true> >(value.to_int());
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
    int captured = 0;
    ac_int<1, false> feed_dependency = 0;
    ac_int<{a_bits}, false> last_a_rows = 0;
    ac_int<{b_bits}, false> last_b_cols = 0;
    ac_int<{bias_bits}, false> bias_packed = 0;

    #pragma hls_unroll
    BIAS_PACK: for (int col = 0; col < {n}; col++) {{
        int col_tile = col / 8;
        int col_local = col % 8;
        bias_packed.set_slc(col_tile * 64 + col_local * 8,
                            {name}_to_gemm_int8(biases[col]));
    }}

{stream_feed_loop}
    #pragma hls_pipeline_init_interval 1
    DRAIN: for (int i = 0; i < {blind + m}; i++) {{
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> drain_valid = 0;
        ac_int<1, false> drain_preload_valid = 0;
        gemm.run(last_a_rows, last_b_cols, bias_packed, drain_preload_valid, drain_valid, c_row, v, l);
        ac_int<1, false> output_valid = v | feed_dependency;
        if (output_valid) {{
            if (v && captured < {m}) {{
                res_T out_pack;
                #pragma hls_unroll
                for (int col = 0; col < {n}; col++) {{
                    int col_tile = col / 8;
                    int col_local = col % 8;
                    ac_int<16, true> raw_val = c_row.template slc<16>(col_tile * 128 + col_local * 16);
                    typename CONFIG_T::accum_t value =
                        static_cast<typename CONFIG_T::accum_t>(raw_val.to_int());
                    out_pack[col] = static_cast<typename res_T::value_type>({assign_expr});
                }}
                res_stream.write(out_pack);
            }}
            captured++;
        }}
    }}

    #pragma hls_pipeline_init_interval 1
    DRAIN_PADDED_ROWS: for (int i = 0; i < {mr - m}; i++) {{
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> drain_valid = 0;
        ac_int<1, false> drain_preload_valid = 0;
        gemm.run(last_a_rows, last_b_cols, bias_packed, drain_preload_valid, drain_valid, c_row, v, l);
    }}
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
    ac_int<1, false> feed_dependency = 0;
    ac_int<{a_bits}, false> last_a_rows = 0;
    ac_int<{b_bits}, false> last_b_cols = 0;
    ac_int<{bias_bits}, false> bias_packed = 0;
    ac_int<{a_bits}, false> preload_a_rows = 0;
    ac_int<{b_bits}, false> preload_b_cols = 0;

    #pragma hls_unroll
    BIAS_PACK_ARRAY: for (int col = 0; col < {n}; col++) {{
        int col_tile = col / 8;
        int col_local = col % 8;
        bias_packed.set_slc(col_tile * 64 + col_local * 8,
                            {name}_to_gemm_int8(biases[col]));
    }}

{array_feed_loop}

    #pragma hls_pipeline_init_interval 1
    DRAIN_ARRAY: for (int i = 0; i < {blind + m}; i++) {{
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        ac_int<1, false> drain_valid = 0;
        ac_int<1, false> drain_preload_valid = 0;
        gemm.run(last_a_rows, last_b_cols, bias_packed, drain_preload_valid, drain_valid, c_row, v, l);
        ac_int<1, false> output_valid = v | feed_dependency;
        if (output_valid) {{
            if (v && captured < {m}) {{
                res_T out_pack;
                #pragma hls_unroll
                for (int col = 0; col < {n}; col++) {{
                    int col_tile = col / 8;
                    int col_local = col % 8;
                    ac_int<16, true> raw_val = c_row.template slc<16>(col_tile * 128 + col_local * 16);
                    typename CONFIG_T::accum_t value =
                        static_cast<typename CONFIG_T::accum_t>(raw_val.to_int());
                    out_pack[col] = static_cast<typename res_T::value_type>({assign_expr});
                }}
                results[captured] = out_pack;
            }}
            captured++;
        }}
    }}

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
        if (&other == this)
            return *this;
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


def gen_tb(name, m, k, n, interface="stream"):
    if interface == "array":
        call_setup = f"""\
    a_beat_t a_rows[{m}];
    b_beat_t weight_cols[{n}];
    res_t results[{m}];

    for (int i = 0; i < {m}; i++) {{
        for (int kk = 0; kk < {k}; kk++) {{
            a_rows[i][kk] = activations[i][kk];
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
"""
        read_result = "        res_t out = results[i];"
    else:
        call_setup = f"""\
    ac_channel<a_beat_t> a_stream;
    ac_channel<b_beat_t> b_stream;
    ac_channel<res_t> res_stream;

    for (int i = 0; i < {m}; i++) {{
        a_beat_t a_beat;
        for (int kk = 0; kk < {k}; kk++) {{
            a_beat[kk] = activations[i][kk];
        }}
        a_stream.write(a_beat);
    }}
    for (int j = 0; j < {n}; j++) {{
        b_beat_t b_beat;
        for (int kk = 0; kk < {k}; kk++) {{
            b_beat[kk] = weights[j][kk];
        }}
        b_stream.write(b_beat);
    }}

#ifdef CCS_SCVERIFY
    CCS_DESIGN({name}_inst)(a_stream, b_stream, biases, res_stream);
#else
    nnet::{name}_gemm_ip_stream<a_beat_t, b_beat_t, int, res_t, {name}_config>(
        a_stream, b_stream, biases, res_stream);
#endif
"""
        read_result = "        res_t out = res_stream.read();"

    return f"""\
#ifdef CCS_SCVERIFY
#include "mc_testbench.h"
#include "mc_scverify.h"
#else
#include <stdio.h>
#endif

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
    typedef int accum_t;
}};

typedef nnet::array<ac_int<8, true>, {m}> a_beat_t;
typedef nnet::array<ac_int<8, true>, {n}> b_beat_t;
typedef nnet::array<ac_int<16, true>, {n}> res_t;

#ifdef CCS_SCVERIFY
CCS_MAIN(int argc, char *argv[]) {{
#else
int main() {{
#endif
    ac_int<8, true> activations[{m}][{k}];
    ac_int<8, true> weights[{n}][{k}];
    int biases[{n}];
    int failed = 0;

    for (int i = 0; i < {m}; i++) {{
        for (int kk = 0; kk < {k}; kk++) {{
            activations[i][kk] = ((i * 3 + kk - 4) & 0x7) - 4;
        }}
    }}

    for (int j = 0; j < {n}; j++) {{
        biases[j] = (j % 5) - 2;
        for (int kk = 0; kk < {k}; kk++) {{
            weights[j][kk] = ((j * 5 - kk + 1) & 0x7) - 3;
        }}
    }}

{call_setup}

    for (int i = 0; i < {m}; i++) {{
{read_result}
        for (int j = 0; j < {n}; j++) {{
            int gemm_acc = biases[j];
            for (int kk = 0; kk < {k}; kk++) {{
                gemm_acc += activations[i][kk].to_int() * weights[j][kk].to_int();
            }}
            if (gemm_acc > 127) gemm_acc = 127;
            else if (gemm_acc < -128) gemm_acc = -128;
            if (out[j].to_int() != gemm_acc) {{
                printf("Mismatch row %d col %d: got %d expected %d\\n",
                       i, j, out[j].to_int(), gemm_acc);
                failed = 1;
            }}
        }}
    }}

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
    typedef int accum_t;
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

directive set -DESIGN_GOAL area
directive set -SPECULATE true
directive set -MERGEABLE true
directive set -REGISTER_THRESHOLD 256
directive set -MEM_MAP_THRESHOLD 32
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


def generate_catapult_pkg(m, k, n, name, output_dir, interface="stream", output_precision=None, gemm_k_spatial=None):
    if interface not in ("stream", "array"):
        raise ValueError(f"Unsupported GEMM interface '{interface}' for {name}; expected stream or array")
    gemm_k_spatial = _validate_gemm_k_spatial(k, gemm_k_spatial)
    out_bits = _output_bits(output_precision)
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
    (pkg_dir / f"{name}_core.v").write_text(grid_v)
    (pkg_dir / "nnet_types.h").write_text(gen_nnet_types_header())
    (pkg_dir / f"{name}_gemm_ip.h").write_text(
        gen_public_header(
            name, m, k, n, grid_rows, grid_cols,
            result_type=output_precision,
            gemm_k_spatial=gemm_k_spatial,
        )
    )
    (pkg_dir / f"{name}_inst.cpp").write_text(gen_inst_cpp(name, m, k, n, interface))
    (pkg_dir / f"{name}_tb.cpp").write_text(gen_tb(name, m, k, n, interface))
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
                "gemm_k_spatial": _validate_gemm_k_spatial(
                    int(item.get("gemm_k", item.get("n_in", 8))),
                    item.get("gemm_k_spatial"),
                ),
            })
        return items
    raise TypeError("Unsupported config format")
