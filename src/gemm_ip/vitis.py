"""
Generate a Vitis HLS blackbox GEMM package.

The generated package is consumable by the Vivado/Vitis GEMM-IP hls4ml backend
through ``GEMM_IP_HEADER`` and ``add_files -blackbox``.

Per-package outputs (``<name>`` is the ``--name`` argument):

  ``<name>_wrapper.cpp``   C API model  (``hls::stream<ap_uint<W>>`` interface)
  ``<name>.v``             RTL wrapper module ``<name>_wrapper``
  ``<name>_wrapper.json``  Blackbox descriptor
  ``run_vitis.tcl``        Standalone Vitis HLS project script (Tcl)
  ``run_vitis.py``         Standalone Vitis 2025.2 Python HLS runner
  ``hls_config.cfg``       Vitis 2025.2 HLS component config

Run the standalone Tcl script with::

    # Vitis 2025.2
    vitis-run --mode hls --tcl run_vitis.tcl --work_dir ./vitis_ws

    # Vitis 2024.x / Vivado 2023.x
    vitis_hls -f run_vitis.tcl
  ``<name>_tb.cpp``        Standalone testbench
  ``<name>_design.cpp``    Standalone design top

When multiple packages are generated together:

  ``gemm_ip_combined.h``   Shape/id dispatch header for hls4ml integration
  ``integration_manifest.json``

Usage::

    # Single package
    python generate_vitis_pkg.py --m 8 --k 8 --n 8 --name gemm_8x8x8 --output_dir ./output

    # From gemm_config.json
    python generate_vitis_pkg.py --config gemm_config.json --output_dir ./output

Stream interface (``--interface stream``, the default):

  Three input streams, one output stream — all ``hls::stream<ap_uint<W>>``::

    void <name>_wrapper(
        hls::stream<ap_uint<A_WIDTH>>& a_stream,    // K beats, M rows packed
        hls::stream<ap_uint<B_WIDTH>>& b_stream,    // K beats, N cols packed
        hls::stream<ap_uint<B_WIDTH>>& bias_stream, // 1 beat,  N bias values packed
        hls::stream<ap_uint<C_WIDTH>>& c_stream     // M beats, N result cols packed
    );

Exact pack/unpack contract matches ``nnet_gemm_stream.h`` and
``stream_gemm_ip_native_beats`` from the Vivado hls4ml backend.
"""

import json
import re
import sys
from pathlib import Path

from gemm_ip.metadata import (
    LANE_WIDTH,
    _safe_name,
    _ceil_div,
    a_stream_width as _a_width,
    b_stream_width as _b_width,
    c_stream_width as _c_width,
    bias_stream_width as _bias_width,
    grid_rows,
    grid_cols,
    tail_mask_hex,
    vm,
    verilog_tile_comment,
    normalize_gemm_config,
    load_vitis_rtl_generator,
    load_vitis_combined_rtl_generator,
)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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


def _output_bits(item):
    """Result-lane width (bits) from the item's output_precision like
    'fixed<16,6,…>'. Drives the GEMM-IP output saturation/packing so the result
    honors output_precision instead of the legacy hardcoded int8 clamp. Returns 8
    (legacy int8) when unset/unparseable. The first ``fixed<>`` field is the
    total bit width."""
    op = item.get("output_precision")
    if not op:
        return 8
    m = re.search(r"u?fixed<\s*(\d+)", str(op))
    return int(m.group(1)) if m else 8


def normalize_vitis_items(cfg):
    items = normalize_gemm_config(cfg)
    for item in items:
        item["backend"] = "vitis"
        item["gemm_k_spatial"] = _validate_gemm_k_spatial(
            item["k"], item.get("gemm_k_spatial")
        )
    return items


def _k_chunks(item):
    return _ceil_div(item["k"], LANE_WIDTH)


def _full_k_spatial(item):
    return item.get("gemm_k_spatial", _k_chunks(item)) == _k_chunks(item)


def _a_bb_width(item):
    # Full-K-spatial uses the NARROW per-beat word: each beat carries one row's
    # K-data as 64*k_chunks bits (one 8-lane tile per K-chunk, at position 0).
    # The wrapper RTL re-inserts the grid_rows row-tile offset internally (routed
    # by beat index), so the deep input FIFO never stores the always-zero padding.
    # Chunked mode keeps the single-chunk width (grid_rows*64).
    if _full_k_spatial(item):
        return 64 * _k_chunks(item)
    return _a_width(item["m"])


def _b_bb_width(item):
    if _full_k_spatial(item):
        return 64 * _k_chunks(item)
    return _b_width(item["n"])


def _total_input_beats(item):
    input_beats = max(item["m"], item["n"])
    return input_beats if _full_k_spatial(item) else _k_chunks(item) * input_beats


# ── C API model (wrapper.cpp) ──────────────────────────────────────────────────


def _gen_wrapper_cpp(item):
    """Generate the C API model for Vitis HLS blackbox integration.

    The model follows the synthesis wrapper stream contract:
      1. Reads 1 bias beat from bias_stream, caches it
      2. Reads K_CHUNKS * max(M, N) row/column beats.  For each K chunk,
         beat t carries A row t's 8 K lanes and B column t's 8 K lanes.
      3. Reconstructs public row/column matrices, computes A @ B + bias
      4. Writes M result beats to c_stream
    """
    name = item["emit_name"]
    m, k_val, n = item["m"], item["k"], item["n"]
    gr, gc = item["grid_rows"], item["grid_cols"]
    aw = _a_bb_width(item)
    bw = _b_bb_width(item)
    biasw = _bias_width(n)
    out_bits = _output_bits(item)
    c_lane = out_bits
    c_tile = 8 * out_bits
    sat_pos = (1 << (out_bits - 1)) - 1
    sat_neg = (1 << (out_bits - 1))
    cw = _c_width(n, out_bits)
    k_chunks = _k_chunks(item)
    full_k_spatial = _full_k_spatial(item)
    input_beats = max(m, n)
    total_input_beats = _total_input_beats(item)
    read_loop = f"""\
    ReadFullK:
    for (int t = 0; t < {input_beats}; t++) {{
        #pragma HLS PIPELINE II=1
        ap_uint<{aw}> a_pkt = a_stream.read();
        ap_uint<{bw}> b_pkt = b_stream.read();
        if (t < {m}) {{
            // Narrow word: one row-tile, all K-chunks at position 0 (no row_tile offset).
            for (int kc = 0; kc < {k_chunks}; kc++) {{
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        a_rows[t][kk] = a_pkt.range(kc * 64 + kl * 8 + 7,
                                                    kc * 64 + kl * 8);
                    }}
                }}
            }}
        }}
        if (t < {n}) {{
            for (int kc = 0; kc < {k_chunks}; kc++) {{
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        b_cols[t][kk] = b_pkt.range(kc * 64 + kl * 8 + 7,
                                                    kc * 64 + kl * 8);
                    }}
                }}
            }}
        }}
    }}""" if full_k_spatial else f"""\
    ReadChunks:
    for (int kc = 0; kc < {k_chunks}; kc++) {{
        for (int t = 0; t < {input_beats}; t++) {{
            #pragma HLS PIPELINE II=1
            ap_uint<{aw}> a_pkt = a_stream.read();
            ap_uint<{bw}> b_pkt = b_stream.read();
            if (t < {m}) {{
                int row_tile = t / 8;
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        a_rows[t][kk] = a_pkt.range(row_tile * 64 + kl * 8 + 7,
                                                    row_tile * 64 + kl * 8);
                    }}
                }}
            }}
            if (t < {n}) {{
                int col_tile = t / 8;
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        b_cols[t][kk] = b_pkt.range(col_tile * 64 + kl * 8 + 7,
                                                    col_tile * 64 + kl * 8);
                    }}
                }}
            }}
        }}
    }}"""

    return f"""\
#include <hls_stream.h>
#include <ap_int.h>

// ── Behavioral C model for {name} ─────────────────────────────────────────────
// Dimensions: M={m}, K={k_val}, N={n}
// Grid: {gr}x{gc} tiles, {k_chunks} K chunks, {total_input_beats} total row/col beats
// K_SPATIAL={item.get("gemm_k_spatial", k_chunks)}
// Stream widths: A={aw}, B={bw}, Bias={biasw}, C={cw}

static ap_int<{c_lane}> saturated_result(ap_int<32> v) {{
    if (v > {sat_pos}) return {sat_pos};
    if (v < -{sat_neg}) return -{sat_neg};
    return v;
}}

void {name}_wrapper(
    hls::stream<ap_uint<{aw}>> &a_stream,
    hls::stream<ap_uint<{bw}>> &b_stream,
    hls::stream<ap_uint<{biasw}>> &bias_stream,
    hls::stream<ap_uint<{cw}>> &c_stream
) {{
    ap_int<8> a_rows[{m}][{k_val}];
    ap_int<8> b_cols[{n}][{k_val}];
    #pragma HLS ARRAY_PARTITION variable=a_rows complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b_cols complete dim=0

    // Bias register
    ap_int<8> bias_vals[{n}];
    #pragma HLS ARRAY_PARTITION variable=bias_vals complete dim=0

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 1: Read bias (1 beat)
    // ═══════════════════════════════════════════════════════════════════════
    {{
        ap_uint<{biasw}> bias_pkt = bias_stream.read();
    BiasUnpack:
        for (int col = 0; col < {gc}; col++) {{
            for (int lane = 0; lane < 8; lane++) {{
                int actual_col = col * 8 + lane;
                if (actual_col < {n}) {{
                    bias_vals[actual_col] = bias_pkt.range(col * 64 + lane * 8 + 7, col * 64 + lane * 8);
                }}
            }}
        }}
    }}

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 2: Read chunk-local row/column beats
    // ═══════════════════════════════════════════════════════════════════════
{read_loop}

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 3: Compute GEMM, add bias, saturate, write results
    // ═══════════════════════════════════════════════════════════════════════
    DrainRows:
    for (int actual_row = 0; actual_row < {m}; actual_row++) {{
        #pragma HLS PIPELINE II=1
        ap_uint<{cw}> out_pkt = 0;

    PackCol:
        for (int c = 0; c < {gc}; c++) {{
            for (int cl = 0; cl < 8; cl++) {{
                int actual_col = c * 8 + cl;
                ap_int<32> acc = 0;
                if (actual_col < {n}) {{
                    acc += bias_vals[actual_col];
                    for (int kk = 0; kk < {k_val}; kk++) {{
                        acc += (ap_int<32>)a_rows[actual_row][kk] * (ap_int<32>)b_cols[actual_col][kk];
                    }}
                }}
                ap_int<{c_lane}> result = saturated_result(acc);
                out_pkt.range(c * {c_tile} + cl * {c_lane} + {c_lane - 1}, c * {c_tile} + cl * {c_lane}) = result;
            }}
        }}
        c_stream.write(out_pkt);
    }}
}}
"""


# ── JSON blackbox descriptor ───────────────────────────────────────────────────


def _gen_json(item):
    """Generate the Vitis HLS blackbox JSON descriptor."""
    name = item["emit_name"]
    m, k_val, n = item["m"], item["k"], item["n"]
    gr, gc = item["grid_rows"], item["grid_cols"]

    k_chunks_val = _k_chunks(item)
    input_beats_val = max(m, n)
    total_input_beats = _total_input_beats(item)
    beh_grid = total_input_beats + max(0, k_val + n - total_input_beats) + m
    total_latency = beh_grid + 4  # Vitis wrapper overhead: bias + tvalid/tready handshake + negedge detect

    desc = {
        "c_function_name": f"{name}_wrapper",
        "rtl_top_module_name": f"{name}_wrapper",
        "c_files": [
            {"c_file": f"{name}/{name}_wrapper.cpp", "cflag": ""}
        ],
        "rtl_files": [
            f"{name}/{name}.v",
        ],
        "gemm_k_spatial": item.get("gemm_k_spatial", k_chunks_val),
        "c_parameters": [
            {
                "c_name": "a_stream",
                "c_port_direction": "in",
                "rtl_ports": {
                    "FIFO_data_read_in": "a_tdata",
                    "FIFO_read_enable": "a_tready",
                    "FIFO_empty_flag": "a_tvalid"
                },
                "c_global": True
            },
            {
                "c_name": "bias_stream",
                "c_port_direction": "in",
                "rtl_ports": {
                    "FIFO_data_read_in": "bias_tdata",
                    "FIFO_read_enable": "bias_tready",
                    "FIFO_empty_flag": "bias_tvalid"
                },
                "c_global": True
            },
            {
                "c_name": "b_stream",
                "c_port_direction": "in",
                "rtl_ports": {
                    "FIFO_data_read_in": "b_tdata",
                    "FIFO_read_enable": "b_tready",
                    "FIFO_empty_flag": "b_tvalid"
                },
                "c_global": True
            },
            {
                "c_name": "c_stream",
                "c_port_direction": "out",
                "rtl_ports": {
                    "FIFO_data_write_out": "c_tdata",
                    "FIFO_write_enable": "c_tvalid",
                    "FIFO_full_flag": "c_tready"
                },
                "c_global": True
            }
        ],
        "rtl_common_signal": {
            "module_clock": "ap_clk",
            "module_reset": "ap_rst",
            "module_clock_enable": "ap_ce",
            "ap_ctrl_chain_protocol_idle": "",
            "ap_ctrl_chain_protocol_start": "",
            "ap_ctrl_chain_protocol_ready": "",
            "ap_ctrl_chain_protocol_done": "",
            "ap_ctrl_chain_protocol_continue": ""
        },
        "rtl_performance": {
            "latency": str(total_latency),
            "II": str(total_input_beats),
            "II_contract": "behavioral_overlap_input_beats"
        },
        "rtl_resource_usage": {
            "FF": "1",
            "LUT": "1",
            "DSP": "0",
            "BRAM": "0",
            "URAM": "0"
        }
    }
    return json.dumps(desc, indent=2) + "\n"


# ── Standalone testbench ───────────────────────────────────────────────────────


def _gen_tb_cpp(item, num_vectors=10):
    """Generate a standalone Vitis HLS testbench driving the hls4ml stream interface.

    Feeds M K-wide A beats + N K-wide B beats via hls::stream, captures
    M N-wide result beats, and compares against a triple-nested-loop golden.
    """
    name = item["emit_name"]
    m, k_val, n = item["m"], item["k"], item["n"]
    nv = max(1, num_vectors)
    out_bits = _output_bits(item)
    c_lane = out_bits
    sat_pos = (1 << (out_bits - 1)) - 1
    sat_neg = (1 << (out_bits - 1))
    a_beat_w = k_val * 8
    b_beat_w = k_val * 8
    c_row_w  = n * out_bits
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#include <hls_stream.h>
#include <ap_int.h>

#define NUM_VECTORS {nv}

static ap_int<{c_lane}> saturated_int8(ap_int<32> v) {{
    if (v > {sat_pos}) return {sat_pos};
    if (v < -{sat_neg}) return -{sat_neg};
    return v;
}}

extern void {name}_design(
    hls::stream<ap_uint<{a_beat_w}>> &a_beat_stream,
    hls::stream<ap_uint<{b_beat_w}>> &b_beat_stream,
    ap_int<8>                        biases[{n}],
    hls::stream<ap_uint<{c_row_w}>> &res_stream
);

int main() {{
    int total_failed = 0;

    for (int vec = 0; vec < NUM_VECTORS; vec++) {{
        hls::stream<ap_uint<{a_beat_w}>> a_beats("a_beats");
        hls::stream<ap_uint<{b_beat_w}>> b_beats("b_beats");
        hls::stream<ap_uint<{c_row_w}>> res_beats("res_beats");

        ap_int<8> activations[{m}][{k_val}];
        ap_int<8> weights[{n}][{k_val}];
        ap_int<8> biases[{n}];

        int seed = vec * 7 + 3;

        // ── Generate deterministic test data ──────────────────────────────
        for (int i = 0; i < {m}; i++) {{
            for (int kk = 0; kk < {k_val}; kk++) {{
                int raw = (i * seed + kk * (vec + 5) - 4) & 0x7F;
                activations[i][kk] = raw - 64;
            }}
        }}

        for (int j = 0; j < {n}; j++) {{
            biases[j] = ((j + vec) % 5) - 2;
            for (int kk = 0; kk < {k_val}; kk++) {{
                int raw = (j * (seed + 3) - kk * (vec + 1) + seed) & 0x7F;
                weights[j][kk] = raw - 64;
            }}
        }}

        // ── Golden reference ───────────────────────────────────────────────
        ap_int<{c_lane}> expected[{m}][{n}];
        for (int i = 0; i < {m}; i++) {{
            for (int j = 0; j < {n}; j++) {{
                ap_int<32> acc = biases[j];
                for (int kk = 0; kk < {k_val}; kk++) {{
                    acc += (ap_int<32>)activations[i][kk] * (ap_int<32>)weights[j][kk];
                }}
                expected[i][j] = saturated_int8(acc);
            }}
        }}

        // ── Stream K-wide A rows ───────────────────────────────────────────
        for (int i = 0; i < {m}; i++) {{
            ap_uint<{a_beat_w}> a_beat = 0;
            for (int kk = 0; kk < {k_val}; kk++)
                a_beat.range(kk * 8 + 7, kk * 8) = activations[i][kk];
            a_beats.write(a_beat);
        }}

        // ── Stream K-wide B columns ────────────────────────────────────────
        for (int j = 0; j < {n}; j++) {{
            ap_uint<{b_beat_w}> b_beat = 0;
            for (int kk = 0; kk < {k_val}; kk++)
                b_beat.range(kk * 8 + 7, kk * 8) = weights[j][kk];
            b_beats.write(b_beat);
        }}

        // ── Call design ────────────────────────────────────────────────────
        {name}_design(a_beats, b_beats, biases, res_beats);

        // ── Check N-wide result rows ───────────────────────────────────────
        int vec_failed = 0;
        for (int i = 0; i < {m}; i++) {{
            ap_uint<{c_row_w}> res = res_beats.read();
            for (int j = 0; j < {n}; j++) {{
                ap_int<{c_lane}> got = res.range(j * {c_lane} + {c_lane - 1}, j * {c_lane});
                if (got != expected[i][j]) {{
                    printf("MISMATCH vec=%d [%d][%d]: got %d expected %d\\n",
                           vec, i, j, (int)got, (int)expected[i][j]);
                    vec_failed = 1;
                }}
            }}
        }}

        if (vec_failed) {{
            printf("Vector %d FAILED\\n", vec);
            total_failed = 1;
        }} else {{
            printf("Vector %d passed\\n", vec);
        }}
    }}

    if (total_failed) {{
        printf("\\nTest FAILED (%d vectors)\\n", NUM_VECTORS);
        return 1;
    }}
    printf("\\nAll %d vectors passed\\n", NUM_VECTORS);
    return 0;
}}
"""


# ── Standalone design top ──────────────────────────────────────────────────────


def _gen_design_cpp(item):
    """Generate a standalone Vitis HLS design top with the hls4ml stream interface.

    For K ≤ 8 (single chunk): streams K-wide beats directly into chunk beats
    without buffering, minimising latency and resource usage.

    For K > 8 (multi-chunk): buffers K-wide A rows and B columns, then replays
    them as 8-lane K-chunk beats into the blackbox wrapper.
    """
    name = item["emit_name"]
    m, k_val, n = item["m"], item["k"], item["n"]
    gr, gc = item["grid_rows"], item["grid_cols"]
    aw = _a_bb_width(item)
    bw = _b_bb_width(item)
    biasw = _bias_width(n)
    out_bits = _output_bits(item)
    c_lane = out_bits
    c_tile = 8 * out_bits
    sat_pos = (1 << (out_bits - 1)) - 1
    sat_neg = (1 << (out_bits - 1))
    cw = _c_width(n, out_bits)
    k_chunks = _k_chunks(item)
    full_k_spatial = _full_k_spatial(item)
    input_beats = max(m, n)
    total_input_beats = _total_input_beats(item)

    a_beat_w = k_val * 8
    b_beat_w = k_val * 8
    c_row_w  = n * out_bits

    # Build the chunk-streaming section: direct for single-chunk, buffered for multi-chunk
    if full_k_spatial:
        chunk_section = f"""\
    // Full K-spatial: one widened blackbox beat per logical row/column
    for (int t = 0; t < {input_beats}; t++) {{
        #pragma HLS PIPELINE II=1
        ap_uint<{aw}> a_pkt = 0;
        ap_uint<{bw}> b_pkt = 0;
        if (t < {m}) {{
            ap_uint<{a_beat_w}> a_beat = a_beat_stream.read();
            // Narrow word: one row-tile, all K-chunks at position 0.
            for (int kc = 0; kc < {k_chunks}; kc++) {{
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        a_pkt.range(kc * 64 + kl * 8 + 7,
                                    kc * 64 + kl * 8) =
                            a_beat.range(kk * 8 + 7, kk * 8);
                    }}
                }}
            }}
        }}
        if (t < {n}) {{
            ap_uint<{b_beat_w}> b_beat = b_beat_stream.read();
            for (int kc = 0; kc < {k_chunks}; kc++) {{
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        b_pkt.range(kc * 64 + kl * 8 + 7,
                                    kc * 64 + kl * 8) =
                            b_beat.range(kk * 8 + 7, kk * 8);
                    }}
                }}
            }}
        }}
        a_chunk.write(a_pkt);
        b_chunk.write(b_pkt);
    }}"""
    elif k_chunks == 1:
        chunk_section = f"""\
    // Single-chunk (K={k_val} <= 8): K-wide beats streamed directly into chunk beats
    for (int t = 0; t < {input_beats}; t++) {{
        #pragma HLS PIPELINE II=1
        ap_uint<{aw}> a_pkt = 0;
        if (t < {m}) {{
            ap_uint<{a_beat_w}> a_beat = a_beat_stream.read();
            int row_tile = t / 8;
            for (int kl = 0; kl < 8; kl++) {{
                int kk = kl;
                if (kk < {k_val}) {{
                    a_pkt.range(row_tile * 64 + kl * 8 + 7, row_tile * 64 + kl * 8) =
                        a_beat.range(kk * 8 + 7, kk * 8);
                }}
            }}
        }}
        a_chunk.write(a_pkt);

        ap_uint<{bw}> b_pkt = 0;
        if (t < {n}) {{
            ap_uint<{b_beat_w}> b_beat = b_beat_stream.read();
            int col_tile = t / 8;
            for (int kl = 0; kl < 8; kl++) {{
                int kk = kl;
                if (kk < {k_val}) {{
                    b_pkt.range(col_tile * 64 + kl * 8 + 7, col_tile * 64 + kl * 8) =
                        b_beat.range(kk * 8 + 7, kk * 8);
                }}
            }}
        }}
        b_chunk.write(b_pkt);
    }}"""
    else:
        chunk_section = f"""\
    // Multi-chunk (K={k_val} > 8, {k_chunks} chunks): buffer then replay
    ap_int<8> a_buf[{m}][{k_val}];
    ap_int<8> b_buf[{n}][{k_val}];
    #pragma HLS ARRAY_PARTITION variable=a_buf complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b_buf complete dim=0

    for (int i = 0; i < {m}; i++) {{
        ap_uint<{a_beat_w}> a_beat = a_beat_stream.read();
        for (int kk = 0; kk < {k_val}; kk++)
            a_buf[i][kk] = a_beat.range(kk * 8 + 7, kk * 8);
    }}

    for (int j = 0; j < {n}; j++) {{
        ap_uint<{b_beat_w}> b_beat = b_beat_stream.read();
        for (int kk = 0; kk < {k_val}; kk++)
            b_buf[j][kk] = b_beat.range(kk * 8 + 7, kk * 8);
    }}

    for (int kc = 0; kc < {k_chunks}; kc++) {{
        for (int t = 0; t < {input_beats}; t++) {{
            ap_uint<{aw}> a_pkt = 0;
            ap_uint<{bw}> b_pkt = 0;

            if (t < {m}) {{
                int row_tile = t / 8;
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        a_pkt.range(row_tile * 64 + kl * 8 + 7, row_tile * 64 + kl * 8) =
                            a_buf[t][kk];
                    }}
                }}
            }}

            if (t < {n}) {{
                int col_tile = t / 8;
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        b_pkt.range(col_tile * 64 + kl * 8 + 7, col_tile * 64 + kl * 8) =
                            b_buf[t][kk];
                    }}
                }}
            }}

            a_chunk.write(a_pkt);
            b_chunk.write(b_pkt);
        }}
    }}"""

    return f"""\
#include <hls_stream.h>
#include <ap_int.h>

extern void {name}_wrapper(
    hls::stream<ap_uint<{aw}>> &a_stream,
    hls::stream<ap_uint<{bw}>> &b_stream,
    hls::stream<ap_uint<{biasw}>> &bias_stream,
    hls::stream<ap_uint<{cw}>> &c_stream
);

// hls4ml-interface standalone top: K-wide streams -> chunked blackbox -> N-wide results
void {name}_design(
    hls::stream<ap_uint<{a_beat_w}>> &a_beat_stream,
    hls::stream<ap_uint<{b_beat_w}>> &b_beat_stream,
    ap_int<8>                        biases[{n}],
    hls::stream<ap_uint<{c_row_w}>> &res_stream
) {{
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Internal chunked streams for the blackbox wrapper
    hls::stream<ap_uint<{aw}>> a_chunk("a_chunk");
    hls::stream<ap_uint<{bw}>> b_chunk("b_chunk");
    hls::stream<ap_uint<{biasw}>> bias_chunk("bias_chunk");
    hls::stream<ap_uint<{cw}>> c_chunk("c_chunk");

    #pragma HLS STREAM variable=a_chunk depth={total_input_beats}
    #pragma HLS STREAM variable=b_chunk depth={total_input_beats}
    #pragma HLS STREAM variable=bias_chunk depth=2
    #pragma HLS STREAM variable=c_chunk depth={m}
    #pragma HLS DATAFLOW

    // -- Pack bias ----------------------------------------------------------
    {{
        ap_uint<{biasw}> bias_pkt = 0;
        for (int c = 0; c < {gc}; c++) {{
            for (int lane = 0; lane < 8; lane++) {{
                int actual_col = c * 8 + lane;
                if (actual_col < {n}) {{
                    bias_pkt.range(c * 64 + lane * 8 + 7, c * 64 + lane * 8) =
                        biases[actual_col];
                }}
            }}
        }}
        bias_chunk.write(bias_pkt);
    }}

{chunk_section}

    // -- Blackbox wrapper --------------------------------------------------
    {name}_wrapper(a_chunk, b_chunk, bias_chunk, c_chunk);

    // -- Unpack chunked output rows -> N-wide hls4ml result beats ----------
    for (int i = 0; i < {m}; i++) {{
        #pragma HLS PIPELINE II=1
        ap_uint<{cw}> out_pkt = c_chunk.read();
        ap_uint<{c_row_w}> row = 0;
        for (int c = 0; c < {gc}; c++) {{
            for (int lane = 0; lane < 8; lane++) {{
                int actual_col = c * 8 + lane;
                if (actual_col < {n}) {{
                    row.range(actual_col * {c_lane} + {c_lane - 1}, actual_col * {c_lane}) =
                        out_pkt.range(c * {c_tile} + lane * {c_lane} + {c_lane - 1}, c * {c_tile} + lane * {c_lane});
                }}
            }}
        }}
        res_stream.write(row);
    }}
}}
"""


# ── Standalone Vitis HLS scripts ───────────────────────────────────────────────


def _gen_tcl(item, output_dir):
    """Generate a Vitis HLS Tcl script for standalone compilation.

    Can be run with either:
      - Vitis 2025.2:  vitis-run --mode hls --tcl run_vitis.tcl --work_dir ./vitis_ws
      - Vitis 2024.x:  vitis_hls -f run_vitis.tcl
      - Vivado 2023.x: vivado_hls -f run_vitis.tcl
    """
    name = item["emit_name"]
    m, k_val, n = item["m"], item["k"], item["n"]
    return f"""\
# Vitis HLS project for {name}_wrapper
# Dimensions: M={m}, K={k_val}, N={n}
# Generated by gemm-ip-gen
#
# Usage:
#   Vitis 2025.2:  vitis-run --mode hls --tcl run_vitis.tcl --work_dir ./vitis_ws
#   Vitis 2024.x:  vitis_hls -f run_vitis.tcl
#   Vivado 2023.x: vivado_hls -f run_vitis.tcl

open_project {name}_proj
set_top {name}_design

add_files {name}/{name}_design.cpp
add_files -tb {name}/{name}_tb.cpp

# Add blackbox descriptor. The JSON lists the C model and RTL files.
add_files -blackbox {name}/{name}_wrapper.json

open_solution "solution1"
set_part {{xcvu13p-flga2577-2-e}}
create_clock -period 5 -name default

config_interface -m_axi_addr64
config_compile -pipeline_loops 1

csim_design
csynth_design
exit
"""


def _gen_hls_config(item):
    """Generate a Vitis 2025.2 HLS component config file."""
    name = item["emit_name"]
    return f"""\
part=xcvu13p-flga2577-2-e

[hls]
syn.top={name}_design
syn.file={name}/{name}_design.cpp
tb.file={name}/{name}_tb.cpp
syn.blackbox.file={name}/{name}_wrapper.json
clock=5ns
syn.compile.pipeline_loops=1
csim.code_analyzer=false
"""


def _gen_python_runner(item):
    """Generate a Vitis 2025.2 Python script for standalone HLS execution."""
    name = item["emit_name"]
    return f"""\
#!/usr/bin/env python3
\"\"\"Run the standalone Vitis HLS component for {name}.

Usage:
  vitis -s run_vitis.py --workspace ./vitis_ws
  vitis -s run_vitis.py --operations C_SIMULATION SYNTHESIS CO_SIMULATION
\"\"\"

import argparse
import json
import os
import tempfile
from pathlib import Path

import vitis


def write_resolved_blackbox_json(package_dir):
    src_json = package_dir / "{name}_wrapper.json"
    resolved_json = Path(tempfile.mkdtemp(prefix="{name}_blackbox_")) / "{name}_wrapper.json"
    desc = json.loads(src_json.read_text())
    desc["c_files"] = [
        {{
            "c_file": str(package_dir / "{name}_wrapper.cpp"),
            "cflag": entry.get("cflag", ""),
        }}
        for entry in desc.get("c_files", [])
    ]
    desc["rtl_top_module_name"] = "{name}_wrapper"
    desc["rtl_files"] = [
        str(package_dir / "{name}.v"),
    ]
    resolved_json.write_text(json.dumps(desc, indent=2) + "\\n")
    return resolved_json


def write_resolved_cfg(package_dir):
    resolved_json = write_resolved_blackbox_json(package_dir)
    resolved = Path(tempfile.mkdtemp(prefix="{name}_hls_cfg_")) / "hls_config.cfg"
    resolved.write_text(
        "\\n".join([
            "part=xcvu13p-flga2577-2-e",
            "",
            "[hls]",
            "syn.top={name}_design",
            f"syn.file={{package_dir / '{name}_design.cpp'}}",
            f"tb.file={{package_dir / '{name}_tb.cpp'}}",
            f"syn.blackbox.file={{resolved_json}}",
            "clock=5ns",
            "syn.compile.pipeline_loops=1",
            "csim.code_analyzer=false",
            "",
        ])
    )
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="{name}_vitis_ws")
    parser.add_argument("--component", default="{name}_component")
    parser.add_argument("--cfg", default="hls_config.cfg")
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["C_SIMULATION", "SYNTHESIS"],
        choices=[
            "C_SIMULATION",
            "SYNTHESIS",
            "CO_SIMULATION",
            "IMPLEMENTATION",
            "ANALYSIS_OPTIMIZATION",
            "PACKAGE",
        ],
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    cfg_path = script_dir / args.cfg
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing HLS config: {{cfg_path}}")
    resolved_cfg_path = write_resolved_cfg(script_dir)

    run_dir = script_dir.parent
    client = vitis.create_client()
    try:
        workspace = Path(args.workspace)
        if not workspace.is_absolute():
            workspace = run_dir / workspace
        workspace.mkdir(parents=True, exist_ok=True)
        client.set_workspace(str(workspace))

        old_cwd = Path.cwd()
        os.chdir(run_dir)
        try:
            comp = client.create_hls_component(
                name=args.component,
                cfg_file=str(resolved_cfg_path),
            )
            comp.report()

            for operation in args.operations:
                print(f"\\n=== Running {{operation}} ===")
                comp.run(operation)
        finally:
            os.chdir(old_cwd)
    finally:
        vitis.dispose()


if __name__ == "__main__":
    main()
"""


# ── Combined dispatch header for hls4ml integration ────────────────────────────


def gen_combined_header(items):
    """Generate the Vitis ``gemm_ip_combined.h`` dispatch header.

    This header is included by hls4ml's ``nnet_gemm_ip.h`` when
    ``GEMM_IP_HEADER`` is defined.  It provides the typed
    ``nnet::gemm_ip_stream<a_beat_T, b_beat_T, bias_T, res_T, CONFIG_T>``
    that the hls4ml backend expects.

    The header packs/unpacks typed ``nnet::array<>`` beats to/from
    ``ap_uint<W>`` streams that the blackbox C model consumes.
    """
    includes = "\n".join(
        f'#include "{item["emit_name"]}/{item["emit_name"]}_gemm_ip.h"'
        for item in items
    )

    stream_branches = []
    for item in items:
        target = item.get("interface", "stream")
        if target == "array":
            continue
        name = item["emit_name"]
        m_val, k_val, n_val = item["m"], item["k"], item["n"]
        shape_cond = f"CONFIG_T::gemm_m == {m_val} && CONFIG_T::gemm_k == {k_val} && CONFIG_T::gemm_n == {n_val}"
        if item.get("gemm_ip_index") is not None:
            shape_cond = f"CONFIG_T::gemm_ip_id == {item['gemm_ip_index']} && {shape_cond}"

        stream_branches.append(f"""\
    if constexpr ({shape_cond}) {{
        {name}_gemm_ip_stream<a_beat_T, b_beat_T, bias_T, res_T, CONFIG_T>(
            a_beat_stream, b_beat_stream, biases, res_stream);
    }}""")

    stream_branches_text = "\n else ".join(stream_branches)
    if not stream_branches_text:
        stream_branches_text = """\
    static_assert(CONFIG_T::gemm_m == 0,
                  "No generated Vitis GEMM IP stream implementation is present.");"""
    else:
        stream_branches_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated Vitis GEMM IP stream implementation matches this CONFIG_T.");
    }"""

    array_branches = []
    for item in items:
        target = item.get("interface", "stream")
        if target != "array":
            continue
        name = item["emit_name"]
        m_val, k_val, n_val = item["m"], item["k"], item["n"]
        shape_cond = f"CONFIG_T::gemm_m == {m_val} && CONFIG_T::gemm_k == {k_val} && CONFIG_T::gemm_n == {n_val}"
        if item.get("gemm_ip_index") is not None:
            shape_cond = f"CONFIG_T::gemm_ip_id == {item['gemm_ip_index']} && {shape_cond}"
        array_branches.append(f"""\
    if constexpr ({shape_cond}) {{
        {name}_gemm_ip_array<a_beat_T, b_beat_T, bias_T, res_T, CONFIG_T>(
            a_beats, b_beats, biases, results);
    }}""")

    array_branches_text = "\n else ".join(array_branches)
    if not array_branches_text:
        array_branches_text = """\
    static_assert(CONFIG_T::gemm_m == 0,
                  "No generated Vitis GEMM IP array implementation is present.");"""
    else:
        array_branches_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated Vitis GEMM IP array implementation matches this CONFIG_T.");
    }"""

    return f"""\
#ifndef GEMM_IP_COMBINED_H_
#define GEMM_IP_COMBINED_H_

// Vitis GEMM blackbox combined dispatch header.
// Generated by gemm-ip-gen for Vitis backend.
// Include path added via -DGEMM_IP_HEADER and -I<pkg_dir>.

#include "hls_stream.h"
#include "ap_int.h"
#include <cstddef>

// nnet::array<> provided by firmware nnet_utils/nnet_types.h
// (always included via nnet_gemm_behavioral.h -> nnet_gemm_ip.h)

{includes}

namespace nnet {{

// ── Stream interface ──────────────────────────────────────────────────────────

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_stream(
    hls::stream<a_beat_T> &a_beat_stream,
    hls::stream<b_beat_T> &b_beat_stream,
    bias_T biases[CONFIG_T::gemm_n],
    hls::stream<res_T> &res_stream
) {{
{stream_branches_text}
}}

// ── Array interface ───────────────────────────────────────────────────────────

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_array(
    a_beat_T a_beats[CONFIG_T::gemm_m],
    b_beat_T b_beats[CONFIG_T::gemm_n],
    bias_T biases[CONFIG_T::gemm_n],
    res_T results[CONFIG_T::gemm_m]
) {{
{array_branches_text}
}}

// ── Simulation helpers are provided by nnet_gemm_behavioral.h,
//     which is always included by nnet_gemm_ip.h.
//     gemm_ip_stream_sim and gemm_ip_array_sim are defined there.

}} // namespace nnet

#endif // GEMM_IP_COMBINED_H_
"""


# ── Per-layer adapter header (included by combined header) ─────────────────────


def _gen_layer_gemm_ip_h(item):
    """Generate ``{name}_gemm_ip.h`` — the typed-beat adapter for this layer.

    This header is what ``gemm_ip_combined.h`` includes.  It defines:
      - ``{name}_gemm_ip_stream`` — packs typed beats → ap_uint, calls blackbox
      - ``{name}_gemm_ip_array``   — same math, array-based

    Protocol: hls4ml feeds M A rows and N B columns, each K-wide.  The
    adapter buffers those public row/column beats and replays them to the
    blackbox as 8-lane K chunks.
    """
    name = item["emit_name"]
    m_val, k_val, n_val = item["m"], item["k"], item["n"]
    gr, gc = item["grid_rows"], item["grid_cols"]
    aw = _a_bb_width(item)
    bw = _b_bb_width(item)
    biasw = _bias_width(n_val)
    out_bits = _output_bits(item)
    c_lane = out_bits
    c_tile = 8 * out_bits
    cw = _c_width(n_val, out_bits)
    k_chunks = _k_chunks(item)
    full_k_spatial = _full_k_spatial(item)
    input_beats = max(m_val, n_val)
    total_input_beats = _total_input_beats(item)

    # ── Narrow packed word (RTL re-expands) ─────────────────────────────────────
    # Full-K: each beat carries one row's K-data as 64*k_chunks bits (one 8-lane
    # tile per K-chunk, position 0).  The wrapper RTL re-inserts the grid_rows /
    # grid_cols tile offset internally, routed by beat index, so the deep input
    # FIFO never stores the always-zero padding (BRAM saving on tiled designs).
    # aw/bw already reflect this narrow width via _a_bb_width/_b_bb_width.
    pingpong_depth = total_input_beats

    if full_k_spatial:
        pack_body = f"""\
    // Single-pass narrow pack: write each row's / column's K-data at position 0
    // (one 8-lane tile per K-chunk).  Constant bit offsets; the wrapper RTL routes
    // the tile to the right grid row/col by beat index.
    for (int t = 0; t < {input_beats}; t++) {{
        #pragma HLS PIPELINE II=1
        ap_uint<{aw}> a_pkt = 0;
        ap_uint<{bw}> b_pkt = 0;

        if (t < {m_val}) {{
            a_beat_T a_beat = a_beat_stream.read();
            for (int kc = 0; kc < {k_chunks}; kc++) {{
                #pragma HLS UNROLL
                for (int kl = 0; kl < 8; kl++) {{
                    #pragma HLS UNROLL
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        a_pkt.range(kc * 64 + kl * 8 + 7, kc * 64 + kl * 8) =
                            static_cast<ap_int<8>>(a_beat[kk]);
                    }}
                }}
            }}
        }}

        if (t < {n_val}) {{
            b_beat_T b_beat = b_beat_stream.read();
            for (int kc = 0; kc < {k_chunks}; kc++) {{
                #pragma HLS UNROLL
                for (int kl = 0; kl < 8; kl++) {{
                    #pragma HLS UNROLL
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        b_pkt.range(kc * 64 + kl * 8 + 7, kc * 64 + kl * 8) =
                            static_cast<ap_int<8>>(b_beat[kk]);
                    }}
                }}
            }}
        }}

        a_packed.write(a_pkt);
        b_packed.write(b_pkt);
    }}"""
    else:
        # Chunked mode replays each row/column across k_chunks output beats, so the
        # input stream (consumed once) is buffered first.
        pack_body = f"""\
    // Buffer the typed row/column beats into flat ap_int<8> arrays.  We avoid an
    // array-of-nnet::array (a_beat_T a_rows[M]) here: Vitis csynth cannot lower the
    // pointer reinterpretation it implies (i9* -> [8 x i9]*) and trips the operator=
    // pointer comparison in nnet::array.  A flat 2D ap_int<8> buffer synthesizes
    // cleanly and matches the blackbox design-side packing.
    ap_int<8> a_rows[{m_val}][{k_val}];
    ap_int<8> b_cols[{n_val}][{k_val}];
    #pragma HLS ARRAY_PARTITION variable=a_rows complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b_cols complete dim=0

    for (int row = 0; row < {m_val}; row++) {{
        #pragma HLS PIPELINE II=1
        a_beat_T a_beat = a_beat_stream.read();
        for (int kk = 0; kk < {k_val}; kk++) {{
            #pragma HLS UNROLL
            a_rows[row][kk] = static_cast<ap_int<8>>(a_beat[kk]);
        }}
    }}
    for (int col = 0; col < {n_val}; col++) {{
        #pragma HLS PIPELINE II=1
        b_beat_T b_beat = b_beat_stream.read();
        for (int kk = 0; kk < {k_val}; kk++) {{
            #pragma HLS UNROLL
            b_cols[col][kk] = static_cast<ap_int<8>>(b_beat[kk]);
        }}
    }}

    // Pack row/column beats as 8-lane K chunks.
    for (int kc = 0; kc < {k_chunks}; kc++) {{
        for (int t = 0; t < {input_beats}; t++) {{
            #pragma HLS PIPELINE II=1
            ap_uint<{aw}> a_pkt = 0;
            ap_uint<{bw}> b_pkt = 0;

            if (t < {m_val}) {{
                int row_tile = t / 8;
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        a_pkt.range(row_tile * 64 + kl * 8 + 7, row_tile * 64 + kl * 8) =
                            static_cast<ap_int<8>>(a_rows[t][kk]);
                    }}
                }}
            }}

            if (t < {n_val}) {{
                int col_tile = t / 8;
                for (int kl = 0; kl < 8; kl++) {{
                    int kk = kc * 8 + kl;
                    if (kk < {k_val}) {{
                        b_pkt.range(col_tile * 64 + kl * 8 + 7, col_tile * 64 + kl * 8) =
                            static_cast<ap_int<8>>(b_cols[t][kk]);
                    }}
                }}
            }}

            a_packed.write(a_pkt);
            b_packed.write(b_pkt);
        }}
    }}"""

    # Narrow-everywhere: pack writes the narrow word straight to a single FIFO and
    # the blackbox re-expands internally.  No expand process, no second FIFO.
    expand_defs = ""

    stream_block = f"""\
    hls::stream<ap_uint<{aw}>> a_packed("a_packed");
    hls::stream<ap_uint<{bw}>> b_packed("b_packed");
    hls::stream<ap_uint<{biasw}>> bias_packed("bias_packed");
    hls::stream<ap_uint<{cw}>> c_packed("c_packed");
    #pragma HLS STREAM variable=a_packed depth={pingpong_depth}
    #pragma HLS STREAM variable=b_packed depth={pingpong_depth}
    #pragma HLS STREAM variable=bias_packed depth=1
    #pragma HLS STREAM variable=c_packed depth={m_val}

    {name}_gemm_ip_pack<a_beat_T, b_beat_T, bias_T, CONFIG_T>(
        a_beat_stream, b_beat_stream, biases, a_packed, b_packed, bias_packed);

    {name}_wrapper(a_packed, b_packed, bias_packed, c_packed);

    {name}_gemm_ip_unpack<res_T, CONFIG_T>(c_packed, res_stream);"""

    return f"""\
#ifndef {name.upper()}_GEMM_IP_H_
#define {name.upper()}_GEMM_IP_H_

// Vitis blackbox adapter for {name} (M={m_val}, K={k_val}, N={n_val}).
// Packs/unpacks typed nnet::array beats to/from ap_uint streams.
// Included by gemm_ip_combined.h under GEMM_IP_HEADER.

#include "hls_stream.h"
#include "ap_int.h"

extern void {name}_wrapper(
    hls::stream<ap_uint<{aw}>> &a_stream,
    hls::stream<ap_uint<{bw}>> &b_stream,
    hls::stream<ap_uint<{biasw}>> &bias_stream,
    hls::stream<ap_uint<{cw}>> &c_stream
);

namespace nnet {{

// Pack stage: read typed row/column beats and the bias vector, emit ap_uint beats
// for the blackbox. a_rows/b_cols are local to this process so the enclosing
// DATAFLOW region stays canonical (only hls::stream FIFOs cross process boundaries).
template <class a_beat_T, class b_beat_T, class bias_T, typename CONFIG_T>
void {name}_gemm_ip_pack(
    hls::stream<a_beat_T> &a_beat_stream,
    hls::stream<b_beat_T> &b_beat_stream,
    bias_T biases[CONFIG_T::gemm_n],
    hls::stream<ap_uint<{aw}>> &a_packed,
    hls::stream<ap_uint<{bw}>> &b_packed,
    hls::stream<ap_uint<{biasw}>> &bias_packed
) {{
    // ── Pack bias ─────────────────────────────────────────────────────────────
    {{
        ap_uint<{biasw}> bias_pkt = 0;
        for (int col = 0; col < {gc}; col++) {{
            for (int lane = 0; lane < 8; lane++) {{
                int actual_col = col * 8 + lane;
                if (actual_col < {n_val}) {{
                    bias_pkt.range(col * 64 + lane * 8 + 7, col * 64 + lane * 8) =
                        static_cast<ap_int<8>>(biases[actual_col]);
                }}
            }}
        }}
        bias_packed.write(bias_pkt);
    }}

{pack_body}
}}

// Unpack stage: read the blackbox result beats and emit typed res beats.
template <class res_T, typename CONFIG_T>
void {name}_gemm_ip_unpack(
    hls::stream<ap_uint<{cw}>> &c_packed,
    hls::stream<res_T> &res_stream
) {{
    for (int i = 0; i < {m_val}; i++) {{
        #pragma HLS PIPELINE II=1
        ap_uint<{cw}> out_pkt = c_packed.read();
        res_T out_beat;
        for (int c = 0; c < {gc}; c++) {{
            for (int lane = 0; lane < 8; lane++) {{
                int actual_col = c * 8 + lane;
                if (actual_col < {n_val}) {{
                    // Blackbox emits signed out_bits-wide results. Read as the
                    // matching ap_int so the assignment value-converts (sign-extends
                    // for integer res_T, value-casts for fixed-point) to res_T.
                    ap_int<{c_lane}> raw_val = (ap_int<{c_lane}>)out_pkt.range(c * {c_tile} + lane * {c_lane} + {c_lane - 1}, c * {c_tile} + lane * {c_lane});
                    out_beat[actual_col] = raw_val;
                }}
            }}
        }}
        res_stream.write(out_beat);
    }}
}}
{expand_defs}
template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void {name}_gemm_ip_stream(
    hls::stream<a_beat_T> &a_beat_stream,
    hls::stream<b_beat_T> &b_beat_stream,
    bias_T biases[CONFIG_T::gemm_n],
    hls::stream<res_T> &res_stream
) {{
    static_assert(CONFIG_T::gemm_m == {m_val}, "Adapter requires matching gemm_m");
    static_assert(CONFIG_T::gemm_k == {k_val}, "Adapter requires matching gemm_k");
    static_assert(CONFIG_T::gemm_n == {n_val}, "Adapter requires matching gemm_n");

    // The blackbox gemm_fc_wrapper is an ap_ctrl_none module: it has no ap_start/
    // ap_done and is driven purely by its FIFO handshakes, so it can only legally
    // be instantiated inside a DATAFLOW region. We keep the pack/blackbox/unpack
    // stages as separate processes connected only by hls::stream FIFOs; that is the
    // canonical dataflow form and avoids the clang-3.9 DataflowCanonicalizer segfault
    // that occurs when local arrays are shared across stages at this scope.
    #pragma HLS DATAFLOW

{stream_block}
}}

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void {name}_gemm_ip_array(
    a_beat_T a_beats[CONFIG_T::gemm_m],
    b_beat_T b_beats[CONFIG_T::gemm_n],
    bias_T biases[CONFIG_T::gemm_n],
    res_T results[CONFIG_T::gemm_m]
) {{
    // Array variant — stream the arrays through the same blackbox. The local
    // streams are fully written before the (dataflow) stream adapter is called,
    // so they are depth-sized to the full beat counts.
    hls::stream<a_beat_T> a_str;
    hls::stream<b_beat_T> b_str;
    hls::stream<res_T> r_str;
    #pragma HLS STREAM variable=a_str depth={m_val}
    #pragma HLS STREAM variable=b_str depth={n_val}
    #pragma HLS STREAM variable=r_str depth={m_val}
    for (unsigned row = 0; row < CONFIG_T::gemm_m; row++) {{
        a_str.write(a_beats[row]);
    }}
    for (unsigned col = 0; col < CONFIG_T::gemm_n; col++) {{
        b_str.write(b_beats[col]);
    }}
    {name}_gemm_ip_stream<a_beat_T, b_beat_T, bias_T, res_T, CONFIG_T>(
        a_str, b_str, biases, r_str);
    // Copy element-wise into the output array. Assigning a whole nnet::array into an
    // array element (results[i] = r_str.read()) invokes nnet::array::operator=, whose
    // self-assignment pointer comparison Vitis csynth rejects on array elements.
    for (unsigned i = 0; i < CONFIG_T::gemm_m; i++) {{
        res_T out_beat = r_str.read();
        for (unsigned j = 0; j < res_T::size; j++) {{
            #pragma HLS UNROLL
            results[i][j] = out_beat[j];
        }}
    }}
}}

}} // namespace nnet

#endif // {name.upper()}_GEMM_IP_H_
"""


# ── Integration manifest ───────────────────────────────────────────────────────


def gen_integration_manifest(items):
    """Generate ``integration_manifest.json`` describing all generated packages."""
    cores = []
    for item in items:
        name = item["emit_name"]
        cores.append({
            "name": name,
            "interface": item.get("interface", "stream"),
            "backend": "vitis",
            "wrapper_symbol": f"{name}_wrapper",
            "json": f"{name}/{name}_wrapper.json",
            "rtl": [
                f"{name}/{name}.v",
            ],
            "c_model": f"{name}/{name}_wrapper.cpp",
            "m": item["m"],
            "k": item["k"],
            "n": item["n"],
            "gemm_k_spatial": item.get("gemm_k_spatial", _k_chunks(item)),
        })

    return json.dumps({
        "package_format": "vitis_blackboxes_v1",
        "backend": "vitis",
        "combined_header": "gemm_ip_combined.h",
        "cores": cores,
    }, indent=2) + "\n"


# ── Top-level generation ───────────────────────────────────────────────────────


def generate_vitis_pkg(item, output_dir):
    """Generate a single Vitis GEMM blackbox package."""
    item = normalize_vitis_items(item)[0]
    pkg_dir = Path(output_dir) / item["emit_name"]
    pkg_dir.mkdir(parents=True, exist_ok=True)

    if item["gemm_k_spatial"] > 1 and item["gemm_k_spatial"] < _k_chunks(item):
        print(
            f"WARNING: {item['emit_name']}: gemm_k_spatial={item['gemm_k_spatial']} is experimental; "
            "tensor-slice partial outputs are INT16 and partial overflow is possible. "
            "Correctness depends on quantized operand ranges and partition size.",
            file=sys.stderr,
        )

    # Generate the Vitis RTL using the combined emitter: behavioral sim model
    # under `ifndef SYNTHESIS (used by cosim/XSIM) and the structural synth
    # wrapper under `else (used by C synthesis). Both share the same port list,
    # so the JSON blackbox binding is unchanged.
    out_bits = _output_bits(item)
    generate_grid_verilog = load_vitis_combined_rtl_generator()
    grid_v = generate_grid_verilog(item["m"], item["k"], item["n"],
                                   module_name=f"{item['emit_name']}_wrapper",
                                   gemm_k_spatial=item["gemm_k_spatial"],
                                   out_bits=out_bits)
    (pkg_dir / f"{item['emit_name']}.v").write_text(grid_v, encoding="utf-8")

    _write_text(pkg_dir / f"{item['emit_name']}_wrapper.cpp", _gen_wrapper_cpp(item))
    _write_text(pkg_dir / f"{item['emit_name']}_wrapper.json", _gen_json(item))
    _write_text(pkg_dir / f"{item['emit_name']}_gemm_ip.h", _gen_layer_gemm_ip_h(item))
    _write_text(pkg_dir / f"{item['emit_name']}_tb.cpp", _gen_tb_cpp(item, num_vectors=item.get('num_vectors', 10)))
    _write_text(pkg_dir / f"{item['emit_name']}_design.cpp", _gen_design_cpp(item))
    _write_text(pkg_dir / "run_vitis.tcl", _gen_tcl(item, output_dir))
    _write_text(pkg_dir / "hls_config.cfg", _gen_hls_config(item))
    _write_text(pkg_dir / "run_vitis.py", _gen_python_runner(item))

    m, k_val, n = item["m"], item["k"], item["n"]
    print(f"Generated Vitis package {pkg_dir}  (M={m}, K={k_val}, N={n}, k_spatial={item['gemm_k_spatial']})")


def generate_from_config_file(config_file, output_dir):
    """Generate Vitis packages from a gemm_config.json file."""
    cfg = _read_json(config_file)
    items = normalize_vitis_items(cfg)

    output_dir = Path(output_dir)
    for item in items:
        generate_vitis_pkg(item, output_dir)

    # Combined dispatch header
    _write_text(output_dir / "gemm_ip_combined.h", gen_combined_header(items))
    _write_text(output_dir / "integration_manifest.json", gen_integration_manifest(items))
    print(f"Generated combined header and manifest in {output_dir}")
