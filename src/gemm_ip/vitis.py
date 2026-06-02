"""
Generate a Vitis HLS blackbox GEMM package.

The generated package is consumable by the Vivado/Vitis GEMM-IP hls4ml backend
through ``GEMM_IP_HEADER`` and ``add_files -blackbox``.

Per-package outputs (``<name>`` is the ``--name`` argument):

  ``<name>_wrapper.cpp``   C API model  (``hls::stream<ap_uint<W>>`` interface)
  ``<name>_wrapper.v``     RTL wrapper (Vitis ap_ctrl + AXI-stream FIFO ports)
  ``<name>_wrapper.json``  Blackbox descriptor
  ``tensor_slice_int8.v``  Copied tensor-slice RTL core
  ``run_vitis.tcl``        Standalone Vitis HLS project script
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
    k_steps,
    tail_mask_hex,
    vm,
    verilog_tile_comment,
    normalize_gemm_config,
    load_vitis_rtl_generator,
)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_file(src, dst):
    if not src.exists():
        raise FileNotFoundError(f"Required RTL source not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


# ── C API model (wrapper.cpp) ──────────────────────────────────────────────────


def _gen_wrapper_cpp(item):
    """Generate the C API model for Vitis HLS blackbox integration.

    The model:
      1. Reads 1 bias beat from bias_stream, caches it
      2. Loops K times (one per K position): reads A and B beats,
         computes outer product of A-column-vector × B-row-vector
      3. Saturates int32 accumulators to int8
      4. Writes M result beats to c_stream

    Protocol per cycle kk:
      - A packet: for each row tile r, byte rl = A[r*8+rl][kk]
      - B packet: for each col tile c, byte cl = B[kk][c*8+cl]
      - Outer product: acc[r][c][rl][cl] += A_byte[rl] * B_byte[cl]
    """
    name = item["emit_name"]
    m, k_val, n = item["m"], item["k"], item["n"]
    gr, gc = item["grid_rows"], item["grid_cols"]
    aw = _a_width(m)
    bw = _b_width(n)
    cw = _c_width(n)

    return f"""\
#include <hls_stream.h>
#include <ap_int.h>

// ── Behavioral C model for {name} ─────────────────────────────────────────────
// Dimensions: M={m}, K={k_val}, N={n}
// Grid: {gr}×{gc} tiles, {k_val} K cycles (1 per K position)
// Stream widths: A={aw}, B={bw}, C={cw}

static ap_int<8> saturated_int8(ap_int<32> v) {{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return v;
}}

void {name}_wrapper(
    hls::stream<ap_uint<{aw}>> &a_stream,
    hls::stream<ap_uint<{bw}>> &b_stream,
    hls::stream<ap_uint<{bw}>> &bias_stream,
    hls::stream<ap_uint<{cw}>> &c_stream
) {{
    // Grid-sized accumulator buffer
    ap_int<32> acc_buf[{gr}][{gc}][8][8];
    #pragma HLS ARRAY_PARTITION variable=acc_buf complete dim=0

    // Bias register
    ap_int<8> bias_vals[{n}];
    #pragma HLS ARRAY_PARTITION variable=bias_vals complete dim=0

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 1: Read bias (1 beat)
    // ═══════════════════════════════════════════════════════════════════════
    {{
        ap_uint<{bw}> bias_pkt = bias_stream.read();
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
    // Phase 2: Clear accumulators
    // ═══════════════════════════════════════════════════════════════════════
    ClearAcc:
    for (int r = 0; r < {gr}; r++) {{
        for (int c = 0; c < {gc}; c++) {{
            for (int rl = 0; rl < 8; rl++) {{
                for (int cl = 0; cl < 8; cl++) {{
                    acc_buf[r][c][rl][cl] = 0;
                }}
            }}
        }}
    }}

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 3: Feed K beats (one per K position), accumulate outer products
    // ═══════════════════════════════════════════════════════════════════════
    FeedK:
    for (int kk = 0; kk < {k_val}; kk++) {{
        #pragma HLS PIPELINE II=1
        ap_uint<{aw}> a_pkt = a_stream.read();
        ap_uint<{bw}> b_pkt = b_stream.read();

    FeedTile:
        for (int r = 0; r < {gr}; r++) {{
            for (int c = 0; c < {gc}; c++) {{
                // Outer product: each A row-lane × each B col-lane
                for (int rl = 0; rl < 8; rl++) {{
                    int actual_row = r * 8 + rl;
                    ap_int<8> a_val = 0;
                    if (actual_row < {m}) {{
                        a_val = a_pkt.range(r * 64 + rl * 8 + 7, r * 64 + rl * 8);
                    }}
                    for (int cl = 0; cl < 8; cl++) {{
                        int actual_col = c * 8 + cl;
                        ap_int<8> b_val = 0;
                        if (actual_col < {n}) {{
                            b_val = b_pkt.range(c * 64 + cl * 8 + 7, c * 64 + cl * 8);
                        }}
                        acc_buf[r][c][rl][cl] += a_val * b_val;
                    }}
                }}
            }}
        }}
    }}

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 4: Add bias, saturate, write results
    // ═══════════════════════════════════════════════════════════════════════
    DrainRows:
    for (int actual_row = 0; actual_row < {m}; actual_row++) {{
        #pragma HLS PIPELINE II=1
        int r = actual_row / 8;
        int rl = actual_row % 8;
        ap_uint<{cw}> out_pkt = 0;

    PackCol:
        for (int c = 0; c < {gc}; c++) {{
            for (int cl = 0; cl < 8; cl++) {{
                int actual_col = c * 8 + cl;
                ap_int<32> acc = acc_buf[r][c][rl][cl];
                if (actual_col < {n}) {{
                    acc += bias_vals[actual_col];
                }}
                ap_int<8> result = saturated_int8(acc);
                out_pkt.range(c * 64 + cl * 8 + 7, c * 64 + cl * 8) = result;
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
    aw = _a_width(m)
    bw = _b_width(n)
    cw = _c_width(n)

    # Latency estimate (1-cycle preload + k feed + pipeline + drain)
    compute_cycles = k_val + (gr - 1 + gc - 1) * 8 + 13
    drain_cycles = gr * 8  # grid_rows*8 output rows
    total_latency = 3 + 1 + compute_cycles + drain_cycles

    desc = {
        "c_function_name": f"{name}_wrapper",
        "rtl_top_module_name": name,
        "c_parameters": [
            {
                "name": "a_stream",
                "direction": "in",
                "type": "hls::stream<ap_uint<{}>>".format(aw),
                "mapped_to": "FIFO",
            },
            {
                "name": "bias_stream",
                "direction": "in",
                "type": "hls::stream<ap_uint<{}>>".format(bw),
                "mapped_to": "FIFO",
            },
            {
                "name": "b_stream",
                "direction": "in",
                "type": "hls::stream<ap_uint<{}>>".format(bw),
                "mapped_to": "FIFO",
            },
            {
                "name": "c_stream",
                "direction": "out",
                "type": "hls::stream<ap_uint<{}>>".format(cw),
                "mapped_to": "FIFO",
            },
        ],
        "c_files": [f"{name}/{name}_wrapper.cpp"],
        "rtl_files": [f"{name}/{name}.v"],
        "c_model_architecture": "dataflow",
        "blackbox_metadata": {
            "gemm_m": m,
            "gemm_k": k_val,
            "gemm_n": n,
            "grid_rows": gr,
            "grid_cols": gc,
            "a_width": aw,
            "b_width": bw,
            "c_width": cw,
            "bias_width": bw,
            "input_type": "int8",
            "weight_type": "int8",
            "output_type": "int8",
            "accumulator_type": "int32",
            "bias_location": "inside_blackbox",
            "interface": "stream",
        },
        "estimated_latency": {
            "min": total_latency,
            "max": total_latency,
            "pipeline_ii": k_val,
        },
    }

    # Clock/reset signals
    desc.setdefault("blackbox_signals", []).extend(
        [
            {"port": "clk", "signal": "clk"},
            {"port": "rst", "signal": "rst"},
        ]
    )
    desc.setdefault("fifo_map", {}).update(
        {
            "a_stream": {
                "data": "a_tdata",
                "valid": "a_tvalid",
                "ready": "a_tready",
            },
            "bias_stream": {
                "data": "bias_tdata",
                "valid": "bias_tvalid",
                "ready": "bias_tready",
            },
            "b_stream": {
                "data": "b_tdata",
                "valid": "b_tvalid",
                "ready": "b_tready",
            },
            "c_stream": {
                "data": "c_tdata",
                "valid": "c_tvalid",
                "ready": "c_tready",
                "last":  "c_tlast",
            },
        }
    )

    return json.dumps(desc, indent=2) + "\n"


# ── Standalone testbench ───────────────────────────────────────────────────────


def _gen_tb_cpp(item):
    """Generate a standalone Vitis HLS testbench.

    Creates random int8 activations/weights/biases, feeds them through the
    blackbox C model one K position per cycle, and compares against a golden
    reference computed with standard GEMM.
    """
    name = item["emit_name"]
    m, k_val, n = item["m"], item["k"], item["n"]
    gr, gc = item["grid_rows"], item["grid_cols"]
    aw = _a_width(m)
    bw = _b_width(n)
    cw = _c_width(n)

    return f"""\
#include <stdio.h>
#include <stdlib.h>
#include <hls_stream.h>
#include <ap_int.h>

// Reference GEMM implementation
static ap_int<8> saturated_int8(ap_int<32> v) {{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return v;
}}

extern void {name}_wrapper(
    hls::stream<ap_uint<{aw}>> &a_stream,
    hls::stream<ap_uint<{bw}>> &b_stream,
    hls::stream<ap_uint<{bw}>> &bias_stream,
    hls::stream<ap_uint<{cw}>> &c_stream
);

int main() {{
    hls::stream<ap_uint<{aw}>> a_stream;
    hls::stream<ap_uint<{bw}>> b_stream;
    hls::stream<ap_uint<{bw}>> bias_stream;
    hls::stream<ap_uint<{cw}>> c_stream;

    // Test data
    ap_int<8> activations[{m}][{k_val}];
    ap_int<8> weights[{n}][{k_val}];
    ap_int<8> biases[{n}];
    int failed = 0;

    // Initialise with deterministic pseudo-random values
    for (int i = 0; i < {m}; i++) {{
        for (int kk = 0; kk < {k_val}; kk++) {{
            activations[i][kk] = ((i * 3 + kk - 4) & 0x7F) - 64;
        }}
    }}
    for (int j = 0; j < {n}; j++) {{
        biases[j] = (j % 5) - 2;
        for (int kk = 0; kk < {k_val}; kk++) {{
            weights[j][kk] = ((j * 5 - kk + 1) & 0x7F) - 64;
        }}
    }}

    // Compute golden reference
    ap_int<8> expected[{m}][{n}];
    for (int i = 0; i < {m}; i++) {{
        for (int j = 0; j < {n}; j++) {{
            ap_int<32> acc = biases[j];
            for (int kk = 0; kk < {k_val}; kk++) {{
                acc += (ap_int<32>)activations[i][kk] * (ap_int<32>)weights[j][kk];
            }}
            expected[i][j] = saturated_int8(acc);
        }}
    }}

    // ── Feed bias ─────────────────────────────────────────────────────────────
    {{
        ap_uint<{bw}> bias_pkt = 0;
        for (int col = 0; col < {gc}; col++) {{
            for (int lane = 0; lane < 8; lane++) {{
                int actual_col = col * 8 + lane;
                if (actual_col < {n}) {{
                    bias_pkt.range(col * 64 + lane * 8 + 7, col * 64 + lane * 8) = biases[actual_col];
                }}
            }}
        }}
        bias_stream.write(bias_pkt);
    }}

    // ── Feed activations + weights  (one K position per cycle) ────────────────
    for (int kk = 0; kk < {k_val}; kk++) {{
        ap_uint<{aw}> a_pkt = 0;
        ap_uint<{bw}> b_pkt = 0;

        // A: pack A[row][kk] for all rows — byte rl in tile r = A[r*8+rl][kk]
        for (int r = 0; r < {gr}; r++) {{
            for (int rl = 0; rl < 8; rl++) {{
                int actual_row = r * 8 + rl;
                ap_int<8> a_val = 0;
                if (actual_row < {m}) {{
                    a_val = activations[actual_row][kk];
                }}
                a_pkt.range(r * 64 + rl * 8 + 7, r * 64 + rl * 8) = a_val;
            }}
        }}

        // B: pack B[kk][col] for all cols — byte cl in tile c = weights[c*8+cl][kk]
        for (int c = 0; c < {gc}; c++) {{
            for (int cl = 0; cl < 8; cl++) {{
                int actual_col = c * 8 + cl;
                ap_int<8> b_val = 0;
                if (actual_col < {n}) {{
                    b_val = weights[actual_col][kk];
                }}
                b_pkt.range(c * 64 + cl * 8 + 7, c * 64 + cl * 8) = b_val;
            }}
        }}

        a_stream.write(a_pkt);
        b_stream.write(b_pkt);
    }}

    // ── Call the wrapper ──────────────────────────────────────────────────────
    {name}_wrapper(a_stream, b_stream, bias_stream, c_stream);

    // ── Read and verify results ───────────────────────────────────────────────
    for (int i = 0; i < {m}; i++) {{
        ap_uint<{cw}> out_pkt = c_stream.read();
        for (int c = 0; c < {gc}; c++) {{
            for (int lane = 0; lane < 8; lane++) {{
                int actual_col = c * 8 + lane;
                if (actual_col < {n}) {{
                    ap_int<8> result = out_pkt.range(c * 64 + lane * 8 + 7, c * 64 + lane * 8);
                    if (result != expected[i][actual_col]) {{
                        printf("MISMATCH [%d][%d]: got %d expected %d\\n",
                               i, actual_col, (int)result, (int)expected[i][actual_col]);
                        failed = 1;
                    }}
                }}
            }}
        }}
    }}

    if (failed) {{
        printf("\\nTest FAILED\\n");
        return 1;
    }}
    printf("\\nTest passed\\n");
    return 0;
}}
"""


# ── Standalone design top ──────────────────────────────────────────────────────


def _gen_design_cpp(item):
    """Generate a standalone Vitis HLS design top that wraps the blackbox.

    Packs memory-mapped arrays into streams (one K position per cycle),
    calls the blackbox wrapper, and unpacks results.
    """
    name = item["emit_name"]
    m, k_val, n = item["m"], item["k"], item["n"]
    gr, gc = item["grid_rows"], item["grid_cols"]
    aw = _a_width(m)
    bw = _b_width(n)
    cw = _c_width(n)

    bw_str = str(bw)
    cw_str = str(cw)
    return f"""\
#include <hls_stream.h>
#include <ap_int.h>

extern void {name}_wrapper(
    hls::stream<ap_uint<{aw}>> &a_stream,
    hls::stream<ap_uint<{bw_str}>> &b_stream,
    hls::stream<ap_uint<{bw_str}>> &bias_stream,
    hls::stream<ap_uint<{cw_str}>> &c_stream
);

// Memory-mapped top for standalone Vitis HLS validation
void {name}_design(
    ap_int<8> activations[{m}][{k_val}],
    ap_int<8> weights[{n}][{k_val}],
    ap_int<8> biases[{n}],
    ap_int<8> results[{m}][{n}]
) {{
    #pragma HLS INTERFACE s_axilite port=activations
    #pragma HLS INTERFACE s_axilite port=weights
    #pragma HLS INTERFACE s_axilite port=biases
    #pragma HLS INTERFACE s_axilite port=results
    #pragma HLS INTERFACE s_axilite port=return

    hls::stream<ap_uint<{aw}>> a_stream("a_stream");
    hls::stream<ap_uint<{bw}>> b_stream("b_stream");
    hls::stream<ap_uint<{bw}>> bias_stream("bias_stream");
    hls::stream<ap_uint<{cw}>> c_stream("c_stream");

    #pragma HLS DATAFLOW

    // Pack and stream bias
    ap_uint<{bw}> bias_pkt = 0;
    for (int c = 0; c < {gc}; c++) {{
        for (int lane = 0; lane < 8; lane++) {{
            int actual_col = c * 8 + lane;
            if (actual_col < {n}) {{
                bias_pkt.range(c * 64 + lane * 8 + 7, c * 64 + lane * 8) = biases[actual_col];
            }}
        }}
    }}
    bias_stream.write(bias_pkt);

    // Stream A and B — one K position per cycle
    for (int kk = 0; kk < {k_val}; kk++) {{
        ap_uint<{aw}> a_pkt = 0;
        ap_uint<{bw}> b_pkt = 0;

        // A: pack A[row][kk] for all rows
        for (int r = 0; r < {gr}; r++) {{
            for (int rl = 0; rl < 8; rl++) {{
                int actual_row = r * 8 + rl;
                if (actual_row < {m}) {{
                    a_pkt.range(r * 64 + rl * 8 + 7, r * 64 + rl * 8) = activations[actual_row][kk];
                }}
            }}
        }}

        // B: pack B[col][kk] for all cols
        for (int c = 0; c < {gc}; c++) {{
            for (int cl = 0; cl < 8; cl++) {{
                int actual_col = c * 8 + cl;
                if (actual_col < {n}) {{
                    b_pkt.range(c * 64 + cl * 8 + 7, c * 64 + cl * 8) = weights[actual_col][kk];
                }}
            }}
        }}

        a_stream.write(a_pkt);
        b_stream.write(b_pkt);
    }}

    // Call wrapper
    {name}_wrapper(a_stream, b_stream, bias_stream, c_stream);

    // Unpack results
    for (int i = 0; i < {m}; i++) {{
        ap_uint<{cw}> out_pkt = c_stream.read();
        for (int c = 0; c < {gc}; c++) {{
            for (int lane = 0; lane < 8; lane++) {{
                int actual_col = c * 8 + lane;
                if (actual_col < {n}) {{
                    results[i][actual_col] = out_pkt.range(c * 64 + lane * 8 + 7, c * 64 + lane * 8);
                }}
            }}
        }}
    }}
}}
"""


# ── Standalone Vitis HLS Tcl script ────────────────────────────────────────────


def _gen_tcl(item, output_dir):
    """Generate a Vitis HLS Tcl script for standalone compilation."""
    name = item["emit_name"]
    m, k_val, n = item["m"], item["k"], item["n"]
    return f"""\
# Vitis HLS project for {name}_wrapper
# Dimensions: M={m}, K={k_val}, N={n}
# Generated by gemm-ip-gen

open_project {name}_proj
set_top {name}_design

add_files {name}/{name}_design.cpp -cflags "-D__SYNTHESIS__"
add_files {name}/{name}_wrapper.cpp -cflags "-D__SYNTHESIS__"
add_files -tb {name}/{name}_tb.cpp

# Add blackbox RTL
add_files {name}/{name}.v

open_solution "solution1"
set_part {{xcvu13p-flga2577-2-e}}
create_clock -period 5 -name default

config_interface -m_axi_addr64
config_compile -pipeline_loops 1

csim_design
csynth_design
exit
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

    branches = []
    for item in items:
        name = item["emit_name"]
        m_val, k_val, n_val = item["m"], item["k"], item["n"]
        gr, gc, ks = item["grid_rows"], item["grid_cols"], item["k_steps"]
        aw = _a_width(m_val)
        bw = _b_width(n_val)
        cw = _c_width(n_val)

        shape_cond = f"CONFIG_T::gemm_m == {m_val} && CONFIG_T::gemm_k == {k_val} && CONFIG_T::gemm_n == {n_val}"
        if item.get("gemm_ip_index") is not None:
            shape_cond = f"CONFIG_T::gemm_ip_id == {item['gemm_ip_index']} && {shape_cond}"

        branches.append(f"""\
    if constexpr ({shape_cond}) {{
        {name}_gemm_ip_stream<a_beat_T, b_beat_T, bias_T, res_T, CONFIG_T>(
            a_beat_stream, b_beat_stream, biases, res_stream);
    }}""")

    branches_text = "\n else ".join(branches)
    if branches:
        branches_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated Vitis GEMM IP implementation matches this CONFIG_T.");
    }"""

    array_branches = []
    for item in items:
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

    return f"""\
#ifndef GEMM_IP_COMBINED_H_
#define GEMM_IP_COMBINED_H_

// Vitis GEMM blackbox combined dispatch header.
// Generated by gemm-ip-gen for Vitis backend.
// Include path added via -DGEMM_IP_HEADER and -I<pkg_dir>.

#include "hls_stream.h"
#include "ap_int.h"
#include <cstddef>

// nnet::array<> definition (needed for typed-beat dispatch)
namespace nnet {{

template <typename T, unsigned N> struct array {{
    typedef T value_type;
    static const unsigned size = N;
    T data[N];
    T &operator[](size_t pos) {{ return data[pos]; }}
    const T &operator[](size_t pos) const {{ return data[pos]; }}
}};

}} // namespace nnet

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
{branches_text}
}}

// ── Array interface ───────────────────────────────────────────────────────────

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_array(
    a_beat_T a_beats[CONFIG_T::gemm_k],
    b_beat_T b_beats[CONFIG_T::gemm_k],
    bias_T biases[CONFIG_T::gemm_n],
    res_T results[CONFIG_T::gemm_m]
) {{
{array_branches_text}
}}

// ── Simulation helper (hls4ml compat) ─────────────────────────────────────────

template <class data_T, class weight_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_stream_sim(
    hls::stream<data_T> &data_stream,
    weight_T weights[CONFIG_T::n_in * CONFIG_T::n_out],
    bias_T biases[CONFIG_T::n_out],
    hls::stream<res_T> &res_stream
) {{
    // Direct scalar computation avoids stream deadlocks in C simulation.
    // Same reference logic as hls4ml's behavioral model.
    typename data_T::value_type activations[CONFIG_T::gemm_m][CONFIG_T::gemm_k];

    for (unsigned i = 0; i < CONFIG_T::gemm_m; i++) {{
        for (unsigned kp = 0; kp < CONFIG_T::gemm_k / data_T::size; kp++) {{
            data_T a_pack = data_stream.read();
            for (unsigned k = 0; k < data_T::size; k++) {{
                activations[i][kp * data_T::size + k] = a_pack[k];
            }}
        }}
    }}

    for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {{
        res_T c_pack;
        for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {{
            typename CONFIG_T::accum_t accum = 0;
            for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {{
                accum += CONFIG_T::template product<typename data_T::value_type, weight_T>::product(
                    activations[m][k], weights[n * CONFIG_T::gemm_k + k]);
            }}
            accum += biases[n];
            c_pack[n] = static_cast<typename res_T::value_type>(accum);
        }}
        res_stream.write(c_pack);
    }}
}}

}} // namespace nnet

#endif // GEMM_IP_COMBINED_H_
"""


# ── Per-layer adapter header (included by combined header) ─────────────────────


def _gen_layer_gemm_ip_h(item):
    """Generate ``{name}_gemm_ip.h`` — the typed-beat adapter for this layer.

    This header is what ``gemm_ip_combined.h`` includes.  It defines:
      - ``{name}_gemm_ip_stream`` — packs typed beats → ap_uint, calls blackbox
      - ``{name}_gemm_ip_array``   — same math, array-based

    Protocol: hls4ml feeds K beats of A (M-wide) and K beats of B (N-wide),
    one per K position.  The adapter packs each beat into the stream layout
    expected by the blackbox (row-major bytes within tiles).
    """
    name = item["emit_name"]
    m_val, k_val, n_val = item["m"], item["k"], item["n"]
    gr, gc = item["grid_rows"], item["grid_cols"]
    aw = _a_width(m_val)
    bw = _b_width(n_val)
    cw = _c_width(n_val)

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
    hls::stream<ap_uint<{bw}>> &bias_stream,
    hls::stream<ap_uint<{cw}>> &c_stream
);

namespace nnet {{

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

    #pragma HLS DATAFLOW

    // Local packed streams
    hls::stream<ap_uint<{aw}>> a_packed("a_packed");
    hls::stream<ap_uint<{bw}>> b_packed("b_packed");
    hls::stream<ap_uint<{bw}>> bias_packed("bias_packed");
    hls::stream<ap_uint<{cw}>> c_packed("c_packed");

    #pragma HLS STREAM variable=a_packed depth={k_val}
    #pragma HLS STREAM variable=b_packed depth={k_val}
    #pragma HLS STREAM variable=bias_packed depth=1
    #pragma HLS STREAM variable=c_packed depth={m_val}

    // ── Pack bias ─────────────────────────────────────────────────────────────
    {{
        ap_uint<{bw}> bias_pkt = 0;
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

    // ── Pack K beats (one per K position) ─────────────────────────────────────
    for (int kk = 0; kk < {k_val}; kk++) {{
        #pragma HLS PIPELINE II=1
        a_beat_T a_beat = a_beat_stream.read();
        b_beat_T b_beat = b_beat_stream.read();

        ap_uint<{aw}> a_pkt = 0;
        ap_uint<{bw}> b_pkt = 0;

        // A: pack a_beat[row] for all rows — one byte per row in each tile
        for (int r = 0; r < {gr}; r++) {{
            for (int rl = 0; rl < 8; rl++) {{
                int actual_row = r * 8 + rl;
                if (actual_row < {m_val}) {{
                    a_pkt.range(r * 64 + rl * 8 + 7, r * 64 + rl * 8) =
                        static_cast<ap_int<8>>(a_beat[actual_row]);
                }}
            }}
        }}

        // B: pack b_beat[col] for all cols — one byte per col in each tile
        for (int c = 0; c < {gc}; c++) {{
            for (int cl = 0; cl < 8; cl++) {{
                int actual_col = c * 8 + cl;
                if (actual_col < {n_val}) {{
                    b_pkt.range(c * 64 + cl * 8 + 7, c * 64 + cl * 8) =
                        static_cast<ap_int<8>>(b_beat[actual_col]);
                }}
            }}
        }}

        a_packed.write(a_pkt);
        b_packed.write(b_pkt);
    }}

    // ── Call blackbox ─────────────────────────────────────────────────────────
    {name}_wrapper(a_packed, b_packed, bias_packed, c_packed);

    // ── Unpack M result beats ─────────────────────────────────────────────────
    for (int i = 0; i < {m_val}; i++) {{
        #pragma HLS PIPELINE II=1
        ap_uint<{cw}> out_pkt = c_packed.read();
        res_T out_beat;
        for (int c = 0; c < {gc}; c++) {{
            for (int lane = 0; lane < 8; lane++) {{
                int actual_col = c * 8 + lane;
                if (actual_col < {n_val}) {{
                    out_beat[actual_col] = out_pkt.range(c * 64 + lane * 8 + 7, c * 64 + lane * 8);
                }}
            }}
        }}
        res_stream.write(out_beat);
    }}
}}

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void {name}_gemm_ip_array(
    a_beat_T a_beats[CONFIG_T::gemm_k],
    b_beat_T b_beats[CONFIG_T::gemm_k],
    bias_T biases[CONFIG_T::gemm_n],
    res_T results[CONFIG_T::gemm_m]
) {{
    // Array variant — stream the arrays through the same blackbox
    hls::stream<a_beat_T> a_str;
    hls::stream<b_beat_T> b_str;
    hls::stream<res_T> r_str;
    for (unsigned kk = 0; kk < CONFIG_T::gemm_k; kk++) {{
        a_str.write(a_beats[kk]);
        b_str.write(b_beats[kk]);
    }}
    {name}_gemm_ip_stream<a_beat_T, b_beat_T, bias_T, res_T, CONFIG_T>(
        a_str, b_str, biases, r_str);
    for (unsigned i = 0; i < CONFIG_T::gemm_m; i++) {{
        results[i] = r_str.read();
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
            "interface": "stream",
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
    pkg_dir = Path(output_dir) / item["emit_name"]
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Generate the Vitis RTL using the shared RTL generator
    generate_grid_verilog = load_vitis_rtl_generator()
    grid_v = generate_grid_verilog(item["m"], item["k"], item["n"],
                                   module_name=f"{item['emit_name']}")
    (pkg_dir / f"{item['emit_name']}.v").write_text(grid_v, encoding="utf-8")

    _write_text(pkg_dir / f"{item['emit_name']}_wrapper.cpp", _gen_wrapper_cpp(item))
    _write_text(pkg_dir / f"{item['emit_name']}_wrapper.json", _gen_json(item))
    _write_text(pkg_dir / f"{item['emit_name']}_gemm_ip.h", _gen_layer_gemm_ip_h(item))
    _write_text(pkg_dir / f"{item['emit_name']}_tb.cpp", _gen_tb_cpp(item))
    _write_text(pkg_dir / f"{item['emit_name']}_design.cpp", _gen_design_cpp(item))
    _write_text(pkg_dir / "run_vitis.tcl", _gen_tcl(item, output_dir))

    m, k_val, n = item["m"], item["k"], item["n"]
    print(f"Generated Vitis package {pkg_dir}  (M={m}, K={k_val}, N={n})")


def generate_from_config_file(config_file, output_dir):
    """Generate Vitis packages from a gemm_config.json file."""
    cfg = _read_json(config_file)
    items = normalize_gemm_config(cfg)

    output_dir = Path(output_dir)
    for item in items:
        item["backend"] = "vitis"
        generate_vitis_pkg(item, output_dir)

    # Combined dispatch header
    _write_text(output_dir / "gemm_ip_combined.h", gen_combined_header(items))
    _write_text(output_dir / "integration_manifest.json", gen_integration_manifest(items))
    print(f"Generated combined header and manifest in {output_dir}")
