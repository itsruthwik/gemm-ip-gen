#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


TENSOR_SLICE_SRC = Path(__file__).resolve().parent / "tensor-slice" / "tensor_slice_int8.v"
LANE_WIDTH = 8


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _copy_file(src, dst):
    if not src.exists():
        raise FileNotFoundError(f"Required RTL source not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _safe_name(name):
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(name))
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "dense_layer"


def _ceil_div(a, b):
    return (a + b - 1) // b


def _chunk_count(dim):
    return _ceil_div(dim, LANE_WIDTH)


def _parse_precision_bits(precision):
    if not isinstance(precision, str):
        return None
    text = precision.strip().lower().replace(" ", "")
    patterns = (
        r"^(?:ap_|ac_)?int<8>$",
        r"^(?:ap_|ac_)?int<8,true>$",
        r"^int<8>$",
    )
    for pattern in patterns:
        if re.match(pattern, text):
            return 8
    return None


def _require_dense_layer(layer_name, layer_cfg):
    if not isinstance(layer_cfg, dict):
        raise ValueError(f"Layer '{layer_name}' must be a JSON object.")

    layer_type = layer_cfg.get("type", layer_cfg.get("class_name"))
    if layer_type not in ["Dense", "GemmStream", "Im2ColGemmStream"]:
        return None

    n_in = layer_cfg.get("n_in")
    n_out = layer_cfg.get("n_out")
    if not isinstance(n_in, int) or n_in <= 0:
        raise ValueError(f"Dense layer '{layer_name}' is missing a valid positive integer n_in.")
    if not isinstance(n_out, int) or n_out <= 0:
        raise ValueError(f"Dense layer '{layer_name}' is missing a valid positive integer n_out.")

    transpose_weights = layer_cfg.get("transpose_weights", True)
    if not isinstance(transpose_weights, bool):
        raise ValueError(f"Dense layer '{layer_name}' must set transpose_weights to a boolean.")

    input_precision = layer_cfg.get("input_precision")
    weight_precision = layer_cfg.get("weight_precision")
    input_bits = _parse_precision_bits(input_precision)
    weight_bits = _parse_precision_bits(weight_precision)
    if input_bits != 8:
        print(f"Warning: Dense layer '{layer_name}' using {input_precision!r} instead of 8-bit. IP will be generated for 8-bit.")
    if weight_bits != 8:
        print(f"Warning: Dense layer '{layer_name}' using {weight_precision!r} instead of 8-bit. IP will be generated for 8-bit.")

    return {
        "name": layer_name,
        "emit_name": _safe_name(layer_name),
        "scaffold_name": f"{_safe_name(layer_name)}_scaffold",
        "type": "Dense",
        "n_in": n_in,
        "n_out": n_out,
        "transpose_weights": transpose_weights,
        "input_precision": input_precision,
        "weight_precision": weight_precision,
        "bias_precision": layer_cfg.get("bias_precision"),
        "output_precision": layer_cfg.get("output_precision"),
        "accum_precision": layer_cfg.get("accum_precision"),
    }


def _collect_dense_layers(config):
    layers = []
    if isinstance(config, dict) and "layers" in config and isinstance(config["layers"], list):
        for idx, layer in enumerate(config["layers"]):
            if not isinstance(layer, dict):
                raise ValueError(f"Layer entry {idx} must be a JSON object.")
            layer_name = layer.get("name") or layer.get("config", {}).get("name") or f"dense_{idx}"
            layer_cfg = layer.get("config", layer)
            dense = _require_dense_layer(layer_name, layer_cfg)
            if dense is not None:
                layers.append(dense)
        return layers

    if not isinstance(config, dict):
        raise ValueError("gemm_config.json must contain a JSON object mapping layer names to configs.")

    for layer_name, layer_cfg in config.items():
        dense = _require_dense_layer(layer_name, layer_cfg)
        if dense is not None:
            layers.append(dense)

    return layers


def _verilog_tile_comment(layer):
    grid_rows = _chunk_count(layer["n_in"])
    grid_cols = _chunk_count(layer["n_out"])
    k_steps = _chunk_count(layer["n_in"])
    return (
        f"// Layer dimensions: n_in={layer['n_in']}, n_out={layer['n_out']}\n"
        f"// Generated RTL IP: 2D output-tile tensor_slice grid behind the gemm8-style blackbox interface.\n"
        f"// GRID_ROWS=ceil(M/8)={grid_rows}, GRID_COLS=ceil(N/8)={grid_cols}, K_STEPS=ceil(K/8)={k_steps}\n"
        f"// v1 protocol: {LANE_WIDTH}-lane chunks, A row tiles={grid_rows}, output column tiles={grid_cols}\n"
        f"// c_row carries GRID_COLS packed 8-lane output tiles per beat.\n"
    )


def _generate_blackbox_verilog(layer):
    scaffold = layer["scaffold_name"]
    module = f"{scaffold}_ccore"
    grid_rows = _chunk_count(layer["n_in"])
    grid_cols = _chunk_count(layer["n_out"])
    k_steps = _chunk_count(layer["n_in"])
    total_tiles = grid_rows * grid_cols
    output_rows = grid_rows * LANE_WIDTH
    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 64
    top_row_slice_latency = max(30, (grid_cols * LANE_WIDTH) + 19)
    bottom_row_slice_latency = top_row_slice_latency + ((grid_rows - 1) * LANE_WIDTH)
    total_cycles = top_row_slice_latency + 1 + output_rows + 16
    slice_instances = "\n".join(
        f"""    tensor_slice slice_r{row}_c{col} (
        .clk(clk),
        .reset(slice_reset),
        .pe_reset(slice_reset),
        .start_mat_mul(slice_start),
        .done_mat_mul(done_slices[{row * grid_cols + col}]),
        .a_in(a_chain[{row}][{col}]),
        .b_in(b_chain[{row}][{col}]),
        .a_out(a_chain[{row}][{col + 1}]),
        .b_out(b_chain[{row + 1}][{col}]),
        .c_data_out(c_data[{row * grid_cols + col}]),
        .c_data_available(c_available[{row * grid_cols + col}]),
        .validity_mask_a_rows(row_valid_masks[{row}]),
        .validity_mask_a_cols_b_rows(k_valid_mask),
        .validity_mask_b_cols(col_valid_masks[{col}]),
        .slice_dtype(2'd0),
        .slice_mode(1'b0),
        .op(3'd0),
        .preload(1'b0),
        .no_rounding(1'b0),
        .final_mat_mul_size(8'd8),
        .a_loc(5'd{row}),
        .b_loc(5'd{col}),
        .latency_config(8'd{top_row_slice_latency + row * LANE_WIDTH})
    );"""
        for row in range(grid_rows)
        for col in range(grid_cols)
    )
    edge_assignments = "\n".join(
        [f"    assign a_chain[{row}][0] = a_edge[{row}];" for row in range(grid_rows)]
        + [f"    assign b_chain[0][{col}] = b_edge[{col}];" for col in range(grid_cols)]
    )
    return f"""`timescale 1ns / 1ps

// This file is generated by gemm_ip_gen/generate_gemm_ip.py.
// It must be compiled together with the copied tensor_slice_int8.v file.
module {module}(
    input clk,
    input rst,
    input en,
    input [{a_width - 1}:0] a_rows,
    input [{b_width - 1}:0] b_cols,
    input in_valid,
    output reg [{c_width - 1}:0] c_row,
    output reg out_valid,
    output reg out_last
);
{_verilog_tile_comment(layer)}
    localparam integer GRID_ROWS = {grid_rows};
    localparam integer GRID_COLS = {grid_cols};
    localparam integer K_STEPS = {k_steps};
    localparam integer TOTAL_TILES = {total_tiles};
    localparam integer OUTPUT_ROWS = {output_rows};
    localparam integer A_WIDTH = {a_width};
    localparam integer B_WIDTH = {b_width};
    localparam integer C_WIDTH = {c_width};
    localparam integer TOP_ROW_SLICE_LATENCY = {top_row_slice_latency};
    localparam integer BOTTOM_ROW_SLICE_LATENCY = {bottom_row_slice_latency};
    localparam integer READOUT_START = TOP_ROW_SLICE_LATENCY + 1;
    localparam integer TOTAL_CYCLES = {total_cycles};

    reg [15:0] cycle;
    reg [15:0] drain_row;
    reg [63:0] a_edge [0:GRID_ROWS-1];
    reg [63:0] b_edge [0:GRID_COLS-1];
    wire [63:0] a_chain [0:GRID_ROWS-1][0:GRID_COLS];
    wire [63:0] b_chain [0:GRID_ROWS][0:GRID_COLS-1];
    wire [63:0] c_data [0:TOTAL_TILES-1];
    wire [TOTAL_TILES-1:0] done_slices;
    wire [TOTAL_TILES-1:0] c_available;
    wire slice_reset;
    wire slice_start;
    reg [7:0] row_valid_masks [0:GRID_ROWS-1];
    reg [7:0] col_valid_masks [0:GRID_COLS-1];
    reg [7:0] k_valid_mask;
    wire [15:0] active_drain_row;
    wire [15:0] active_drain_row_tile;
    integer i;

    assign slice_reset = rst;
    assign slice_start = en && (cycle < TOTAL_CYCLES);
    assign active_drain_row = cycle - READOUT_START;
    assign active_drain_row_tile = active_drain_row >> 3;

{edge_assignments}

{slice_instances}

    function [7:0] tail_mask;
        input integer total;
        input integer chunk;
        integer remain;
        integer lane;
        begin
            remain = total - (chunk * 8);
            tail_mask = 8'd0;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                if (lane < remain) begin
                    tail_mask[lane] = 1'b1;
                end
            end
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            cycle <= 16'd0;
            drain_row <= 16'd0;
            c_row <= {c_width}'d0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
            for (i = 0; i < GRID_ROWS; i = i + 1) begin
                a_edge[i] <= 64'd0;
                row_valid_masks[i] <= tail_mask({layer['n_in']}, i);
            end
            for (i = 0; i < GRID_COLS; i = i + 1) begin
                b_edge[i] <= 64'd0;
                col_valid_masks[i] <= tail_mask({layer['n_out']}, i);
            end
            k_valid_mask <= tail_mask({layer['n_in']}, K_STEPS - 1);
        end else if (en) begin
            if (cycle < K_STEPS) begin
                if (in_valid) begin
                    for (i = 0; i < GRID_ROWS; i = i + 1) begin
                        // Each A edge packet carries one 8-lane row tile for
                        // this K step. For 64-wide A, this is 8 packed packets.
                        a_edge[i] <= a_rows[i * 64 +: 64];
                    end
                    for (i = 0; i < GRID_COLS; i = i + 1) begin
                        // Each B edge packet carries one 8-lane column tile for
                        // this K step. For 64-wide B, this is 8 packed packets.
                        b_edge[i] <= b_cols[i * 64 +: 64];
                    end
                end
            end

            if (cycle >= READOUT_START && cycle < READOUT_START + OUTPUT_ROWS) begin
                drain_row <= active_drain_row;
                for (i = 0; i < GRID_COLS; i = i + 1) begin
                    c_row[i * 64 +: 64] <= c_data[((active_drain_row_tile * GRID_COLS) + i)];
                end
                out_valid <= 1'b1;
                out_last <= (active_drain_row == OUTPUT_ROWS - 1);
            end else begin
                drain_row <= 16'd0;
                c_row <= {c_width}'d0;
                out_valid <= 1'b0;
                out_last <= 1'b0;
            end

            cycle <= (cycle == (TOTAL_CYCLES - 1)) ? 16'd0 : (cycle + 16'd1);
        end
    end

endmodule
"""


def _generate_header():
    return f"""#ifndef NNET_GEMM_IP_H_
#define NNET_GEMM_IP_H_

#include "ac_channel.h"
#include "nnet_common.h"
#include "nnet_mult.h"

namespace nnet {{

template <class data_T, class res_T, typename CONFIG_T>
void gemm_ip_dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

InitAccum:
    for (int j = 0; j < CONFIG_T::n_out; j++) {{
        //#pragma HLS UNROLL
        acc[j] = (typename CONFIG_T::accum_t)biases[j];
    }}

Multiply:
    for (int i = 0; i < CONFIG_T::n_in; i++) {{
        //#pragma HLS UNROLL
        for (int j = 0; j < CONFIG_T::n_out; j++) {{
            //#pragma HLS UNROLL
            int weight_idx = CONFIG_T::transpose_weights ? j * CONFIG_T::n_in + i : i * CONFIG_T::n_out + j;
            acc[j] += static_cast<typename CONFIG_T::accum_t>(data[i]) *
                      static_cast<typename CONFIG_T::accum_t>(weights[weight_idx]);
        }}
    }}

Cast:
    for (int j = 0; j < CONFIG_T::n_out; j++) {{
        //#pragma HLS UNROLL
        res[j] = cast<typename CONFIG_T::accum_t, res_T, CONFIG_T>(acc[j]);
    }}
}}

template <class data_T, class res_T, typename CONFIG_T>
void gemm_ip_dense_stream(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream,
                          typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                          typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
    typename data_T::value_type data[CONFIG_T::n_in];
    typename res_T::value_type res[CONFIG_T::n_out];
    const int input_chunks = DIV_ROUNDUP(CONFIG_T::n_in, data_T::size);
    const int output_chunks = DIV_ROUNDUP(CONFIG_T::n_out, res_T::size);

ReadData:
    for (int chunk = 0; chunk < input_chunks; chunk++) {{
        data_T in_pkt = data_stream.read();
    ReadLane:
        for (int lane = 0; lane < data_T::size; lane++) {{
            int idx = chunk * data_T::size + lane;
            if (idx < CONFIG_T::n_in) {{
                data[idx] = in_pkt[lane];
            }}
        }}
    }}

    gemm_ip_dense<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

WriteData:
    for (int chunk = 0; chunk < output_chunks; chunk++) {{
        res_T out_pkt;
    WriteLane:
        for (int lane = 0; lane < res_T::size; lane++) {{
            int idx = chunk * res_T::size + lane;
            out_pkt[lane] = (idx < CONFIG_T::n_out) ? res[idx] : (typename res_T::value_type)0;
        }}
        res_stream.write(out_pkt);
    }}
}}

}} // namespace nnet

#endif // NNET_GEMM_IP_H_
"""


def _generate_source(layer):
    scaffold = layer["scaffold_name"]
    cfg = f"{scaffold}_config"
    input_chunks = _chunk_count(layer["n_in"])
    output_chunks = _chunk_count(layer["n_out"])
    grid_rows = _chunk_count(layer["n_in"])
    grid_cols = _chunk_count(layer["n_out"])
    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 64
    return f"""#include <ac_channel.h>
#include <ac_int.h>
#include "{scaffold}_stream_types.h"
#include "../nnet_gemm_ip.h"

namespace {scaffold}_pkg {{

template <typename T, int N>
struct vec_packet {{
    typedef T value_type;
    static const int size = N;
    T data[N];
    inline T &operator[](int idx) {{ return data[idx]; }}
    inline const T &operator[](int idx) const {{ return data[idx]; }}
}};

struct {cfg} {{
    static const int n_in = {layer['n_in']};
    static const int n_out = {layer['n_out']};
    static const int in_chunks = {input_chunks};
    static const int out_chunks = {output_chunks};
    static const int lane_width = {LANE_WIDTH};
    static const int grid_rows = {grid_rows};
    static const int grid_cols = {grid_cols};
    static const int a_stream_width = {a_width};
    static const int b_stream_width = {b_width};
    static const int c_stream_width = {c_width};
    static const bool transpose_weights = {'true' if layer['transpose_weights'] else 'false'};
    typedef ac_int<8, true> data_t;
    typedef ac_int<16, true> res_t;
    typedef ac_int<8, true> weight_t;
    typedef ac_int<16, true> bias_t;
    typedef ac_int<32, true> accum_t;
}};

template <class data_T, class res_T>
void {scaffold}_dense_stream(
    ac_channel<data_T> &data_stream,
    ac_channel<res_T> &res_stream,
    typename {cfg}::weight_t weights[{cfg}::n_in * {cfg}::n_out],
    typename {cfg}::bias_t biases[{cfg}::n_out]
) {{
    nnet::gemm_ip_dense_stream<data_T, res_T, {cfg}>(data_stream, res_stream, weights, biases);
}}

}} // namespace {scaffold}_pkg

#pragma hls_design top
void {scaffold}_stream(
    ac_channel<AStream_t> &a_stream,
    ac_channel<BStream_t> &b_stream,
    ac_channel<CStream_t> &c_stream
) {{
    // Placeholder top for Catapult. Functional C++ validation uses the
    // behavioral nnet::gemm_ip_dense_stream wrapper in the generated testbench.
    if (a_stream.available(1) && b_stream.available(1)) {{
        AStream_t a_pkt = a_stream.read();
        BStream_t b_pkt = b_stream.read();
        CStream_t c_pkt;
        c_pkt.ctdata = 0;
        c_pkt.ctuser = a_pkt.ctuser | b_pkt.ctuser;
        c_pkt.ctlast = a_pkt.ctlast & b_pkt.ctlast;
        c_stream.write(c_pkt);
    }}
}}
"""


def _generate_stream_types_header(layer):
    scaffold = layer["scaffold_name"]
    guard = f"{scaffold.upper()}_STREAM_TYPES_H_"
    grid_rows = _chunk_count(layer["n_in"])
    grid_cols = _chunk_count(layer["n_out"])
    a_width = grid_rows * 64
    b_width = grid_cols * 64
    c_width = grid_cols * 64
    return f"""#ifndef {guard}
#define {guard}

#include <ac_int.h>

struct AStream_t {{
    ac_int<{a_width}, false> ctdata;
    ac_int<1, false> ctuser;
    ac_int<1, false> ctlast;
}};

struct BStream_t {{
    ac_int<{b_width}, false> ctdata;
    ac_int<1, false> ctuser;
    ac_int<1, false> ctlast;
}};

struct CStream_t {{
    ac_int<{c_width}, false> ctdata;
    ac_int<1, false> ctuser;
    ac_int<1, false> ctlast;
}};

#endif // {guard}
"""


def _generate_tb(layer):
    scaffold = layer["scaffold_name"]
    cfg = f"{scaffold}_config"
    name = scaffold
    input_chunks = _chunk_count(layer["n_in"])
    output_chunks = _chunk_count(layer["n_out"])
    return f"""#include <stdio.h>
#include <ac_int.h>
#include <ac_channel.h>
#include "../../nnet_gemm_ip.h"

namespace {name}_tb {{

template <typename T, int N>
struct vec_packet {{
    typedef T value_type;
    static const int size = N;
    T data[N];
    inline T &operator[](int idx) {{ return data[idx]; }}
    inline const T &operator[](int idx) const {{ return data[idx]; }}
}};

struct {cfg} {{
    static const int n_in = {layer['n_in']};
    static const int n_out = {layer['n_out']};
    static const int in_chunks = {input_chunks};
    static const int out_chunks = {output_chunks};
    static const int lane_width = {LANE_WIDTH};
    static const bool transpose_weights = {'true' if layer['transpose_weights'] else 'false'};
    typedef ac_int<8, true> data_t;
    typedef ac_int<16, true> res_t;
    typedef ac_int<8, true> weight_t;
    typedef ac_int<16, true> bias_t;
    typedef ac_int<32, true> accum_t;
}};

static void compute_reference(
    const ac_int<8, true> *input,
    const ac_int<8, true> *weights,
    const ac_int<16, true> *biases,
    ac_int<16, true> *expected
) {{
    for (int j = 0; j < {cfg}::n_out; j++) {{
        ac_int<32, true> acc = biases[j];
        for (int i = 0; i < {cfg}::n_in; i++) {{
            int weight_idx = {cfg}::transpose_weights ? j * {cfg}::n_in + i : i * {cfg}::n_out + j;
            acc += input[i] * weights[weight_idx];
        }}
        expected[j] = acc;
    }}
}}

int run() {{
    ac_channel<vec_packet<ac_int<8, true>, {LANE_WIDTH}> > data_stream;
    ac_channel<vec_packet<ac_int<16, true>, {LANE_WIDTH}> > res_stream;
    ac_int<8, true> input[{cfg}::n_in];
    ac_int<8, true> weights[{cfg}::n_in * {cfg}::n_out];
    ac_int<16, true> biases[{cfg}::n_out];
    ac_int<16, true> expected[{cfg}::n_out];
    int failed = 0;

    for (int i = 0; i < {cfg}::n_in; i++) {{
        input[i] = (i % 13) - 6;
    }}
    for (int j = 0; j < {cfg}::n_out; j++) {{
        biases[j] = (j % 5) - 2;
    }}
    for (int i = 0; i < {cfg}::n_in * {cfg}::n_out; i++) {{
        weights[i] = ((i * 7) % 17) - 8;
    }}

    compute_reference(input, weights, biases, expected);

    for (int chunk = 0; chunk < {cfg}::in_chunks; chunk++) {{
        vec_packet<ac_int<8, true>, {LANE_WIDTH}> in_pkt;
        for (int lane = 0; lane < {LANE_WIDTH}; lane++) {{
            int idx = chunk * {LANE_WIDTH} + lane;
            in_pkt[lane] = (idx < {cfg}::n_in) ? input[idx] : (ac_int<8, true>)0;
        }}
        data_stream.write(in_pkt);
    }}

    nnet::gemm_ip_dense_stream<
        vec_packet<ac_int<8, true>, {LANE_WIDTH}>,
        vec_packet<ac_int<16, true>, {LANE_WIDTH}>,
        {cfg}
    >(
        data_stream,
        res_stream,
        weights,
        biases
    );

    for (int chunk = 0; chunk < {cfg}::out_chunks; chunk++) {{
        vec_packet<ac_int<16, true>, {LANE_WIDTH}> out_pkt = res_stream.read();
        for (int lane = 0; lane < {LANE_WIDTH}; lane++) {{
            int idx = chunk * {LANE_WIDTH} + lane;
            if (idx < {cfg}::n_out && out_pkt[lane] != expected[idx]) {{
                printf("Mismatch {name}[%d]: got %d expected %d\\n", idx, out_pkt[lane].to_int(), expected[idx].to_int());
                failed = 1;
            }}
        }}
    }}

    return failed;
}}

}} // namespace {name}_tb

int main() {{
    int failed = {name}_tb::run();
    if (failed) {{
        printf("Test FAILED\\n");
        return 1;
    }}
    printf("Test passed\\n");
    return 0;
}}
"""


def _generate_tcl(layer, package_dir):
    scaffold = layer["scaffold_name"]
    top = f"{scaffold}_stream"
    cpp = f"{scaffold}.cpp"
    tb = f"testbench/{scaffold}_tb.cpp"
    grid_rows = _chunk_count(layer["n_in"])
    grid_cols = _chunk_count(layer["n_out"])
    a_word_width = grid_rows * 64 + 2
    b_word_width = grid_cols * 64 + 2
    c_word_width = grid_cols * 64 + 2
    return f"""set project_name "{scaffold}_blackbox_stream"
set solution_name "{scaffold}_bb"
set script_dir [file dirname [info script]]

project new -name $project_name
solution new $solution_name
solution options defaults
solution options set /Output/OutputVerilog true
solution options set /Output/GenerateCycleNetlist false
options set Input/CompilerFlags {{-DBLACKBOX_FLOW}}

solution file add [file join $script_dir {cpp}] -type C++
solution file add [file join $script_dir {tb}] -type C++

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
solution design set {top} -top
go analyze
go compile

solution library add nangate-45nm_beh -- -rtlsyntool DesignCompiler -vendor Nangate -technology 045nm
solution library add amba
solution library add ML_amba
go libraries

directive set -CLOCKS {{clk {{-CLOCK_PERIOD 10.0 -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME 5.0 -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND both -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low}}}}
go assembly

directive set /{top}/a_stream:rsc -MAP_TO_MODULE amba.ccs_axi4stream_in
directive set /{top}/a_stream -WORD_WIDTH {a_word_width}
directive set /{top}/a_stream:rsc -PACKING_MODE sidebyside

directive set /{top}/b_stream:rsc -MAP_TO_MODULE amba.ccs_axi4stream_in
directive set /{top}/b_stream -WORD_WIDTH {b_word_width}
directive set /{top}/b_stream:rsc -PACKING_MODE sidebyside

directive set /{top}/c_stream:rsc -MAP_TO_MODULE amba.ccs_axi4stream_out
directive set /{top}/c_stream -WORD_WIDTH {c_word_width}
directive set /{top}/c_stream:rsc -PACKING_MODE sidebyside

go architect
go allocate
go extract

project save

puts ""
puts "Generated Catapult scaffold for {scaffold} in {package_dir}"
"""


def _generate_layer_manifest(layer, package_dir):
    manifest = {
        "layer_name": layer["name"],
        "scaffold_name": layer["scaffold_name"],
        "type": layer["type"],
        "n_in": layer["n_in"],
        "n_out": layer["n_out"],
        "transpose_weights": layer["transpose_weights"],
        "precision": {
            "input": layer["input_precision"],
            "weight": layer["weight_precision"],
            "bias": layer["bias_precision"],
            "output": layer["output_precision"],
            "accum": layer["accum_precision"],
        },
        "shared_header": "../nnet_gemm_ip.h",
        "files": [
            "../nnet_gemm_ip.h",
            f"{layer['scaffold_name']}_stream_types.h",
            f"{layer['scaffold_name']}.cpp",
            f"{layer['scaffold_name']}_ccore.v",
            "tensor_slice_int8.v",
            f"run_{layer['scaffold_name']}_catapult.tcl",
            f"testbench/{layer['scaffold_name']}_tb.cpp",
        ],
        "notes": [
            "Dense-layer Catapult package generated from gemm_config.json",
            "Shared ../nnet_gemm_ip.h one directory up provides the behavioral GEMM implementation",
            "RTL blackbox uses layer-width packed stream payloads for Catapult",
            "Generated ccore stitches copied tensor_slice_int8.v instances into an RTL IP",
        ],
    }
    _write_text(package_dir / "layer_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _generate_package(layer, output_dir):
    package_dir = output_dir / layer["scaffold_name"]
    tb_dir = package_dir / "testbench"
    package_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    _write_text(package_dir / f"{layer['scaffold_name']}.cpp", _generate_source(layer))
    _write_text(package_dir / f"{layer['scaffold_name']}_stream_types.h", _generate_stream_types_header(layer))
    _write_text(package_dir / f"{layer['scaffold_name']}_ccore.v", _generate_blackbox_verilog(layer))
    _copy_file(TENSOR_SLICE_SRC, package_dir / "tensor_slice_int8.v")
    _write_text(package_dir / f"run_{layer['scaffold_name']}_catapult.tcl", _generate_tcl(layer, package_dir.name))
    _write_text(tb_dir / f"{layer['scaffold_name']}_tb.cpp", _generate_tb(layer))
    _generate_layer_manifest(layer, package_dir)


def generate_gemm_ip(config_file, output_dir):
    config = _read_json(config_file)
    layers = _collect_dense_layers(config)
    if not layers:
        raise ValueError("No Dense layers were found in gemm_config.json.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(output_dir / "nnet_gemm_ip.h", _generate_header())

    top_manifest = {
        "config_file": str(Path(config_file).resolve()),
        "generated_from": "gemm_ip_gen/generate_gemm_ip.py",
        "tensor_slice_source": str(TENSOR_SLICE_SRC),
        "package_format": "per-dense-layer-catapult-tensor-slice-ccore",
        "layer_count": len(layers),
        "shared_header": "nnet_gemm_ip.h",
        "layers": [],
    }

    for layer in layers:
        _generate_package(layer, output_dir)
        top_manifest["layers"].append(
            {
                "name": layer["name"],
                "scaffold_name": layer["scaffold_name"],
                "package_dir": layer["scaffold_name"],
                "manifest": f"{layer['scaffold_name']}/layer_manifest.json",
                "n_in": layer["n_in"],
                "n_out": layer["n_out"],
                "transpose_weights": layer["transpose_weights"],
            }
        )

    _write_text(output_dir / "top_manifest.json", json.dumps(top_manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate GEMM IP from gemm_config.json")
    parser.add_argument("config_file", help="Path to gemm_config.json")
    parser.add_argument("output_dir", help="Directory to output generated IP files")
    args = parser.parse_args()
    generate_gemm_ip(args.config_file, args.output_dir)


if __name__ == "__main__":
    main()
