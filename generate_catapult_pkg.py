#!/usr/bin/env python3
"""
Generate a Catapult GEMM package with a single hls4ml-native C wrapper.

The generated public API is a header that exports:

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_stream(
    ac_channel<a_beat_T>& a_beat_stream,
    ac_channel<b_beat_T>& b_beat_stream,
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T>& res_stream
);

The wrapper performs the internal packing needed for the tensor-slice hardblock,
adds bias in the wrapper, and emits typed ac_channel outputs suitable for
direct inclusion through GEMM_IP_HEADER in hls4ml.
"""
import argparse
import json
from pathlib import Path
import sys


def _load_tensor_slice_generators():
    ts_dir = Path(__file__).resolve().parent / "tensor-slice"
    ts_dir_str = str(ts_dir)
    if ts_dir_str not in sys.path:
        sys.path.insert(0, ts_dir_str)
    from generate_verilog_grid import generate_grid_verilog
    return generate_grid_verilog


def latency_cycles(k, grid_rows, grid_cols):
    return (grid_cols - 1) * 8 + k + 10


def dead_cycles_raw(grid_cols):
    return (grid_cols - 1) * 8 + 10


def dead_cycles(grid_cols):
    return dead_cycles_raw(grid_cols) + 1


def gen_public_header(name, m, k, n, grid_rows, grid_cols):
    a_bits = grid_rows * 64
    b_bits = grid_cols * 64
    c_bits = grid_cols * 128
    mr = grid_rows * 8
    first_out = latency_cycles(k, grid_rows, grid_cols)
    blind = dead_cycles(grid_cols)
    drain = blind + mr

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
        c_row.set_slc(0, a_rows);
        c_row.set_slc({a_bits}, b_cols);
        c_row[0] = c_row[0] ^ in_valid[0];
        out_valid = in_valid;
        out_last = in_valid;
#else
        static ac_int<{a_bits}, false> a_buf[{k}];
        static ac_int<{b_bits}, false> b_buf[{k}];
        static int clk_cnt = 0;
        static bool running = false;

        c_row = 0;
        out_valid = 0;
        out_last = 0;

        if (in_valid) {{
            if (!running) {{
                clk_cnt = 0;
                running = true;
            }}
            if (clk_cnt < {k}) {{
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
                    if (actual_row < {m} && actual_col < {n}) {{
                        for (int kk = 0; kk < {k}; kk++) {{
                            ac_int<8, true> a_el = a_buf[kk].slc<8>(row_tile * 64 + row_local * 8);
                            ac_int<8, true> b_el = b_buf[kk].slc<8>(ct * 64 + cl * 8);
                            acc += a_el * b_el;
                        }}
                    }}
                    ac_int<8, true> sat_val;
                    if (acc > 127) sat_val = 127;
                    else if (acc < -128) sat_val = -128;
                    else sat_val = acc;
                    row_out.set_slc(ct * 128 + cl * 8, sat_val);
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
void {name}_gemm_ip_stream(
    ac_channel<a_beat_T> &a_beat_stream,
    ac_channel<b_beat_T> &b_beat_stream,
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
) {{
    static_assert(CONFIG_T::gemm_m == {m}, "Generated GEMM wrapper requires matching gemm_m.");
    static_assert(CONFIG_T::gemm_k == {k}, "Generated GEMM wrapper requires matching gemm_k.");
    static_assert(CONFIG_T::gemm_n == {n}, "Generated GEMM wrapper requires matching gemm_n.");
    static_assert(a_beat_T::size == CONFIG_T::gemm_m,
                  "a_beat_T must carry one tensor-slice activation beat across GEMM rows.");
    static_assert(CONFIG_T::transpose_weights,
                  "Generated GEMM IP wrapper expects transposed weights.");
    static_assert(b_beat_T::size == CONFIG_T::gemm_n,
                  "b_beat_T must carry one tensor-slice weight beat across GEMM output columns.");
    static_assert(res_T::size == CONFIG_T::gemm_n,
                  "res_T must carry one full GEMM result row.");

    static {name}_ccore gemm;
    int captured = 0;
    ac_int<{a_bits}, false> last_a_rows = 0;
    ac_int<{b_bits}, false> last_b_cols = 0;

    #pragma hls_pipeline_init_interval 1
    FEED: for (int kk = 0; kk < {k}; kk++) {{
        ac_int<{a_bits}, false> a_rows = 0;
        ac_int<{b_bits}, false> b_cols = 0;
        a_beat_T activation_rows = a_beat_stream.read();
        b_beat_T weight_cols = b_beat_stream.read();

        #pragma hls_unroll
        ROW_PACK: for (int row = 0; row < {m}; row++) {{
            int row_tile = row / 8;
            int row_local = row % 8;
            a_rows.set_slc(row_tile * 64 + row_local * 8,
                           {name}_to_gemm_int8(activation_rows[row]));
        }}

        #pragma hls_unroll
        COL_PACK: for (int col = 0; col < {n}; col++) {{
            int col_tile = col / 8;
            int col_local = col % 8;
            b_cols.set_slc(col_tile * 64 + col_local * 8,
                           {name}_to_gemm_int8(weight_cols[col]));
        }}

        last_a_rows = a_rows;
        last_b_cols = b_cols;
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        gemm.run(last_a_rows, last_b_cols, 1, c_row, v, l);
    }}

    #pragma hls_pipeline_init_interval 1
    DRAIN_WRITE: for (int i = 0; i < {blind + m}; i++) {{
        ac_int<{c_bits}, false> c_row;
        ac_int<1, false> v, l;
        gemm.run(last_a_rows, last_b_cols, 0, c_row, v, l);
        if (v) {{
            if (captured < {m}) {{
                res_T out_pack;
                #pragma hls_unroll
                for (int col = 0; col < {n}; col++) {{

                    int col_tile = col / 8;
                    int col_local = col % 8;
                    ac_int<8, true> raw_val = c_row.template slc<8>(col_tile * 128 + col_local * 8);
                    typename CONFIG_T::accum_t biased =
                        static_cast<typename CONFIG_T::accum_t>(raw_val.to_int()) +
                        static_cast<typename CONFIG_T::accum_t>(biases[col]);
                    out_pack[col] = static_cast<typename res_T::value_type>(biased);
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
        gemm.run(last_a_rows, last_b_cols, 0, c_row, v, l);
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


def gen_tb(name, m, k, n):
    return f"""\
#include <stdio.h>

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

int main() {{
    ac_channel<a_beat_t> a_beat_stream;
    ac_channel<b_beat_t> b_beat_stream;
    ac_channel<res_t> res_stream;
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

    for (int kk = 0; kk < {k}; kk++) {{
        a_beat_t a_beat;
        b_beat_t b_beat;
        for (int i = 0; i < {m}; i++) {{
            a_beat[i] = activations[i][kk];
        }}
        for (int j = 0; j < {n}; j++) {{
            b_beat[j] = weights[j][kk];
        }}
        a_beat_stream.write(a_beat);
        b_beat_stream.write(b_beat);
    }}

    nnet::{name}_gemm_ip_stream<a_beat_t, b_beat_t, int, res_t, {name}_config>(
        a_beat_stream, b_beat_stream, biases, res_stream);

    for (int i = 0; i < {m}; i++) {{
        res_t out = res_stream.read();
        for (int j = 0; j < {n}; j++) {{
            int acc = biases[j];
            int gemm_acc = 0;
            for (int kk = 0; kk < {k}; kk++) {{
                gemm_acc += activations[i][kk].to_int() * weights[j][kk].to_int();
            }}
            if (gemm_acc > 127) gemm_acc = 127;
            else if (gemm_acc < -128) gemm_acc = -128;
            acc += gemm_acc;
            if (out[j].to_int() != acc) {{
                printf("Mismatch row %d col %d: got %d expected %d\\n",
                       i, j, out[j].to_int(), acc);
                failed = 1;
            }}
        }}
    }}

    if (failed) {{
        printf("Test FAILED\\n");
        return 1;
    }}
    printf("Test passed\\n");
    return 0;
}}
"""


def gen_inst_cpp(name, m, k, n):
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

typedef nnet::array<ac_int<8, true>, {m}> a_beat_t;
typedef nnet::array<ac_int<8, true>, {n}> b_beat_t;
typedef nnet::array<ac_int<16, true>, {n}> res_t;

#pragma hls_design top
void {name}_top(
    ac_channel<a_beat_t> &a_beat_stream,
    ac_channel<b_beat_t> &b_beat_stream,
    int biases[{n}],
    ac_channel<res_t> &res_stream
) {{
    nnet::{name}_gemm_ip_stream<a_beat_t, b_beat_t, int, res_t, {name}_config>(
        a_beat_stream, b_beat_stream, biases, res_stream);
}}
"""


def gen_tcl(name, m, k, n):
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
solution design set {name}_top -top
go analyze
go compile

solution library add nangate-45nm_beh -- -rtlsyntool DesignCompiler -vendor Nangate -technology 045nm
solution library add ccs_sample_mem
solution library add ccs_sample_rom
go libraries

directive set -CLOCKS {{clk {{-CLOCK_PERIOD 10.0 -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME 5.0 -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND both -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low}}}}

# Phase 5: Map streams to real streaming resources
directive set /{name}_top/a_beat_stream:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
directive set /{name}_top/b_beat_stream:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
directive set /{name}_top/biases:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
directive set /{name}_top/res_stream:rsc -MAP_TO_MODULE ccs_ioport.ccs_out_wait

go assembly
go architect
go allocate
go schedule
go extract

project save
puts "{name} Catapult run complete."
"""


def gen_combined_header(items):
    includes = "\n".join(f'#include "{item["name"]}/{item["name"]}_gemm_ip.h"' for item in items)
    branches = []
    for item in items:
        branches.append(
            f"""\
    if constexpr (CONFIG_T::gemm_m == {item["m"]} &&
                  CONFIG_T::gemm_k == {item["k"]} &&
                  CONFIG_T::gemm_n == {item["n"]}) {{
        {item["name"]}_gemm_ip_stream<a_beat_T, b_beat_T, bias_T, res_T, CONFIG_T>(
            a_beat_stream, b_beat_stream, biases, res_stream);
    }}"""
        )
    branches_text = " else ".join(branches)
    return f"""\
#ifndef GEMM_IP_COMBINED_H_
#define GEMM_IP_COMBINED_H_

#include "ac_channel.h"
{includes}

namespace nnet {{

template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_stream(
    ac_channel<a_beat_T> &a_beat_stream,
    ac_channel<b_beat_T> &b_beat_stream,
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
) {{
    {branches_text} else {{
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated GEMM IP implementation matches this CONFIG_T shape.");
    }}
}}

template <class data_T, class weight_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_stream_sim(
    ac_channel<data_T> &data_stream,
    weight_T weights[CONFIG_T::n_in * CONFIG_T::n_out],
    bias_T biases[CONFIG_T::n_out],
    ac_channel<res_T> &res_stream
) {{
    typedef nnet::array<typename data_T::value_type, CONFIG_T::gemm_m> a_beat_t;
    typedef nnet::array<weight_T, CONFIG_T::gemm_n> b_beat_t;
    ac_channel<a_beat_t> a_beat_stream;
    ac_channel<b_beat_t> b_beat_stream;
    data_T activation_rows[CONFIG_T::gemm_m];

    static_assert(CONFIG_T::transpose_weights,
                  "GEMM IP simulation helper expects transposed weight storage.");

    for (unsigned int row = 0; row < CONFIG_T::gemm_m; row++) {{
        activation_rows[row] = data_stream.read();
    }}

    for (unsigned int kk = 0; kk < CONFIG_T::gemm_k; kk++) {{
        a_beat_t a_beat;
        b_beat_t b_beat;
        for (unsigned int row = 0; row < CONFIG_T::gemm_m; row++) {{
            a_beat[row] = activation_rows[row][kk];
        }}
        for (unsigned int col = 0; col < CONFIG_T::gemm_n; col++) {{
            b_beat[col] = weights[col * CONFIG_T::gemm_k + kk];
        }}
        a_beat_stream.write(a_beat);
        b_beat_stream.write(b_beat);
    }}

    gemm_ip_stream<a_beat_t, b_beat_t, bias_T, res_T, CONFIG_T>(
        a_beat_stream, b_beat_stream, biases, res_stream);
}}

}} // namespace nnet

#endif
"""


def gen_integration_manifest(items):
    cores = []
    for item in items:
        cores.append({
            "name": item["name"],
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


def generate_catapult_pkg(m, k, n, name, output_dir):
    pkg_dir = Path(output_dir) / name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8

    generate_grid_verilog = _load_tensor_slice_generators()
    grid_v = generate_grid_verilog(m, k, n, module_name=f"{name}_core")
    ts_src = Path(__file__).parent / "tensor-slice" / "tensor_slice_int8.v"
    (pkg_dir / f"{name}_core.v").write_text(grid_v + "\n\n" + ts_src.read_text())
    (pkg_dir / "nnet_types.h").write_text(gen_nnet_types_header())
    (pkg_dir / f"{name}_gemm_ip.h").write_text(
        gen_public_header(name, m, k, n, grid_rows, grid_cols)
    )
    (pkg_dir / f"{name}_inst.cpp").write_text(gen_inst_cpp(name, m, k, n))
    (pkg_dir / f"{name}_tb.cpp").write_text(gen_tb(name, m, k, n))
    (pkg_dir / "run_catapult.tcl").write_text(gen_tcl(name, m, k, n))
    print(f"Generated {pkg_dir}  (M={m}, K={k}, N={n})")


def _normalize_config_items(cfg):
    if isinstance(cfg, list):
        return cfg
    if isinstance(cfg, dict):
        if "m" in cfg and "k" in cfg and "n" in cfg and "name" in cfg:
            return [cfg]
        items = []
        for name, item in cfg.items():
            items.append({
                "name": name,
                "m": item.get("gemm_m", 1),
                "k": item["n_in"],
                "n": item["n_out"],
            })
        return items
    raise TypeError("Unsupported config format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Catapult HLS package for a GEMM IP")
    parser.add_argument("--config", type=str, help="JSON config (list of GEMMs)")
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--name", type=str, default="gemm_8x8x8")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        items = _normalize_config_items(cfg)
        for item in items:
            generate_catapult_pkg(item["m"], item["k"], item["n"], item["name"], args.output_dir)
        (Path(args.output_dir) / "gemm_ip_combined.h").write_text(gen_combined_header(items))
        (Path(args.output_dir) / "integration_manifest.json").write_text(gen_integration_manifest(items) + "\n")
        (Path(args.output_dir) / "catapult_gemm_blackboxes.tcl").write_text(gen_blackbox_tcl(items))
    else:
        generate_catapult_pkg(args.m, args.k, args.n, args.name, args.output_dir)
