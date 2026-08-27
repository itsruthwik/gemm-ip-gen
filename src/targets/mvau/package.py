"""mvau package assembly: emit a Vitis-HLS RTL-blackbox GEMM IP package.

Generalizes the cosim-validated ``temp_space/mvau-spike``. Per package ``<name>``:

    <name>_core.v      shim (FINN mvu_vvu_axi wrapped) -- module name == c_function_name
    <name>_core.cpp    the C twin (blackbox behavioral model, csim)
    <name>_top.cpp     the DUT: feed -> blackbox -> drain, in a HLS dataflow region
    <name>.json        blackbox descriptor (FIFO ports, ap_ctrl_none, CE, 5 ap_ctrl keys)
    <name>_tb.cpp      self-checking testbench
    <name>_gemm_ip.h   hls4ml-facing adapter (minimal for now)
    run_vitis.tcl      csim + csynth + cosim
    rtl_static/*.sv    vendored FINN cores (copied in)
"""

import json
import shutil
from pathlib import Path

import geometry as _geom
import rtl as _rtl
import golden as _golden

_RTL_STATIC = Path(__file__).resolve().parent / "rtl_static"
_STATIC_SOURCES = ["mvu_vvu_axi.sv", "replay_buffer.sv",
                   "mvu_4sx4u.sv", "mvu_8sx8u_dsp48.sv", "mvu_vvu_8sx9_dsp58.sv"]

_PLAN_KEYS = ("weight_precision", "input_precision", "output_precision", "part",
              "clock_period_ns", "reuse_factor", "strategy", "target_cycles",
              "parallelization_factor", "n_tiles", "weights")


def _plan_kwargs(cfg):
    return {k: cfg[k] for k in _PLAN_KEYS if k in cfg and cfg[k] is not None}


def cbits(plan):
    """Byte-aligned width of the requantized C-row beat: N results at out_width."""
    return ((plan["n"] * plan["output_width"]) + 7) // 8 * 8


def bias_acc_codes(bias, product_frac, n):
    """Scale per-column real bias to the accumulator (2^product_frac) domain.
    Returns a list of N ints, or None if no (or all-zero) bias."""
    if not bias:
        return None
    codes = [int(round(float(b) * (1 << product_frac))) for b in bias]
    if len(codes) != n:
        raise ValueError(f"bias length {len(codes)} != N {n}")
    return codes if any(codes) else None


def _dataflow_top(name, plan, bias_codes=None):
    """The DUT: feed -> blackbox (raw integer matmul) -> affine requant drain, in
    a dataflow region. Per vector: NF raw beats in, one N-wide C-row beat out.

    Drain (Keras order: matmul, bias, quantize): add per-column bias in the
    accumulator (2^(fa+fb)) domain, reinterpret as fixed-point, cast to the output
    ap_fixed with AP_RND (round-half-up) + AP_SAT. bias_codes is the bias already
    scaled to that domain (None => no bias; the two-operand GEMMs have none).
    """
    t = plan["tile"]
    m = plan["num_input_vectors"]
    WB, AB, PB = t["weight_stream_width_ba"], t["input_stream_width_ba"], t["output_stream_width_ba"]
    PE, SF, NF, ACCU = t["pe"], t["sf"], t["nf"], t["accu_width"]
    N, outW, outI = plan["n"], plan["output_width"], plan["output_int"]
    pfrac = plan["product_frac"]
    CB = cbits(plan)
    wbeats, abeats = m * NF * SF, m * SF
    pad = ' ' * (len(name) + 6)
    if bias_codes:
        bias_decl = (f"static const long {name}_bias[{N}] = {{"
                     + ", ".join(str(c) for c in bias_codes) + "};\n")
        bias_add = f" + {name}_bias[oc]"
    else:
        bias_decl, bias_add = "", ""
    return f"""#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// output precision: fixed<{outW},{outI}> with round-half-up + saturate
typedef ap_fixed<{outW}, {outI}, AP_RND, AP_SAT> {name}_result_t;
{bias_decl}
void {name}_core(hls::stream<ap_uint<{WB}> >&, hls::stream<ap_uint<{AB}> >&,
{pad}hls::stream<ap_uint<{PB}> >&);

static void feed_w(hls::stream<ap_uint<{WB}> >& in, hls::stream<ap_uint<{WB}> >& out) {{
    for (int i = 0; i < {wbeats}; i++) out.write(in.read());
}}
static void feed_a(hls::stream<ap_uint<{AB}> >& in, hls::stream<ap_uint<{AB}> >& out) {{
    for (int i = 0; i < {abeats}; i++) out.write(in.read());
}}

// Affine requant drain: raw ACCU codes (+bias) -> N-wide requantized C row.
static void requant(hls::stream<ap_uint<{PB}> >& in, hls::stream<ap_uint<{CB}> >& out) {{
    for (int vec = 0; vec < {m}; vec++) {{
        ap_uint<{CB}> crow = 0;
        for (int nf = 0; nf < {NF}; nf++) {{
            ap_uint<{PB}> ob = in.read();
            for (int pe = 0; pe < {PE}; pe++) {{
                int oc = nf * {PE} + pe;
                ap_int<{ACCU}> raw = ob.range(pe * {ACCU} + {ACCU} - 1, pe * {ACCU});
                ap_int<64> v = (ap_int<64>)raw{bias_add};         // matmul + bias, accum domain
                ap_fixed<64, {64 - pfrac}> rv;
                rv.range(63, 0) = (ap_uint<64>)v;                 // code -> fixed (frac={pfrac})
                {name}_result_t r = rv;                           // rescale + round + saturate
                crow.range(oc * {outW} + {outW} - 1, oc * {outW}) = r.range({outW} - 1, 0);
            }}
        }}
        out.write(crow);
    }}
}}

void {name}(hls::stream<ap_uint<{WB}> >& w_in, hls::stream<ap_uint<{AB}> >& a_in,
{' ' * (len(name) + 1)}hls::stream<ap_uint<{CB}> >& c_out) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{WB}> > w_s;
    hls::stream<ap_uint<{AB}> > a_s;
    hls::stream<ap_uint<{PB}> > p_s;
#pragma HLS STREAM variable=w_s depth=4
#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=p_s depth=4
    feed_w(w_in, w_s);
    feed_a(a_in, a_s);
    {name}_core(w_s, a_s, p_s);   // <-- FINN MVU RTL blackbox (pure integer matmul)
    requant(p_s, c_out);
}}
"""


def _blackbox_json(name, t):
    WB, AB, PB = t["weight_stream_width_ba"], t["input_stream_width_ba"], t["output_stream_width_ba"]
    fn = f"{name}_core"
    dsp = t["dsp_estimate"]
    return json.dumps({
        "c_function_name": fn,
        "rtl_top_module_name": fn,     # MUST equal c_function_name (Vitis cosim gotcha)
        "c_files": [{"c_file": f"{fn}.cpp", "cflag": ""}],
        "rtl_files": [f"{fn}.v"] + [f"rtl_static/{s}" for s in _STATIC_SOURCES],
        "c_parameters": [
            {"c_name": "w", "c_port_direction": "in",
             "rtl_ports": {"FIFO_data_read_in": "w_dout", "FIFO_read_enable": "w_read", "FIFO_empty_flag": "w_empty_n"}},
            {"c_name": "a", "c_port_direction": "in",
             "rtl_ports": {"FIFO_data_read_in": "a_dout", "FIFO_read_enable": "a_read", "FIFO_empty_flag": "a_empty_n"}},
            {"c_name": "p", "c_port_direction": "out",
             "rtl_ports": {"FIFO_data_write_out": "p_din", "FIFO_write_enable": "p_write", "FIFO_full_flag": "p_full_n"}},
        ],
        "rtl_common_signal": {
            "module_clock": "ap_clk",
            "module_reset": "ap_rst",
            "module_clock_enable": "ap_ce",
            "ap_ctrl_chain_protocol_idle": "",
            "ap_ctrl_chain_protocol_start": "",
            "ap_ctrl_chain_protocol_ready": "",
            "ap_ctrl_chain_protocol_done": "",
            "ap_ctrl_chain_protocol_continue": "",
        },
        # Deterministic (RTL is a fixed pipeline): latency + II are exact functions
        # of the fold, verified against XSIM (geometry.latency_cycles / output_ii).
        "rtl_performance": {"latency": str(t["latency_cycles"]), "II": str(t["ii"])},
        # FINN cost-model DSP for the tile; FF/LUT are rough per-DSP hints. (For a
        # blackbox these are what csynth reports; the real count comes from Vivado.)
        "rtl_resource_usage": {"FF": str(30 * dsp), "LUT": str(40 * dsp),
                               "DSP": str(dsp), "BRAM": "0", "URAM": "0"},
    }, indent=2) + "\n"


def _run_vitis_tcl(name, part, clock_ns):
    return f"""# mvau blackbox package: FINN MVU wrapped by {name}_core.v, blackboxed into {name}.
open_project -reset {name}_proj
set_top {name}
add_files {name}_top.cpp
add_files -blackbox {name}.json
add_files -tb {name}_tb.cpp
open_solution -reset sol1
set_part {part}
create_clock -period {clock_ns} -name default

csim_design
csynth_design
cosim_design

exit
"""


def _gemm_ip_header(name, plan):
    """The dedicated hls4ml-facing IP for one gemm config: an HLS C++ dataflow IP
    with the internal FINN-MVU blackbox. Repacks hls4ml beats -> FINN beats,
    runs the blackbox, requant-drains (runtime bias) -> hls4ml C row. The combined
    header routes nnet::gemm_* to this by CONFIG_T::gemm_ip_id (template dispatch).

    io_stream weightless (projections) implemented; two-operand + io_parallel
    follow the same pattern.
    """
    t = plan["tile"]
    m = plan["num_input_vectors"]
    WB, AB, PB = t["weight_stream_width_ba"], t["input_stream_width_ba"], t["output_stream_width_ba"]
    PE, SF, NF, ACCU = t["pe"], t["sf"], t["nf"], t["accu_width"]
    WW, AW, K, N = t["weight_width"], t["activation_width"], plan["k"], plan["n"]
    outW, outI, pfrac = plan["output_width"], plan["output_int"], plan["product_frac"]
    guard = ' ' * 6
    return f"""#ifndef {name.upper()}_GEMM_IP_H_
#define {name.upper()}_GEMM_IP_H_
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// internal FINN-MVU blackbox (shim {name}_core.v; C twin {name}_core.cpp)
void {name}_core(hls::stream<ap_uint<{WB}> >&, hls::stream<ap_uint<{AB}> >&,
{' ' * (len(name) + 6)}hls::stream<ap_uint<{PB}> >&);

namespace nnet {{

// Dedicated IP for gemm config M={m} K={K} N={N} (core={t['compute_core']},
// PE={PE} SIMD={SF and t['simd']} SF={SF} NF={NF}). Baked geometry; templated on the
// hls4ml stream/config types so the combined header can route to it by id.

// repack: {m} rows of K={K} activations -> {m}*SF={m * SF} FINN beats of SIMD={t['simd']}.
template <class data_T>
void {name}_repack_a(hls::stream<data_T> &a_stream, hls::stream<ap_uint<{AB}> > &a_s) {{
    for (unsigned mm = 0; mm < {m}; mm++) {{
        ap_int<{AW}> arow[{K}];
        #pragma HLS ARRAY_PARTITION variable=arow complete
        for (unsigned kp = 0; kp < {K} / data_T::size; kp++) {{
            data_T beat = a_stream.read();
            for (unsigned j = 0; j < data_T::size; j++) arow[kp * data_T::size + j] = beat[j].range({AW} - 1, 0);
        }}
        for (unsigned sf = 0; sf < {SF}; sf++) {{
            ap_uint<{AB}> ab = 0;
            for (unsigned s = 0; s < {t['simd']}; s++)
                ab.range(s * {AW} + {AW} - 1, s * {AW}) = (ap_uint<{AW}>)arow[sf * {t['simd']} + s];
            a_s.write(ab);
        }}
    }}
}}

// weight-stationary feed: source B^T from CONFIG_T::gemm_weight_cols() (weight_cols[n][k]),
// re-emit as {m}*NF*SF FINN weight beats in (nf outer, sf inner) order, PE-packed.
template <typename CONFIG_T>
void {name}_feed_w(hls::stream<ap_uint<{WB}> > &w_s) {{
    typename CONFIG_T::weight_col_t *wc = CONFIG_T::gemm_weight_cols();
    for (unsigned mm = 0; mm < {m}; mm++)
        for (unsigned nf = 0; nf < {NF}; nf++)
            for (unsigned sf = 0; sf < {SF}; sf++) {{
                ap_uint<{WB}> wb = 0;
                for (unsigned pe = 0; pe < {PE}; pe++)
                    for (unsigned s = 0; s < {t['simd']}; s++) {{
                        unsigned nn = nf * {PE} + pe, kk = sf * {t['simd']} + s;
                        ap_int<{WW}> wv = wc[nn][kk].range({WW} - 1, 0);
                        wb.range((pe * {t['simd']} + s) * {WW} + {WW} - 1, (pe * {t['simd']} + s) * {WW}) = (ap_uint<{WW}>)wv;
                    }}
                w_s.write(wb);
            }}
}}

// requant drain (runtime per-column bias, Keras order): raw ACCU -> fixed -> +bias -> round+sat.
template <class res_T, typename CONFIG_T>
void {name}_drain(hls::stream<ap_uint<{PB}> > &p_s, hls::stream<res_T> &res_stream,
{' ' * (len(name) + 7)}typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
    typedef ap_fixed<{outW}, {outI}, AP_RND, AP_SAT> result_t;
    for (unsigned mm = 0; mm < {m}; mm++) {{
        res_T crow;
        for (unsigned nf = 0; nf < {NF}; nf++) {{
            ap_uint<{PB}> ob = p_s.read();
            for (unsigned pe = 0; pe < {PE}; pe++) {{
                unsigned oc = nf * {PE} + pe;
                ap_int<{ACCU}> raw = ob.range(pe * {ACCU} + {ACCU} - 1, pe * {ACCU});
                ap_fixed<64, {64 - pfrac}> rv;
                rv.range(63, 0) = (ap_uint<64>)(ap_int<64>)raw;   // code -> fixed (frac={pfrac})
                rv += biases[oc];                                 // + bias (fixed-point, aligned)
                result_t r = rv;                                  // rescale + round + saturate
                crow[oc] = r;
            }}
        }}
        res_stream.write(crow);
    }}
}}

// The dedicated IP: hls4ml io_stream weightless GEMM -> internal MVU blackbox.
template <class data_T, class res_T, typename CONFIG_T>
void {name}_gemm_stream_weightless(hls::stream<data_T> &a_stream, hls::stream<res_T> &res_stream,
{' ' * (len(name) + 28)}typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{WB}> > w_s;
    hls::stream<ap_uint<{AB}> > a_s;
    hls::stream<ap_uint<{PB}> > p_s;
#pragma HLS STREAM variable=w_s depth={NF * SF + 2}
#pragma HLS STREAM variable=a_s depth={SF + 2}
#pragma HLS STREAM variable=p_s depth={NF + 2}
    {name}_repack_a<data_T>(a_stream, a_s);
    {name}_feed_w<CONFIG_T>(w_s);
    {name}_core(w_s, a_s, p_s);
    {name}_drain<res_T, CONFIG_T>(p_s, res_stream, biases);
}}

}} // namespace nnet
#endif
"""


def generate_mvau_pkg(shape, name, output_dir, **cfg):
    """Emit a full mvau blackbox package into ``<output_dir>/<name>/``."""
    if cfg.get("interface") == "array":
        raise ValueError(
            f"mvau target does not support io_parallel (interface=array) for '{name}'. "
            "The FINN MVU is a streaming AXIS core; use io_stream (interface=stream).")
    plan = _geom.fold_plan(*shape, **_plan_kwargs(cfg))
    t = plan["tile"]
    m = plan["num_input_vectors"]
    part = cfg.get("part") or "xcvu13p-flga2577-2-e"
    clock_ns = cfg.get("clock_period_ns") or 5

    pkg = Path(output_dir) / name
    (pkg / "rtl_static").mkdir(parents=True, exist_ok=True)

    # FORCE_BEHAVIORAL=0 -> real DSP48/DSP58 primitives (impl-ready; cosim runs them via
    # XSIM unisim models). Set force_behavioral=True in cfg for unisim-free behavioral cosim.
    force_behavioral = bool(cfg.get("force_behavioral", False))
    (pkg / f"{name}_core.v").write_text(
        _rtl.generate_shim(shape, module_name=f"{name}_core",
                           force_behavioral=force_behavioral, tile=t))
    (pkg / f"{name}_core.cpp").write_text(
        _golden.generate_core_twin(shape, func_name=f"{name}_core", plan=plan))
    bias_codes = bias_acc_codes(cfg.get("bias"), plan["product_frac"], plan["n"])
    (pkg / f"{name}_top.cpp").write_text(_dataflow_top(name, plan, bias_codes=bias_codes))
    (pkg / f"{name}.json").write_text(_blackbox_json(name, t))
    (pkg / f"{name}_tb.cpp").write_text(
        _golden.generate_tb(shape, top_name=name, func_name=f"{name}_core", plan=plan,
                            bias_codes=bias_codes))
    (pkg / f"{name}_gemm_ip.h").write_text(_gemm_ip_header(name, plan))
    (pkg / "run_vitis.tcl").write_text(_run_vitis_tcl(name, part, clock_ns))

    for s in _STATIC_SOURCES:
        shutil.copy(_RTL_STATIC / s, pkg / "rtl_static" / s)
    shutil.copy(_RTL_STATIC / "FINN_COMMIT.txt", pkg / "rtl_static" / "FINN_COMMIT.txt")

    return str(pkg)


# ── batch orchestration (multi-package; used by the CLI --config path) ─────────

def gen_sources_tcl(items):
    """The RTL-blackbox seam hls4ml's Vitis writer sources: add each config's
    blackbox JSON. Paths are resolved relative to this file's own directory
    (`[file dirname [info script]]`) since hls4ml sources it with only $tcldir in
    scope (not $script_dir)."""
    lines = ["# Generated by gemm-ip-gen (mvau): add each config's blackbox JSON.\n",
             "set _mvau_pkgdir [file dirname [info script]]\n"]
    for it in items:
        nm = it.get("emit_name") or it["name"]
        lines.append(f"add_files -blackbox [file join $_mvau_pkgdir {nm} {nm}.json]\n")
    return "".join(lines)


def gen_combined_header(items):
    """The whole-model header hls4ml includes when GEMM_IP_HEADER is set. Includes
    each per-config dedicated IP, then routes nnet::gemm_* to it by CONFIG_T's
    gemm_ip_id via template specialization (c++11-safe: only the matching IP
    instantiates, so the other configs' shapes never compile against this CONFIG_T).
    """
    def _nm(it):
        return it.get("emit_name") or it["name"]

    def _id(it):
        v = it.get("gemm_ip_index")
        return int(v) if v is not None else None

    incs = "\n".join(f'#include "{_nm(it)}/{_nm(it)}_gemm_ip.h"' for it in items)
    specs = []
    for it in items:
        i, nm = _id(it), _nm(it)
        if i is None:
            continue
        specs.append(
            f"template <> struct mvau_ip<{i}> {{\n"
            f"    template <class data_T, class res_T, typename CONFIG_T>\n"
            f"    static void stream_weightless(hls::stream<data_T> &a, hls::stream<res_T> &r,\n"
            f"                                  typename CONFIG_T::bias_t b[CONFIG_T::n_out]) {{\n"
            f"        {nm}_gemm_stream_weightless<data_T, res_T, CONFIG_T>(a, r, b);\n"
            f"    }}\n"
            f"}};")
    specs_s = "\n".join(specs)
    return f"""#ifndef GEMM_IP_COMBINED_H_
#define GEMM_IP_COMBINED_H_
#include <hls_stream.h>
{incs}

namespace nnet {{

// id -> dedicated IP dispatch (specialized per gemm config below).
template <int ID> struct mvau_ip;
{specs_s}

// io_stream weightless entry hls4ml calls; routes to the config's IP by id.
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_weightless(hls::stream<data_T> &a_stream, hls::stream<res_T> &res_stream,
                            typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
    mvau_ip<CONFIG_T::gemm_ip_id>::template stream_weightless<data_T, res_T, CONFIG_T>(
        a_stream, res_stream, biases);
}}

}} // namespace nnet
#endif // GEMM_IP_COMBINED_H_
"""


def gen_integration_manifest(items):
    cores = []
    for it in items:
        nm = it.get("emit_name") or it["name"]
        cores.append({"name": nm, "kind": "rtl_blackbox", "tool": "vitis",
                      "entity": f"{nm}_core", "rtl": f"{nm}/{nm}_core.v",
                      "json": f"{nm}/{nm}.json",
                      "m": it.get("m"), "k": it.get("k"), "n": it.get("n")})
    return json.dumps({"tool": "vitis", "flow": "rtl_blackbox",
                       "header": "gemm_ip_combined.h", "cores": cores}, indent=2) + "\n"


def run_vitis_smoke(cases=None, keep=False):
    """Placeholder for a standalone vitis-run smoke test (mirrors generic target)."""
    raise NotImplementedError("mvau run_vitis_smoke: pending")
