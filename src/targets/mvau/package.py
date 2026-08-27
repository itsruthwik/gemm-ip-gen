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
import weightpack as _wpack

_RTL_STATIC = Path(__file__).resolve().parent / "rtl_static"
# memstream is the weight-stationary weight ROM (baked from <name>_weights.dat);
# always vendored -- the weightless path instantiates it, the (future) two-operand
# streamed path simply leaves it uninstantiated.
_STATIC_SOURCES = ["mvu_vvu_axi.sv", "replay_buffer.sv", "memstream.sv",
                   "mvu_4sx4u.sv", "mvu_8sx8u_dsp48.sv", "mvu_vvu_8sx9_dsp58.sv"]
_WEIGHTS_DAT = "{name}_weights.dat"   # per-IP memstream $readmemh init, in rtl_static/


def _core_rtl_files(name, prefix=""):
    """The complete, ordered RTL source set for one core: the generated shim top
    first, then the vendored FINN cores under rtl_static/. `prefix` (e.g. the
    package-relative "<name>/") is prepended so the same list serves both the
    per-IP blackbox JSON (prefix="") and the integration manifest."""
    return ([f"{prefix}{name}_core.v"]
            + [f"{prefix}rtl_static/{s}" for s in _STATIC_SOURCES])

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


def _dataflow_top(name, plan, bias_codes=None, weights_in_core=False):
    """The DUT: feed -> blackbox (raw integer matmul) -> affine requant drain, in
    a dataflow region. Per vector: NF raw beats in, one N-wide C-row beat out.

    Drain (Keras order: matmul, bias, quantize): add per-column bias in the
    accumulator (2^(fa+fb)) domain, reinterpret as fixed-point, cast to the output
    ap_fixed with AP_RND (round-half-up) + AP_SAT. bias_codes is the bias already
    scaled to that domain (None => no bias; the two-operand GEMMs have none).

    ``weights_in_core`` (weight-stationary): the blackbox bakes its weights in the
    memstream, so the top has no weight input and no ``feed_w`` -- only activations
    flow in. The streamed variant (``w_in`` + ``feed_w``) is reserved for two-operand.
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
    core_decl = (f"void {name}_core(hls::stream<ap_uint<{AB}> >&,\n{pad}hls::stream<ap_uint<{PB}> >&);"
                 if weights_in_core else
                 f"void {name}_core(hls::stream<ap_uint<{WB}> >&, hls::stream<ap_uint<{AB}> >&,\n{pad}hls::stream<ap_uint<{PB}> >&);")
    feed_w_fn = "" if weights_in_core else f"""static void feed_w(hls::stream<ap_uint<{WB}> >& in, hls::stream<ap_uint<{WB}> >& out) {{
    for (int i = 0; i < {wbeats}; i++) out.write(in.read());
}}
"""
    indent = ' ' * (len(name) + 1)
    if weights_in_core:
        top_sig = f"void {name}(hls::stream<ap_uint<{AB}> >& a_in,\n{indent}hls::stream<ap_uint<{CB}> >& c_out)"
        w_stream_decl, w_stream_pragma = "", ""
        feed_calls = (f"    feed_a(a_in, a_s);\n"
                      f"    {name}_core(a_s, p_s);   // <-- FINN MVU RTL blackbox (weights baked in memstream)")
    else:
        top_sig = f"void {name}(hls::stream<ap_uint<{WB}> >& w_in, hls::stream<ap_uint<{AB}> >& a_in,\n{indent}hls::stream<ap_uint<{CB}> >& c_out)"
        w_stream_decl = f"    hls::stream<ap_uint<{WB}> > w_s;\n"
        w_stream_pragma = f"#pragma HLS STREAM variable=w_s depth=4\n"
        feed_calls = (f"    feed_w(w_in, w_s);\n    feed_a(a_in, a_s);\n"
                      f"    {name}_core(w_s, a_s, p_s);   // <-- FINN MVU RTL blackbox (pure integer matmul)")
    return f"""#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// output precision: fixed<{outW},{outI}> with round-half-up + saturate
typedef ap_fixed<{outW}, {outI}, AP_RND, AP_SAT> {name}_result_t;
{bias_decl}
{core_decl}

{feed_w_fn}static void feed_a(hls::stream<ap_uint<{AB}> >& in, hls::stream<ap_uint<{AB}> >& out) {{
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

{top_sig} {{
#pragma HLS DATAFLOW
{w_stream_decl}    hls::stream<ap_uint<{AB}> > a_s;
    hls::stream<ap_uint<{PB}> > p_s;
{w_stream_pragma}#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=p_s depth=4
{feed_calls}
    requant(p_s, c_out);
}}
"""


def _blackbox_json(name, t, weights_in_core=False):
    WB, AB, PB = t["weight_stream_width_ba"], t["input_stream_width_ba"], t["output_stream_width_ba"]
    fn = f"{name}_core"
    dsp = t["dsp_estimate"]
    # weight-stationary: weights are baked in the memstream, so the blackbox has no
    # weight FIFO -- only the activation input and the result output.
    a_param = {"c_name": "a", "c_port_direction": "in",
               "rtl_ports": {"FIFO_data_read_in": "a_dout", "FIFO_read_enable": "a_read", "FIFO_empty_flag": "a_empty_n"}}
    p_param = {"c_name": "p", "c_port_direction": "out",
               "rtl_ports": {"FIFO_data_write_out": "p_din", "FIFO_write_enable": "p_write", "FIFO_full_flag": "p_full_n"}}
    w_param = {"c_name": "w", "c_port_direction": "in",
               "rtl_ports": {"FIFO_data_read_in": "w_dout", "FIFO_read_enable": "w_read", "FIFO_empty_flag": "w_empty_n"}}
    c_params = [a_param, p_param] if weights_in_core else [w_param, a_param, p_param]
    return json.dumps({
        "c_function_name": fn,
        "rtl_top_module_name": fn,     # MUST equal c_function_name (Vitis cosim gotcha)
        "c_files": [{"c_file": f"{fn}.cpp", "cflag": ""}],
        "rtl_files": _core_rtl_files(name),
        "c_parameters": c_params,
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


def _gemm_ip_header(name, plan, weights_in_core=True):
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
    core_hdr_pad = ' ' * (len(name) + 6)
    core_decl = (f"void {name}_core(hls::stream<ap_uint<{AB}> >&,\n{core_hdr_pad}hls::stream<ap_uint<{PB}> >&);"
                 if weights_in_core else
                 f"void {name}_core(hls::stream<ap_uint<{WB}> >&, hls::stream<ap_uint<{AB}> >&,\n{core_hdr_pad}hls::stream<ap_uint<{PB}> >&);")
    # weightless (weight-stationary): weights are baked in the RTL memstream, so no
    # feed_w / weight stream. Streamed feed_w kept for the future two-operand path.
    if weights_in_core:
        feed_w_tmpl = ""
        ws_streams = (f"    hls::stream<ap_uint<{AB}> > a_s;\n"
                      f"    hls::stream<ap_uint<{PB}> > p_s;\n"
                      f"#pragma HLS STREAM variable=a_s depth={SF + 2}\n"
                      f"#pragma HLS STREAM variable=p_s depth={NF + 2}")
        ws_calls = (f"    {name}_repack_a<data_T>(a_stream, a_s);\n"
                    f"    {name}_core(a_s, p_s);\n"
                    f"    {name}_drain<res_T, CONFIG_T>(p_s, res_stream, biases);")
    else:
        feed_w_tmpl = f"""// weight-stationary feed: source B^T from CONFIG_T::gemm_weight_cols() (weight_cols[n][k]),
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

"""
        ws_streams = (f"    hls::stream<ap_uint<{WB}> > w_s;\n"
                      f"    hls::stream<ap_uint<{AB}> > a_s;\n"
                      f"    hls::stream<ap_uint<{PB}> > p_s;\n"
                      f"#pragma HLS STREAM variable=w_s depth={NF * SF + 2}\n"
                      f"#pragma HLS STREAM variable=a_s depth={SF + 2}\n"
                      f"#pragma HLS STREAM variable=p_s depth={NF + 2}")
        ws_calls = (f"    {name}_repack_a<data_T>(a_stream, a_s);\n"
                    f"    {name}_feed_w<CONFIG_T>(w_s);\n"
                    f"    {name}_core(w_s, a_s, p_s);\n"
                    f"    {name}_drain<res_T, CONFIG_T>(p_s, res_stream, biases);")
    return f"""#ifndef {name.upper()}_GEMM_IP_H_
#define {name.upper()}_GEMM_IP_H_
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// internal FINN-MVU blackbox (shim {name}_core.v; C twin {name}_core.cpp)
{core_decl}

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

{feed_w_tmpl}// requant drain (runtime per-column bias, Keras order): raw ACCU -> fixed -> +bias -> round+sat.
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
{ws_streams}
{ws_calls}
}}

}} // namespace nnet
#endif
"""


def _synth_weights(n, k, ww, seed=42):
    """Deterministic in-range B ([K][N]) when no real weights are supplied (unit
    tests / standalone). Same small [-3, 3]-ish spread the old streamed TB used, so
    the self-check stays representative; narrow-safe (never the most-negative code)."""
    wrange = (1 << (ww - 1)) - 1
    wmod = min(7, 2 * wrange + 1)
    off = wrange if wmod == 2 * wrange + 1 else 3
    return [[((oo + kk + seed) % wmod) - off for oo in range(n)] for kk in range(k)]


def _weight_matrix_as_B(cfg, n, k, ww):
    """Return baked weights as a plain ``[K][N]`` int list. Uses the hls4ml weights
    (``cfg['weight_matrix']``, already ``[K, N]`` from load_weight_dat) when present,
    else a deterministic synthetic matrix so tests / standalone packages still bake."""
    wm = cfg.get("weight_matrix")
    if wm is None:
        return _synth_weights(n, k, ww)
    B = [[int(wm[kk][oo]) for oo in range(n)] for kk in range(k)]
    if len(B) != k or (B and len(B[0]) != n):
        raise ValueError(f"weight_matrix shape != ({k}, {n})")
    return B


def generate_mvau_pkg(shape, name, output_dir, **cfg):
    """Emit a full mvau blackbox package into ``<output_dir>/<name>/``.

    Weight-stationary: the constant weights are packed into the FINN memstream
    ``$readmemh`` init (``rtl_static/<name>_weights.dat``) and baked into the shim,
    C twin and TB. The blackbox has no weight port -- only activations in, C row out.
    """
    if cfg.get("interface") == "array":
        raise ValueError(
            f"mvau target does not support io_parallel (interface=array) for '{name}'. "
            "The FINN MVU is a streaming AXIS core; use io_stream (interface=stream).")
    plan = _geom.fold_plan(*shape, **_plan_kwargs(cfg))
    t = plan["tile"]
    m = plan["num_input_vectors"]
    part = cfg.get("part") or "xcvu13p-flga2577-2-e"
    clock_ns = cfg.get("clock_period_ns") or 5
    N, K, PE, SIMD, WW = plan["n"], plan["k"], t["pe"], t["simd"], t["weight_width"]

    pkg = Path(output_dir) / name
    (pkg / "rtl_static").mkdir(parents=True, exist_ok=True)

    # Pack the baked weights into the memstream init, alongside the vendored RTL.
    B = _weight_matrix_as_B(cfg, N, K, WW)
    dat_name = _WEIGHTS_DAT.format(name=name)
    dat_path = pkg / "rtl_static" / dat_name
    dat_path.write_text(_wpack.pack_memstream_hex(
        B, N, K, PE, SIMD, WW, word_bits=t["weight_stream_width_ba"]))
    # Absolute $readmemh path: relative is unresolvable in Vitis cosim's XSIM dir
    # (empirically -- see jojo-track); the package builds in place, so the absolute
    # path computed here stays valid for csim/cosim/impl.
    init_file = str(dat_path.resolve())

    # FORCE_BEHAVIORAL=0 -> real DSP48/DSP58 primitives (impl-ready; cosim runs them via
    # XSIM unisim models). Set force_behavioral=True in cfg for unisim-free behavioral cosim.
    force_behavioral = bool(cfg.get("force_behavioral", False))
    (pkg / f"{name}_core.v").write_text(
        _rtl.generate_shim(shape, module_name=f"{name}_core",
                           force_behavioral=force_behavioral, tile=t,
                           weights_in_core=True, init_file=init_file))
    (pkg / f"{name}_core.cpp").write_text(
        _golden.generate_core_twin(shape, func_name=f"{name}_core", plan=plan, baked_weights=B))
    bias_codes = bias_acc_codes(cfg.get("bias"), plan["product_frac"], plan["n"])
    (pkg / f"{name}_top.cpp").write_text(
        _dataflow_top(name, plan, bias_codes=bias_codes, weights_in_core=True))
    (pkg / f"{name}.json").write_text(_blackbox_json(name, t, weights_in_core=True))
    (pkg / f"{name}_tb.cpp").write_text(
        _golden.generate_tb(shape, top_name=name, func_name=f"{name}_core", plan=plan,
                            bias_codes=bias_codes, baked_weights=B))
    (pkg / f"{name}_gemm_ip.h").write_text(_gemm_ip_header(name, plan, weights_in_core=True))
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
        core = {"name": nm, "kind": "rtl_blackbox", "tool": "vitis",
                "entity": f"{nm}_core", "rtl": f"{nm}/{nm}_core.v",
                # full RTL set (shim + vendored FINN cores), not just the shim
                "rtl_files": _core_rtl_files(nm, prefix=f"{nm}/"),
                "json": f"{nm}/{nm}.json",
                "m": it.get("m"), "k": it.get("k"), "n": it.get("n")}
        # weight-stationary: the packed memstream init is a data dependency the shim
        # $readmemh's by absolute path -- record it so downstream keeps it with the
        # package (it cannot go in rtl_files: Vitis rejects a .dat as blackbox RTL).
        core["weight_data"] = f"{nm}/rtl_static/{_WEIGHTS_DAT.format(name=nm)}"
        cores.append(core)
    return json.dumps({"tool": "vitis", "flow": "rtl_blackbox",
                       "header": "gemm_ip_combined.h", "cores": cores}, indent=2) + "\n"


def run_vitis_smoke(cases=None, keep=False):
    """Placeholder for a standalone vitis-run smoke test (mirrors generic target)."""
    raise NotImplementedError("mvau run_vitis_smoke: pending")
