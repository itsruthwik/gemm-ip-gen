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
import re
import shutil
from pathlib import Path

import geometry as _geom
import rtl as _rtl
import golden as _golden
import weightpack as _wpack

_RTL_STATIC = Path(__file__).resolve().parent / "rtl_static"
# memstream is the weight-stationary weight ROM (baked from <name>_weights.dat);
# always vendored -- the const_weights path instantiates it, the (future) two-operand
# streamed path simply leaves it uninstantiated.
_STATIC_SOURCES = ["mvu_vvu_axi.sv", "replay_buffer.sv", "memstream.sv",
                   "mvu_4sx4u.sv", "mvu_8sx8u_dsp48.sv", "mvu_vvu_8sx9_dsp58.sv"]
_WEIGHTS_DAT = "{name}_weights.dat"   # per-IP memstream $readmemh init, in rtl_static/
# XSIM requires all-or-none `timescale across the design. The vendored FINN cores and the
# generated shim carry none (fine standalone), but the hls4ml RTL they integrate with does
# -> cosim elaboration fails. Stamp a matching timescale onto every mvau RTL file.
_TIMESCALE = "`timescale 1ns / 1ps\n"


def _with_timescale(text):
    return text if "`timescale" in text else _TIMESCALE + text


def _rv_width(accu, bias_in_core):
    """Just-wide-enough intermediate width for a requant drain's ``rv``. With the
    bias add, ``CONFIG_T::bias_t`` is unknown at generation time, so keep headroom
    (>= 32b) plus a couple guard bits; without it, ACCU + 1 guard bit suffices."""
    return (max(accu, 32) + 2) if bias_in_core else (accu + 1)


def _rv_decl_assign(width, pfrac, raw_expr, indent=""):
    """``ap_fixed`` decl + range-assign for a requant drain's ``rv``, narrowed to
    *width* bits (was a fixed 64-bit intermediate)."""
    return (f"{indent}ap_fixed<{width}, {width - pfrac}> rv;\n"
            f"{indent}rv.range({width - 1}, 0) = (ap_uint<{width}>)(ap_int<{width}>){raw_expr};")


def _apmaxw(bits):
    """AP_INT_MAX_W setting for a design whose widest ap_uint is ``bits`` wide.
    The default cap is 1024; K-tiled result beats (KT*PB) blow past it. Round up to
    the next KiB with a KiB of headroom (min 1024 = the default, so narrow cases are
    unaffected)."""
    return max(1024, ((bits + 1023) // 1024 + 1) * 1024)


#: any ap_int / ap_uint / ap_fixed / ap_ufixed declaration, capturing the bit width.
_AP_WIDTH_RE = re.compile(r"ap_u?(?:int|fixed)<\s*(\d+)")


def _with_ap_int_max_w(src):
    """Guarantee a generated C++/header TU raises ``AP_INT_MAX_W`` above the ap_int
    default (1024) whenever it declares a wider ``ap_uint``/``ap_fixed``. Scans the
    emitted source for its widest ap type and prepends an ``#ifndef``-guarded define
    ahead of the includes. Idempotent: a TU that already sets the macro (the grid
    tops/twins size it from the result beat) is left untouched, and a TU whose types
    all fit the default gets nothing — so narrow single-tile packages are unchanged.

    Without this, a layer that folds to a single tile with a wide beat (e.g. a wide
    dense layer at moderate ReuseFactor) aborts csim with 'Bitwidth exceeds the
    default max value 1024'."""
    if "AP_INT_MAX_W" in src:
        return src
    widths = [int(w) for w in _AP_WIDTH_RE.findall(src)]
    if not widths or max(widths) <= 1024:
        return src
    apmax = _apmaxw(max(widths))
    return (f"#ifndef AP_INT_MAX_W\n#define AP_INT_MAX_W {apmax}   "
            f"// widest ap type ({max(widths)}b) exceeds the 1024-bit default\n#endif\n"
            + src)


def _dat_name(name, ti, n_tiles):
    """memstream ``$readmemh`` init filename for tile ``ti``. Single-tile keeps the
    original ``<name>_weights.dat``; N-tiling suffixes ``_t{ti}`` per column slice."""
    if n_tiles <= 1:
        return _WEIGHTS_DAT.format(name=name)
    return f"{name}_weights_t{ti}.dat"


def _kdat_name(name, ti):
    """memstream ``$readmemh`` init filename for K-tile ``ti`` (one per K-row slice)."""
    return f"{name}_weights_k{ti}.dat"


def _core_rtl_files(name, prefix=""):
    """The complete, ordered RTL source set for one core: the generated shim top
    first, then the vendored FINN cores under rtl_static/. `prefix` (e.g. the
    package-relative "<name>/") is prepended so the same list serves both the
    per-IP blackbox JSON (prefix="") and the integration manifest."""
    return ([f"{prefix}{name}_core.v"]
            + [f"{prefix}rtl_static/{s}" for s in _STATIC_SOURCES])

_PLAN_KEYS = ("weight_precision", "input_precision", "output_precision", "part",
              "clock_period_ns", "reuse_factor", "strategy", "target_cycles",
              "parallelization_factor", "n_tiles", "k_tiles", "weights")


def _plan_kwargs(cfg):
    return {k: cfg[k] for k in _PLAN_KEYS if k in cfg and cfg[k] is not None}


def _resolve_plan(shape, cfg):
    """Resolve the MVU plan for *shape* (see :func:`geometry.resolve_plan`): the fold is
    user-directed here — the tiling knobs (``n_tiles``/``k_tiles``) and ReuseFactor come
    from the config and ``fold_plan`` derives the rest. No mapper / cost model on this branch."""
    return _geom.resolve_plan(shape, **cfg)


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
    NT, NTILE = plan["n_tiles"], plan["n_tile"]
    PB_TOTAL = NT * PB   # concatenated result beat: the n_tiles tiles side by side
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
    core_decl = (f"void {name}_core(hls::stream<ap_uint<{AB}> >&,\n{pad}hls::stream<ap_uint<{PB_TOTAL}> >&);"
                 if weights_in_core else
                 f"void {name}_core(hls::stream<ap_uint<{WB}> >&, hls::stream<ap_uint<{AB}> >&,\n{pad}hls::stream<ap_uint<{PB_TOTAL}> >&);")
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

// Affine requant drain: raw ACCU codes (+bias) -> N-wide requantized C row. The
// input beat concatenates the n_tiles tiles; tile ti lane pe (bits ti*PB+pe*ACCU)
// is global column ti*n_tile + nf*PE + pe.
static void requant(hls::stream<ap_uint<{PB_TOTAL}> >& in, hls::stream<ap_uint<{CB}> >& out) {{
    for (int vec = 0; vec < {m}; vec++) {{
        ap_uint<{CB}> crow = 0;
        for (int nf = 0; nf < {NF}; nf++) {{
            ap_uint<{PB_TOTAL}> ob = in.read();
            for (int ti = 0; ti < {NT}; ti++)
                for (int pe = 0; pe < {PE}; pe++) {{
                    int oc = ti * {NTILE} + nf * {PE} + pe;
                    ap_int<{ACCU}> raw = ob.range(ti * {PB} + pe * {ACCU} + {ACCU} - 1, ti * {PB} + pe * {ACCU});
                    ap_int<{_rv_width(ACCU, bool(bias_codes))}> v = (ap_int<{_rv_width(ACCU, bool(bias_codes))}>)raw{bias_add};         // matmul + bias, accum domain
{_rv_decl_assign(_rv_width(ACCU, bool(bias_codes)), pfrac, "v", ' ' * 20)}   // code -> fixed (frac={pfrac})
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
    hls::stream<ap_uint<{PB_TOTAL}> > p_s;
{w_stream_pragma}#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=p_s depth=4
{feed_calls}
    requant(p_s, c_out);
}}
"""


def _blackbox_json(name, t, weights_in_core=False, tiles=1, resources=None):
    WB, AB, PB = t["weight_stream_width_ba"], t["input_stream_width_ba"], t["output_stream_width_ba"]
    fn = f"{name}_core"
    # Cost-model resource estimate over all copies.K·copies.N tiles: DSP + BRAM18 from
    # geometry/cost (validated against Vivado OOC synth). If the caller didn't resolve a
    # plan (legacy path), fall back to the per-tile DSP × tiles and no BRAM.
    res = resources or {"dsp": t["dsp_estimate"] * int(tiles), "bram18": 0}
    dsp, bram = int(res["dsp"]), int(res.get("bram18", 0))
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
            "ap_ctrl_chain_protocol_idle": "ap_idle",
            "ap_ctrl_chain_protocol_start": "ap_start",
            "ap_ctrl_chain_protocol_ready": "ap_ready",
            "ap_ctrl_chain_protocol_done": "ap_done",
            "ap_ctrl_chain_protocol_continue": "ap_continue",
        },
        # Deterministic (RTL is a fixed pipeline): latency + II are exact functions
        # of the fold, verified against XSIM (geometry.latency_cycles / output_ii).
        "rtl_performance": {"latency": str(t["latency_cycles"]), "II": str(t["ii"])},
        # JSON has no comment syntax; this underscore-key carries a note in the file.
        "_comment": "FINN cost-model DSP estimate scaled by the tile count; FF/LUT/BRAM "
                    "are rough hints, not synthesis-accurate (TODO: measure).",
        # DSP + BRAM18 over all tiles from the cost model; FF/LUT are rough per-DSP hints.
        "rtl_resource_usage": {"FF": str(30 * dsp), "LUT": str(40 * dsp),
                               "DSP": str(dsp), "BRAM": str(bram), "URAM": "0"},
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


def _gemm_ip_header(name, plan, weights_in_core=True, bias_in_core=True):
    """The dedicated hls4ml-facing IP for one gemm config: an HLS C++ dataflow IP
    with the internal FINN-MVU blackbox. Repacks hls4ml beats -> FINN beats,
    runs the blackbox, requant-drains (runtime bias) -> hls4ml C row. The combined
    header routes nnet::gemm_* to this by CONFIG_T::gemm_ip_id (template dispatch).

    io_stream const_weights (projections) implemented; two-operand + io_parallel
    follow the same pattern.
    """
    t = plan["tile"]
    m = plan["num_input_vectors"]
    WB, AB, PB = t["weight_stream_width_ba"], t["input_stream_width_ba"], t["output_stream_width_ba"]
    PE, SF, NF, ACCU = t["pe"], t["sf"], t["nf"], t["accu_width"]
    NT, NTILE = plan["n_tiles"], plan["n_tile"]
    PB_TOTAL = NT * PB   # concatenated result beat: the n_tiles tiles side by side
    WW, AW, K, N = t["weight_width"], t["activation_width"], plan["k"], plan["n"]
    KPAD = plan["k_pad"]   # K padded to a multiple of SIMD; arow is KPAD-wide, pad lanes 0
    pfrac = plan["product_frac"]
    guard = ' ' * 6
    core_hdr_pad = ' ' * (len(name) + 6)
    core_decl = (f"void {name}_core(hls::stream<ap_uint<{AB}> >&,\n{core_hdr_pad}hls::stream<ap_uint<{PB_TOTAL}> >&);"
                 if weights_in_core else
                 f"void {name}_core(hls::stream<ap_uint<{WB}> >&, hls::stream<ap_uint<{AB}> >&,\n{core_hdr_pad}hls::stream<ap_uint<{PB_TOTAL}> >&);")
    # const_weights (weight-stationary): weights are baked in the RTL memstream, so no
    # feed_w / weight stream. Streamed feed_w kept for the future two-operand path.
    if weights_in_core:
        feed_w_tmpl = ""
        ws_streams = (f"    hls::stream<ap_uint<{AB}> > a_s;\n"
                      f"    hls::stream<ap_uint<{PB_TOTAL}> > p_s;\n"
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
                      f"    hls::stream<ap_uint<{PB_TOTAL}> > p_s;\n"
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
        ap_int<{AW}> arow[{KPAD}];
        #pragma HLS ARRAY_PARTITION variable=arow complete
        for (unsigned i = 0; i < {KPAD}; i++) {{
            #pragma HLS UNROLL
            arow[i] = 0;   // zero-fill the K-pad lanes (their weights are 0 -> bit-exact)
        }}
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
void {name}_drain(hls::stream<ap_uint<{PB_TOTAL}> > &p_s, hls::stream<res_T> &res_stream,
{' ' * (len(name) + 7)}typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
    typedef typename res_T::value_type result_t;
    for (unsigned mm = 0; mm < {m}; mm++) {{
        res_T crow;
        for (unsigned nf = 0; nf < {NF}; nf++) {{
            ap_uint<{PB_TOTAL}> ob = p_s.read();
            for (unsigned ti = 0; ti < {NT}; ti++)     // tile ti lane pe -> global column
                for (unsigned pe = 0; pe < {PE}; pe++) {{
                    unsigned oc = ti * {NTILE} + nf * {PE} + pe;
                    ap_int<{ACCU}> raw = ob.range(ti * {PB} + pe * {ACCU} + {ACCU} - 1, ti * {PB} + pe * {ACCU});
{_rv_decl_assign(_rv_width(ACCU, bias_in_core), pfrac, "raw", ' ' * 20)}   // code -> fixed (frac={pfrac})
{('                    rv += biases[oc];                                 // + bias (fixed-point, aligned)' if bias_in_core else '')}
                    result_t r = rv;                                  // convert per res_T's rounding/overflow modes
                    crow[oc] = r;
                }}
        }}
        res_stream.write(crow);
    }}
}}

// The dedicated IP: hls4ml io_stream const_weights GEMM -> internal MVU blackbox.
template <class data_T, class res_T, typename CONFIG_T>
void {name}_gemm_stream_const_weights(hls::stream<data_T> &a_stream, hls::stream<res_T> &res_stream,
{' ' * (len(name) + 28)}typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
#pragma HLS DATAFLOW
{ws_streams}
{ws_calls}
}}

}} // namespace nnet
#endif
"""


def _kt_dataflow_top(name, plan, bias_codes=None):
    """DUT for the K-tiled blackbox (fully-spatial per-tile, SF=NF=1). The activation
    arrives as one wide beat of ``KT*AB`` (all K_pad activations); the blackbox emits
    one wide beat of ``KT*PB`` holding the ``KT`` per-tile PARTIALS for all N columns.
    The drain SUMS the partials per column (accumulator widened by ceil(log2 KT)),
    then adds bias and requantizes -- the K-tiling counterpart of the single-tile drain."""
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AB, PB, ACCU = t["input_stream_width_ba"], t["output_stream_width_ba"], t["accu_width"]
    PE, NF, MW = t["pe"], t["nf"], t["mw"]
    SFT = MW // t["simd"]
    KT, NT = plan["k_tiles"], plan["n_tiles"]
    A_TOTAL, PB_TOTAL = KT * AB, NT * KT * PB
    ACCU_SUM = plan["accu_sum"]
    N, outW, outI = plan["n"], plan["output_width"], plan["output_int"]
    NTILE = N // NT
    pfrac = plan["product_frac"]
    CB = cbits(plan)
    pad = ' ' * (len(name) + 6)
    if bias_codes:
        bias_decl = (f"static const long {name}_bias[{N}] = {{"
                     + ", ".join(str(c) for c in bias_codes) + "};\n")
        bias_add = f" + {name}_bias[oc]"
    else:
        bias_decl, bias_add = "", ""
    indent = ' ' * (len(name) + 1)
    apmax = _apmaxw(PB_TOTAL)
    return f"""#define AP_INT_MAX_W {apmax}   // wide K-tiled beats (KT*PB={PB_TOTAL}) exceed the 1024-bit default
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// output precision: fixed<{outW},{outI}> with round-half-up + saturate
typedef ap_fixed<{outW}, {outI}, AP_RND, AP_SAT> {name}_result_t;
{bias_decl}
void {name}_core(hls::stream<ap_uint<{A_TOTAL}> >&,
{pad}hls::stream<ap_uint<{PB_TOTAL}> >&);

static void feed_a(hls::stream<ap_uint<{A_TOTAL}> >& in, hls::stream<ap_uint<{A_TOTAL}> >& out) {{
    for (int i = 0; i < {m * SFT}; i++) out.write(in.read());
}}

// Grid requant drain: each vector emits NF={NF} partial beats; sum the KT K-partials per
// column (accum widened to {ACCU_SUM} = ACCU + ceil(log2 KT)) and concatenate the NT N-slices
// (tile (j,i) at [(j*KT+i)*PB]), then + bias and quantize (Keras order).
static void requant(hls::stream<ap_uint<{PB_TOTAL}> >& in, hls::stream<ap_uint<{CB}> >& out) {{
    for (int vec = 0; vec < {m}; vec++) {{
        ap_uint<{CB}> crow = 0;
        for (int nf = 0; nf < {NF}; nf++) {{
            ap_uint<{PB_TOTAL}> ob = in.read();
            for (int j = 0; j < {NT}; j++)
            for (int pe = 0; pe < {PE}; pe++) {{
                int oc = j * {NTILE} + nf * {PE} + pe;
                ap_int<{ACCU_SUM}> raw = 0;
                for (int i = 0; i < {KT}; i++)
                    raw += (ap_int<{ACCU}>)ob.range((j * {KT} + i) * {PB} + pe * {ACCU} + {ACCU} - 1, (j * {KT} + i) * {PB} + pe * {ACCU});
                ap_int<{_rv_width(ACCU_SUM, bool(bias_codes))}> v = (ap_int<{_rv_width(ACCU_SUM, bool(bias_codes))}>)raw{bias_add};         // Σ partials + bias, accum domain
{_rv_decl_assign(_rv_width(ACCU_SUM, bool(bias_codes)), pfrac, "v", ' ' * 16)}   // code -> fixed (frac={pfrac})
                {name}_result_t r = rv;                           // rescale + round + saturate
                crow.range(oc * {outW} + {outW} - 1, oc * {outW}) = r.range({outW} - 1, 0);
            }}
        }}
        out.write(crow);
    }}
}}

void {name}(hls::stream<ap_uint<{A_TOTAL}> >& a_in,
{indent}hls::stream<ap_uint<{CB}> >& c_out) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{A_TOTAL}> > a_s;
    hls::stream<ap_uint<{PB_TOTAL}> > p_s;
#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=p_s depth=4
    feed_a(a_in, a_s);
    {name}_core(a_s, p_s);   // <-- FINN MVU RTL blackbox ({KT} K-tiles, partials)
    requant(p_s, c_out);
}}
"""


def _kt_gemm_ip_header(name, plan, bias_in_core=True):
    """hls4ml-facing IP for a K-tiled gemm config (fully-spatial per-tile). Repacks the
    K activations into one wide ``KT*AB`` beat, runs the blackbox, and drains by summing
    the KT partials per column (runtime bias) -- the K-tiling twin of ``_gemm_ip_header``."""
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AB, PB, ACCU = t["input_stream_width_ba"], t["output_stream_width_ba"], t["accu_width"]
    PE, NF, MW, SIMD = t["pe"], t["nf"], t["mw"], t["simd"]
    SFT = MW // SIMD
    KT, NT = plan["k_tiles"], plan["n_tiles"]
    A_TOTAL, PB_TOTAL = KT * AB, NT * KT * PB
    ACCU_SUM = plan["accu_sum"]
    AW, K, N = t["activation_width"], plan["k"], plan["n"]
    NTILE = N // NT
    KPAD = plan["k_pad"]
    pfrac = plan["product_frac"]
    core_hdr_pad = ' ' * (len(name) + 6)
    apmax = _apmaxw(PB_TOTAL)
    return f"""#ifndef {name.upper()}_GEMM_IP_H_
#define {name.upper()}_GEMM_IP_H_
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W {apmax}   // wide K-tiled beats (KT*PB={PB_TOTAL}) exceed the 1024-bit default
#endif
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// internal FINN-MVU blackbox (K-tiled shim {name}_core.v; C twin {name}_core.cpp)
void {name}_core(hls::stream<ap_uint<{A_TOTAL}> >&,
{core_hdr_pad}hls::stream<ap_uint<{PB_TOTAL}> >&);

namespace nnet {{

// Dedicated K-tiled IP for gemm config M={m} K={K} N={N} (core={t['compute_core']},
// KT={KT} K-tiles of SIMD={t['simd']} each, PE={t['pe']} full-N). Baked geometry.

// repack: {m} rows of K={K} activations -> {m} wide beats of KT*AB (all K_pad lanes,
// pad lanes 0 -> bit-exact; tile ti reads lanes [ti*AB +: AB]).
template <class data_T>
void {name}_repack_a(hls::stream<data_T> &a_stream, hls::stream<ap_uint<{A_TOTAL}> > &a_s) {{
    for (unsigned mm = 0; mm < {m}; mm++) {{
        ap_int<{AW}> arow[{KPAD}];
        #pragma HLS ARRAY_PARTITION variable=arow complete
        for (unsigned i = 0; i < {KPAD}; i++) {{
            #pragma HLS UNROLL
            arow[i] = 0;   // zero-fill the K-pad lanes (their weights are 0 -> bit-exact)
        }}
        for (unsigned kp = 0; kp < {K} / data_T::size; kp++) {{
            data_T beat = a_stream.read();
            for (unsigned j = 0; j < data_T::size; j++) arow[kp * data_T::size + j] = beat[j].range({AW} - 1, 0);
        }}
        // emit SF_tile beats/vector; band sf carries each tile's SIMD lanes at [ti*AB +: AB]
        for (unsigned sf = 0; sf < {SFT}; sf++) {{
            ap_uint<{A_TOTAL}> ab = 0;
            for (unsigned ti = 0; ti < {KT}; ti++)
                for (unsigned s = 0; s < {SIMD}; s++)
                    ab.range(ti * {AB} + s * {AW} + {AW} - 1, ti * {AB} + s * {AW}) =
                        (ap_uint<{AW}>)arow[ti * {MW} + sf * {SIMD} + s];
            a_s.write(ab);
        }}
    }}
}}

// K-tiled requant drain: sum the KT partials per column, + runtime bias, round+sat.
template <class res_T, typename CONFIG_T>
void {name}_drain(hls::stream<ap_uint<{PB_TOTAL}> > &p_s, hls::stream<res_T> &res_stream,
{' ' * (len(name) + 7)}typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
    typedef typename res_T::value_type result_t;
    for (unsigned mm = 0; mm < {m}; mm++) {{
        res_T crow;
        for (unsigned nf = 0; nf < {NF}; nf++) {{     // NF partial beats/vector, PE columns each
            ap_uint<{PB_TOTAL}> ob = p_s.read();
            for (unsigned j = 0; j < {NT}; j++)
            for (unsigned pe = 0; pe < {PE}; pe++) {{
                unsigned oc = j * {NTILE} + nf * {PE} + pe;
                ap_int<{ACCU_SUM}> raw = 0;
                for (unsigned i = 0; i < {KT}; i++)
                    raw += (ap_int<{ACCU}>)ob.range((j * {KT} + i) * {PB} + pe * {ACCU} + {ACCU} - 1, (j * {KT} + i) * {PB} + pe * {ACCU});
{_rv_decl_assign(_rv_width(ACCU_SUM, bias_in_core), pfrac, "raw", ' ' * 16)}   // code -> fixed (frac={pfrac})
{('                rv += biases[oc];                                 // + bias (fixed-point, aligned)' if bias_in_core else '')}
                result_t r = rv;                                  // convert per res_T's rounding/overflow modes
                crow[oc] = r;
            }}
        }}
        res_stream.write(crow);
    }}
}}

// The dedicated K-tiled IP: hls4ml io_stream const_weights GEMM -> internal MVU blackbox.
template <class data_T, class res_T, typename CONFIG_T>
void {name}_gemm_stream_const_weights(hls::stream<data_T> &a_stream, hls::stream<res_T> &res_stream,
{' ' * (len(name) + 28)}typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{A_TOTAL}> > a_s;
    hls::stream<ap_uint<{PB_TOTAL}> > p_s;
#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=p_s depth=4
    {name}_repack_a<data_T>(a_stream, a_s);
    {name}_core(a_s, p_s);
    {name}_drain<res_T, CONFIG_T>(p_s, res_stream, biases);
}}

}} // namespace nnet
#endif
"""


def _2op_dataflow_top(name, plan):
    """DUT for the two-operand blackbox (NF=1 single tile): pure passthrough of both A and
    B streams into the core, then the affine requant drain (no bias; act×act product scale
    ``fa+fb``). B is buffered/replayed inside the RTL, so the top just forwards its beats."""
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AB, PB, ACCU = t["input_stream_width_ba"], t["output_stream_width_ba"], t["accu_width"]
    PE, SF, NF = t["pe"], t["sf"], t["nf"]
    N, WW = plan["n"], t["weight_width"]
    BB = ((N * WW) + 7) // 8 * 8
    K = plan["k_pad"]
    outW, outI, pfrac = plan["output_width"], plan["output_int"], plan["product_frac"]
    CB = cbits(plan)
    abeats, bbeats = m * SF, K
    pad = ' ' * (len(name) + 6)
    indent = ' ' * (len(name) + 1)
    return f"""#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// output precision: fixed<{outW},{outI}> with round-half-up + saturate
typedef ap_fixed<{outW}, {outI}, AP_RND, AP_SAT> {name}_result_t;

void {name}_core(hls::stream<ap_uint<{AB}> >&, hls::stream<ap_uint<{BB}> >&,
{pad}hls::stream<ap_uint<{PB}> >&);

static void feed_a(hls::stream<ap_uint<{AB}> >& in, hls::stream<ap_uint<{AB}> >& out) {{
    for (int i = 0; i < {abeats}; i++) out.write(in.read());
}}
static void feed_b(hls::stream<ap_uint<{BB}> >& in, hls::stream<ap_uint<{BB}> >& out) {{
    for (int i = 0; i < {bbeats}; i++) out.write(in.read());
}}

// Affine requant drain (no bias): NF raw-ACCU beats -> one N-wide requantized C row.
// Beat nf lane pe holds output column nf*PE+pe.
static void requant(hls::stream<ap_uint<{PB}> >& in, hls::stream<ap_uint<{CB}> >& out) {{
    for (int vec = 0; vec < {m}; vec++) {{
        ap_uint<{CB}> crow = 0;
        for (int nf = 0; nf < {NF}; nf++) {{
            ap_uint<{PB}> ob = in.read();
            for (int pe = 0; pe < {PE}; pe++) {{
                int oc = nf * {PE} + pe;
                ap_int<{ACCU}> raw = ob.range(pe * {ACCU} + {ACCU} - 1, pe * {ACCU});
{_rv_decl_assign(_rv_width(ACCU, False), pfrac, "raw", ' ' * 16)}   // code -> fixed (frac={pfrac})
                {name}_result_t r = rv;                           // rescale + round + saturate
                crow.range(oc * {outW} + {outW} - 1, oc * {outW}) = r.range({outW} - 1, 0);
            }}
        }}
        out.write(crow);
    }}
}}

void {name}(hls::stream<ap_uint<{AB}> >& a_in, hls::stream<ap_uint<{BB}> >& b_in,
{indent}hls::stream<ap_uint<{CB}> >& c_out) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{AB}> > a_s;
    hls::stream<ap_uint<{BB}> > b_s;
    hls::stream<ap_uint<{PB}> > p_s;
#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=b_s depth=4
#pragma HLS STREAM variable=p_s depth=4
    feed_a(a_in, a_s);
    feed_b(b_in, b_s);
    {name}_core(a_s, b_s, p_s);   // <-- FINN MVU RTL blackbox (B loaded+replayed in-core)
    requant(p_s, c_out);
}}
"""


def _2op_kt_dataflow_top(name, plan):
    """DUT for the K-tiled two-operand blackbox: passthrough of the wide A beat + the B
    stream into the core, then the summing requant drain (sum the gk per-tile partials per
    column, accumulator widened to accu_sum; no bias; act×act scale fa+fb)."""
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AB, PB, ACCU = t["input_stream_width_ba"], t["output_stream_width_ba"], t["accu_width"]
    PE, NF, MW = t["pe"], t["nf"], t["mw"]
    SFT = MW // t["simd"]
    gk, nt = plan["k_tiles"], plan["n_tiles"]
    A_TOTAL, PB_TOTAL = gk * AB, nt * gk * PB
    ACCU_SUM = plan["accu_sum"]
    N, WW = plan["n"], t["weight_width"]
    NTILE = N // nt
    BB = ((N * WW) + 7) // 8 * 8
    K = plan["k_pad"]
    outW, outI, pfrac = plan["output_width"], plan["output_int"], plan["product_frac"]
    CB = cbits(plan)
    apmax = _apmaxw(PB_TOTAL)
    pad = ' ' * (len(name) + 6)
    indent = ' ' * (len(name) + 1)
    return f"""#define AP_INT_MAX_W {apmax}   // concatenated partial beat (gk*PB={PB_TOTAL}) exceeds the 1024-bit default
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

typedef ap_fixed<{outW}, {outI}, AP_RND, AP_SAT> {name}_result_t;

void {name}_core(hls::stream<ap_uint<{A_TOTAL}> >&, hls::stream<ap_uint<{BB}> >&,
{pad}hls::stream<ap_uint<{PB_TOTAL}> >&);

static void feed_a(hls::stream<ap_uint<{A_TOTAL}> >& in, hls::stream<ap_uint<{A_TOTAL}> >& out) {{
    for (int i = 0; i < {m * SFT}; i++) out.write(in.read());
}}
static void feed_b(hls::stream<ap_uint<{BB}> >& in, hls::stream<ap_uint<{BB}> >& out) {{
    for (int i = 0; i < {K}; i++) out.write(in.read());
}}

// Grid requant drain (no bias): each vector emits NF={NF} partial beats; sum the gk K-partials
// per column (accum widened to {ACCU_SUM} = ACCU + ceil(log2 gk)) and concatenate the nt
// N-slices (tile (j,i) at [(j*gk+i)*PB]), then quantize.
static void requant(hls::stream<ap_uint<{PB_TOTAL}> >& in, hls::stream<ap_uint<{CB}> >& out) {{
    for (int vec = 0; vec < {m}; vec++) {{
        ap_uint<{CB}> crow = 0;
        for (int nf = 0; nf < {NF}; nf++) {{
            ap_uint<{PB_TOTAL}> ob = in.read();
            for (int j = 0; j < {nt}; j++)
            for (int pe = 0; pe < {PE}; pe++) {{
                int oc = j * {NTILE} + nf * {PE} + pe;
                ap_int<{ACCU_SUM}> raw = 0;
                for (int i = 0; i < {gk}; i++)
                    raw += (ap_int<{ACCU}>)ob.range((j * {gk} + i) * {PB} + pe * {ACCU} + {ACCU} - 1, (j * {gk} + i) * {PB} + pe * {ACCU});
{_rv_decl_assign(_rv_width(ACCU_SUM, False), pfrac, "raw", ' ' * 16)}
                {name}_result_t r = rv;
                crow.range(oc * {outW} + {outW} - 1, oc * {outW}) = r.range({outW} - 1, 0);
            }}
        }}
        out.write(crow);
    }}
}}

void {name}(hls::stream<ap_uint<{A_TOTAL}> >& a_in, hls::stream<ap_uint<{BB}> >& b_in,
{indent}hls::stream<ap_uint<{CB}> >& c_out) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{A_TOTAL}> > a_s;
    hls::stream<ap_uint<{BB}> > b_s;
    hls::stream<ap_uint<{PB_TOTAL}> > p_s;
#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=b_s depth=4
#pragma HLS STREAM variable=p_s depth=4
    feed_a(a_in, a_s);
    feed_b(b_in, b_s);
    {name}_core(a_s, b_s, p_s);   // <-- FINN MVU K-tiled blackbox (gk partials, B in-core)
    requant(p_s, c_out);
}}
"""


def _2op_gemm_ip_header(name, plan, bias_in_core=False):
    """hls4ml-facing two-operand IP: ``<name>_gemm_stream<data0_T,data1_T,res_T,CONFIG_T>``
    (repack A -> shim activations, repack B -> shim N-wide K-row beats, internal MVU blackbox,
    requant drain). Requires B streamed **row-major** (data1_T::size == N, one K-row per beat);
    hls4ml must set SecondOperandRowMajor=True to route here. Handles the three shim forms:
    kt (K-tiled register / fully-spatial, wide activation, summed partials), nt (N-tiled memstream,
    broadcast activation, concatenated), single (temporal memstream)."""
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AB, PB, ACCU = t["input_stream_width_ba"], t["output_stream_width_ba"], t["accu_width"]
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    AW, WW = t["activation_width"], t["weight_width"]
    N, K, KPAD = plan["n"], plan["k"], plan["k_pad"]
    BB = ((N * WW) + 7) // 8 * 8
    NT_, NTILE = plan["n_tiles"], plan["n_tile"]
    KT_ = plan.get("k_tiles", 1)
    pfrac = plan["product_frac"]
    nt_form = NT_ > 1
    kt_form = (not nt_form) and (KT_ > 1 or (SF == 1 and NF == 1))
    gk = KT_ if kt_form else 1
    if kt_form:
        a_width, p_width = gk * AB, gk * PB
    elif nt_form:
        a_width, p_width = AB, NT_ * PB
    else:
        a_width, p_width = AB, PB
    apmax = _apmaxw(p_width)
    guard = (f"#ifndef AP_INT_MAX_W\n#define AP_INT_MAX_W {apmax}\n#endif\n"
             if p_width > 1024 else "")
    core_pad = ' ' * (len(name) + 6)

    # repack A: kt packs one wide K_pad beat/vector; single/nt pack SF SIMD-beats/vector.
    if kt_form:
        repack_a = f"""template <class data0_T>
void {name}_repack_a(hls::stream<data0_T> &a_stream, hls::stream<ap_uint<{a_width}> > &a_s) {{
    for (unsigned mm = 0; mm < {m}; mm++) {{
        ap_int<{AW}> arow[{KPAD}];
        #pragma HLS ARRAY_PARTITION variable=arow complete
        for (unsigned i = 0; i < {KPAD}; i++) {{
            #pragma HLS UNROLL
            arow[i] = 0;
        }}
        for (unsigned kp = 0; kp < {K} / data0_T::size; kp++) {{
            data0_T beat = a_stream.read();
            for (unsigned j = 0; j < data0_T::size; j++) arow[kp * data0_T::size + j] = beat[j].range({AW} - 1, 0);
        }}
        ap_uint<{a_width}> ab = 0;
        for (unsigned kk = 0; kk < {KPAD}; kk++)
            ab.range(kk * {AW} + {AW} - 1, kk * {AW}) = (ap_uint<{AW}>)arow[kk];
        a_s.write(ab);
    }}
}}"""
    else:
        repack_a = f"""template <class data0_T>
void {name}_repack_a(hls::stream<data0_T> &a_stream, hls::stream<ap_uint<{AB}> > &a_s) {{
    for (unsigned mm = 0; mm < {m}; mm++) {{
        ap_int<{AW}> arow[{KPAD}];
        #pragma HLS ARRAY_PARTITION variable=arow complete
        for (unsigned i = 0; i < {KPAD}; i++) {{
            #pragma HLS UNROLL
            arow[i] = 0;
        }}
        for (unsigned kp = 0; kp < {K} / data0_T::size; kp++) {{
            data0_T beat = a_stream.read();
            for (unsigned j = 0; j < data0_T::size; j++) arow[kp * data0_T::size + j] = beat[j].range({AW} - 1, 0);
        }}
        for (unsigned sf = 0; sf < {SF}; sf++) {{
            ap_uint<{AB}> ab = 0;
            for (unsigned s = 0; s < {SIMD}; s++)
                ab.range(s * {AW} + {AW} - 1, s * {AW}) = (ap_uint<{AW}>)arow[sf * {SIMD} + s];
            a_s.write(ab);
        }}
    }}
}}"""

    # requant drain: kt sums gk partials; nt concatenates; single walks NF beats.
    if kt_form:
        ACCU_SUM = plan["accu_sum"]
        drain_body = f"""        ap_uint<{p_width}> ob = p_s.read();
        for (unsigned oc = 0; oc < {N}; oc++) {{
            ap_int<{ACCU_SUM}> raw = 0;
            for (unsigned ti = 0; ti < {gk}; ti++)
                raw += (ap_int<{ACCU}>)ob.range(ti * {PB} + oc * {ACCU} + {ACCU} - 1, ti * {PB} + oc * {ACCU});
{_rv_decl_assign(_rv_width(ACCU_SUM, bias_in_core), pfrac, "raw", ' ' * 12)}
{('            rv += biases[oc];' if bias_in_core else '')}
            crow[oc] = (result_t)rv;
        }}"""
    elif nt_form:
        drain_body = f"""        ap_uint<{p_width}> ob = p_s.read();
        for (unsigned ti = 0; ti < {NT_}; ti++)
            for (unsigned pe = 0; pe < {PE}; pe++) {{
                unsigned oc = ti * {NTILE} + pe;
                ap_int<{ACCU}> raw = ob.range(ti * {PB} + pe * {ACCU} + {ACCU} - 1, ti * {PB} + pe * {ACCU});
{_rv_decl_assign(_rv_width(ACCU, bias_in_core), pfrac, "raw", ' ' * 16)}
{('                rv += biases[oc];' if bias_in_core else '')}
                crow[oc] = (result_t)rv;
            }}"""
    else:
        drain_body = f"""        for (unsigned nf = 0; nf < {NF}; nf++) {{
            ap_uint<{p_width}> ob = p_s.read();
            for (unsigned pe = 0; pe < {PE}; pe++) {{
                unsigned oc = nf * {PE} + pe;
                ap_int<{ACCU}> raw = ob.range(pe * {ACCU} + {ACCU} - 1, pe * {ACCU});
{_rv_decl_assign(_rv_width(ACCU, bias_in_core), pfrac, "raw", ' ' * 16)}
{('                rv += biases[oc];' if bias_in_core else '')}
                crow[oc] = (result_t)rv;
            }}
        }}"""

    return f"""#ifndef {name.upper()}_GEMM_IP_H_
#define {name.upper()}_GEMM_IP_H_
{guard}#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// internal FINN-MVU blackbox (two-operand shim {name}_core.v; C twin {name}_core.cpp)
void {name}_core(hls::stream<ap_uint<{a_width}> >&, hls::stream<ap_uint<{BB}> >&,
{core_pad}hls::stream<ap_uint<{p_width}> >&);

namespace nnet {{

// Dedicated two-operand IP for gemm config M={m} K={K} N={N} (core={t['compute_core']}).
// B MUST arrive row-major: data1_T::size == N, one K-row per beat (SecondOperandRowMajor).

{repack_a}

// repack B: {K} row-major K-row beats (N-wide) -> shim beats, zero-padded K -> {KPAD}.
template <class data1_T>
void {name}_repack_b(hls::stream<data1_T> &b_stream, hls::stream<ap_uint<{BB}> > &b_s) {{
    for (unsigned k = 0; k < {K}; k++) {{
        data1_T beat = b_stream.read();
        ap_uint<{BB}> bb = 0;
        for (unsigned n = 0; n < {N}; n++) bb.range(n * {WW} + {WW} - 1, n * {WW}) = beat[n].range({WW} - 1, 0);
        b_s.write(bb);
    }}
    for (unsigned k = {K}; k < {KPAD}; k++) b_s.write(0);   // zero-pad K -> K_pad (bit-exact)
}}

// requant drain (no per-column bias for two-operand; biases[] are zero from hls4ml): raw ACCU
// -> fixed (frac={pfrac}) -> + bias -> convert per res_T's rounding/overflow modes.
template <class res_T, typename CONFIG_T>
void {name}_drain(hls::stream<ap_uint<{p_width}> > &p_s, hls::stream<res_T> &res_stream,
{' ' * (len(name) + 7)}typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
    typedef typename res_T::value_type result_t;
    for (unsigned mm = 0; mm < {m}; mm++) {{
        res_T crow;
{drain_body}
        res_stream.write(crow);
    }}
}}

// The dedicated IP: hls4ml io_stream two-operand GEMM -> internal MVU blackbox.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void {name}_gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
{' ' * (len(name) + 17)}hls::stream<res_T> &res_stream,
{' ' * (len(name) + 17)}typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{a_width}> > a_s;
    hls::stream<ap_uint<{BB}> > b_s;
    hls::stream<ap_uint<{p_width}> > p_s;
#pragma HLS STREAM variable=a_s depth={SF + 2}
#pragma HLS STREAM variable=b_s depth={KPAD + 2}
#pragma HLS STREAM variable=p_s depth={NF + 2}
    {name}_repack_a<data0_T>(a_stream, a_s);
    {name}_repack_b<data1_T>(b_stream, b_s);
    {name}_core(a_s, b_s, p_s);
    {name}_drain<res_T, CONFIG_T>(p_s, res_stream, biases);
}}

}} // namespace nnet
#endif
"""


def _2op_blackbox_json(name, t, n, ww, tiles=1, resources=None):
    """Blackbox JSON for the two-operand core: two input FIFOs (a, b) + one output (p)."""
    AB, PB = t["input_stream_width_ba"], t["output_stream_width_ba"]
    fn = f"{name}_core"
    # Cost-model resource estimate over all tiles (see _blackbox_json).
    res = resources or {"dsp": t["dsp_estimate"] * int(tiles), "bram18": 0}
    dsp, bram = int(res["dsp"]), int(res.get("bram18", 0))
    a_param = {"c_name": "a", "c_port_direction": "in",
               "rtl_ports": {"FIFO_data_read_in": "a_dout", "FIFO_read_enable": "a_read", "FIFO_empty_flag": "a_empty_n"}}
    b_param = {"c_name": "b", "c_port_direction": "in",
               "rtl_ports": {"FIFO_data_read_in": "b_dout", "FIFO_read_enable": "b_read", "FIFO_empty_flag": "b_empty_n"}}
    p_param = {"c_name": "p", "c_port_direction": "out",
               "rtl_ports": {"FIFO_data_write_out": "p_din", "FIFO_write_enable": "p_write", "FIFO_full_flag": "p_full_n"}}
    return json.dumps({
        "c_function_name": fn,
        "rtl_top_module_name": fn,
        "c_files": [{"c_file": f"{fn}.cpp", "cflag": ""}],
        "rtl_files": _core_rtl_files(name),
        "c_parameters": [a_param, b_param, p_param],
        "rtl_common_signal": {
            "module_clock": "ap_clk",
            "module_reset": "ap_rst",
            "module_clock_enable": "ap_ce",
            "ap_ctrl_chain_protocol_idle": "ap_idle",
            "ap_ctrl_chain_protocol_start": "ap_start",
            "ap_ctrl_chain_protocol_ready": "ap_ready",
            "ap_ctrl_chain_protocol_done": "ap_done",
            "ap_ctrl_chain_protocol_continue": "ap_continue",
        },
        "rtl_performance": {"latency": str(t["latency_cycles"]), "II": str(t["ii"])},
        "_comment": "FINN cost-model DSP estimate scaled by the tile count; FF/LUT/BRAM "
                    "are rough hints, not synthesis-accurate (TODO: measure).",
        "rtl_resource_usage": {"FF": str(30 * dsp), "LUT": str(40 * dsp),
                               "DSP": str(dsp), "BRAM": str(bram), "URAM": "0"},
    }, indent=2) + "\n"


def generate_two_operand_pkg(shape, name, output_dir, **cfg):
    """Emit a two-operand (``gemm_stream``) mvau blackbox package into ``<output_dir>/<name>/``.

    Both operands are runtime streams: A activations, B (= the MVU weight matrix) fed as
    N-wide K-row beats and loaded into ``memstream`` at runtime, replayed across the M rows
    of A. No baked weights; no bias. MVP: single tile, NF=1 (PE=N), any SF."""
    if cfg.get("interface") == "array":
        raise ValueError(f"mvau two-operand does not support io_parallel for '{name}'.")
    # The mvau two-operand IP consumes B row-major (N-wide beats, one contraction row per
    # beat). A manifest that explicitly routes a col-major two-operand node here is a
    # misconfiguration -- hls4ml must set SecondOperandRowMajor=True. (Absent == standalone
    # generation, which is row-major by construction, so only reject an explicit False.)
    if cfg.get("second_operand_row_major") is False:
        raise ValueError(
            f"mvau two-operand IP '{name}' requires SecondOperandRowMajor=True (B row-major, "
            "N-wide beats); the manifest declares col-major B. Set SecondOperandRowMajor on the "
            "hls4ml layer, or route this node to the generic/soft target.")
    plan = _resolve_plan(shape, cfg)
    t = plan["tile"]
    kt = plan.get("k_tiles", 1)
    nt = plan["n_tiles"]
    if t["compute_core"] == "mvu_4sx4u_dsp48e1":
        raise ValueError("two-operand cannot use mvu_4sx4u/DSP48E1 (requires NARROW_WEIGHTS, "
                         "which a runtime B operand cannot guarantee); target DSP48E2/DSP58.")
    # Shim selection:
    #   grid shim (nt>1 or gk>1): an nt×gk grid of MVU cores (memstream per tile, DEPTH>=2),
    #     activation K-sliced + broadcast over N-slices, partials summed over K + concatenated
    #     over N. Also serves the fully-spatial single tile (SF=NF=1) via the register path.
    #   memstream single-tile shim: the untiled temporal fold (DEPTH=NF*SF>=2).
    use_kt = (kt > 1) or (t["sf"] == 1 and t["nf"] == 1)
    part = cfg.get("part") or "xcvu13p-flga2577-2-e"
    clock_ns = cfg.get("clock_period_ns") or 5
    pkg = Path(output_dir) / name
    (pkg / "rtl_static").mkdir(parents=True, exist_ok=True)

    force_behavioral = bool(cfg.get("force_behavioral", False))
    if nt > 1:
        shim, top_src = (_rtl.generate_two_operand_nt_shim, _2op_kt_dataflow_top(name, plan))
    elif use_kt:
        shim, top_src = (_rtl.generate_two_operand_kt_shim, _2op_kt_dataflow_top(name, plan))
    else:
        shim, top_src = (_rtl.generate_two_operand_shim, _2op_dataflow_top(name, plan))
    (pkg / f"{name}_core.v").write_text(_with_timescale(
        shim(shape, module_name=f"{name}_core",
             force_behavioral=force_behavioral, tile=t, plan=plan)))
    (pkg / f"{name}_core.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_2op_core_twin(shape, func_name=f"{name}_core", plan=plan)))
    (pkg / f"{name}_top.cpp").write_text(_with_ap_int_max_w(top_src))
    (pkg / f"{name}.json").write_text(
        _2op_blackbox_json(name, t, plan["n"], t["weight_width"], tiles=kt * nt,
                           resources=plan.get("resources")))
    (pkg / f"{name}_tb.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_2op_tb(shape, top_name=name, func_name=f"{name}_core", plan=plan,
                                n_nodes=cfg.get("n_nodes", 6))))
    bias_in_core = bool(cfg.get("bias_in_core", False))
    (pkg / f"{name}_gemm_ip.h").write_text(_with_ap_int_max_w(
        _2op_gemm_ip_header(name, plan, bias_in_core=bias_in_core)))
    (pkg / "run_vitis.tcl").write_text(_run_vitis_tcl(name, part, clock_ns))

    for s in _STATIC_SOURCES:
        (pkg / "rtl_static" / s).write_text(_with_timescale((_RTL_STATIC / s).read_text()))
    shutil.copy(_RTL_STATIC / "FINN_COMMIT.txt", pkg / "rtl_static" / "FINN_COMMIT.txt")
    return str(pkg)


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
    klog = len(wm)
    if klog > k or (klog and len(wm[0]) != n):
        raise ValueError(f"weight_matrix shape ({klog},{len(wm[0]) if klog else '?'}) "
                         f"does not fit ({k}, {n})")
    # Zero-pad K -> k (K-padding so SIMD reaches its cap): padded rows are all-zero, and
    # the repack zero-fills the matching activation lanes, so the result is bit-exact.
    B = [[int(wm[kk][oo]) for oo in range(n)] for kk in range(klog)]
    B += [[0] * n for _ in range(k - klog)]
    return B


def hoist_shared_static(items, output_dir):
    """Multi-IP dedup: the vendored FINN static RTL (mvu_vvu_axi.sv, etc.) is byte-
    identical across IPs. Vitis rejects the SAME blackbox RTL file being contributed by
    more than one blackbox core ("multiple blackbox RTL file name ... reused") -- even at
    an identical path. So copy the static set once to <output_dir>/rtl_static_shared/ and
    let only the FIRST IP's JSON own it (repointed there); every other IP lists only its
    own unique <name>_core.v. All cores compile into one HLS project, so the static
    modules added by the first blackbox resolve for the rest. Per-IP rtl_static/ copies
    stay in place (standalone verify + the per-IP weights .dat)."""
    out = Path(output_dir)
    shared = out / "rtl_static_shared"
    shared.mkdir(parents=True, exist_ok=True)
    for s in _STATIC_SOURCES:
        (shared / s).write_text(_with_timescale((_RTL_STATIC / s).read_text()))
    shutil.copy(_RTL_STATIC / "FINN_COMMIT.txt", shared / "FINN_COMMIT.txt")
    for idx, it in enumerate(items):
        nm = it["name"]
        jf = out / nm / f"{nm}.json"
        if not jf.is_file():
            continue
        j = json.loads(jf.read_text())
        core = [f for f in j.get("rtl_files", []) if not f.startswith("rtl_static/")]
        statics = ([f"../rtl_static_shared/{s}" for s in _STATIC_SOURCES] if idx == 0 else [])
        j["rtl_files"] = core + statics   # only the first IP carries the shared statics
        jf.write_text(json.dumps(j, indent=2))


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
    plan = _resolve_plan(shape, cfg)
    t = plan["tile"]
    m = plan["num_input_vectors"]
    part = cfg.get("part") or "xcvu13p-flga2577-2-e"
    clock_ns = cfg.get("clock_period_ns") or 5
    # K_pad (>= logical K, a multiple of SIMD) is the K the RTL/memstream see; the weight
    # matrix is zero-padded to it and the repack zero-fills the extra activation lanes.
    N, K, PE, SIMD, WW = plan["n"], plan["k_pad"], t["pe"], t["simd"], t["weight_width"]
    NT, NTILE = plan["n_tiles"], plan["n_tile"]
    KT = plan.get("k_tiles", 1)

    pkg = Path(output_dir) / name
    (pkg / "rtl_static").mkdir(parents=True, exist_ok=True)

    # Pack the baked weights into the memstream init(s), alongside the vendored RTL.
    #   grid (K-tiling, possibly N-tiling): one memstream per (N-slice j, K-slice i) — the
    #     [MW_tile][NTILE] block; each reduces its K-slice for its N-slice -> a partial.
    #   N-tiling only: one memstream per N-column slice ([K][NTILE]).
    B = _weight_matrix_as_B(cfg, N, K, WW)
    MW = t["mw"]                       # per-tile K rows (= K_pad/KT = SF_tile*SIMD)
    init_files = []
    if KT > 1:
        for j in range(NT):
            for i in range(KT):
                B_ji = [[B[i * MW + kk][j * NTILE + oo] for oo in range(NTILE)]
                        for kk in range(MW)]
                dat_path = pkg / "rtl_static" / _kdat_name(name, j * KT + i)
                dat_path.write_text(_wpack.pack_memstream_hex(
                    B_ji, NTILE, MW, PE, SIMD, WW, word_bits=t["weight_stream_width_ba"]))
                init_files.append(str(dat_path.resolve()))
    else:
        for ti in range(NT):
            B_ti = [[B[kk][ti * NTILE + oo] for oo in range(NTILE)] for kk in range(K)]
            dat_path = pkg / "rtl_static" / _dat_name(name, ti, NT)
            dat_path.write_text(_wpack.pack_memstream_hex(
                B_ti, NTILE, K, PE, SIMD, WW, word_bits=t["weight_stream_width_ba"]))
            # Absolute $readmemh path: relative is unresolvable in Vitis cosim's XSIM dir
            # (empirically -- see jojo-track); the package builds in place, so the
            # absolute path computed here stays valid for csim/cosim/impl.
            init_files.append(str(dat_path.resolve()))

    # FORCE_BEHAVIORAL=0 -> real DSP48/DSP58 primitives (impl-ready; cosim runs them via
    # XSIM unisim models). Set force_behavioral=True in cfg for unisim-free behavioral cosim.
    force_behavioral = bool(cfg.get("force_behavioral", False))
    (pkg / f"{name}_core.v").write_text(_with_timescale(
        _rtl.generate_shim(shape, module_name=f"{name}_core",
                           force_behavioral=force_behavioral, tile=t,
                           weights_in_core=True, init_files=init_files,
                           n_tiles=NT, k_tiles=KT)))
    (pkg / f"{name}_core.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_core_twin(shape, func_name=f"{name}_core", plan=plan, baked_weights=B)))
    bias_codes = bias_acc_codes(cfg.get("bias"), plan["product_frac"], plan["n"])
    top_src = (_kt_dataflow_top(name, plan, bias_codes=bias_codes) if KT > 1
               else _dataflow_top(name, plan, bias_codes=bias_codes, weights_in_core=True))
    (pkg / f"{name}_top.cpp").write_text(_with_ap_int_max_w(top_src))
    (pkg / f"{name}.json").write_text(_blackbox_json(
        name, t, weights_in_core=True, tiles=KT * NT, resources=plan.get("resources")))
    (pkg / f"{name}_tb.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_tb(shape, top_name=name, func_name=f"{name}_core", plan=plan,
                            bias_codes=bias_codes, baked_weights=B,
                            n_nodes=cfg.get("n_nodes", 6))))
    bias_in_core = bool(cfg.get("bias_in_core", True))
    ip_hdr = (_kt_gemm_ip_header(name, plan, bias_in_core=bias_in_core) if KT > 1
              else _gemm_ip_header(name, plan, weights_in_core=True, bias_in_core=bias_in_core))
    (pkg / f"{name}_gemm_ip.h").write_text(_with_ap_int_max_w(ip_hdr))
    (pkg / "run_vitis.tcl").write_text(_run_vitis_tcl(name, part, clock_ns))

    for s in _STATIC_SOURCES:
        (pkg / "rtl_static" / s).write_text(_with_timescale((_RTL_STATIC / s).read_text()))
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

    # weights_in_core True (or missing) -> const_weights (baked-B) IP; False -> two-operand
    # (runtime-B) IP. A two-operand IP emits <name>_gemm_stream instead of
    # <name>_gemm_stream_const_weights, so it needs the two-operand dispatcher below.
    wl_items = [it for it in items if it.get("weights_in_core", True)]
    two_op_items = [it for it in items if not it.get("weights_in_core", True)]

    specs = []
    for it in wl_items:
        i, nm = _id(it), _nm(it)
        if i is None:
            continue
        specs.append(
            f"template <> struct mvau_ip<{i}> {{\n"
            f"    template <class data_T, class res_T, typename CONFIG_T>\n"
            f"    static void stream_const_weights(hls::stream<data_T> &a, hls::stream<res_T> &r,\n"
            f"                                  typename CONFIG_T::bias_t b[CONFIG_T::n_out]) {{\n"
            f"        #pragma HLS INLINE\n"
            f"        {nm}_gemm_stream_const_weights<data_T, res_T, CONFIG_T>(a, r, b);\n"
            f"    }}\n"
            f"}};")
    specs_s = "\n".join(specs)

    two_op_specs = []
    for it in two_op_items:
        i, nm = _id(it), _nm(it)
        if i is None:
            continue
        two_op_specs.append(
            f"template <> struct mvau_ip_stream<{i}> {{\n"
            f"    template <class data0_T, class data1_T, class res_T, typename CONFIG_T>\n"
            f"    static void stream(hls::stream<data0_T> &a, hls::stream<data1_T> &b,\n"
            f"                       hls::stream<res_T> &r,\n"
            f"                       typename CONFIG_T::bias_t bias[CONFIG_T::n_out]) {{\n"
            f"        #pragma HLS INLINE\n"
            f"        {nm}_gemm_stream<data0_T, data1_T, res_T, CONFIG_T>(a, b, r, bias);\n"
            f"    }}\n"
            f"}};")
    two_op_specs_s = "\n".join(two_op_specs)

    # The two-operand dispatcher + public gemm_stream entry are only emitted when a
    # two-operand IP exists (no soft two-operand primary; an unrouted id is a compile
    # error, the correct signal that a gemm_stream layer wasn't given an IP).
    two_op_block = "" if not two_op_specs else f"""
// id -> two-operand (runtime-B) IP dispatch.
template <int ID> struct mvau_ip_stream;
{two_op_specs_s}

// io_stream two-operand entry hls4ml calls; routes to the config's IP by id.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
                 hls::stream<res_T> &res_stream,
                 typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
    #pragma HLS INLINE
    mvau_ip_stream<CONFIG_T::gemm_ip_id>::template stream<data0_T, data1_T, res_T, CONFIG_T>(
        a_stream, b_stream, res_stream, biases);
}}
"""

    return f"""#ifndef GEMM_IP_COMBINED_H_
#define GEMM_IP_COMBINED_H_
#include <hls_stream.h>
{incs}

namespace nnet {{

// id -> dedicated IP dispatch (specialized per gemm config below).
template <int ID> struct mvau_ip;
{specs_s}

// io_stream const_weights entry hls4ml calls; routes to the config's IP by id.
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(hls::stream<data_T> &a_stream, hls::stream<res_T> &res_stream,
                            typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {{
    #pragma HLS INLINE
    mvau_ip<CONFIG_T::gemm_ip_id>::template stream_const_weights<data_T, res_T, CONFIG_T>(
        a_stream, res_stream, biases);
}}
{two_op_block}
}} // namespace nnet
#endif // GEMM_IP_COMBINED_H_
"""


def _resolve_manifest_plan(it):
    """Resolve the MVU plan for a manifest item, or ``None`` if it can't be resolved
    (missing shape / mapper error). Re-resolves the plan (cheap) so the manifest carries
    the same estimate/fold the emitted blackbox JSON does; shared by the resource,
    reuse-factor and warning reporting below so each item is only resolved once."""
    try:
        m, k, n = int(it["m"]), int(it["k"]), int(it["n"])
    except (KeyError, TypeError, ValueError):
        return None
    cfg = {key: it.get(key) for key in
           ("input_precision", "weight_precision", "output_precision", "part",
            "clock_period_ns", "reuse_factor", "strategy", "target_cycles",
            "parallelization_factor", "n_tiles", "k_tiles", "weights",
            "weights_in_core", "name")}
    try:
        return _resolve_plan((m, k, n), cfg)
    except Exception:
        return None


def _reuse_factor_warning(name, plan):
    """hls4ml's ReuseFactor-snap warning text (see run_atlas_flow.py's
    ``_snap_reuse_factor`` for the generic target), extended with what mvau actually
    achieved for the snapped request."""
    tile = plan["tile"]
    return (f'WARNING: Invalid ReuseFactor={plan["requested_reuse_factor"]} in layer "{name}".'
            f' mvau achieved ReuseFactor={plan["achieved_reuse_factor"]} '
            f'(PE={tile["pe"]}, SIMD={tile["simd"]}, per-vector II={tile["nf"] * tile["sf"]}).')


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
        # weight-stationary: the packed memstream init(s) are data dependencies the
        # shim $readmemh's by absolute path -- record them so downstream keeps them
        # with the package (they cannot go in rtl_files: Vitis rejects a .dat as
        # blackbox RTL). One file per N-tile column slice.
        nt = int(it.get("n_tiles", 1) or 1)
        try:
            kt = int(it.get("k_tiles", 1) or 1)   # resolved tile count; "auto" -> ignore here
        except (TypeError, ValueError):
            kt = 1
        if kt > 1:
            # grid: one .dat per (N-slice j, K-slice i), flat index j*kt+i
            core["weight_data"] = [f"{nm}/rtl_static/{_kdat_name(nm, j * kt + i)}"
                                   for j in range(nt) for i in range(kt)]
        else:
            core["weight_data"] = [f"{nm}/rtl_static/{_dat_name(nm, ti, nt)}" for ti in range(nt)]
        # Resolve the plan once for this item: cost-model resource estimate plus the
        # ReuseFactor reporting (requested/achieved/snapped, PE/SIMD, per-vector II).
        # Best-effort: skip these fields if the node has no resolvable shape.
        plan = _resolve_manifest_plan(it)
        if plan is not None:
            if plan.get("resources") is not None:
                core["resources"] = plan["resources"]
            tile = plan["tile"]
            core["reuse_factor_requested"] = plan["requested_reuse_factor"]
            core["reuse_factor_achieved"] = plan["achieved_reuse_factor"]
            core["reuse_factor_snapped"] = bool(plan["reuse_factor_snapped"])
            core["pe"] = tile["pe"]
            core["simd"] = tile["simd"]
            core["ii_per_vector"] = tile["nf"] * tile["sf"]
            if "n_tiles" in plan:
                core["n_tiles"] = plan["n_tiles"]
            if "k_tiles" in plan:
                core["k_tiles"] = plan["k_tiles"]
            if plan["reuse_factor_snapped"]:
                print(_reuse_factor_warning(nm, plan))
        cores.append(core)
    return json.dumps({"tool": "vitis", "flow": "rtl_blackbox",
                       "header": "gemm_ip_combined.h", "cores": cores}, indent=2) + "\n"


def run_vitis_smoke(package=None, cases=None, keep=False):
    """Run Vitis csim + csynth + cosim on generated mvau package(s) via ``vitis-run``.

    Returns a process-style code: 0 = all pass, 1 = a failure, 2 = ``vitis-run`` not found
    (skipped). With *package* given, run that package dir. Otherwise build the built-in smoke
    set (*cases*, or a default covering the single-tile, K-tiled and N-tiled paths) into a
    temp dir under ``temp_space/`` and run each. A package passes iff its self-checking TB
    reports ``MVAU_PKG PASS`` and cosim finishes ``PASS``."""
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path as _Path

    if shutil.which("vitis-run") is None:
        print("vitis-run not found on PATH (source the Vitis settings); skipping mvau rtl_test",
              file=sys.stderr)
        return 2

    def _run_one(pkg):
        pkg = _Path(pkg)
        r = subprocess.run(["vitis-run", "--tcl", "run_vitis.tcl", "--mode", "hls"],
                           cwd=str(pkg), text=True, capture_output=True)
        log = (r.stdout or "") + (r.stderr or "")
        ok = ("MVAU_PKG PASS" in log and "MVAU_PKG FAIL" not in log
              and "co-simulation finished: PASS" in log and r.returncode == 0)
        if not ok:
            print(f"[mvau rtl_test] {pkg.name}: FAIL\n{log[-3000:]}", file=sys.stderr)
        return ok

    if package is not None:
        return 0 if _run_one(package) else 1

    # Built-in smoke set: covers single-tile temporal, K-tiled (baked + two-operand,
    # NF>1 and SF_tile>1) and N-tiled paths. Small shapes keep cosim quick.
    default = [
        {"shape": (4, 4, 4), "reuse_factor": 1, "weights_in_core": True},    # single tile
        {"shape": (4, 6, 8), "reuse_factor": 4, "weights_in_core": False},   # 2op K-tile, NF>1
        {"shape": (2, 12, 4), "reuse_factor": 8, "weights_in_core": True},   # baked K-tile, SF_tile>1
        {"shape": (2, 3, 8), "n_tiles": 2, "weights_in_core": True},         # N-tile (K fits one core)
        {"shape": (4, 6, 8), "reuse_factor": 2, "n_tiles": 2,
         "weights_in_core": False},                                          # two-operand combined N+K grid
        {"shape": (2, 6, 8), "reuse_factor": 2, "n_tiles": 2,
         "weights_in_core": True},                                           # baked combined N+K grid
    ]
    smoke = cases or default
    root = _Path(__file__).resolve().parents[3] / "temp_space" / "mvau_rtl_smoke"
    root.mkdir(parents=True, exist_ok=True)   # repo-local scratch only (never /tmp)
    outdir = _Path(tempfile.mkdtemp(dir=str(root)))
    rc = 0
    try:
        for i, c in enumerate(smoke):
            shp = c["shape"]
            nm = f"smoke_{i}"
            opts = {"weight_precision": "fixed<8,4>", "input_precision": "fixed<8,4>",
                    "output_precision": "fixed<16,6>", "part": "xcve2802-vsvh1760-2MP-e-S",
                    "clock_period_ns": 5,
                    **{k: v for k, v in c.items() if k != "shape"}}
            gen = (generate_two_operand_pkg if not opts.get("weights_in_core", True)
                   else generate_mvau_pkg)
            pkg = gen(shp, nm, str(outdir), **opts)
            if not _run_one(pkg):
                rc = 1
    finally:
        if not keep:
            shutil.rmtree(outdir, ignore_errors=True)
    return rc
