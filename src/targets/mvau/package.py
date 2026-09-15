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

from . import geometry as _geom
from . import rtl as _rtl
from . import golden as _golden
from . import weightpack as _wpack

_RTL_STATIC = Path(__file__).resolve().parent / "rtl_static"
# memstream is the weight-stationary weight ROM (baked from <name>_weights.dat);
# always vendored -- the const_weights path instantiates it, the two-operand shims
# (see _2op_* emitters) leave it uninstantiated.
_STATIC_SOURCES = ["mvu_vvu_axi.sv", "replay_buffer.sv", "memstream.sv",
                   "mvu_pkg.sv", "mvu.sv", "add_multi.sv", "mvu_vvu_8sx9_dsp58.sv",
                   "dynamic_load_2op.sv"]
_WEIGHTS_DAT = "{name}_weights.dat"   # per-IP memstream $readmemh init, in rtl_static/
# XSIM requires all-or-none `timescale across the design. The vendored FINN cores and the
# generated shim carry none (fine standalone), but the hls4ml RTL they integrate with does
# -> cosim elaboration fails. Stamp a matching timescale onto every mvau RTL file.
_TIMESCALE = "`timescale 1ns / 1ps\n"



def _glue_pipeline_fn(m):
    """Function-level PIPELINE for the repack/drain glue when a node is one row
    (M == 1): Vitis removes the single-trip row loop, so a loop-level pragma is
    dropped and the process stays an unpipelined ap_ctrl_chain leaf that costs a
    start/done handshake per frame. Pipelining the whole function makes it a
    flushable pipeline (II 1 across frames)."""
    return "    #pragma HLS PIPELINE II=1\n" if int(m) == 1 else ""



def _drain_pipeline_fn(m, t):
    """Function-level PIPELINE for the drain only when the node is one row AND
    the core emits one output beat per cycle (SF*NF == 1). Measured on the fc
    set: at RF 1 the unpipelined drain's per-frame handshake caps the interval
    at 2 (fc_tiny 1 -> 2 without it), but at RF >= 2 the interval is already
    the core's RF and a function-pipelined (flushable) drain instead adds SF-1
    cycles to every frame's latency (fc_large 18 -> 25). Loop-level pipelining
    for M > 1 nodes is unaffected."""
    if int(m) != 1:
        return ""
    return "    #pragma HLS PIPELINE II=1\n" if int(t["sf"]) * int(t["nf"]) == 1 else ""


def _glue_pipeline_loop(m):
    """Loop-level PIPELINE for the row loop when a node is M > 1 rows: one row
    per cycle inside the frame; the per-frame handshake is amortised over M."""
    return "" if int(m) == 1 else "        #pragma HLS PIPELINE II=1\n"


def _with_timescale(text):
    return text if "`timescale" in text else _TIMESCALE + text


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
              "clock_period_ns", "reuse_factor", "fold_axis", "n_tiles", "k_tiles",
              "weights")


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


def _dataflow_top(name, plan):
    """The DUT: a straight passthrough into the blackbox, in a dataflow region.
    One external activation row in, one external result row out -- per vector.

    All K-padding (zero-fill to k_pad, the SF-way SIMD fan-out) and N-padding
    (N-tile stitching, dropping the pad tail columns) now happen INSIDE
    ``{name}_core``'s RTL wrapper (and its C twin, ``golden.py``): the boundary
    this top sees is already ``K*activation_width`` in / ``N*out_width`` out,
    matching hls4ml's own unpadded TDATA widths exactly. So this top does no
    repacking of its own -- it is only the dataflow region Vitis needs to drop
    the RTL blackbox into.

    Weight-stationary: the blackbox bakes its weights (and bias) in the memstream,
    so the top has no weight input -- only activations flow in. Two-operand IPs
    have their own top emitter (``_2op_dataflow_top``).
    """
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AW, N, outW = t["activation_width"], plan["n"], plan["output_width"]
    K = plan["k"]
    AB = K * AW      # raw, unpadded activation beat (one hls4ml row)
    PB = N * outW    # raw, unpadded result beat (one hls4ml row)
    pad = ' ' * (len(name) + 6)
    core_decl = f"void {name}_core(hls::stream<ap_uint<{AB}> >&,\n{pad}hls::stream<ap_uint<{PB}> >&);"
    indent = ' ' * (len(name) + 1)
    top_sig = f"void {name}(hls::stream<ap_uint<{AB}> >& a_in,\n{indent}hls::stream<ap_uint<{PB}> >& c_out)"
    return f"""#include <hls_stream.h>
#include <ap_int.h>

{core_decl}

// Pure passthrough: {name}_core's RTL wrapper does all K/N padding and the
// row<->beat fan-out/stitching internally, so the boundary here is already
// hls4ml's own unpadded row width on both sides -- {m} rows in, {m} rows out.
{top_sig} {{
    {name}_core(a_in, c_out);   // <-- FINN MVU RTL blackbox (weights + bias baked; already requantized)
}}
"""


def _blackbox_json(name, t, tiles=1, resources=None):
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
    c_params = [a_param, p_param]
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


def _gemm_ip_header(name, plan):
    """The dedicated hls4ml-facing IP for one gemm config: an HLS C++ dataflow IP
    with the internal FINN-MVU blackbox. The blackbox's RTL wrapper now does ALL
    K/N padding and row<->beat fan-out/stitching internally (see ``rtl.py``'s
    ``_generate_ws_shim`` and its C twin in ``golden.py``), so its port widths are
    already hls4ml's own unpadded ``data_T``/``res_T`` row widths
    (``K*activation_width`` in, ``N*out_width`` out). The glue here is therefore a
    PURE bit-reinterpretation -- one fully-unrolled concatenation per row, zero
    additional pipeline cycles -- not a lane-by-lane repack/drain loop. The
    combined header routes nnet::gemm_* to this by CONFIG_T::gemm_ip_id (template
    dispatch).

    Weight-stationary only (weights + bias baked in the RTL memstream / bias ROM);
    two-operand IPs use ``_2op_gemm_ip_header``.
    """
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AW, K, N = t["activation_width"], plan["k"], plan["n"]
    outW = plan["output_width"]
    AB = K * AW      # raw, unpadded activation beat -- must equal data_T::size*AW
    PB = N * outW    # raw, unpadded result beat -- must equal res_T::size*out_width
    core_hdr_pad = ' ' * (len(name) + 6)
    core_decl = f"void {name}_core(hls::stream<ap_uint<{AB}> >&,\n{core_hdr_pad}hls::stream<ap_uint<{PB}> >&);"
    ws_streams = (f"    hls::stream<ap_uint<{AB}> > a_s;\n"
                  f"    hls::stream<ap_uint<{PB}> > p_s;\n"
                  f"#pragma HLS STREAM variable=a_s depth=4\n"
                  f"#pragma HLS STREAM variable=p_s depth=4")
    ws_calls = (f"    {name}_repack_a<data_T>(a_stream, a_s);\n"
                f"    {name}_core(a_s, p_s);\n"
                f"    {name}_drain<res_T, CONFIG_T>(p_s, res_stream);")
    return f"""#ifndef {name.upper()}_GEMM_IP_H_
#define {name.upper()}_GEMM_IP_H_
#include <hls_stream.h>
#include <ap_int.h>

// internal FINN-MVU blackbox (shim {name}_core.v; C twin {name}_core.cpp) -- ALL
// K/N padding, bias (if any), and the shift/round-half-up/wrap to out_width are
// baked/performed inside it (RTL wrapper + its C twin), so its ports already sit
// at hls4ml's own unpadded row widths.
{core_decl}

namespace nnet {{

// Dedicated IP for gemm config M={m} K={K} N={N} (core={t['compute_core']}).
// Baked geometry; templated on the hls4ml stream/config types so the combined
// header can route to it by id.

// Pure bit-reinterpretation: hls4ml delivers one full, unpadded K-wide row per
// beat (data_T::size == K by construction -- enforced below) -- reinterpret it as
// one K*AW-bit word via a single fully-unrolled concatenation. No lane-by-lane
// pipelined loop, no padding: {name}_core's RTL wrapper does the K-padding.
template <class data_T>
void {name}_repack_a(hls::stream<data_T> &a_stream, hls::stream<ap_uint<{AB}> > &a_s) {{
    static_assert(data_T::size == {K},
        "{name}: hls4ml must deliver one full, unpadded K-wide row per stream beat");
{_glue_pipeline_fn(m)}    for (unsigned mm = 0; mm < {m}; mm++) {{
{_glue_pipeline_loop(m)}        data_T beat = a_stream.read();
        ap_uint<{AB}> ab;
        for (unsigned j = 0; j < {K}; j++) {{
            #pragma HLS UNROLL
            ab.range(j * {AW} + {AW} - 1, j * {AW}) = beat[j].range({AW} - 1, 0);
        }}
        a_s.write(ab);
    }}
}}

// Pure bit-reinterpretation: {name}_core's RTL wrapper already emits one full,
// unpadded N-wide row of requantized out_width-bit-per-lane codes per beat (bias,
// shift, round-half-up, wrap, N-tile stitching and pad-column drop all happened
// inside it -- see golden.py's core twin and the RTL requant stage). Slice one
// lane per column via a single fully-unrolled loop; loaded via .range() (raw bit
// pattern), never a value-preserving conversion, since the value is already
// rounded/wrapped and re-converting would corrupt it.
template <class res_T, typename CONFIG_T>
void {name}_drain(hls::stream<ap_uint<{PB}> > &p_s, hls::stream<res_T> &res_stream) {{
    typedef typename res_T::value_type result_t;
{_drain_pipeline_fn(m, t)}    for (unsigned mm = 0; mm < {m}; mm++) {{
{_glue_pipeline_loop(m)}        ap_uint<{PB}> ob = p_s.read();
        res_T crow;
        for (unsigned oc = 0; oc < {N}; oc++) {{
            #pragma HLS UNROLL
            ap_uint<{outW}> raw = ob.range(oc * {outW} + {outW} - 1, oc * {outW});
            result_t tmp;
            tmp.range() = raw;     // load the already-requantized bit pattern as-is
            crow[oc] = tmp;
        }}
        res_stream.write(crow);
    }}
}}

// The dedicated IP: hls4ml io_stream const_weights GEMM -> internal MVU blackbox.
// Bias (when present) is the baked constant above, never a function argument -- this
// is the only signature; hls4ml's call site never passes a bias parameter.
template <class data_T, class res_T, typename CONFIG_T>
void {name}_gemm_stream_const_weights(hls::stream<data_T> &a_stream, hls::stream<res_T> &res_stream) {{
#pragma HLS DATAFLOW
{ws_streams}
{ws_calls}
}}

}} // namespace nnet
#endif
"""


def _kt_dataflow_top(name, plan):
    """DUT for the K-tiled blackbox: a straight passthrough into the blackbox, in a
    dataflow region. One external activation row in, one external result row out --
    per vector.

    All K-padding (zero-fill to the grid's total k_pad, the per-tile SF_tile-way SIMD
    fan-out), the K-tile partial-sum, bias bake, and N-padding (N-tile stitching,
    dropping the pad tail columns) now happen INSIDE ``{name}_core``'s RTL wrapper
    (and its C twin, ``golden.py``): the boundary this top sees is already
    ``K*activation_width`` in / ``N*out_width`` out, matching hls4ml's own unpadded
    TDATA widths exactly -- the K-tiling counterpart of ``_dataflow_top``."""
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AW, N, outW = t["activation_width"], plan["n"], plan["output_width"]
    K = plan["k"]
    AB = K * AW      # raw, unpadded activation beat (one hls4ml row)
    PB = N * outW    # raw, unpadded result beat (one hls4ml row)
    pad = ' ' * (len(name) + 6)
    core_decl = f"void {name}_core(hls::stream<ap_uint<{AB}> >&,\n{pad}hls::stream<ap_uint<{PB}> >&);"
    indent = ' ' * (len(name) + 1)
    apmax = _apmaxw(max(AB, PB))
    top_sig = f"void {name}(hls::stream<ap_uint<{AB}> >& a_in,\n{indent}hls::stream<ap_uint<{PB}> >& c_out)"
    return f"""#define AP_INT_MAX_W {apmax}   // raw K-wide activation row may exceed the 1024-bit default
#include <hls_stream.h>
#include <ap_int.h>

{core_decl}

// Pure passthrough: {name}_core's RTL wrapper does all K/N padding, the K-tile
// partial-sum, and the row<->beat fan-out/stitching internally, so the boundary
// here is already hls4ml's own unpadded row width on both sides -- {m} rows in,
// {m} rows out.
{top_sig} {{
    {name}_core(a_in, c_out);   // <-- FINN MVU RTL blackbox (K-tiles summed, weights + bias baked; already requantized)
}}
"""


def _kt_gemm_ip_header(name, plan):
    """hls4ml-facing IP for a K-tiled gemm config. The blackbox's RTL wrapper now does
    ALL K/N padding, the K-tile partial-sum, and row<->beat fan-out/stitching internally
    (see ``rtl.py``'s ``_generate_kt_shim`` and its C twin in ``golden.py``), so its port
    widths are already hls4ml's own unpadded ``data_T``/``res_T`` row widths
    (``K*activation_width`` in, ``N*out_width`` out). The glue here is therefore a PURE
    bit-reinterpretation -- one fully-unrolled concatenation per row, zero additional
    pipeline cycles -- the K-tiling twin of ``_gemm_ip_header``."""
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AW, K, N = t["activation_width"], plan["k"], plan["n"]
    outW = plan["output_width"]
    AB = K * AW      # raw, unpadded activation beat -- must equal data_T::size*AW
    PB = N * outW    # raw, unpadded result beat -- must equal res_T::size*out_width
    core_hdr_pad = ' ' * (len(name) + 6)
    apmax = _apmaxw(max(AB, PB))
    KT = plan["k_tiles"]
    return f"""#ifndef {name.upper()}_GEMM_IP_H_
#define {name.upper()}_GEMM_IP_H_
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W {apmax}   // raw K-wide activation row may exceed the 1024-bit default
#endif
#include <hls_stream.h>
#include <ap_int.h>

// internal FINN-MVU blackbox (K-tiled grid shim {name}_core.v; C twin {name}_core.cpp) --
// ALL K/N padding, the {KT}-way K-tile partial-sum, bias (if any), and the
// shift/round-half-up/wrap to out_width are baked/performed inside it (RTL wrapper +
// its C twin), so its ports already sit at hls4ml's own unpadded row widths.
void {name}_core(hls::stream<ap_uint<{AB}> >&,
{core_hdr_pad}hls::stream<ap_uint<{PB}> >&);

namespace nnet {{

// Dedicated K-tiled IP for gemm config M={m} K={K} N={N} (core={t['compute_core']},
// KT={KT} K-tiles). Baked geometry; templated on the hls4ml stream/config types so
// the combined header can route to it by id.

// Pure bit-reinterpretation: hls4ml delivers one full, unpadded K-wide row per beat
// (data_T::size == K by construction -- enforced below) -- reinterpret it as one
// K*AW-bit word via a single fully-unrolled concatenation. No lane-by-lane pipelined
// loop, no padding: {name}_core's RTL wrapper does the K-padding and K-tile split.
template <class data_T>
void {name}_repack_a(hls::stream<data_T> &a_stream, hls::stream<ap_uint<{AB}> > &a_s) {{
    static_assert(data_T::size == {K},
        "{name}: hls4ml must deliver one full, unpadded K-wide row per stream beat");
{_glue_pipeline_fn(m)}    for (unsigned mm = 0; mm < {m}; mm++) {{
{_glue_pipeline_loop(m)}        data_T beat = a_stream.read();
        ap_uint<{AB}> ab;
        for (unsigned j = 0; j < {K}; j++) {{
            #pragma HLS UNROLL
            ab.range(j * {AW} + {AW} - 1, j * {AW}) = beat[j].range({AW} - 1, 0);
        }}
        a_s.write(ab);
    }}
}}

// Pure bit-reinterpretation: {name}_core's RTL wrapper already emits one full,
// unpadded N-wide row of requantized out_width-bit-per-lane codes per beat (the
// {KT} K-tile partials summed, bias, shift, round-half-up, wrap, N-tile stitching
// and pad-column drop all happened inside it -- see golden.py's core twin and the
// RTL requant stage). Slice one lane per column via a single fully-unrolled loop;
// loaded via .range() (raw bit pattern), never a value-preserving conversion,
// since the value is already rounded/wrapped and re-converting would corrupt it.
template <class res_T, typename CONFIG_T>
void {name}_drain(hls::stream<ap_uint<{PB}> > &p_s, hls::stream<res_T> &res_stream) {{
    typedef typename res_T::value_type result_t;
{_drain_pipeline_fn(m, t)}    for (unsigned mm = 0; mm < {m}; mm++) {{
{_glue_pipeline_loop(m)}        ap_uint<{PB}> ob = p_s.read();
        res_T crow;
        for (unsigned oc = 0; oc < {N}; oc++) {{
            #pragma HLS UNROLL
            ap_uint<{outW}> raw = ob.range(oc * {outW} + {outW} - 1, oc * {outW});
            result_t tmp;
            tmp.range() = raw;     // load the already-requantized bit pattern as-is
            crow[oc] = tmp;
        }}
        res_stream.write(crow);
    }}
}}

// The dedicated K-tiled IP: hls4ml io_stream const_weights GEMM -> internal MVU blackbox.
// Bias (when present) is the baked constant above -- there is only ever this one
// signature; hls4ml's call site never passes a bias parameter.
template <class data_T, class res_T, typename CONFIG_T>
void {name}_gemm_stream_const_weights(hls::stream<data_T> &a_stream, hls::stream<res_T> &res_stream) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{AB}> > a_s;
    hls::stream<ap_uint<{PB}> > p_s;
#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=p_s depth=4
    {name}_repack_a<data_T>(a_stream, a_s);
    {name}_core(a_s, p_s);
    {name}_drain<res_T, CONFIG_T>(p_s, res_stream);
}}

}} // namespace nnet
#endif
"""


def _2op_dataflow_top(name, plan):
    """DUT for the two-operand blackbox (single-tile temporal fold, DEPTH=NF*SF>=2):
    passthrough of A, the affine requant drain (no bias; act×act product scale
    ``fa+fb``), and a ``feed_b`` gearbox that reindexes hls4ml's wide B beat down to
    ``dynamic_load_2op``'s narrow input beat -- AT MOST a 1-wide-beat register, no
    reorder buffer (see jojo-track/defer/mvau-two-operand-dynamic-load/plan.md,
    "Input width gearbox").

    Mode A (``mode=0``, row-major B): one N-wide K-row arrives per beat (K beats
    total); feed_b holds it in a 1xN register and drains it PE at a time, NF
    sub-beats (nf=0..NF-1, nf-fast) -- matches the loader's Mode A writer
    (nf-fast/simd-mid/sf-slow: rows arrive in natural k=sf*SIMD+simd order).

    Mode B (``mode=1``, col-major B): one K-wide column arrives per beat (N beats
    total, natural column order c=nf*PE+pe, pe-fast/nf-slow); feed_b holds it in a
    1xK register and drains it SIMD at a time, SF sub-beats (sf=0..SF-1, sf-fast)
    -- matches the loader's Mode B (transposed) writer (sf-fast/pe-mid/nf-slow)."""
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AB, PB = t["input_stream_width_ba"], t["output_stream_width_ba"]
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    N, WW = plan["n"], t["weight_width"]
    K = plan["k_pad"]
    outW = plan["output_width"]
    CB = cbits(plan)
    abeats = m * SF
    pad = ' ' * (len(name) + 6)
    indent = ' ' * (len(name) + 1)
    mode = plan.get("mode", 0)

    LANES_RAW = PE if mode == 0 else SIMD
    BWn = ((LANES_RAW * WW + 7) // 8) * 8   # narrow beat into the core (module idat width)

    if mode == 0:
        # Mode A: wide beat = one N-wide K-row (K beats); drain NF PE-wide sub-beats
        # per row (nf-fast), lane pe = row bits [(nf*PE+pe)*WW +: WW].
        WROW, WROW_BEATS, SUBBEATS = N, K, NF
        feed_b_body = f"""static void feed_b(hls::stream<ap_uint<{((WROW * WW + 7) // 8) * 8}> >& in,
                    hls::stream<ap_uint<{BWn}> >& out) {{
    for (int k = 0; k < {WROW_BEATS}; k++) {{
        ap_uint<{((WROW * WW + 7) // 8) * 8}> row = in.read();   // 1xN register (one arriving wide beat)
        for (int nf = 0; nf < {SUBBEATS}; nf++) {{
#pragma HLS PIPELINE II=1
            ap_uint<{BWn}> nb = 0;
            for (int pe = 0; pe < {PE}; pe++) {{
#pragma HLS UNROLL
                nb.range(pe * {WW} + {WW} - 1, pe * {WW}) =
                    row.range((nf * {PE} + pe) * {WW} + {WW} - 1, (nf * {PE} + pe) * {WW});
            }}
            out.write(nb);
        }}
    }}
}}"""
    else:
        # Mode B: wide beat = one K-wide column (N beats, natural column order
        # c=nf*PE+pe pe-fast); drain SF SIMD-wide sub-beats per column (sf-fast),
        # lane s = column bits [(sf*SIMD+s)*WW +: WW].
        WROW, WROW_BEATS, SUBBEATS = K, N, SF
        feed_b_body = f"""static void feed_b(hls::stream<ap_uint<{((WROW * WW + 7) // 8) * 8}> >& in,
                    hls::stream<ap_uint<{BWn}> >& out) {{
    for (int c = 0; c < {WROW_BEATS}; c++) {{
        ap_uint<{((WROW * WW + 7) // 8) * 8}> col = in.read();   // 1xK register (one arriving wide beat)
        for (int sf = 0; sf < {SUBBEATS}; sf++) {{
#pragma HLS PIPELINE II=1
            ap_uint<{BWn}> nb = 0;
            for (int s = 0; s < {SIMD}; s++) {{
#pragma HLS UNROLL
                nb.range(s * {WW} + {WW} - 1, s * {WW}) =
                    col.range((sf * {SIMD} + s) * {WW} + {WW} - 1, (sf * {SIMD} + s) * {WW});
            }}
            out.write(nb);
        }}
    }}
}}"""
    BB_top = ((WROW * WW + 7) // 8) * 8   # top-level (hls4ml-facing) wide-beat width

    return f"""#include <hls_stream.h>
#include <ap_int.h>

void {name}_core(hls::stream<ap_uint<{AB}> >&, hls::stream<ap_uint<{BWn}> >&,
{pad}hls::stream<ap_uint<{PB}> >&);

static void feed_a(hls::stream<ap_uint<{AB}> >& in, hls::stream<ap_uint<{AB}> >& out) {{
    for (int i = 0; i < {abeats}; i++) out.write(in.read());
}}
{feed_b_body}

// Pure unpack (no bias -- two-operand GEMM never has one): {name}_core already
// shift/round-half-up/wrapped each lane to out_width. Beat nf lane pe holds output
// column nf*PE+pe. Walks one beat (one p_s.read()) per loop iteration -- rather
// than NF reads per row inside one iteration -- so a PIPELINE II=1 loop can
// actually schedule at II=1 per beat (II=NF per row, same throughput either way).
static void unpack(hls::stream<ap_uint<{PB}> >& in, hls::stream<ap_uint<{CB}> >& out) {{
    ap_uint<{CB}> crow = 0;
    for (int bt = 0; bt < {m * NF}; bt++) {{
        int nf = bt % {NF};
        ap_uint<{PB}> ob = in.read();
        for (int pe = 0; pe < {PE}; pe++) {{
            int oc = nf * {PE} + pe;
            if (oc < {N}) {{   // drop the N-pad tail columns (untiled: n_tile == n)
            crow.range(oc * {outW} + {outW} - 1, oc * {outW}) =
                ob.range(pe * {outW} + {outW} - 1, pe * {outW});
            }}
        }}
        if (nf == {NF} - 1) out.write(crow);
    }}
}}

void {name}(hls::stream<ap_uint<{AB}> >& a_in, hls::stream<ap_uint<{BB_top}> >& b_in,
{indent}hls::stream<ap_uint<{CB}> >& c_out) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{AB}> > a_s;
    hls::stream<ap_uint<{BWn}> > b_s;
    hls::stream<ap_uint<{PB}> > p_s;
#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=b_s depth=4
#pragma HLS STREAM variable=p_s depth=4
    feed_a(a_in, a_s);
    feed_b(b_in, b_s);
    {name}_core(a_s, b_s, p_s);   // <-- FINN MVU RTL blackbox (B loaded+replayed in-core; already requantized)
    unpack(p_s, c_out);
}}
"""


def _2op_gemm_ip_header(name, plan):
    """hls4ml-facing two-operand IP: ``<name>_gemm_stream<data0_T,data1_T,res_T,CONFIG_T>``
    (repack A -> shim activations, repack B -> the loader's narrow beat via the HLS
    feed_b gearbox, internal MVU blackbox, requant drain). Single-tile only -- 2-op
    only ever folds within one MVU tile (no N/K-tiling; see
    jojo-track/defer/mvau-two-operand-dynamic-load/plan.md's "Cleanup: collapse 2-op
    to a single dynamic_load_2op tile"), any depth including the fully-spatial
    DEPTH==1 case (SF=NF=1), both B-layout modes (``SecondOperandRowMajor``)."""
    t = plan["tile"]
    m = plan["num_input_vectors"]
    AB, PB = t["input_stream_width_ba"], t["output_stream_width_ba"]
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    AW, WW = t["activation_width"], t["weight_width"]
    N, K, KPAD = plan["n"], plan["k"], plan["k_pad"]
    mode = plan.get("mode", 0)
    # BB: the dynamic_load_2op loader's own NARROW input beat (core-side, module idat
    # width) -- PE-wide (Mode A) or SIMD-wide (Mode B). The hls4ml-facing wide beat
    # (data1_T, N-wide row / K-wide column) is reindexed down to this by repack_b below
    # (the HLS feed_b gearbox), never materializing more than one arriving wide beat.
    LANES_RAW = PE if mode == 0 else SIMD
    BB = ((LANES_RAW * WW) + 7) // 8 * 8
    outW = plan["output_width"]
    a_width, p_width = AB, PB
    apmax = _apmaxw(max(a_width, p_width))
    guard = (f"#ifndef AP_INT_MAX_W\n#define AP_INT_MAX_W {apmax}\n#endif\n"
             if max(a_width, p_width) > 1024 else "")
    core_pad = ' ' * (len(name) + 6)

    # repack A: pack SF SIMD-wide beats/vector (SF==1 -> one beat/vector).
    repack_a = f"""template <class data0_T>
void {name}_repack_a(hls::stream<data0_T> &a_stream, hls::stream<ap_uint<{AB}> > &a_s) {{
{_glue_pipeline_fn(m)}    for (unsigned mm = 0; mm < {m}; mm++) {{
{_glue_pipeline_loop(m)}        ap_int<{AW}> arow[{KPAD}];
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

    # Pure unpack (no arithmetic): {name}_core already shift/round-half-up/wrapped
    # each lane to out_width. Two-operand GEMM never carries a real bias (has_bias is
    # always False by construction). Loaded via .range() (raw bit pattern), never a
    # value-preserving conversion, since the value is already rounded/wrapped.
    #
    # NF>1 here means the drain must read NF beats per output row. Reading all NF
    # beats inside a single PIPELINE II=1 loop iteration is unschedulable at II=1
    # (Vitis emits HLS-200-880 and silently falls back to II=NF per row anyway --
    # same throughput, but the [verify] step flags the warning as a failure), so this
    # walks one beat per loop iteration (trip count m*NF) and only fires
    # res_stream.write on the last beat of each row -- true II=1 per beat, II=NF per
    # row (NF==1 -> one beat per row, same as before).
    drain_body = f"""        unsigned nf = bt % {NF};
        ap_uint<{p_width}> ob = p_s.read();
        for (unsigned pe = 0; pe < {PE}; pe++) {{
            unsigned local_oc = nf * {PE} + pe;
            if (local_oc < {N}) {{   // drop the N-pad tail columns
            unsigned oc = local_oc;
            ap_uint<{outW}> raw = ob.range(pe * {outW} + {outW} - 1, pe * {outW});
            result_t tmp; tmp.range() = raw; crow[oc] = tmp;
            }}
        }}
        if (nf == {NF} - 1) res_stream.write(crow);"""

    drain_fn = f"""template <class res_T, typename CONFIG_T>
void {name}_drain(hls::stream<ap_uint<{p_width}> > &p_s, hls::stream<res_T> &res_stream) {{
    typedef typename res_T::value_type result_t;
    res_T crow;
    for (unsigned bt = 0; bt < {m * NF}; bt++) {{
{drain_body}
    }}
}}"""

    # repack B: reindex hls4ml's wide beat down to the loader's narrow beat (the HLS
    # feed_b gearbox -- see rtl.py's dynamic_load_2op instantiation and
    # jojo-track/defer/mvau-two-operand-dynamic-load/plan.md, "Input width gearbox").
    # Materializes AT MOST one arriving wide beat (a 1xN or 1xK register), never a
    # reorder buffer. Padding (K -> KPAD rows for Mode A, N -> PE*NF columns for Mode
    # B) is a zero-filled pass with no stream read, matching the old zero-pad semantics
    # bit-exact.
    if mode == 0:
        # Mode A: KPAD row-slots (K real + zero-pad), each split into NF PE-wide
        # sub-beats (nf-fast) -- matches the loader's Mode A writer order.
        repack_b = f"""template <class data1_T>
void {name}_repack_b(hls::stream<data1_T> &b_stream, hls::stream<ap_uint<{BB}> > &b_s) {{
    static_assert(data1_T::size == {N},
        "{name}: hls4ml must deliver one N-wide K-row per beat (row-major B)");
    for (unsigned k = 0; k < {KPAD}; k++) {{
        ap_uint<{N * WW}> row = 0;
        if (k < {K}) {{
            data1_T beat = b_stream.read();
            for (unsigned n = 0; n < {N}; n++)
                row.range(n * {WW} + {WW} - 1, n * {WW}) = beat[n].range({WW} - 1, 0);
        }}
        for (unsigned nf = 0; nf < {NF}; nf++) {{
            ap_uint<{BB}> nb = 0;
            for (unsigned pe = 0; pe < {PE}; pe++)
                nb.range(pe * {WW} + {WW} - 1, pe * {WW}) =
                    row.range((nf * {PE} + pe) * {WW} + {WW} - 1, (nf * {PE} + pe) * {WW});
            b_s.write(nb);
        }}
    }}
}}"""
        b_s_depth = KPAD * NF + 2
    else:
        # Mode B: PE*NF column-slots (N real + zero-pad), each split into SF SIMD-wide
        # sub-beats (sf-fast) -- matches the loader's (transposed) Mode B writer order;
        # natural column order c=nf*PE+pe (pe-fast) matches hls4ml's own beat order.
        repack_b = f"""template <class data1_T>
void {name}_repack_b(hls::stream<data1_T> &b_stream, hls::stream<ap_uint<{BB}> > &b_s) {{
    static_assert(data1_T::size == {K},
        "{name}: hls4ml must deliver one K-wide column per beat (col-major B)");
    for (unsigned c = 0; c < {PE * NF}; c++) {{
        ap_uint<{KPAD * WW}> col = 0;
        if (c < {N}) {{
            data1_T beat = b_stream.read();
            for (unsigned k = 0; k < {K}; k++)
                col.range(k * {WW} + {WW} - 1, k * {WW}) = beat[k].range({WW} - 1, 0);
        }}
        for (unsigned sf = 0; sf < {SF}; sf++) {{
            ap_uint<{BB}> nb = 0;
            for (unsigned s = 0; s < {SIMD}; s++)
                nb.range(s * {WW} + {WW} - 1, s * {WW}) =
                    col.range((sf * {SIMD} + s) * {WW} + {WW} - 1, (sf * {SIMD} + s) * {WW});
            b_s.write(nb);
        }}
    }}
}}"""
        b_s_depth = PE * NF * SF + 2

    return f"""#ifndef {name.upper()}_GEMM_IP_H_
#define {name.upper()}_GEMM_IP_H_
{guard}#include <hls_stream.h>
#include <ap_int.h>

// internal FINN-MVU blackbox (two-operand shim {name}_core.v; C twin {name}_core.cpp) --
// sums any K-tile partials and shift/round-half-up/wraps to out_width internally.
void {name}_core(hls::stream<ap_uint<{a_width}> >&, hls::stream<ap_uint<{BB}> >&,
{core_pad}hls::stream<ap_uint<{p_width}> >&);

namespace nnet {{

// Dedicated two-operand IP for gemm config M={m} K={K} N={N} (core={t['compute_core']}).
// B layout selected by SecondOperandRowMajor: MODE={mode} -- row-major (data1_T::size ==
// N, one K-row per beat) when True/unset, col-major (data1_T::size == K, one N-column
// per beat) when False.

{repack_a}

{repack_b}

// pure unpack drain: no bias for two-operand GEMM (has_bias is always False by
// construction) -- {name}_core already requantized each lane to out_width; this
// only slices lanes and loads them via .range() (raw bit pattern).
{drain_fn}

// The dedicated IP: hls4ml io_stream two-operand GEMM -> internal MVU blackbox.
// Two-operand GEMM never has a real bias, so hls4ml's call site carries no bias
// parameter at all -- there is only ever this one signature.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void {name}_gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
{' ' * (len(name) + 17)}hls::stream<res_T> &res_stream) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{a_width}> > a_s;
    hls::stream<ap_uint<{BB}> > b_s;
    hls::stream<ap_uint<{p_width}> > p_s;
#pragma HLS STREAM variable=a_s depth={SF + 2}
#pragma HLS STREAM variable=b_s depth={b_s_depth}
#pragma HLS STREAM variable=p_s depth={NF + 2}
    {name}_repack_a<data0_T>(a_stream, a_s);
    {name}_repack_b<data1_T>(b_stream, b_s);
    {name}_core(a_s, b_s, p_s);
    {name}_drain<res_T, CONFIG_T>(p_s, res_stream);
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

    Both operands are runtime streams: A activations, B (= the MVU weight matrix)
    loaded at runtime into the forked ``dynamic_load_2op`` module (double-buffered
    ping-pong), replayed across the M rows of A. No baked weights; no bias. 2-op
    only ever folds within ONE MVU tile -- no N/K-tiling (see
    jojo-track/defer/mvau-two-operand-dynamic-load/plan.md's "Cleanup: collapse
    2-op to a single dynamic_load_2op tile"); any depth (including the
    fully-spatial DEPTH==1 case), both B-layout modes."""
    if cfg.get("interface") == "array":
        raise ValueError(f"mvau two-operand does not support io_parallel for '{name}'.")
    # V1 mode selector (locked 2026-09-14): reuse SecondOperandRowMajor as the loader
    # layout knob, no new JSON key. True/unset (standalone generation is row-major by
    # construction) -> Mode A (row-major B, N-wide beats); explicit False -> Mode B
    # (col-major B, K-wide beats) via the forked dynamic_load_2op loader's MODE=1.
    mode = 0 if cfg.get("second_operand_row_major") is not False else 1
    plan = _resolve_plan(shape, cfg)
    plan["mode"] = mode
    t = plan["tile"]
    kt = plan.get("k_tiles", 1)
    nt = plan["n_tiles"]
    if nt > 1 or kt > 1:
        # 2-op dropped N/K-tiling entirely (untested/unused in practice -- see the
        # plan's "Cleanup" section); a config that still asks for it is a bug upstream.
        raise NotImplementedError(
            f"mvau two-operand IP '{name}': n_tiles/k_tiles>1 is not supported (2-op "
            "only ever folds within one MVU tile). Got n_tiles={nt} k_tiles={kt}.")
    part = cfg.get("part") or "xcvu13p-flga2577-2-e"
    clock_ns = cfg.get("clock_period_ns") or 5
    pkg = Path(output_dir) / name
    (pkg / "rtl_static").mkdir(parents=True, exist_ok=True)

    force_behavioral = bool(cfg.get("force_behavioral", False))
    shim, top_src = _rtl.generate_two_operand_shim, _2op_dataflow_top(name, plan)
    (pkg / f"{name}_core.v").write_text(_with_timescale(
        shim(shape, module_name=f"{name}_core",
             force_behavioral=force_behavioral, tile=t, plan=plan, mode=mode)))
    (pkg / f"{name}_core.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_2op_core_twin(shape, func_name=f"{name}_core", plan=plan)))
    (pkg / f"{name}_top.cpp").write_text(_with_ap_int_max_w(top_src))
    (pkg / f"{name}.json").write_text(
        _2op_blackbox_json(name, t, plan["n"], t["weight_width"], tiles=1,
                           resources=plan.get("resources")))
    (pkg / f"{name}_tb.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_2op_tb(shape, top_name=name, func_name=f"{name}_core", plan=plan,
                                n_nodes=cfg.get("n_nodes", 6))))
    (pkg / f"{name}_gemm_ip.h").write_text(_with_ap_int_max_w(
        _2op_gemm_ip_header(name, plan)))
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
    NTILE_REAL = N // NT                # unpadded per-tile column count
    init_files = []
    # N-pad columns (oo >= NTILE_REAL, only when NTILE > NTILE_REAL) get an all-zero
    # weight lane: the drain never reads that lane's result (see the local_oc guards
    # in the emitters), but the memstream word must still exist and be well-formed.
    if KT > 1:
        for j in range(NT):
            for i in range(KT):
                B_ji = [[(B[i * MW + kk][j * NTILE_REAL + oo] if oo < NTILE_REAL else 0)
                         for oo in range(NTILE)]
                        for kk in range(MW)]
                dat_path = pkg / "rtl_static" / _kdat_name(name, j * KT + i)
                dat_path.write_text(_wpack.pack_memstream_hex(
                    B_ji, NTILE, MW, PE, SIMD, WW, word_bits=t["weight_stream_width_ba"]))
                init_files.append(str(dat_path.resolve()))
    else:
        for ti in range(NT):
            B_ti = [[(B[kk][ti * NTILE_REAL + oo] if oo < NTILE_REAL else 0)
                     for oo in range(NTILE)] for kk in range(K)]
            dat_path = pkg / "rtl_static" / _dat_name(name, ti, NT)
            dat_path.write_text(_wpack.pack_memstream_hex(
                B_ti, NTILE, K, PE, SIMD, WW, word_bits=t["weight_stream_width_ba"]))
            # Absolute $readmemh path: relative is unresolvable in Vitis cosim's XSIM dir
            # (empirically -- see jojo-track); the package builds in place, so the
            # absolute path computed here stays valid for csim/cosim/impl.
            init_files.append(str(dat_path.resolve()))

    # has_bias is the single gate. Pre-has_bias manifests (the field absent from
    # cfg entirely) fall back to "does the given bias look real" -- not a hardcoded
    # True -- so an old caller that never supplied a bias keeps its old "no bias"
    # behavior instead of newly raising (see status.md, has_bias-from-tensor). The
    # bias is now baked into the RTL requant stage (bias ROM) and its C twin, not
    # the downstream drain -- computed here so both get the same codes.
    _bias = cfg.get("bias")
    _has_bias_default = _bias is not None and any(_bias)
    bias_codes = _wpack.bias_acc_codes(_bias, plan["product_frac"], plan["n"],
                                       bool(cfg.get("has_bias", _has_bias_default)))

    # FORCE_BEHAVIORAL=0 -> real DSP48/DSP58 primitives (impl-ready; cosim runs them via
    # XSIM unisim models). Set force_behavioral=True in cfg for unisim-free behavioral cosim.
    force_behavioral = bool(cfg.get("force_behavioral", False))
    (pkg / f"{name}_core.v").write_text(_with_timescale(
        _rtl.generate_shim(shape, module_name=f"{name}_core",
                           force_behavioral=force_behavioral, tile=t,
                           weights_in_core=True, init_files=init_files,
                           n_tiles=NT, k_tiles=KT, bias_codes=bias_codes,
                           raw_k=plan["k"], raw_n=plan["n"])))
    (pkg / f"{name}_core.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_core_twin(shape, func_name=f"{name}_core", plan=plan,
                                   baked_weights=B, bias_codes=bias_codes)))
    top_src = (_kt_dataflow_top(name, plan) if KT > 1
               else _dataflow_top(name, plan))
    (pkg / f"{name}_top.cpp").write_text(_with_ap_int_max_w(top_src))
    (pkg / f"{name}.json").write_text(_blackbox_json(
        name, t, tiles=KT * NT, resources=plan.get("resources")))
    (pkg / f"{name}_tb.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_tb(shape, top_name=name, func_name=f"{name}_core", plan=plan,
                            bias_codes=bias_codes, baked_weights=B,
                            n_nodes=cfg.get("n_nodes", 6))))
    ip_hdr = (_kt_gemm_ip_header(name, plan) if KT > 1
              else _gemm_ip_header(name, plan))
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
            f"    // Bias, when this item has one, is baked as a compile-time constant inside\n"
            f"    // {nm}_gemm_ip.h -- never a function argument -- so there is only ever this\n"
            f"    // one signature.\n"
            f"    template <class data_T, class res_T, typename CONFIG_T>\n"
            f"    static void stream_const_weights(hls::stream<data_T> &a, hls::stream<res_T> &r) {{\n"
            f"        #pragma HLS INLINE\n"
            f"        {nm}_gemm_stream_const_weights<data_T, res_T, CONFIG_T>(a, r);\n"
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
            f"    // Two-operand GEMM never has a real bias -- there is only ever this one\n"
            f"    // signature.\n"
            f"    template <class data0_T, class data1_T, class res_T, typename CONFIG_T>\n"
            f"    static void stream(hls::stream<data0_T> &a, hls::stream<data1_T> &b,\n"
            f"                       hls::stream<res_T> &r) {{\n"
            f"        #pragma HLS INLINE\n"
            f"        {nm}_gemm_stream<data0_T, data1_T, res_T, CONFIG_T>(a, b, r);\n"
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

// io_stream two-operand entry hls4ml calls; routes to the config's IP by id. Two-operand
// GEMM never has a real bias, so hls4ml's call site never passes one -- there is only
// ever this one signature.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
                 hls::stream<res_T> &res_stream) {{
    #pragma HLS INLINE
    mvau_ip_stream<CONFIG_T::gemm_ip_id>::template stream<data0_T, data1_T, res_T, CONFIG_T>(
        a_stream, b_stream, res_stream);
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

// io_stream const_weights entry hls4ml calls; routes to the config's IP by id. Bias
// (when an item has one) is baked as a compile-time constant inside that item's
// <name>_gemm_ip.h -- never a function argument -- so there is only ever this one
// signature.
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(hls::stream<data_T> &a_stream, hls::stream<res_T> &res_stream) {{
    #pragma HLS INLINE
    mvau_ip<CONFIG_T::gemm_ip_id>::template stream_const_weights<data_T, res_T, CONFIG_T>(
        a_stream, res_stream);
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
            "clock_period_ns", "reuse_factor", "fold_axis", "n_tiles", "k_tiles",
            "weights", "weights_in_core", "name")}
    try:
        return _resolve_plan((m, k, n), cfg)
    except Exception:
        return None


def _reuse_factor_warnings(plan):
    """hls4ml's ReuseFactor-legalization warning text (see run_atlas_flow.py's
    ``_snap_reuse_factor`` for the v-generic target); ``resolve_fold`` already renders
    the warning strings (with the layer name baked in via the ``name`` config key)."""
    return plan.get("fold_warnings", [])


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
        # Resolve the plan once for this item: DSP estimate plus ReuseFactor reporting
        # (requested/legalized/effective reuse, PE/SIMD, padded N). Best-effort: skip
        # these fields if the node has no resolvable shape.
        plan = _resolve_manifest_plan(it)
        if plan is not None:
            if plan.get("resources") is not None:
                core["resources"] = plan["resources"]
            tile = plan["tile"]
            core["reuse_factor_requested"] = plan["requested_reuse_factor"]
            core["reuse_factor"] = plan["reuse_factor"]
            core["effective_reuse"] = plan["effective_reuse"]
            core["n_pad"] = plan["n_pad"]
            core["fold_axis"] = plan["fold_axis"]
            core["pe"] = tile["pe"]
            core["simd"] = tile["simd"]
            core["ii_per_vector"] = tile["nf"] * tile["sf"]
            if "n_tiles" in plan:
                core["n_tiles"] = plan["n_tiles"]
            if "k_tiles" in plan:
                core["k_tiles"] = plan["k_tiles"]
            for w in _reuse_factor_warnings(plan):
                print(w)
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
