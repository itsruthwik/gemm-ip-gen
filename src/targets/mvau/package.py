"""mvau package assembly: emit a Vitis-HLS RTL-blackbox GEMM IP package.

Generalizes a cosim-validated single-tile spike. Per package ``<name>``:

    <name>_core.v      shim (FINN mvu_vvu_axi wrapped) -- module name == c_function_name
    <name>_core.cpp    the C twin (blackbox behavioral model, csim)
    <name>_top.cpp     the DUT: feed -> blackbox -> drain, in a HLS dataflow region
    <name>.json        blackbox descriptor (FIFO ports, CE, the 5 ap_ctrl_chain_protocol_* keys)
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
from gemm_ip.quant import _truncates

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



def public_fn(name):
    """The one function hls4ml calls for this node -- and, being an RTL blackbox, the
    JSON ``c_function_name`` and the RTL module name too (Vitis instantiates by it)."""
    return f"gemm_stream_{name}"


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
    core_decl = f"void {public_fn(name)}(hls::stream<ap_uint<{AB}> >&,\n{pad}hls::stream<ap_uint<{PB}> >&);"
    indent = ' ' * (len(name) + 1)
    top_sig = f"void {name}(hls::stream<ap_uint<{AB}> >& a_in,\n{indent}hls::stream<ap_uint<{PB}> >& c_out)"
    return f"""#include <hls_stream.h>
#include <ap_int.h>

{core_decl}

// Passthrough feed/drain processes around the blackbox: Vitis will not pass a
// top-level argument straight into a blackbox (HLS 214-149), so the DUT is a
// dataflow region with one-beat-per-iteration copies on each side. The boundary
// is already hls4ml's own unpadded row width on both sides -- {m} rows in, {m}
// rows out; the copies add no repacking.
static void feed_a(hls::stream<ap_uint<{AB}> >& in, hls::stream<ap_uint<{AB}> >& out) {{
    for (int i = 0; i < {m}; i++) {{
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }}
}}
static void drain_c(hls::stream<ap_uint<{PB}> >& in, hls::stream<ap_uint<{PB}> >& out) {{
    for (int i = 0; i < {m}; i++) {{
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }}
}}

{top_sig} {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{AB}> > a_s;
    hls::stream<ap_uint<{PB}> > p_s;
#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=p_s depth=4
    feed_a(a_in, a_s);
    {public_fn(name)}(a_s, p_s);   // <-- FINN MVU RTL blackbox (weights + bias baked; already requantized)
    drain_c(p_s, c_out);
}}
"""


def _call_performance(t, m, load_beats=0):
    """``rtl_performance`` for the blackbox JSON, in the units Vitis reads them:
    per INVOCATION of the function, i.e. per node of ``m`` rows. These are
    scheduling hints for Vitis only; measured results come from cosim and Vivado.

    ``II`` is the node interval: the ports' per-row cadence ``SF*NF`` times the
    rows, or, for a two-operand node, the B load beats if those take longer
    (``load_beats``, one B row / column per beat; the next node's B loads into
    the loader's spare bank while this node computes).

    ``latency`` is first input beat in to last row out. For a two-operand node the
    first A row is not admitted to the core until the whole B bank is written,
    so the load beats come first; then the row-port latency, then the remaining
    rows at the per-row cadence. Per-row values stay in the manifest as
    ``ii_per_row`` / ``port_latency``."""
    ii_row = int(t["ii_per_row"])
    ii_call = max(int(m) * ii_row, int(load_beats))
    latency = int(load_beats) + int(t["port_latency"]) + (int(m) - 1) * ii_row
    return {"latency": str(latency), "II": str(ii_call)}


def _blackbox_json(name, t, m, tiles=1, resources=None):
    WB, AB, PB = t["weight_stream_width_ba"], t["input_stream_width_ba"], t["output_stream_width_ba"]
    fn = public_fn(name)
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
        "c_files": [{"c_file": f"{name}_core.cpp", "cflag": ""}],   # C twin file; the function inside is `fn`
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
        # Per invocation (one node of m rows), the unit Vitis reads: II = node interval
        # = m*SF*NF, latency = first row in -> last row out. Built on the measured
        # row-port numbers (geometry.port_latency_cycles, SF*NF per row).
        "rtl_performance": _call_performance(t, m),
        # JSON has no comment syntax; this underscore-key carries a note in the file.
        "_comment": "latency/II are per invocation (one node of m rows), from the measured "
                    "row-port values (SF*NF cycles per row). DSP: FINN cost model "
                    "PE*ceil(SIMD/3) per tile, scaled by the tile count (within a few percent "
                    "of post-route). FF/LUT: rough per-DSP hints only; BRAM from the plan.",
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


def ws_ports(plan):
    """Boundary contract of the weight-stationary (plain, K-tiled, N-tiled) shim:
    one raw ``K*activation_width`` row per beat in, one raw ``N*out_width`` row per
    beat out, ``M`` beats each per node. Exact bit counts, no byte alignment -- the
    same numbers hls4ml computes from its own stream types."""
    t = plan["tile"]
    K, N = plan["k"], plan["n"]
    return {"a": K * t["activation_width"], "a_beats": plan["num_input_vectors"],
            "p": N * plan["output_width"], "p_beats": plan["num_input_vectors"]}


def _gemm_ip_header(name, plan, two_operand=False):
    """The per-node header hls4ml's build includes: the DECLARATION of this node's
    public function, ``gemm_stream_<name>`` -- which is the RTL blackbox itself.
    hls4ml calls it directly from its top dataflow region (with its own pack/unpack
    processes converting its array streams to these packed beats), so there is no
    HLS-side wrapper, no nested dataflow region and no per-id dispatch. The shim
    does ALL K/N padding, the SF fan-out, the B loader gearbox (two-operand), the
    NF-beat stitching, bias and requant internally, so the ports are exactly
    hls4ml's own row widths."""
    fn = public_fn(name)
    pad = " " * (len(fn) + 6)
    if two_operand:
        ports = _rtl.two_operand_ports(plan, plan["tile"], plan.get("mode", 0))
        b_what = "N-row" if plan.get("mode", 0) == 0 else "K-column"
        args = (f"hls::stream<ap_uint<{ports['a']}> >&,   // A: one K-row/beat, {ports['a_beats']} beats/node" + "\n"
                + f"{pad}hls::stream<ap_uint<{ports['b']}> >&,   // B: one {b_what}/beat, {ports['b_beats']} beats/node" + "\n"
                + f"{pad}hls::stream<ap_uint<{ports['p']}> >&);  // C: one N-row/beat, {ports['p_beats']} beats/node")
        widest = max(ports["a"], ports["b"], ports["p"])
        kind = "two-operand, B at runtime"
    else:
        ports = ws_ports(plan)
        args = (f"hls::stream<ap_uint<{ports['a']}> >&,   // A: one K-row/beat, {ports['a_beats']} beats/node" + "\n"
                + f"{pad}hls::stream<ap_uint<{ports['p']}> >&);  // C: one N-row/beat, {ports['p_beats']} beats/node")
        widest = max(ports["a"], ports["p"])
        kind = "weight-stationary, weights + bias baked"
    guard = ("#ifndef AP_INT_MAX_W" + "\n" + f"#define AP_INT_MAX_W {_apmaxw(widest)}" + "\n" + "#endif" + "\n"
             if widest > 1024 else "")
    m, K, N = plan["num_input_vectors"], plan["k"], plan["n"]
    return f"""#ifndef {name.upper()}_GEMM_IP_H_
#define {name.upper()}_GEMM_IP_H_
{guard}#include <hls_stream.h>
#include <ap_int.h>

// {fn}: the FINN-MVU RTL blackbox for gemm config M={m} K={K} N={N} ({kind}).
// hls4ml calls this directly from its top dataflow region on packed bit streams (its
// pack/unpack processes convert its array beats); the shim does all padding, fan-out,
// stitching and requant internally, so the ports are hls4ml's own unpadded row widths.
void {fn}({args}

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
    core_decl = f"void {public_fn(name)}(hls::stream<ap_uint<{AB}> >&,\n{pad}hls::stream<ap_uint<{PB}> >&);"
    indent = ' ' * (len(name) + 1)
    apmax = _apmaxw(max(AB, PB))
    top_sig = f"void {name}(hls::stream<ap_uint<{AB}> >& a_in,\n{indent}hls::stream<ap_uint<{PB}> >& c_out)"
    return f"""#define AP_INT_MAX_W {apmax}   // raw K-wide activation row may exceed the 1024-bit default
#include <hls_stream.h>
#include <ap_int.h>

{core_decl}

// Passthrough feed/drain processes around the blackbox: Vitis will not pass a
// top-level argument straight into a blackbox (HLS 214-149). No repacking: the
// wrapper's boundary is already hls4ml's own unpadded row width on both sides --
// {m} rows in, {m} rows out.
static void feed_a(hls::stream<ap_uint<{AB}> >& in, hls::stream<ap_uint<{AB}> >& out) {{
    for (int i = 0; i < {m}; i++) {{
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }}
}}
static void drain_c(hls::stream<ap_uint<{PB}> >& in, hls::stream<ap_uint<{PB}> >& out) {{
    for (int i = 0; i < {m}; i++) {{
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }}
}}

{top_sig} {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{AB}> > a_s;
    hls::stream<ap_uint<{PB}> > p_s;
#pragma HLS STREAM variable=a_s depth=4
#pragma HLS STREAM variable=p_s depth=4
    feed_a(a_in, a_s);
    {public_fn(name)}(a_s, p_s);   // <-- FINN MVU RTL blackbox (K-tiles summed, weights + bias baked; already requantized)
    drain_c(p_s, c_out);
}}
"""


def _2op_dataflow_top(name, plan):
    """DUT for the two-operand blackbox: a straight passthrough into the blackbox. The shim's boundary is already hls4ml's own raw beat widths
    (one K-row of A, one B row/column, one N-row of C per beat -- see
    ``rtl.two_operand_ports``), and the wide-to-narrow loader gearbox, K-padding and
    NF-beat stitching all live inside the RTL, so this top does no repacking of its
    own -- it is only the dataflow region Vitis needs to drop the RTL blackbox into."""
    t = plan["tile"]
    mode = plan.get("mode", 0)
    ports = _rtl.two_operand_ports(plan, t, mode)
    AB, BB, CB = ports["a"], ports["b"], ports["p"]
    A_BEATS, B_BEATS = ports["a_beats"], ports["b_beats"]
    pad = ' ' * (len(name) + 6)
    indent = ' ' * (len(name) + 1)
    return f"""#include <hls_stream.h>
#include <ap_int.h>

void {public_fn(name)}(hls::stream<ap_uint<{AB}> >&, hls::stream<ap_uint<{BB}> >&,
{pad}hls::stream<ap_uint<{CB}> >&);

// Passthrough feed/drain processes around the blackbox: Vitis will not pass a
// top-level argument straight into a blackbox (HLS 214-149). No repacking: the
// shim's boundary is already hls4ml's own raw beat widths.
static void feed_a(hls::stream<ap_uint<{AB}> >& in, hls::stream<ap_uint<{AB}> >& out) {{
    for (int i = 0; i < {A_BEATS}; i++) {{
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }}
}}
static void feed_b(hls::stream<ap_uint<{BB}> >& in, hls::stream<ap_uint<{BB}> >& out) {{
    for (int i = 0; i < {B_BEATS}; i++) {{
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }}
}}
static void drain_c(hls::stream<ap_uint<{CB}> >& in, hls::stream<ap_uint<{CB}> >& out) {{
    for (int i = 0; i < {A_BEATS}; i++) {{
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }}
}}

void {name}(hls::stream<ap_uint<{AB}> >& a_in, hls::stream<ap_uint<{BB}> >& b_in,
{indent}hls::stream<ap_uint<{CB}> >& c_out) {{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<{AB}> > a_s;
    hls::stream<ap_uint<{BB}> > b_s;
    hls::stream<ap_uint<{CB}> > p_s;
#pragma HLS STREAM variable=a_s depth={max(A_BEATS, 2)}
#pragma HLS STREAM variable=b_s depth={max(B_BEATS, 2)}
#pragma HLS STREAM variable=p_s depth=4
    feed_a(a_in, a_s);
    feed_b(b_in, b_s);
    {public_fn(name)}(a_s, b_s, p_s);   // <-- FINN MVU RTL blackbox (B loaded+replayed in-core; already requantized)
    drain_c(p_s, c_out);
}}
"""


def _2op_blackbox_json(name, plan, resources=None):
    """Blackbox JSON for the two-operand core: two input FIFOs (a, b) + one output (p),
    at the shim's raw boundary widths (``rtl.two_operand_ports``)."""
    t = plan["tile"]
    m = plan["num_input_vectors"]
    mode = plan.get("mode", 0)
    # loader beats per node: one whole B row (mode 0) / column (mode 1) per cycle,
    # no narrow sub-beat gearbox (see generate_two_operand_shim / dynamic_load_2op.sv)
    load_beats = plan["k_pad"] if mode == 0 else plan["n"]
    fn = public_fn(name)
    res = resources or {"dsp": t["dsp_estimate"], "bram18": 0}
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
        "c_files": [{"c_file": f"{name}_core.cpp", "cflag": ""}],   # C twin file; the function inside is `fn`
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
        # Per invocation (one node of m rows), the unit Vitis reads. II is the node
        # interval: m*SF*NF of compute, or the B load beats (k_pad row-major, n
        # col-major -- the loader takes one whole B row/column per cycle) when those
        # take longer. Latency is first row in -> last row out with B resident; the
        # first node of a run additionally waits for its whole B load.
        "rtl_performance": _call_performance(t, m, load_beats=load_beats),
        "_comment": "latency/II are per invocation (one node of m rows): compute m*SF*NF "
                    "or the B load beats, whichever is larger. DSP: FINN cost model "
                    "PE*ceil(SIMD/3). FF/LUT: rough per-DSP hints only.",
        "rtl_resource_usage": {"FF": str(30 * dsp), "LUT": str(40 * dsp),
                               "DSP": str(dsp), "BRAM": str(bram), "URAM": "0"},
    }, indent=2) + "\n"


def generate_two_operand_pkg(shape, name, output_dir, **cfg):
    """Emit a two-operand (``gemm_stream``) mvau blackbox package into ``<output_dir>/<name>/``.

    Both operands are runtime streams: A activations, B (= the MVU weight matrix)
    loaded at runtime into the forked ``dynamic_load_2op`` module (double-buffered
    ping-pong), replayed across the M rows of A. No baked weights; no bias. 2-op
    only ever folds within ONE MVU tile -- no N/K-tiling (multi-tile 2-op and the
    register/grid form were retired); any depth (including the
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
            f"only ever folds within one MVU tile). Got n_tiles={nt} k_tiles={kt}.")
    part = cfg.get("part") or "xcvu13p-flga2577-2-e"
    clock_ns = cfg.get("clock_period_ns") or 5
    pkg = Path(output_dir) / name
    (pkg / "rtl_static").mkdir(parents=True, exist_ok=True)

    force_behavioral = bool(cfg.get("force_behavioral", False))
    shim, top_src = _rtl.generate_two_operand_shim, _2op_dataflow_top(name, plan)
    (pkg / f"{name}_core.v").write_text(_with_timescale(
        shim(shape, module_name=public_fn(name),
             force_behavioral=force_behavioral, tile=t, plan=plan, mode=mode)))
    (pkg / f"{name}_core.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_2op_core_twin(shape, func_name=public_fn(name), plan=plan)))
    (pkg / f"{name}_top.cpp").write_text(_with_ap_int_max_w(top_src))
    (pkg / f"{name}.json").write_text(
        _2op_blackbox_json(name, plan, resources=plan.get("resources")))
    (pkg / f"{name}_tb.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_2op_tb(shape, top_name=name, func_name=public_fn(name), plan=plan,
                                n_nodes=cfg.get("n_nodes", 6))))
    (pkg / f"{name}_gemm_ip.h").write_text(_with_ap_int_max_w(
        _gemm_ip_header(name, plan, two_operand=True)))
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
            # (found empirically); the package builds in place, so the absolute path
            # computed here stays valid for csim/cosim/impl.
            init_files.append(str(dat_path.resolve()))

    # has_bias is the single gate. Pre-has_bias manifests (the field absent from
    # cfg entirely) fall back to "does the given bias look real" -- not a hardcoded
    # True -- so an old caller that never supplied a bias keeps its old "no bias"
    # behavior instead of newly raising (see status.md, has_bias-from-tensor). The
    # bias is now baked into the RTL requant stage (bias ROM) and its C twin, not
    # the downstream drain -- computed here so both get the same codes.
    _bias = cfg.get("bias")
    _has_bias_default = _bias is not None and any(_bias)
    _bake_bias = bool(cfg.get("has_bias", _has_bias_default))
    # A bias finer than the products (input_frac + weight_frac) lifts the accumulator:
    # the raw sum is shifted left by acc_lshift before the add, so every bias is an exact
    # code. Set on the plan and its tile before anything renders the requant stage.
    _lsh = _geom.acc_lshift_for_bias(_bias if _bake_bias else None, plan["product_frac"])
    plan["acc_lshift"] = plan["tile"]["acc_lshift"] = _lsh
    bias_codes = _wpack.bias_acc_codes(_bias, plan["product_frac"] + _lsh, plan["n"], _bake_bias,
                                       exact=True)
    # Truncating (TRN) result: the requant stage only rounds half-up, and
    # floor(x / 2^s) == round_half_up(x - 2^(s-1), s), so the half is folded into the
    # baked bias codes (created for a bias-free layer). One list feeds the RTL ROM,
    # the C twin and the testbench, so all three floor together.
    _shift = _geom.requant_shift(plan)
    if _truncates(cfg.get("output_precision")) and _shift > 0:
        _half = 1 << (_shift - 1)
        bias_codes = [int(c) - _half for c in (bias_codes if bias_codes is not None
                                               else [0] * plan["n"])]

    # FORCE_BEHAVIORAL=0 -> real DSP48/DSP58 primitives (impl-ready; cosim runs them via
    # XSIM unisim models). Set force_behavioral=True in cfg for unisim-free behavioral cosim.
    force_behavioral = bool(cfg.get("force_behavioral", False))
    (pkg / f"{name}_core.v").write_text(_with_timescale(
        _rtl.generate_shim(shape, module_name=public_fn(name),
                           force_behavioral=force_behavioral, tile=t,
                           weights_in_core=True, init_files=init_files,
                           n_tiles=NT, k_tiles=KT, bias_codes=bias_codes,
                           raw_k=plan["k"], raw_n=plan["n"])))
    (pkg / f"{name}_core.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_core_twin(shape, func_name=public_fn(name), plan=plan,
                                   baked_weights=B, bias_codes=bias_codes)))
    top_src = (_kt_dataflow_top(name, plan) if KT > 1
               else _dataflow_top(name, plan))
    (pkg / f"{name}_top.cpp").write_text(_with_ap_int_max_w(top_src))
    (pkg / f"{name}.json").write_text(_blackbox_json(
        name, t, plan["num_input_vectors"], tiles=KT * NT, resources=plan.get("resources")))
    (pkg / f"{name}_tb.cpp").write_text(_with_ap_int_max_w(
        _golden.generate_tb(shape, top_name=name, func_name=public_fn(name), plan=plan,
                            bias_codes=bias_codes, baked_weights=B,
                            n_nodes=cfg.get("n_nodes", 6))))
    ip_hdr = _gemm_ip_header(name, plan)
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
    """The whole-model header hls4ml includes when GEMM_IP_HEADER is set: just the
    per-node declaration headers. Each node's public function is ``gemm_stream_<name>``
    (see ``public_fn``), which hls4ml calls by name from its top dataflow region, so
    there is no per-id template dispatch any more -- a blackbox is a concrete function
    and could not be reached through one anyway."""
    def _nm(it):
        return it.get("emit_name") or it["name"]
    incs = "\n".join(f'#include "{_nm(it)}/{_nm(it)}_gemm_ip.h"' for it in items)
    return f"""#ifndef GEMM_IP_COMBINED_MVAU_H_
#define GEMM_IP_COMBINED_MVAU_H_
// mvau (Vitis RTL blackbox) package: one declaration per GEMM node. hls4ml calls each
// node's gemm_stream_<name> directly on packed bit streams; see the per-node headers
// for the exact port contract.
{incs}
#endif // GEMM_IP_COMBINED_MVAU_H_
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
            "weights", "weights_in_core", "name", "pe", "simd")}
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
    ap_int_max_w = None
    for it in items:
        nm = it.get("emit_name") or it["name"]
        core = {"name": nm, "kind": "rtl_blackbox", "tool": "vitis",
                # the one function hls4ml calls == the RTL module == the JSON c_function_name
                "function": public_fn(nm),
                "entity": public_fn(nm), "rtl": f"{nm}/{nm}_core.v",
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
            # the packed-stream port contract of `function` (exact bit widths + beats
            # per node), so an integrator can check hls4ml's side against it
            if it.get("weights_in_core", True):
                core["ports"] = ws_ports(plan)
            else:
                mode = 0 if it.get("second_operand_row_major") is not False else 1
                pp = _rtl.two_operand_ports(plan, tile, mode)
                core["ports"] = {k: pp[k] for k in ("a", "a_beats", "b", "b_beats", "p", "p_beats")}
            widest_port = max(
                width for port, width in core["ports"].items() if not port.endswith("_beats")
            )
            if widest_port > 1024:
                required = _apmaxw(widest_port)
                ap_int_max_w = required if ap_int_max_w is None else max(ap_int_max_w, required)
            for w in _reuse_factor_warnings(plan):
                print(w)
        cores.append(core)
    manifest = {"tool": "vitis", "flow": "rtl_blackbox",
                "header": "gemm_ip_combined.h", "cores": cores}
    if ap_int_max_w is not None:
        # Package consumer contract: this definition must also reach the hls4ml
        # design translation unit, not only gemm-ip-gen's generated core TUs.
        manifest["compile_definitions"] = {"AP_INT_MAX_W": ap_int_max_w}
    return json.dumps(manifest, indent=2) + "\n"


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
        # (no two-operand N-tiled case: 2-op only ever folds within one MVU tile and
        # generate_two_operand_pkg rejects n_tiles > 1)
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
