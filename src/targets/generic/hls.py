"""C++ emitters for the `generic` behavioral-HLS GEMM target (tool: Vitis HLS).

No RTL blackbox / hardblock: these emit plain synthesizable C++ that Vitis HLS
turns into RTL. The four entry points mirror the hls4ml Vitis seam
(``nnet::gemm_{stream,stream_weightless,array,array_weightless}``) and are the
behavioral triple-loop from ``nnet_gemm_behavioral.h`` promoted to synthesizable
form (HLS pragmas, ``ap_fixed`` arithmetic, ``nnet::array`` beats, per-shape
``CONFIG_T``). One templated definition covers every shape.
"""

import re


def _ap_type(precision, default):
    """Normalise an hls4ml/Catapult precision string to a Vitis ap_* type.

    Accepts ``ap_fixed<...>`` (kept), ``fixed<...>``/``ufixed<...>`` (→ ap_),
    ``int<W>``/``uint<W>`` (→ ap_int/ap_uint), Catapult ``ac_fixed<W,I,...>`` and
    ``ac_int<W,true|false>``. Falls back to *default* when unset/unparseable.
    """
    if not precision:
        return default
    p = str(precision).replace(" ", "")
    if p.startswith("ap_"):
        return p
    if p.startswith("ac_int<"):
        m = re.match(r"ac_int<(\d+),(true|false)>", p)
        if m:
            return (f"ap_int<{m.group(1)}>" if m.group(2) == "true"
                    else f"ap_uint<{m.group(1)}>")
    if p.startswith("ac_fixed<"):
        inner = p[len("ac_fixed<"):].rstrip(">")
        parts = inner.split(",")
        wi = ",".join(parts[:2])
        return f"ap_fixed<{wi}>"
    if p.startswith("fixed<"):
        return "ap_" + p
    if p.startswith("ufixed<"):
        return "ap_" + p
    m = re.match(r"u?int<(\d+)>", p)
    if m:
        return ("ap_uint<" if p.startswith("u") else "ap_int<") + m.group(1) + ">"
    return default


# ── The four synthesizable entry points (shape-independent template) ─────────────
#
# Identical microarchitecture to the whole-model combined header (see
# _GEMM_IP_COMBINED_FUNCS): the row (M) loop is pipelined at II=CONFIG_T::reuse_factor,
# the N (output) and K (contraction) loops are fully UNROLLed, and the multiplier count
# is capped by CONFIG_T::multiplier_limit. Strategy/ReuseFactor reach the core purely
# through CONFIG_T (which config_header emits), so this standalone package and the flow
# synthesize the same RTL:
#   - Latency  (reuse_factor=1): II=1, multiplier_limit=gemm_k*gemm_n -> full array.
#   - Resource (reuse_factor=R): II=R, multiplier_limit=ceil(gemm_k*gemm_n/R) -> shared.
# The only difference from the combined funcs is the signature: here the weightless
# entries take the weight ROM as an explicit argument (the standalone top feeds it from
# <name>_weights.h) rather than sourcing it from CONFIG_T::gemm_weight_cols().
_GEMM_IP_FUNCS = r"""
namespace nnet {

// gemm_array — io_parallel, TWO activation operands (e.g. attention QK^T / A.V).
template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_array(a_row_T a_rows[CONFIG_T::gemm_m], b_col_T b_cols[CONFIG_T::gemm_n],
                res_row_T results[CONFIG_T::gemm_m], bias_T biases[CONFIG_T::gemm_n]) {
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit
    GEMM_ARRAY_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        GEMM_ARRAY_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t accum = 0;
            GEMM_ARRAY_K: for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
                #pragma HLS UNROLL
                accum += (typename CONFIG_T::accum_t)(a_rows[m][k] * b_cols[n][k]);
            }
            accum += biases[n];
            // Write the output element directly: the io_parallel caller partitions
            // results[] complete, and a whole-row nnet::array operator= copy under the
            // pipelined M loop is not a transformable instruction for Vitis HLS.
            results[m][n] = accum;
        }
    }
}

// gemm_array_weightless — io_parallel, weights baked into the IP. The weight ROM
// is passed in by the top from <name>_weights.h (not an external port, and not
// routed through CONFIG_T).
template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_array_weightless(a_row_T a_rows[CONFIG_T::gemm_m], b_col_T weight_cols[CONFIG_T::gemm_n],
                           res_row_T results[CONFIG_T::gemm_m], bias_T biases[CONFIG_T::gemm_n]) {
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit
    GEMM_AWL_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        GEMM_AWL_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t accum = 0;
            GEMM_AWL_K: for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
                #pragma HLS UNROLL
                accum += (typename CONFIG_T::accum_t)(a_rows[m][k] * weight_cols[n][k]);
            }
            accum += biases[n];
            // Direct element write (see gemm_array): avoids the whole-row operator=
            // copy that Vitis cannot transform under complete partition + pipelined M.
            results[m][n] = accum;
        }
    }
}

// gemm_stream — io_stream, TWO activation operands. B is read into local storage
// (operand residency), then C = A * B streams out. One K-wide beat per A row.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
                 hls::stream<res_T> &res_stream, typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    static_assert(data0_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(data1_T::size == CONFIG_T::gemm_k, "B column height must equal gemm_k.");
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

    data1_T b_cols[CONFIG_T::gemm_n];
    GEMM_STREAM_READB: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
        #pragma HLS PIPELINE II=1
        b_cols[n] = b_stream.read();
    }
    GEMM_STREAM_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        data0_T a_row = a_stream.read();
        res_T c_row;
        GEMM_STREAM_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t accum = 0;
            GEMM_STREAM_K: for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
                #pragma HLS UNROLL
                accum += (typename CONFIG_T::accum_t)(a_row[k] * b_cols[n][k]);
            }
            accum += biases[n];
            c_row[n] = accum;
        }
        res_stream.write(c_row);
    }
}

// gemm_stream_weightless — io_stream, weights baked into the IP. The weight ROM is
// passed in by the top from <name>_weights.h (not a stream, not via CONFIG_T).
// One K-wide beat per A row (data_T::size == gemm_k).
template <class data_T, class b_col_T, class res_T, typename CONFIG_T>
void gemm_stream_weightless(hls::stream<data_T> &data_stream, b_col_T weight_cols[CONFIG_T::gemm_n],
                            hls::stream<res_T> &res_stream, typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    static_assert(data_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

    GEMM_SWL_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        data_T a_row = data_stream.read();
        res_T c_row;
        GEMM_SWL_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t accum = 0;
            GEMM_SWL_K: for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
                #pragma HLS UNROLL
                accum += (typename CONFIG_T::accum_t)(a_row[k] * weight_cols[n][k]);
            }
            accum += biases[n];
            c_row[n] = accum;
        }
        res_stream.write(c_row);
    }
}

} // namespace nnet
"""


def nnet_types_header():
    """Minimal self-contained nnet::array<T,N> (matches hls4ml's interface)."""
    return """#ifndef NNET_TYPES_H_
#define NNET_TYPES_H_

#include <cstddef>

namespace nnet {

// Fixed-size packed beat, interface-compatible with hls4ml's nnet::array.
template <typename T, unsigned N> struct array {
    typedef T value_type;
    static const unsigned size = N;
    T data[N];
    T &operator[](std::size_t pos) { return data[pos]; }
    const T &operator[](std::size_t pos) const { return data[pos]; }
    array &operator=(const array &other) {
        for (unsigned i = 0; i < N; i++) {
            #pragma HLS UNROLL
            data[i] = other.data[i];
        }
        return *this;
    }
};

} // namespace nnet

#endif // NNET_TYPES_H_
"""


def gemm_ip_header(name):
    """The four synthesizable entry points (shape-generic). Strategy/ReuseFactor are
    honored via CONFIG_T (reuse_factor / multiplier_limit), so the funcs need no
    per-shape specialisation here — config_header emits the knobs into the struct."""
    funcs = _GEMM_IP_FUNCS
    return (
        f"#ifndef {name.upper()}_GEMM_IP_H_\n"
        f"#define {name.upper()}_GEMM_IP_H_\n\n"
        "#include <ap_fixed.h>\n"
        "#include <ap_int.h>\n"
        "#include <hls_stream.h>\n"
        '#include "nnet_types.h"\n'
        f"{funcs}\n"
        f"#endif // {name.upper()}_GEMM_IP_H_\n"
    )


# ── Whole-model combined header (the hls4ml Vitis integration seam) ───────────────
#
# Unlike the per-shape RTL blackbox targets, the generic funcs are one templated
# definition that covers every layer, so there is no per-core dispatch: the four
# entry points below ARE the combined header. Their signatures match the hls4ml
# contract exactly (the "declaration only" prototypes in nnet_gemm_ip.h /
# nnet_gemm_stream.h): the weightless entries take NO weight argument and source the
# constant columns from CONFIG_T::gemm_weight_cols() (the ROM the writer injects into
# each layer's CONFIG_T), and gemm_stream_weightless unpacks narrow input beats into a
# gemm_k-wide row. Header-only + synthesizable -> no add_files (see sources tcl).
#
# Microarchitecture (mirrors hls4ml's nnet_dense_latency): each streamed/array A row
# is one pipelined region — the N (output) and K (contraction) loops are fully
# UNROLLed, the row (M) loop is pipelined at II=CONFIG_T::reuse_factor, and the
# multiplier count is capped by CONFIG_T::multiplier_limit. So Strategy/ReuseFactor
# reach the core through CONFIG_T (hls4ml injects both into each layer's gemm config):
#   - Latency  (reuse_factor=1): II=1, multiplier_limit=gemm_k*gemm_n -> the full
#     K*N multiplier array fires per row, one row/cycle (the DenseLatency point).
#   - Resource (reuse_factor=R): II=R, multiplier_limit=ceil(gemm_k*gemm_n/R) -> the
#     scheduler shares that many multipliers across R cycles/row (DenseResource point).
_GEMM_IP_COMBINED_FUNCS = r"""
namespace nnet {

// gemm_array — io_parallel, TWO activation operands (attention QK^T / A.V).
template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_array(a_row_T a_rows[CONFIG_T::gemm_m], b_col_T b_cols[CONFIG_T::gemm_n],
                res_row_T results[CONFIG_T::gemm_m], bias_T biases[CONFIG_T::gemm_n]) {
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit
    GEMM_ARRAY_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        GEMM_ARRAY_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t accum = 0;
            GEMM_ARRAY_K: for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
                #pragma HLS UNROLL
                accum += (typename CONFIG_T::accum_t)(a_rows[m][k] * b_cols[n][k]);
            }
            accum += biases[n];
            // Write the output element directly: the io_parallel caller partitions
            // results[] complete, and a whole-row nnet::array operator= copy under the
            // pipelined M loop is not a transformable instruction for Vitis HLS.
            results[m][n] = accum;
        }
    }
}

// gemm_array_weightless — io_parallel, constant operand held by the IP. No weight
// argument: the columns come from CONFIG_T::gemm_weight_cols() (contract signature).
template <class a_row_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_array_weightless(a_row_T a_rows[CONFIG_T::gemm_m], res_row_T results[CONFIG_T::gemm_m],
                           bias_T biases[CONFIG_T::gemm_n]) {
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit
    typename CONFIG_T::weight_col_t *weight_cols = CONFIG_T::gemm_weight_cols();
    GEMM_AWL_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        GEMM_AWL_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t accum = 0;
            GEMM_AWL_K: for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
                #pragma HLS UNROLL
                accum += (typename CONFIG_T::accum_t)(a_rows[m][k] * weight_cols[n][k]);
            }
            accum += biases[n];
            // Direct element write (see gemm_array): avoids the whole-row operator=
            // copy that Vitis cannot transform under complete partition + pipelined M.
            results[m][n] = accum;
        }
    }
}

// gemm_stream — io_stream, TWO activation operands. B is read into local storage
// (operand residency), then C = A * B streams out.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
                 hls::stream<res_T> &res_stream, typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    static_assert(data0_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(data1_T::size == CONFIG_T::gemm_k, "B column height must equal gemm_k.");
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

    data1_T b_cols[CONFIG_T::gemm_n];
    GEMM_STREAM_READB: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
        #pragma HLS PIPELINE II=1
        b_cols[n] = b_stream.read();
    }
    GEMM_STREAM_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        data0_T a_row = a_stream.read();
        res_T c_row;
        GEMM_STREAM_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t accum = 0;
            GEMM_STREAM_K: for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
                #pragma HLS UNROLL
                accum += (typename CONFIG_T::accum_t)(a_row[k] * b_cols[n][k]);
            }
            accum += biases[n];
            c_row[n] = accum;
        }
        res_stream.write(c_row);
    }
}

// gemm_stream_weightless — io_stream, constant operand held by the IP. No weight
// argument (contract signature): columns come from CONFIG_T::gemm_weight_cols(). The
// input may arrive as several narrower beats (gemm_k / data_T::size) that are packed
// into a gemm_k-wide row, mirroring the hls4ml behavioral entry.
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_weightless(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
                            typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit
    typedef nnet::array<typename data_T::value_type, CONFIG_T::gemm_k> a_row_T;
    typename CONFIG_T::weight_col_t *weight_cols = CONFIG_T::gemm_weight_cols();

    GEMM_SWL_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        a_row_T a_row;
        GEMM_SWL_READA: for (unsigned kp = 0; kp < CONFIG_T::gemm_k / data_T::size; kp++) {
            #pragma HLS UNROLL
            data_T a_pack = data_stream.read();
            for (unsigned k = 0; k < data_T::size; k++) {
                #pragma HLS UNROLL
                a_row[kp * data_T::size + k] = a_pack[k];
            }
        }
        res_T c_row;
        GEMM_SWL_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t accum = 0;
            GEMM_SWL_K: for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
                #pragma HLS UNROLL
                accum += (typename CONFIG_T::accum_t)(a_row[k] * weight_cols[n][k]);
            }
            accum += biases[n];
            c_row[n] = accum;
        }
        res_stream.write(c_row);
    }
}

} // namespace nnet
"""


def combined_header():
    """The whole-model integration header included by the firmware when
    GEMM_IP_HEADER is set: the four contract entry points, one template each,
    shape-generic. Deliberately includes no nnet_types.h — the firmware already
    provides nnet::array / hls::stream — so there is no type redefinition."""
    return (
        "#ifndef GEMM_IP_COMBINED_H_\n"
        "#define GEMM_IP_COMBINED_H_\n\n"
        "#include <hls_stream.h>\n"
        f"{_GEMM_IP_COMBINED_FUNCS}\n"
        "#endif // GEMM_IP_COMBINED_H_\n"
    )


def config_header(name, m, k, n, input_precision=None, weight_precision=None,
                  output_precision=None, bias_precision=None, accum_precision=None,
                  weights_in_core=False, strategy="latency", reuse_factor=1):
    """Per-shape config struct + beat typedefs."""
    rf = max(1, int(reuse_factor or 1))
    mult_limit = -(-(int(k) * int(n)) // rf)   # ceil(gemm_k*gemm_n / reuse_factor)
    input_t = _ap_type(input_precision, "ap_fixed<16,6>")
    weight_t = _ap_type(weight_precision, input_t)
    result_t = _ap_type(output_precision, "ap_fixed<16,6>")
    bias_t = _ap_type(bias_precision, result_t)
    accum_t = _ap_type(accum_precision, "ap_fixed<32,12>")
    return f"""#ifndef {name.upper()}_CONFIG_H_
#define {name.upper()}_CONFIG_H_

#include <ap_fixed.h>
#include <ap_int.h>
#include "nnet_types.h"

typedef {input_t} {name}_input_t;
typedef {weight_t} {name}_weight_t;
typedef {result_t} {name}_result_t;
typedef {bias_t} {name}_bias_t;

typedef nnet::array<{name}_input_t, {k}> {name}_a_row_t;
typedef nnet::array<{name}_weight_t, {k}> {name}_b_col_t;
typedef nnet::array<{name}_result_t, {n}> {name}_res_row_t;

struct {name}_config {{
    static const unsigned n_in      = {k};
    static const unsigned n_out     = {n};
    static const unsigned n_patches = {m};
    static const unsigned gemm_m    = {m};
    static const unsigned gemm_k    = {k};
    static const unsigned gemm_n    = {n};
    static const bool transpose_weights = true;
    static const unsigned reuse_factor = {rf};
    static const unsigned multiplier_limit = {mult_limit};
    typedef {name}_bias_t bias_t;
    typedef {accum_t} accum_t;
}};

#endif // {name.upper()}_CONFIG_H_
"""


def weights_header(name, m, k, n, weight_matrix):
    """Column-major weight ROM + gemm_weight_cols() accessor (weight-stationary).

    ``weight_matrix`` is B shaped ``[K, N]``; column n is ``B[:, n]`` so
    ``weight_cols[n][k] == B[k][n]``.
    """
    cols = []
    for col in range(n):
        vals = ", ".join(str(int(weight_matrix[row][col])) for row in range(k))
        cols.append(f"    {{ {{ {vals} }} }}")
    rom = ",\n".join(cols)
    return f"""#ifndef {name.upper()}_WEIGHTS_H_
#define {name.upper()}_WEIGHTS_H_

#include "{name}_config.h"

// Weight-stationary ROM: column-major, weight_cols[n][k] == B[k][n]. The top feeds
// this array into the weightless entry point (weights are the IP's own, baked here).
static {name}_b_col_t {name}_weight_cols_rom[{n}] = {{
{rom}
}};

#endif // {name.upper()}_WEIGHTS_H_
"""


def top_cpp(name, m, k, n, interface="array", weights_in_core=False):
    """The Vitis synthesis top (set_top) that calls the chosen entry point."""
    inc_w = f'#include "{name}_weights.h"\n' if weights_in_core else ""
    head = (
        f'#include "{name}_config.h"\n'
        f'#include "{name}_gemm_ip.h"\n'
        f"{inc_w}\n"
    )
    if interface == "array" and not weights_in_core:
        return head + f"""void {name}(
    {name}_a_row_t a_rows[{m}],
    {name}_b_col_t b_cols[{n}],
    {name}_res_row_t results[{m}],
    {name}_config::bias_t biases[{n}]
) {{
    nnet::gemm_array<{name}_a_row_t, {name}_b_col_t, {name}_config::bias_t,
                     {name}_res_row_t, {name}_config>(a_rows, b_cols, results, biases);
}}
"""
    if interface == "array" and weights_in_core:
        return head + f"""void {name}(
    {name}_a_row_t a_rows[{m}],
    {name}_res_row_t results[{m}],
    {name}_config::bias_t biases[{n}]
) {{
    nnet::gemm_array_weightless<{name}_a_row_t, {name}_b_col_t, {name}_config::bias_t,
                                {name}_res_row_t, {name}_config>(
        a_rows, {name}_weight_cols_rom, results, biases);
}}
"""
    if interface == "stream" and not weights_in_core:
        return head + f"""void {name}(
    hls::stream<{name}_a_row_t> &a_stream,
    hls::stream<{name}_b_col_t> &b_stream,
    hls::stream<{name}_res_row_t> &res_stream,
    {name}_config::bias_t biases[{n}]
) {{
    nnet::gemm_stream<{name}_a_row_t, {name}_b_col_t, {name}_res_row_t, {name}_config>(
        a_stream, b_stream, res_stream, biases);
}}
"""
    # stream + weights_in_core
    return head + f"""void {name}(
    hls::stream<{name}_a_row_t> &a_stream,
    hls::stream<{name}_res_row_t> &res_stream,
    {name}_config::bias_t biases[{n}]
) {{
    nnet::gemm_stream_weightless<{name}_a_row_t, {name}_b_col_t, {name}_res_row_t, {name}_config>(
        a_stream, {name}_weight_cols_rom, res_stream, biases);
}}
"""
