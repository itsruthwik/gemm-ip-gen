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

_GEMM_IP_FUNCS = r"""
namespace nnet {

// gemm_array — io_parallel, TWO activation operands (e.g. attention QK^T / A.V).
template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_array(a_row_T a_rows[CONFIG_T::gemm_m], b_col_T b_cols[CONFIG_T::gemm_n],
                res_row_T results[CONFIG_T::gemm_m], bias_T biases[CONFIG_T::gemm_n]) {
    GEMM_ARRAY_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        res_row_T c_row;
        GEMM_ARRAY_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS PIPELINE II=1
            typename CONFIG_T::accum_t accum = 0;
            GEMM_ARRAY_K: for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
                #pragma HLS UNROLL
                accum += (typename CONFIG_T::accum_t)(a_rows[m][k] * b_cols[n][k]);
            }
            accum += biases[n];
            c_row[n] = accum;
        }
        results[m] = c_row;
    }
}

// gemm_array_weightless — io_parallel, weights baked into the IP. The weight ROM
// is passed in by the top from <name>_weights.h (not an external port, and not
// routed through CONFIG_T).
template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_array_weightless(a_row_T a_rows[CONFIG_T::gemm_m], b_col_T weight_cols[CONFIG_T::gemm_n],
                           res_row_T results[CONFIG_T::gemm_m], bias_T biases[CONFIG_T::gemm_n]) {
    GEMM_AWL_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        res_row_T c_row;
        GEMM_AWL_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS PIPELINE II=1
            typename CONFIG_T::accum_t accum = 0;
            GEMM_AWL_K: for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
                #pragma HLS UNROLL
                accum += (typename CONFIG_T::accum_t)(a_rows[m][k] * weight_cols[n][k]);
            }
            accum += biases[n];
            c_row[n] = accum;
        }
        results[m] = c_row;
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

    data1_T b_cols[CONFIG_T::gemm_n];
    GEMM_STREAM_READB: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
        #pragma HLS PIPELINE II=1
        b_cols[n] = b_stream.read();
    }
    GEMM_STREAM_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        data0_T a_row = a_stream.read();
        res_T c_row;
        GEMM_STREAM_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS PIPELINE II=1
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

    GEMM_SWL_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        data_T a_row = data_stream.read();
        res_T c_row;
        GEMM_SWL_N: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            #pragma HLS PIPELINE II=1
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
    """The four synthesizable entry points (shape-independent)."""
    return (
        f"#ifndef {name.upper()}_GEMM_IP_H_\n"
        f"#define {name.upper()}_GEMM_IP_H_\n\n"
        "#include <ap_fixed.h>\n"
        "#include <ap_int.h>\n"
        "#include <hls_stream.h>\n"
        '#include "nnet_types.h"\n'
        f"{_GEMM_IP_FUNCS}\n"
        f"#endif // {name.upper()}_GEMM_IP_H_\n"
    )


def config_header(name, m, k, n, input_precision=None, weight_precision=None,
                  output_precision=None, bias_precision=None, accum_precision=None,
                  weights_in_core=False):
    """Per-shape config struct + beat typedefs."""
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
