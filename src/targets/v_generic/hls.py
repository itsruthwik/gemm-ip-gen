"""C++ emitters for the `v-generic` behavioral-HLS GEMM target (tool: Vitis HLS).

No RTL blackbox / hardblock: these emit plain synthesizable C++ that Vitis HLS
turns into RTL. The four entry points mirror the hls4ml Vitis seam
(``nnet::gemm_{stream,stream_const_weights,array,array_const_weights}``) and are the
behavioral triple-loop from ``nnet_gemm_behavioral.h`` promoted to synthesizable
form (HLS pragmas, ``ap_fixed`` arithmetic, ``nnet::array`` beats, per-shape
``CONFIG_T``). One templated definition covers every shape.

One bias contract, one kernel body per entry shape
-----------------------------------------------------------------
The four entry points (``gemm_array``, ``gemm_array_const_weights``, ``gemm_stream``,
``gemm_stream_const_weights``) have exactly one body (``_GEMM_IP_COMBINED_FUNCS``,
the resource core), shared by both callers:

* ``combined_header()`` -- the whole-model hls4ml integration -- embeds it under
  the entry points' own names (no per-layer dispatch needed: every layer uses the
  same body). The ``gemm_rf`` / ``gemm_ip_has_bias`` traits still key per layer on
  ``CONFIG_T::gemm_ip_id`` for reuse factor and bias.
* ``gemm_ip_header()`` -- the standalone self-test package built by
  ``generate_generic_pkg`` -- embeds the same body under the entry points' own
  names (a standalone package is always exactly one layer).

Both callers' ``CONFIG_T`` carry the same accessor contract: the const-weight
entries take no weight/bias arguments at all, reading
``CONFIG_T::gemm_weight_beats()`` / ``CONFIG_T::gemm_bias()`` instead (the
standalone ``config_header()`` backs these with the same baked
``<name>_weights.h`` / ``<name>_bias.h`` ROMs the old explicit-argument signature
used to wire in from ``top_cpp()``), and gate the add on the same
``gemm_ip_has_bias<CONFIG_T::gemm_ip_id>`` trait (``combined_header()`` builds one
specialization per manifest layer; ``gemm_ip_header()`` builds the single
specialization its one layer needs). The two-operand entries in both callers
never take a bias at all (a two-operand GEMM never owns one) and share the
``gemm_row_resource`` regime core (``_GEMM_ROW_RESOURCE_CORE``).
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



# ── Resource-kernel per-row core (mirrors hls4ml's nnet_dense_resource.h) ─────────
#
# hls4ml's dense_resource picks one of three schedules by reuse_factor (rf) vs n_in
# (== gemm_k): rf<=n_in, rf>n_in with rf%n_in==0, or the general remainder case. Each
# unrolls `block_factor = ceil(n_in*n_out/rf)` multipliers per ReuseLoop iteration (rf
# iterations, PIPELINE II=1). This is the same recurrence, transliterated variable-for-
# variable (n_in->gemm_k, n_out->gemm_n, data->a_row, res->c_row) with ONE change: the
# baseline's flat row-major `weights[index]` (index = k*n_out + n) becomes a beat array:
# column-major `weight_cols[n][k]` (== B[k][n], the default) or, when the layer's ROM was
# packed under SecondOperandRowMajor (ROW_MAJOR template flag == CONFIG_T::weights_row_major
# in the combined header), row-major `weight_cols[k][n]`. Wherever baseline reads
# weights[index], here we do `k = index / gemm_n; n = index % gemm_n;` and pick the
# beat index order by ROW_MAJOR.
#
# One core (three regime functions + a compile-time dispatcher) is shared by all four
# resource entry points in each header (standalone and combined each get their own
# copy, since the two headers are textually independent) to avoid a 4x duplication of
# the three regimes.
_GEMM_ROW_RESOURCE_CORE = r"""
template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T, bool ROW_MAJOR = false, bool HAS_BIAS = true>
void gemm_row_resource_rf_leq_nin(a_row_T &a_row, b_col_T weight_cols[ROW_MAJOR ? CONFIG_T::gemm_k : CONFIG_T::gemm_n],
                                   bias_T *biases, res_row_T &c_row) {
    const int rufactor = gemm_rf<CONFIG_T>::reuse_factor;
    const int nin = CONFIG_T::gemm_k;
    const int nout = CONFIG_T::gemm_n;
    const int multfactor = rufactor < nin ? rufactor : nin;
    const int multiplier_limit = (nin * nout + multfactor - 1) / multfactor;
    const int block_factor = (nin * nout + rufactor - 1) / rufactor;
    const int multscale = multiplier_limit / nout;

    #pragma HLS INLINE off
    #pragma HLS function_instantiate variable=weight_cols,biases
    // The caller's row buffers are nnet::array structs; partition their element arrays
    // here (dim=0) so the unrolled result/operand accesses are registers, not a RAM port
    // bound (at N=128 a 2-port RAM c_row capped the reuse loop at II=64).
    #pragma HLS ARRAY_PARTITION variable=c_row complete dim=0
    #pragma HLS ARRAY_PARTITION variable=a_row complete dim=0
    typename CONFIG_T::accum_t acc[CONFIG_T::gemm_n];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    // Mirror hls4ml's Vitis nnet_dense_resource override: pin the accumulate add to
    // fabric so Versal Vitis HLS does not fuse the multiply-accumulate into a
    // multi-cycle DSP58 dot-product primitive (which silently diverges from the C
    // model in cosim), and bind each product to a plain DSP multiplier below.
    // TODO: Ruthwik check again
    #pragma HLS bind_op variable=acc op=add impl=fabric

    GRR_LEQ_INIT: for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = HAS_BIAS ? (typename CONFIG_T::accum_t)biases[iacc] : (typename CONFIG_T::accum_t)0;
    }

    GRR_LEQ_REUSE: for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        int w_index = ir;
        int in_index = ir;
        int out_index = 0;
        int acc_step = 0;
        GRR_LEQ_MULT: for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            // weights[w_index] (flat [n_in][n_out]) -> weight_cols[n][k]; the
            // recurrence already tracks (in_index, out_index) == (k, n) for the
            // pair weights[w_index] semantically belongs to, so no decode of
            // w_index itself is needed (only used here for its own recurrence).
            typename CONFIG_T::accum_t mult = (typename CONFIG_T::accum_t)(a_row[in_index] * (ROW_MAJOR ? weight_cols[in_index][out_index] : weight_cols[out_index][in_index]));
            // TODO: Ruthwik check again
            #pragma HLS bind_op variable=mult op=mul impl=dsp
            acc[out_index] += mult;
            w_index += rufactor;
            in_index += rufactor;
            if (in_index >= nin) in_index = ir;
            if (acc_step + 1 >= multscale) { acc_step = 0; out_index++; }
            else acc_step++;
        }
    }

    GRR_LEQ_RESULT: for (int ires = 0; ires < nout; ires++) {
        #pragma HLS UNROLL
        c_row[ires] = acc[ires];
    }
}

template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T, bool ROW_MAJOR = false, bool HAS_BIAS = true>
void gemm_row_resource_rf_gt_nin_rem0(a_row_T &a_row, b_col_T weight_cols[ROW_MAJOR ? CONFIG_T::gemm_k : CONFIG_T::gemm_n],
                                       bias_T *biases, res_row_T &c_row) {
    const int nin = CONFIG_T::gemm_k;
    const int nout = CONFIG_T::gemm_n;
    const int rufactor = gemm_rf<CONFIG_T>::reuse_factor < nin * nout ? gemm_rf<CONFIG_T>::reuse_factor : nin * nout;
    const int block_factor = (nin * nout + gemm_rf<CONFIG_T>::reuse_factor - 1) / gemm_rf<CONFIG_T>::reuse_factor;

    #pragma HLS INLINE off
    #pragma HLS function_instantiate variable=weight_cols,biases
    // The caller's row buffers are nnet::array structs; partition their element arrays
    // here (dim=0) so the unrolled result/operand accesses are registers, not a RAM port
    // bound (at N=128 a 2-port RAM c_row capped the reuse loop at II=64).
    #pragma HLS ARRAY_PARTITION variable=c_row complete dim=0
    #pragma HLS ARRAY_PARTITION variable=a_row complete dim=0
    typename CONFIG_T::accum_t acc[CONFIG_T::gemm_n];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    // Mirror hls4ml's Vitis nnet_dense_resource override: pin the accumulate add to
    // fabric so Versal Vitis HLS does not fuse the multiply-accumulate into a
    // multi-cycle DSP58 dot-product primitive (which silently diverges from the C
    // model in cosim), and bind each product to a plain DSP multiplier below.
    // TODO: Ruthwik check again
    #pragma HLS bind_op variable=acc op=add impl=fabric

    GRR_REM0_INIT: for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = HAS_BIAS ? (typename CONFIG_T::accum_t)biases[iacc] : (typename CONFIG_T::accum_t)0;
    }

    int in_index = 0;
    int outstep = 0;
    const int outscale = rufactor / nin;

    int outidx[/*rufactor*/ gemm_rf<CONFIG_T>::reuse_factor];
    GRR_REM0_INDEX: for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        if ((ir + 1) % nin == 0) outstep++;
    }

    GRR_REM0_REUSE: for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        int w_index = ir;
        int out_index = outidx[ir];
        GRR_REM0_MULT: for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            // in_index is fixed for the whole MultLoop (one k per ir); out_index
            // is the tracked n. weights[w_index] -> weight_cols[out_index][in_index].
            typename CONFIG_T::accum_t mult = (typename CONFIG_T::accum_t)(a_row[in_index] * (ROW_MAJOR ? weight_cols[in_index][out_index] : weight_cols[out_index][in_index]));
            // TODO: Ruthwik check again
            #pragma HLS bind_op variable=mult op=mul impl=dsp
            acc[out_index] += mult;
            w_index += rufactor;
            if (w_index >= nin * nout) break;
            out_index += outscale;
        }
        in_index++;
        if (in_index >= nin) in_index = 0;
    }

    GRR_REM0_RESULT: for (int ires = 0; ires < nout; ires++) {
        #pragma HLS UNROLL
        c_row[ires] = acc[ires];
    }
}

template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T, bool ROW_MAJOR = false, bool HAS_BIAS = true>
void gemm_row_resource_rf_gt_nin(a_row_T &a_row, b_col_T weight_cols[ROW_MAJOR ? CONFIG_T::gemm_k : CONFIG_T::gemm_n],
                                  bias_T *biases, res_row_T &c_row) {
    const int rufactor = gemm_rf<CONFIG_T>::reuse_factor;
    const int nin = CONFIG_T::gemm_k;
    const int nout = CONFIG_T::gemm_n;
    const int multfactor = rufactor < nin ? rufactor : nin;
    const int multiplier_limit = (nin * nout + multfactor - 1) / multfactor;
    const int block_factor = (nin * nout + rufactor - 1) / rufactor;

    #pragma HLS INLINE off
    #pragma HLS function_instantiate variable=weight_cols,biases
    // The caller's row buffers are nnet::array structs; partition their element arrays
    // here (dim=0) so the unrolled result/operand accesses are registers, not a RAM port
    // bound (at N=128 a 2-port RAM c_row capped the reuse loop at II=64).
    #pragma HLS ARRAY_PARTITION variable=c_row complete dim=0
    #pragma HLS ARRAY_PARTITION variable=a_row complete dim=0
    typename CONFIG_T::accum_t acc[CONFIG_T::gemm_n];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    // NB: unlike the rf<=nin kernels, here acc accumulates the already-reduced mult[]
    // partials (not a product directly), so acc's add has no adjacent multiply to fuse
    // into a DSP58 dot-product -- the fabric pin goes on the mult[] reduction and the
    // dsp pin on tmpmult below, matching hls4ml's own rf_gt_nin dense_resource kernel.

    GRR_GT_INIT: for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = HAS_BIAS ? (typename CONFIG_T::accum_t)biases[iacc] : (typename CONFIG_T::accum_t)0;
    }

    GRR_GT_REUSE: for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        typename CONFIG_T::accum_t tmpmult[/*block_factor*/ (CONFIG_T::gemm_k * CONFIG_T::gemm_n
                                                               + gemm_rf<CONFIG_T>::reuse_factor - 1)
                                                              / gemm_rf<CONFIG_T>::reuse_factor];
        #pragma HLS ARRAY_PARTITION variable=tmpmult complete
        // TODO: Ruthwik check again
        #pragma HLS bind_op variable=tmpmult op=mul impl=dsp

        GRR_GT_MULT: for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int in_index = w_index % nin;
            if (w_index >= nin * nout) continue;
            // multfactor == nin here (rufactor > nin), so w_index/multfactor == the
            // output index n (matches out_index below, which uses the same divisor).
            int n = w_index / multfactor;
            tmpmult[im] = (typename CONFIG_T::accum_t)(a_row[in_index] * (ROW_MAJOR ? weight_cols[in_index][n] : weight_cols[n][in_index]));
        }

        typename CONFIG_T::accum_t mult[/*multiplier_limit*/ (CONFIG_T::gemm_k * CONFIG_T::gemm_n
                                                                + ((gemm_rf<CONFIG_T>::reuse_factor < CONFIG_T::gemm_k)
                                                                   ? gemm_rf<CONFIG_T>::reuse_factor : CONFIG_T::gemm_k) - 1)
                                                               / ((gemm_rf<CONFIG_T>::reuse_factor < CONFIG_T::gemm_k)
                                                                  ? gemm_rf<CONFIG_T>::reuse_factor : CONFIG_T::gemm_k)];
        #pragma HLS ARRAY_PARTITION variable=mult complete
        // TODO: Ruthwik check again
        #pragma HLS bind_op variable=mult op=add impl=fabric

        GRR_GT_RESETMULT: for (int imult = 0; imult < multiplier_limit; imult++) {
            #pragma HLS UNROLL
            mult[imult] = 0;
        }

        GRR_GT_ACC1: for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int out_index = w_index / multfactor;
            if (out_index >= multiplier_limit) continue;
            mult[out_index] += tmpmult[im];
        }

        GRR_GT_ACC2: for (int im = 0; im < multiplier_limit; im++) {
            #pragma HLS UNROLL
            // If RF > N_IN then multiplier_limit == n_out (see baseline).
            acc[im] += mult[im];
        }
    }

    GRR_GT_RESULT: for (int ires = 0; ires < nout; ires++) {
        #pragma HLS UNROLL
        c_row[ires] = acc[ires];
    }
}

// gemm_row_resource — dispatches to the regime matching gemm_rf<CONFIG_T>::reuse_factor vs
// CONFIG_T::gemm_k, exactly as hls4ml's dense_resource dispatches on n_in. Reused by
// gemm_array/_const_weights/gemm_stream/_const_weights below (M loop wraps this once
// per row; no PIPELINE on the M loop itself -- the II=1 pipelining is on the inner
// ReuseLoop inside each regime, per hls4ml's own structure).
template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T, bool ROW_MAJOR = false, bool HAS_BIAS = true>
void gemm_row_resource(a_row_T &a_row, b_col_T weight_cols[ROW_MAJOR ? CONFIG_T::gemm_k : CONFIG_T::gemm_n],
                        bias_T *biases, res_row_T &c_row) {
    #pragma HLS INLINE off
    if (gemm_rf<CONFIG_T>::reuse_factor <= CONFIG_T::gemm_k) {
        gemm_row_resource_rf_leq_nin<a_row_T, b_col_T, bias_T, res_row_T, CONFIG_T, ROW_MAJOR, HAS_BIAS>(
            a_row, weight_cols, biases, c_row);
    } else if (gemm_rf<CONFIG_T>::reuse_factor % CONFIG_T::gemm_k == 0) {
        gemm_row_resource_rf_gt_nin_rem0<a_row_T, b_col_T, bias_T, res_row_T, CONFIG_T, ROW_MAJOR, HAS_BIAS>(
            a_row, weight_cols, biases, c_row);
    } else {
        gemm_row_resource_rf_gt_nin<a_row_T, b_col_T, bias_T, res_row_T, CONFIG_T, ROW_MAJOR, HAS_BIAS>(
            a_row, weight_cols, biases, c_row);
    }
}
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


_GEMM_RF_PASSTHROUGH = r'''namespace nnet {
// gemm_rf<CONFIG_T>: the reuse factor / multiplier cap the kernels schedule against.
// Standalone packages have one config per package, so this is a plain pass-through of
// CONFIG_T; the whole-model combined header overrides it per layer (see
// _gemm_rf_trait) with the value the runner validated/snapped in the manifest.
template <typename CONFIG_T> struct gemm_rf {
    static const unsigned reuse_factor = CONFIG_T::reuse_factor;
    static const unsigned multiplier_limit = CONFIG_T::multiplier_limit;
};
} // namespace nnet
'''


def gemm_ip_header(name, has_bias=True):
    """The four synthesizable entry points (shape-generic).

    Embeds the exact same kernel bodies the whole-model combined header uses
    (``_GEMM_IP_COMBINED_FUNCS``, the resource core) under the entry points' own
    names -- a standalone package is always exactly one layer, so no runtime
    dispatch is needed here. Bias comes from ``CONFIG_T::gemm_bias()`` (which
    ``config_header()`` backs with the baked ``<name>_bias_rom``) and, for the
    const-weight entries, the weight ROM comes from ``CONFIG_T::gemm_weight_beats()``
    the same way. ``has_bias`` (default True, matching the previous always-add
    behavior of this standalone self-test path) picks whether the one
    ``gemm_ip_id`` this package defines gets a ``gemm_ip_has_bias`` specialization
    of ``false`` -- exactly the trait the combined header's const-weight kernels
    gate on, so a standalone package generated with ``has_bias=False`` proves the
    same no-add code path a whole-model build would take for that layer."""
    funcs = _GEMM_IP_COMBINED_FUNCS
    has_bias_trait = (
        "namespace nnet {\n"
        "template <unsigned id> struct gemm_ip_has_bias { static const bool value = true; };\n"
        + ("" if has_bias else
           "template <> struct gemm_ip_has_bias<0> { static const bool value = false; };\n")
        + "} // namespace nnet\n"
    )
    return (
        f"#ifndef {name.upper()}_GEMM_IP_H_\n"
        f"#define {name.upper()}_GEMM_IP_H_\n\n"
        "#include <ap_fixed.h>\n"
        "#include <ap_int.h>\n"
        "#include <hls_stream.h>\n"
        '#include "nnet_types.h"\n'
        f"{_GEMM_RF_PASSTHROUGH}\n"
        f"{has_bias_trait}\n"
        f"{funcs}\n"
        f"#endif // {name.upper()}_GEMM_IP_H_\n"
    )


# ── Whole-model combined header (the hls4ml Vitis integration seam) ───────────────
#
# Unlike the per-shape RTL blackbox targets, the generic funcs are one templated
# definition that covers every layer, so there is no per-core dispatch: the four
# entry points below ARE the combined header. Their signatures match the hls4ml
# contract exactly (the "declaration only" prototypes in nnet_gemm_ip.h /
# nnet_gemm_stream.h): the const_weights entries take NO weight argument and source
# the constant operand from CONFIG_T::gemm_weight_beats() (the ROM the writer
# injects into each layer's CONFIG_T, column- or row-major per
# SecondOperandRowMajor), and gemm_stream_const_weights unpacks narrow input beats
# into a gemm_k-wide row. Header-only + synthesizable -> no add_files (see sources
# tcl). Built on the shared gemm_row_resource core (three reuse-factor regimes,
# mirroring hls4ml's nnet_dense_resource dispatch on rf vs n_in); ReuseFactor
# reaches the core through gemm_rf<CONFIG_T> (hls4ml injects it into each layer's
# gemm config; the generic target validates/snaps it, see gemm_rf_override below).
_GEMM_IP_COMBINED_FUNCS = r"""
namespace nnet {
""" + _GEMM_ROW_RESOURCE_CORE + r"""

// gemm_array — io_parallel, TWO activation operands. Extrapolation beyond baseline
// (dense_resource has no two-operand kernel): b_cols is already resident, so the
// per-row resource core runs over it directly, treating b_cols as weight_cols. No bias
// (contract signature; a two-operand GEMM never owns one) — gemm_row_resource still
// wants a bias operand, so a zero constant stands in for it.
template <class a_row_T, class b_col_T, class res_row_T, typename CONFIG_T>
void gemm_array(a_row_T a_rows[CONFIG_T::gemm_m], b_col_T b_cols[CONFIG_T::gemm_n],
                res_row_T results[CONFIG_T::gemm_m]) {
    GEMM_ARRAY_RES_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        gemm_row_resource<a_row_T, b_col_T, typename CONFIG_T::accum_t, res_row_T, CONFIG_T,
                          /*ROW_MAJOR=*/false, /*HAS_BIAS=*/false>(
            a_rows[m], b_cols, nullptr, results[m]);
    }
}

// gemm_array_const_weights — io_parallel, constant operand from CONFIG_T::gemm_weight_beats()
// (contract signature: no weight argument; beat layout per CONFIG_T::weights_row_major).
// Bias, like the weight ROM, is read through the config (CONFIG_T::gemm_bias()) rather
// than a function parameter. Same ARRAY_RESHAPE / BIND_STORAGE pragmas as baseline
// dense_resource, applied to weight_cols.
template <class a_row_T, class res_row_T, typename CONFIG_T>
void gemm_array_const_weights(a_row_T a_rows[CONFIG_T::gemm_m], res_row_T results[CONFIG_T::gemm_m]) {
    typename CONFIG_T::weight_beat_t *weight_cols = CONFIG_T::gemm_weight_beats();
    typename CONFIG_T::bias_t *biases = CONFIG_T::gemm_bias();
    const int block_factor = (CONFIG_T::gemm_k * CONFIG_T::gemm_n + gemm_rf<CONFIG_T>::reuse_factor - 1)
                              / gemm_rf<CONFIG_T>::reuse_factor;
    #pragma HLS ARRAY_RESHAPE   variable=weight_cols block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete
    if (gemm_rf<CONFIG_T>::reuse_factor > 1) {
        #pragma HLS BIND_STORAGE variable=weight_cols type=rom_np impl=bram
    }
    GEMM_AWL_RES_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        gemm_row_resource<a_row_T, typename CONFIG_T::weight_beat_t, typename CONFIG_T::bias_t, res_row_T, CONFIG_T,
                          CONFIG_T::weights_row_major, gemm_ip_has_bias<CONFIG_T::gemm_ip_id>::value>(
            a_rows[m], weight_cols, biases, results[m]);
    }
}

// gemm_stream — io_stream, TWO activation operands (extrapolation, see gemm_array above).
// No bias (contract signature) — same zero-constant stand-in as gemm_array.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
                 hls::stream<res_T> &res_stream) {
    static_assert(data0_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(data1_T::size == CONFIG_T::gemm_k, "B column height must equal gemm_k.");
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");

    data1_T b_cols[CONFIG_T::gemm_n];
    GEMM_STREAM_RES_READB: for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
        #pragma HLS PIPELINE II=1
        b_cols[n] = b_stream.read();
    }
    GEMM_STREAM_RES_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        data0_T a_row = a_stream.read();
        res_T c_row;
        gemm_row_resource<data0_T, data1_T, typename CONFIG_T::accum_t, res_T, CONFIG_T,
                          /*ROW_MAJOR=*/false, /*HAS_BIAS=*/false>(
            a_row, b_cols, nullptr, c_row);
        res_stream.write(c_row);
    }
}

// gemm_stream_const_weights — io_stream, constant operand from CONFIG_T::gemm_weight_beats()
// (beat layout per CONFIG_T::weights_row_major)
// (contract signature). Bias is read through the config (CONFIG_T::gemm_bias()) rather
// than a function parameter. Input may arrive as several narrower beats packed into a
// gemm_k-wide row, mirroring the array const-weights entry above.
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream) {
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");
    typedef nnet::array<typename data_T::value_type, CONFIG_T::gemm_k> a_row_T;
    typename CONFIG_T::weight_beat_t *weight_cols = CONFIG_T::gemm_weight_beats();
    typename CONFIG_T::bias_t *biases = CONFIG_T::gemm_bias();
    const int block_factor = (CONFIG_T::gemm_k * CONFIG_T::gemm_n + gemm_rf<CONFIG_T>::reuse_factor - 1)
                              / gemm_rf<CONFIG_T>::reuse_factor;
    #pragma HLS ARRAY_RESHAPE   variable=weight_cols block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete
    if (gemm_rf<CONFIG_T>::reuse_factor > 1) {
        #pragma HLS BIND_STORAGE variable=weight_cols type=rom_np impl=bram
    }

    GEMM_SWL_RES_M: for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        a_row_T a_row;
        GEMM_SWL_RES_READA: for (unsigned kp = 0; kp < CONFIG_T::gemm_k / data_T::size; kp++) {
            #pragma HLS UNROLL
            data_T a_pack = data_stream.read();
            for (unsigned k = 0; k < data_T::size; k++) {
                #pragma HLS UNROLL
                a_row[kp * data_T::size + k] = a_pack[k];
            }
        }
        res_T c_row;
        gemm_row_resource<a_row_T, typename CONFIG_T::weight_beat_t, typename CONFIG_T::bias_t, res_T, CONFIG_T,
                          CONFIG_T::weights_row_major, gemm_ip_has_bias<CONFIG_T::gemm_ip_id>::value>(
            a_row, weight_cols, biases, c_row);
        res_stream.write(c_row);
    }
}

} // namespace nnet
"""


def _gemm_rf_trait(items):
    """Emit gemm_rf<CONFIG_T> for the combined header: per-layer reuse factor and
    multiplier cap taken from the manifest (which the generic target itself has
    validated/snapped against hls4ml's reuse-factor rules), overriding the raw value
    hls4ml baked into CONFIG_T. Items with no gemm_ip_index or no reuse_factor fall
    through to CONFIG_T."""
    specs = []
    for item in items:
        idx = item.get("gemm_ip_index")
        rf = item.get("reuse_factor")
        if idx is None or rf is None:
            continue
        rf = max(1, int(rf))
        k = int(item.get("k", item.get("gemm_k", item.get("n_in", 0))))
        n = int(item.get("n", item.get("gemm_n", item.get("n_out", 0))))
        mult_limit = -(-(k * n) // rf)
        specs.append(
            f"template <> struct gemm_rf_override<{idx}> {{ static const bool set = true; "
            f"static const unsigned reuse_factor = {rf}; "
            f"static const unsigned multiplier_limit = {mult_limit}; }};")
    specs_txt = "\n".join(specs)
    return (
        "namespace nnet {\n"
        "// Per-layer reuse factor from the manifest (validated/snapped by the generic\n"
        "// target itself), keyed on CONFIG_T::gemm_ip_id; layers without an entry use\n"
        "// CONFIG_T's own.\n"
        "template <unsigned id> struct gemm_rf_override { static const bool set = false; "
        "static const unsigned reuse_factor = 1; static const unsigned multiplier_limit = 1; };\n"
        f"{specs_txt}\n"
        "template <typename CONFIG_T> struct gemm_rf {\n"
        "    static const unsigned reuse_factor = gemm_rf_override<CONFIG_T::gemm_ip_id>::set\n"
        "        ? gemm_rf_override<CONFIG_T::gemm_ip_id>::reuse_factor : CONFIG_T::reuse_factor;\n"
        "    static const unsigned multiplier_limit = gemm_rf_override<CONFIG_T::gemm_ip_id>::set\n"
        "        ? gemm_rf_override<CONFIG_T::gemm_ip_id>::multiplier_limit : CONFIG_T::multiplier_limit;\n"
        "};\n"
        "} // namespace nnet\n"
    )


def _gemm_has_bias_trait(items):
    """Emit the gemm_ip_has_bias<id> compile-time trait: primary = true (has a real
    bias), with a specialization per const-weight item whose manifest has_bias is
    False, so that item's const-weights entry folds away its bias add entirely
    (CONFIG_T::gemm_bias() may still return an all-zero ROM in that case, but the
    resource kernels never read it when the trait says false)."""
    specs = []
    for item in items:
        idx = item.get("gemm_ip_index")
        if idx is None:
            continue
        if item.get("has_bias") is False:
            specs.append(
                f"template <> struct gemm_ip_has_bias<{idx}> {{ "
                f"static const bool value = false; }};")
    specs_txt = "\n".join(specs)
    return (
        "namespace nnet {\n"
        "// Per-layer has_bias from the manifest, keyed on CONFIG_T::gemm_ip_id; layers\n"
        "// with no entry (or has_bias True) default to true (the fallback CONFIG_T::gemm_bias()\n"
        "// accessor still gets called, but every add of its result is gated by this trait).\n"
        "template <unsigned id> struct gemm_ip_has_bias { static const bool value = true; };\n"
        f"{specs_txt}\n"
        "} // namespace nnet\n"
    )


def combined_header(items=None):
    """The whole-model integration header included by the firmware when
    GEMM_IP_HEADER is set: the four contract entry points, one template each,
    shape-generic (resource core only). Deliberately includes no nnet_types.h —
    the firmware already provides nnet::array / hls::stream — so there is no type
    redefinition.

    ``items`` (the manifest's per-layer gemm items) still drives compile-time
    per-layer dispatch for reuse factor (``gemm_rf``) and bias
    (``gemm_ip_has_bias``), keyed on ``CONFIG_T::gemm_ip_id``, but every layer
    shares the same (resource) kernel body -- there is no per-layer strategy
    choice anymore."""
    items = items or []
    funcs = _GEMM_IP_COMBINED_FUNCS
    rf_trait = _gemm_rf_trait(items)
    has_bias_trait = _gemm_has_bias_trait(items)
    return (
        "#ifndef GEMM_IP_COMBINED_H_\n"
        "#define GEMM_IP_COMBINED_H_\n\n"
        "#include <hls_stream.h>\n"
        f"{rf_trait}\n"
        f"{has_bias_trait}\n"
        f"{funcs}\n"
        "#endif // GEMM_IP_COMBINED_H_\n"
    )


def config_header(name, m, k, n, input_precision=None, weight_precision=None,
                  output_precision=None, bias_precision=None, accum_precision=None,
                  weights_in_core=False, reuse_factor=1):
    """Per-shape config struct + beat typedefs.

    Exposes ``gemm_bias()`` (and, when ``weights_in_core``, ``gemm_weight_beats()``)
    on the config struct -- the same compile-time-constant accessor contract the
    whole-model combined header's CONFIG_T carries -- so the single shared kernel
    body in ``gemm_ip_header()`` can source bias/weights from CONFIG_T exactly like
    the combined header does. ``{name}_bias.h`` (and ``{name}_weights.h``) are
    included from inside this header, right after the typedefs they need
    (``{name}_bias_t`` / ``{name}_b_col_t``); those files also include this one, but
    the include guard makes the round trip a no-op, so ordering is inconsequential
    at the call site (``top_cpp`` no longer needs to include them itself)."""
    rf = max(1, int(reuse_factor or 1))
    mult_limit = -(-(int(k) * int(n)) // rf)   # ceil(gemm_k*gemm_n / reuse_factor)
    input_t = _ap_type(input_precision, "ap_fixed<16,6>")
    weight_t = _ap_type(weight_precision, input_t)
    result_t = _ap_type(output_precision, "ap_fixed<16,6>")
    bias_t = _ap_type(bias_precision, result_t)
    accum_t = _ap_type(accum_precision, "ap_fixed<32,12>")
    weights_inc = f'#include "{name}_weights.h"\n' if weights_in_core else ""
    weight_accessor = (
        f"    static weight_beat_t *gemm_weight_beats() {{ return {name}_weight_cols_rom; }}\n"
        if weights_in_core else ""
    )
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

#include "{name}_bias.h"
{weights_inc}
struct {name}_config {{
    static const unsigned n_in      = {k};
    static const unsigned n_out     = {n};
    static const unsigned n_patches = {m};
    static const unsigned gemm_m    = {m};
    static const unsigned gemm_k    = {k};
    static const unsigned gemm_n    = {n};
    // A standalone package is always exactly one layer -- gemm_ip_id is fixed at 0
    // purely so the shared kernel bodies' gemm_ip_has_bias<CONFIG_T::gemm_ip_id>
    // lookup (see gemm_ip_header()) has something to key on; there is no
    // multi-layer dispatch here (contrast the combined header's per-layer ids).
    static const unsigned gemm_ip_id = 0;
    static const bool transpose_weights = true;
    static const bool weights_row_major = false;
    static const unsigned reuse_factor = {rf};
    static const unsigned multiplier_limit = {mult_limit};
    typedef {name}_bias_t bias_t;
    typedef {accum_t} accum_t;
    typedef {name}_weight_t weight_t;
    typedef {name}_b_col_t weight_beat_t;

    static bias_t *gemm_bias() {{ return {name}_bias_rom; }}
{weight_accessor}}};

#endif // {name.upper()}_CONFIG_H_
"""


def bias_header(name, n, bias_t, bias_values):
    """Bias ROM: baked as a compile-time constant, like the weight ROM, so it never
    becomes a top-level port. ``bias_values`` is a list of N reals (or None/empty
    for "no bias", which bakes an all-zero array -- mathematically identical to
    omitting the add, so the top signature never needs a has_bias-gated overload)."""
    vals = list(bias_values) if bias_values else [0] * n
    if len(vals) != n:
        raise ValueError(f"bias length {len(vals)} != N {n}")
    body = ", ".join(str(v) for v in vals)
    return f"""#ifndef {name.upper()}_BIAS_H_
#define {name.upper()}_BIAS_H_

#include "{name}_config.h"

// Bias ROM: baked as a compile-time constant (same mechanism as the weight ROM),
// referenced directly inside the top -- never a function argument, so it never
// becomes a port.
static {bias_t} {name}_bias_rom[{n}] = {{ {body} }};

#endif // {name.upper()}_BIAS_H_
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
// this array into the const_weights entry point (weights are the IP's own, baked here).
static {name}_b_col_t {name}_weight_cols_rom[{n}] = {{
{rom}
}};

#endif // {name.upper()}_WEIGHTS_H_
"""


def top_cpp(name, m, k, n, interface="array", weights_in_core=False):
    """The Vitis synthesis top (set_top) that calls the chosen entry point.

    Bias and (for a const-weight layer) the weight ROM are baked constants (see
    ``bias_header`` / ``weights_header``) the kernel itself reads through
    ``CONFIG_T::gemm_bias()`` / ``CONFIG_T::gemm_weight_beats()`` -- the same
    accessor contract ``combined_header()`` uses -- so neither is a function
    argument here, and neither ever appears as a top-level port.
    """
    head = (
        f'#include "{name}_config.h"\n'
        f'#include "{name}_gemm_ip.h"\n'
    )
    if interface == "array" and not weights_in_core:
        return head + f"""void {name}(
    {name}_a_row_t a_rows[{m}],
    {name}_b_col_t b_cols[{n}],
    {name}_res_row_t results[{m}]
) {{
    nnet::gemm_array<{name}_a_row_t, {name}_b_col_t, {name}_res_row_t, {name}_config>(
        a_rows, b_cols, results);
}}
"""
    if interface == "array" and weights_in_core:
        return head + f"""void {name}(
    {name}_a_row_t a_rows[{m}],
    {name}_res_row_t results[{m}]
) {{
    nnet::gemm_array_const_weights<{name}_a_row_t, {name}_res_row_t, {name}_config>(
        a_rows, results);
}}
"""
    if interface == "stream" and not weights_in_core:
        return head + f"""void {name}(
    hls::stream<{name}_a_row_t> &a_stream,
    hls::stream<{name}_b_col_t> &b_stream,
    hls::stream<{name}_res_row_t> &res_stream
) {{
    nnet::gemm_stream<{name}_a_row_t, {name}_b_col_t, {name}_res_row_t, {name}_config>(
        a_stream, b_stream, res_stream);
}}
"""
    # stream + weights_in_core
    return head + f"""void {name}(
    hls::stream<{name}_a_row_t> &a_stream,
    hls::stream<{name}_res_row_t> &res_stream
) {{
    nnet::gemm_stream_const_weights<{name}_a_row_t, {name}_res_row_t, {name}_config>(
        a_stream, res_stream);
}}
"""
