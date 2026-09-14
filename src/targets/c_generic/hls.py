"""C++ emitters for the `c-generic` behavioral-HLS GEMM target (tool: Catapult).

The Catapult twin of ``targets/v-generic/hls.py``: same public API shape
(``combined_header`` / ``gemm_ip_header`` / ``config_header`` / ``bias_header`` /
``weights_header`` / ``top_cpp``), same four hls4ml-facing entry-point signatures
(``nnet::gemm_array``, ``nnet::gemm_array_const_weights``, ``nnet::gemm_stream``,
``nnet::gemm_stream_const_weights``), but in Catapult idiom: ``ac_fixed`` / ``ac_int``
arithmetic, ``ac_channel`` streams, ``#pragma hls_pipeline_init_interval`` instead of
``#pragma HLS PIPELINE``. No ``#pragma HLS``, no ``hls::stream``, no ``ap_*`` type may
appear anywhere in emitted text (tests grep for all three).

Kernel body
-----------
Rather than re-deriving hls4ml's three reuse-factor regimes the way
``v-generic/hls.py`` does (transliterated into a bespoke ``gemm_row_resource`` core),
this target copies ``nnet_dense_resource.h`` verbatim (its ``tree_sum_t`` and the
three ``dense_resource_rf_*`` regime functions plus the dispatcher) into a
private namespace, ``c_generic_dense``, together with the small slice of
``nnet_common.h`` / ``nnet_mult.h`` it needs to compile standalone (``DIV_ROUNDUP``
/ ``MIN``, ``product::mult``, a plain ``cast<>``) -- so this package never
``#include``s anything from the hls4ml tree. The four contract entry points are
thin row-loop wrappers around ``c_generic_dense::dense_resource`` (the
``nnet_einsum_dense.h`` idiom: unroll the free (M) dimension, call the per-row
dense kernel once per row) plus, for the two streaming entries, the
``nnet_dense_stream.h`` beat-gather/beat-write idiom.

Weight layout
-------------
``dense_resource``'s own index arithmetic (traced from the copied RF regime
loops, not the module's prose) actually wants a flat ``weights[n_in*n_out]``
array indexed **out-major**, ``n*n_in+k`` -- i.e. ``[n_out][n_in]`` flattened,
the transpose of the naive row-major-by-input guess an earlier draft of this
docstring made (the RF<=n_in loop's ``w_index`` walks the *whole* n_in block
for a fixed ``out_index`` before advancing it, and the RF>n_in loop's
``in_index = w_index % nin`` says the same thing: ``n_in`` is the
fastest-varying component of the flat index, so ``n_out`` is the slower/outer
one). For the *const-weight* entries the ROM is baked at emit time directly in
that ``n*n_in+k`` layout -- ``weights_row_major`` only controls how the
*incoming* ``weight_matrix`` argument is read while building the ROM literal,
never a runtime transpose. For the *two-operand* entries (``gemm_array`` /
``gemm_stream``) B arrives at run time as column beats (``weight_cols[n][k]
== B[k][n]``, the fixed convention the hls4ml Catapult seam uses for
two-operand GEMMs, see ``nnet_gemm_ip.h``); those beats are copied into a
local ``weight_t[gemm_k*gemm_n]`` scratch array at the same ``n*gemm_k+k``
index while copying (B is not known until run time) before calling the same
``dense_resource``.
"""

import re

# ── Rounding/overflow mode map (Catapult ac_fixed/ac_int quantization & overflow) ──
MODE_MAP = {
    "RND": "AC_RND",
    "RND_CONV": "AC_RND_CONV",
    "TRN": "AC_TRN",
    "RND_ZERO": "AC_RND_ZERO",
    "WRAP": "AC_WRAP",
    "SAT": "AC_SAT",
    "SAT_SYM": "AC_SAT_SYM",
}
_DEFAULT_Q = "AC_TRN"
_DEFAULT_O = "AC_WRAP"


def _mode(name, default):
    if not name:
        return default
    name = name.upper()
    if name.startswith("AP_"):
        name = name[3:]
    if name.startswith("AC_"):
        name = name[3:]
    return MODE_MAP.get(name, default)


def ac_type(precision, default="ac_fixed<16,6,true>"):
    """Normalise an hls4ml/Vitis/Catapult precision string to an ac_fixed/ac_int type.

    Accepts ``ac_fixed<W,I[,S[,Q[,O]]]>`` / ``ac_int<W,S>`` (kept, with textual
    rounding-mode names remapped through MODE_MAP), ``ap_fixed<W,I[,Q[,O]]>`` /
    ``ap_ufixed<...>`` / ``ap_int<W>`` / ``ap_uint<W>``, and hls4ml's own
    ``fixed<W,I[,Q[,O]]>`` / ``ufixed<...>`` / ``int<W>`` / ``uint<W>``. Falls back
    to *default* when unset/unparseable.
    """
    if not precision:
        return default
    p = str(precision).replace(" ", "")

    if p.startswith("ac_int<"):
        m = re.match(r"ac_int<(\d+),(true|false)>", p)
        if m:
            return p
        m = re.match(r"ac_int<(\d+)>", p)
        if m:
            return f"ac_int<{m.group(1)},true>"
        return default

    if p.startswith("ac_fixed<"):
        inner = p[len("ac_fixed<"):].rstrip(">")
        parts = [x for x in inner.split(",") if x != ""]
        w, i = parts[0], parts[1]
        signed = parts[2] if len(parts) > 2 else "true"
        q = _mode(parts[3], _DEFAULT_Q) if len(parts) > 3 else _DEFAULT_Q
        o = _mode(parts[4], _DEFAULT_O) if len(parts) > 4 else _DEFAULT_O
        return f"ac_fixed<{w},{i},{signed},{q},{o}>"

    m = re.match(r"ap_u?int<(\d+)>", p)
    if m:
        signed = "false" if p.startswith("ap_uint") else "true"
        return f"ac_int<{m.group(1)},{signed}>"

    m = re.match(r"ap_u?fixed<([^>]+)>", p)
    if m:
        unsigned = p.startswith("ap_ufixed")
        parts = [x for x in m.group(1).split(",") if x != ""]
        w, i = parts[0], parts[1]
        q = _mode(parts[2], _DEFAULT_Q) if len(parts) > 2 else _DEFAULT_Q
        o = _mode(parts[3], _DEFAULT_O) if len(parts) > 3 else _DEFAULT_O
        signed = "false" if unsigned else "true"
        return f"ac_fixed<{w},{i},{signed},{q},{o}>"

    m = re.match(r"u?int<(\d+)>", p)
    if m:
        signed = "false" if p.startswith("u") else "true"
        return f"ac_int<{m.group(1)},{signed}>"

    m = re.match(r"u?fixed<([^>]+)>", p)
    if m:
        unsigned = p.startswith("u")
        parts = [x for x in m.group(1).split(",") if x != ""]
        w, i = parts[0], parts[1]
        q = _mode(parts[2], _DEFAULT_Q) if len(parts) > 2 else _DEFAULT_Q
        o = _mode(parts[3], _DEFAULT_O) if len(parts) > 3 else _DEFAULT_O
        signed = "false" if unsigned else "true"
        return f"ac_fixed<{w},{i},{signed},{q},{o}>"

    return default


# Backward/forward-compatible alias matching v-generic/hls.py's private helper name.
_ap_type = ac_type


# ── Copied verbatim (pruned to what compiles standalone) from
# hls4ml-gemm/hls4ml/templates/catapult/nnet_utils/{nnet_common,nnet_mult,
# nnet_dense_resource}.h, namespaced so it can never collide with hls4ml's own
# copy of the same file when both trees are on the include path together. ─────
DENSE_NAMESPACE = "c_generic_dense"

_DENSE_RESOURCE_BODY = r"""
#ifndef C_GENERIC_DENSE_MACROS_
#define C_GENERIC_DENSE_MACROS_
#define C_GENERIC_DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define C_GENERIC_MIN(n, d) (n > d ? d : n)
#endif // C_GENERIC_DENSE_MACROS_

namespace c_generic_dense {

namespace product {
// Copied from hls4ml's nnet_mult.h (nnet::product::mult), pruned to the one
// variant this kernel needs ('normal' product).
template <class x_T, class w_T> class mult {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) { return a * w; }
};
} // namespace product

// Copied from hls4ml's nnet_mult.h (nnet::cast), pruned to the plain-cast
// overload (this kernel's data/weight types are never the 1-bit binary special
// case that overload set exists for).
template <class data_T, class res_T, typename CONFIG_T> inline res_T cast(typename CONFIG_T::accum_t x) {
    return (res_T)x;
}

// Copied verbatim from nnet_dense_resource.h.
template <class T, int N> struct tree_sum_t {
    static T sum(const T *p) { return tree_sum_t<T, N / 2>::sum(p) + tree_sum_t<T, N - N / 2>::sum(p + N / 2); }
};
template <class T> struct tree_sum_t<T, 1> {
    static T sum(const T *p) { return p[0]; }
};

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_leq_nin(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = C_GENERIC_MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    const int multiplier_limit = C_GENERIC_DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
    const int block_factor = C_GENERIC_DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit / CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_IN");

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc_part[CONFIG_T::n_out][multscale];

#pragma hls_unroll
InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    #pragma hls_unroll
    InitPart:
        for (int ip = 0; ip < multscale; ip++) {
            acc_part[iacc][ip] = 0;
        }
    }

#pragma hls_pipeline_init_interval 1
ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {

        int w_index = ir;
        int in_index = ir;
        int out_index = 0;
        int acc_step = 0;

    #pragma hls_unroll
    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            if (rufactor == 1) {
                acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                    CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));
            } else {
                acc_part[out_index][acc_step] += static_cast<typename CONFIG_T::accum_t>(
                    CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));
            }

            w_index += rufactor;
            in_index += rufactor;
            if (in_index >= nin) {
                in_index = ir;
            }
            if (acc_step + 1 >= multscale) {
                acc_step = 0;
                out_index++;
            } else {
                acc_step++;
            }
        }
    }

    if (rufactor > 1) {
    #pragma hls_unroll
    ReducePart:
        for (int io = 0; io < nout; io++) {
            acc[io] += tree_sum_t<typename CONFIG_T::accum_t, multscale>::sum(acc_part[io]);
        }
    }

#pragma hls_unroll
Result:
    for (unsigned int ires = 0; ires < CONFIG_T::n_out; ires++) {
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_gt_nin_rem0(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = C_GENERIC_MIN(CONFIG_T::reuse_factor, CONFIG_T::n_in * CONFIG_T::n_out);
    const int multfactor = C_GENERIC_MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    const int multiplier_limit = C_GENERIC_DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
    const int block_factor = C_GENERIC_DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin && rufactor % nin == 0) && "This function is correct only for RF > N_IN && RF % N_IN == 0");

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

#pragma hls_unroll
InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    unsigned int w_index;
    int in_index = 0;
    int out_index;
    int outstep = 0;
    const int outscale = rufactor / nin;

    int outidx[rufactor];
IndexLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        if ((ir + 1) % nin == 0) {
            outstep++;
        }
    }

#pragma hls_pipeline_init_interval 1
ReuseLoop:
    for (unsigned int ir = 0; ir < rufactor; ir++) {

        w_index = ir;
        out_index = outidx[ir];

    #pragma hls_unroll
    MultLoop:
        for (unsigned int im = 0; im < block_factor; im++) {
            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));

            w_index += rufactor;
            if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out)
                break;
            out_index += outscale;
        }

        in_index++;
        if (in_index >= nin) {
            in_index = 0;
        }
    }

#pragma hls_unroll
Result:
    for (unsigned int ires = 0; ires < CONFIG_T::n_out; ires++) {
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_gt_nin(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                              typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                              typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = C_GENERIC_MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    const int multiplier_limit = C_GENERIC_DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
    const int block_factor = C_GENERIC_DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin) && "This function is correct only for RF > N_IN");

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

#pragma hls_unroll
InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

#pragma hls_pipeline_init_interval 1
ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        typename CONFIG_T::accum_t tmpmult[block_factor];

    #pragma hls_unroll
    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            unsigned int w_index = ir + rufactor * im;
            int in_index = w_index % nin;
            if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out)
                continue;
            tmpmult[im] =
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]);
        }

        typename CONFIG_T::accum_t mult[multiplier_limit];

    #pragma hls_unroll
    ResetMult:
        for (int imult = 0; imult < multiplier_limit; imult++) {
            mult[imult] = 0;
        }

    #pragma hls_unroll
    AccumLoop1:
        for (int im = 0; im < block_factor; im++) {
            int w_index = ir + rufactor * im;
            int out_index = w_index / multfactor;
            if (out_index >= multiplier_limit)
                continue;
            mult[out_index] += tmpmult[im];
        }

    #pragma hls_unroll
    AccumLoop2:
        for (int im = 0; im < multiplier_limit; im++) {
            acc[im] += mult[im];
        }
    }

#pragma hls_unroll
Result:
    for (unsigned int ires = 0; ires < CONFIG_T::n_out; ires++) {
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        dense_resource_rf_leq_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else if (CONFIG_T::reuse_factor % CONFIG_T::n_in == 0) {
        dense_resource_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource_rf_gt_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

} // namespace c_generic_dense
"""


# ── Shared four entry points -- ONE templated definition each, generic over any
# CONFIG_T carrying the fields the hls4ml-gemm Catapult writer's own per-layer
# gemm config struct exposes (``gemm_m``/``gemm_k``/``gemm_n``, ``gemm_ip_id``,
# ``reuse_factor``, ``weight_t``/``bias_t``/``accum_t``, a ``product<x_T,w_T>``
# alias, and -- for const-weight layers -- ``weights_row_major`` +
# ``weight_beat_t``/``gemm_weight_beats()``/``gemm_bias()``): see
# hls4ml-gemm/hls4ml/backends/catapult/passes/gemm_templates.py. This module's own
# ``_config_struct()``/``config_header()`` bake a struct with exactly the same
# field set (never a private ``dense_conf`` shape) so ONE set of entry points here
# serves both the whole-model integration header (CONFIG_T is the writer's own
# config) and this target's standalone self-test package (CONFIG_T is baked by
# this module) -- earlier drafts of this file defined a separate nested
# ``dense_conf``/``gemm_weight_rom()`` shape that only the self-baked struct had,
# which left the whole-model header's ``gemm_stream<...,config25>`` etc. calls
# instantiating against the writer's own struct and failing to compile (no
# ``dense_conf`` member) -- fixed by converging on the writer's contract
# everywhere. ``c_generic_bridge_conf<CONFIG_T>`` renames ``gemm_k``/``gemm_n`` to
# ``dense_resource``'s own ``n_in``/``n_out`` naming (the one field pair the
# contract does not share with hls4ml's own ``nnet::gemm_config``); everything
# else is read off CONFIG_T directly. Mirrors v-generic/hls.py's
# _GEMM_IP_COMBINED_FUNCS in shape: an einsum_dense-style row loop (unroll the
# free M dimension, call the per-row dense kernel once per row) around
# c_generic_dense::dense_resource, plus the nnet_dense_stream.h beat idiom for the
# two streaming entries. ────────────────────────────────────────────────────
_GEMM_IP_COMBINED_FUNCS = r"""
namespace nnet {

// Bridges any CONFIG_T satisfying the writer's GEMM config contract (gemm_k/
// gemm_n/reuse_factor/weight_t/bias_t/accum_t/product<>) to the field names
// c_generic_dense::dense_resource itself uses (n_in/n_out).
template <typename CONFIG_T>
struct c_generic_bridge_conf {
    static const unsigned n_in = CONFIG_T::gemm_k;
    static const unsigned n_out = CONFIG_T::gemm_n;
    static const unsigned reuse_factor = CONFIG_T::reuse_factor;
    typedef typename CONFIG_T::weight_t weight_t;
    typedef typename CONFIG_T::bias_t bias_t;
    typedef typename CONFIG_T::accum_t accum_t;
    template <class x_T, class w_T> using product = typename CONFIG_T::template product<x_T, w_T>;
};

// Same bridge, for the two-operand entries. The writer's own two-operand config's
// ``weight_t`` is aliased to the WHOLE B beat type (nnet::array<T,K>), not its
// scalar element type -- it exists only for cast<>'s binary-quantizer SFINAE, per
// hls4ml-gemm's gemm_two_operand_config_template docstring -- so ``dense_resource``
// (which uses ``CONFIG_T::weight_t`` as its flat weight array's ELEMENT type) must
// be handed the real scalar type explicitly rather than reading it off CONFIG_T.
template <typename CONFIG_T, typename WeightScalarT>
struct c_generic_two_op_bridge_conf {
    static const unsigned n_in = CONFIG_T::gemm_k;
    static const unsigned n_out = CONFIG_T::gemm_n;
    static const unsigned reuse_factor = CONFIG_T::reuse_factor;
    typedef WeightScalarT weight_t;
    typedef typename CONFIG_T::bias_t bias_t;
    typedef typename CONFIG_T::accum_t accum_t;
    template <class x_T, class w_T> using product = typename CONFIG_T::template product<x_T, w_T>;
};

// gemm_array -- io_parallel, TWO activation operands. B arrives as column beats
// (weight_cols[n][k] == B[k][n], the fixed two-operand convention); copied/
// transposed into a local dense_resource-layout scratch array at RUN time (B is
// not known until then) before the per-row kernel runs. No bias (a two-operand
// GEMM never owns one) -- a zero-valued local bias array stands in.
template <class a_row_T, class b_col_T, class res_row_T, typename CONFIG_T>
void gemm_array(a_row_T a_rows[CONFIG_T::gemm_m], b_col_T weight_cols[CONFIG_T::gemm_n],
                res_row_T results[CONFIG_T::gemm_m]) {
    typedef c_generic_two_op_bridge_conf<CONFIG_T, typename b_col_T::value_type> bconf;
    typename bconf::weight_t local_w[CONFIG_T::gemm_k * CONFIG_T::gemm_n];
#pragma hls_unroll
    for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
    #pragma hls_unroll
        for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
            local_w[n * CONFIG_T::gemm_k + k] = weight_cols[n][k];
        }
    }
    static typename bconf::bias_t zero_bias[CONFIG_T::gemm_n] = {};
    for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        typename a_row_T::value_type a_row[CONFIG_T::gemm_k];
        for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
            a_row[k] = a_rows[m][k];
        }
        typename res_row_T::value_type c_row[CONFIG_T::gemm_n];
        c_generic_dense::dense_resource<typename a_row_T::value_type, typename res_row_T::value_type, bconf>(
            a_row, c_row, local_w, zero_bias);
        for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            results[m][n] = c_row[n];
        }
    }
}

// gemm_array_const_weights -- io_parallel, constant operand read through
// CONFIG_T::gemm_weight_beats() (beat layout per CONFIG_T::weights_row_major, the
// same layout-agnostic convention hls4ml's own nnet::gemm_weight_at uses) into a
// local dense_resource-layout scratch array at RUN time, then dense_resource runs
// as normal. Bias likewise from gemm_bias() (baked all-zero when the layer has
// none, see bias_header()).
template <class a_row_T, class res_row_T, typename CONFIG_T>
void gemm_array_const_weights(a_row_T a_rows[CONFIG_T::gemm_m], res_row_T results[CONFIG_T::gemm_m]) {
    typedef c_generic_bridge_conf<CONFIG_T> bconf;
    // CONFIG_T is shared with the hls4ml-gemm Catapult writer's own per-layer
    // config struct (see this module's field-set docstring above), which only
    // exposes the beat-layout ROM (gemm_weight_beats()/weights_row_major) --
    // never a pre-flattened dense_resource-order array -- so the transpose
    // into dense_resource order still happens here, but fully unrolled: every
    // index (n, k) is a compile-time constant against a ROM of compile-time
    // constants, so Catapult constant-folds this into pure wiring (zero
    // cycles) instead of scheduling it as a rolled runtime copy.
    typename CONFIG_T::weight_beat_t *wbeats = CONFIG_T::gemm_weight_beats();
    typename bconf::weight_t local_w[CONFIG_T::gemm_k * CONFIG_T::gemm_n];
#pragma hls_unroll
    for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
    #pragma hls_unroll
        for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
            local_w[n * CONFIG_T::gemm_k + k] = CONFIG_T::weights_row_major ? wbeats[k][n] : wbeats[n][k];
        }
    }
    typename bconf::bias_t *biases = CONFIG_T::gemm_bias();
    for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        typename a_row_T::value_type a_row[CONFIG_T::gemm_k];
        for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
            a_row[k] = a_rows[m][k];
        }
        typename res_row_T::value_type c_row[CONFIG_T::gemm_n];
        c_generic_dense::dense_resource<typename a_row_T::value_type, typename res_row_T::value_type, bconf>(
            a_row, c_row, local_w, biases);
        for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
            results[m][n] = c_row[n];
        }
    }
}

// gemm_stream -- io_stream, TWO activation operands. B is drained fully into a
// local column-beat buffer ONCE, before the row loop (mandatory operand
// residency: B is reused across A's M rows), transposed into dense_resource
// layout, then each row is handled by the same three-region shape
// nnet_dense_stream.h's nnet::dense() uses: a DataPrepare-equivalent gather
// (outer beat loop pipelined, inner beat-unpack unrolled), the dense_resource
// call itself (no wrapper pragma of ours -- see gemm_stream_const_weights
// below for why), then a ResWrite-equivalent pack (outer beat loop pipelined,
// inner pack unrolled). The row loop itself is a plain (unpipelined) loop,
// mirroring how hls4ml calls nnet::dense() once per frame -- never one single
// pipeline pragma spanning gather+kernel+pack.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(ac_channel<data0_T> &a_stream, ac_channel<data1_T> &b_stream, ac_channel<res_T> &res_stream) {
    static_assert(data0_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(data1_T::size == CONFIG_T::gemm_k, "B column height must equal gemm_k.");
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");

    typedef c_generic_two_op_bridge_conf<CONFIG_T, typename data1_T::value_type> bconf;
    data1_T b_cols[CONFIG_T::gemm_n];
#pragma hls_pipeline_init_interval 1
GemmStreamReadB:
    for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
        b_cols[n] = b_stream.read();
    }
    typename bconf::weight_t local_w[CONFIG_T::gemm_k * CONFIG_T::gemm_n];
#pragma hls_unroll
    for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
    #pragma hls_unroll
        for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
            local_w[n * CONFIG_T::gemm_k + k] = b_cols[n][k];
        }
    }
    static typename bconf::bias_t zero_bias[CONFIG_T::gemm_n] = {};

GemmStreamM:
    for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        typename data0_T::value_type a_row[CONFIG_T::gemm_k];
#pragma hls_pipeline_init_interval 1
    GemmStreamGatherA:
        for (unsigned kp = 0; kp < CONFIG_T::gemm_k / data0_T::size; kp++) {
            data0_T beat = a_stream.read();
        #pragma hls_unroll
        GemmStreamGatherAUnpack:
            for (unsigned k = 0; k < data0_T::size; k++) {
                a_row[kp * data0_T::size + k] = beat[k];
            }
        }

        typename res_T::value_type c_row[CONFIG_T::gemm_n];
        c_generic_dense::dense_resource<typename data0_T::value_type, typename res_T::value_type, bconf>(
            a_row, c_row, local_w, zero_bias);

#pragma hls_pipeline_init_interval 1
    GemmStreamPackRes:
        for (unsigned np = 0; np < CONFIG_T::gemm_n / res_T::size; np++) {
            res_T c_beat;
        #pragma hls_unroll
        GemmStreamPackResPack:
            for (unsigned n = 0; n < res_T::size; n++) {
                c_beat[n] = c_row[np * res_T::size + n];
            }
            res_stream.write(c_beat);
        }
    }
}

// gemm_stream_const_weights -- io_stream, constant operand read through
// CONFIG_T::gemm_weight_beats() (see gemm_array_const_weights above). Mirrors
// nnet_dense_stream.h's nnet::dense() shape exactly, per row: a DataPrepare
// gather loop (A may arrive as several narrower beats packed into a gemm_k-wide
// row; outer beat loop pipelined, inner beat-unpack unrolled), the
// dense_resource call itself with NO wrapper pragma of our own (hls4ml's own
// dense_wrapper() only adds a `hls_pipeline_init_interval ce_reuse_factor`
// pragma for the LATENCY strategy; for RESOURCE -- what this target always
// uses -- dense_wrapper() calls dense_resource() directly and relies entirely
// on dense_resource's own internal ReuseLoop pragma, so nesting an outer
// pipeline pragma around the call here would only mis-schedule it), then a
// ResWrite pack loop (outer beat loop pipelined, inner pack unrolled). The row
// (M) loop itself is a plain unpipelined loop -- never one pipeline pragma
// spanning gather+kernel+pack the way an earlier draft of this emitter did.
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream) {
    static_assert(CONFIG_T::gemm_k % data_T::size == 0, "gemm_k must be a whole number of input beats.");
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");

    typedef c_generic_bridge_conf<CONFIG_T> bconf;
    // See gemm_array_const_weights above: fully-unrolled transpose of the
    // compile-time-constant beat ROM, so Catapult folds it to wiring.
    typename CONFIG_T::weight_beat_t *wbeats = CONFIG_T::gemm_weight_beats();
    typename bconf::weight_t local_w[CONFIG_T::gemm_k * CONFIG_T::gemm_n];
#pragma hls_unroll
    for (unsigned n = 0; n < CONFIG_T::gemm_n; n++) {
    #pragma hls_unroll
        for (unsigned k = 0; k < CONFIG_T::gemm_k; k++) {
            local_w[n * CONFIG_T::gemm_k + k] = CONFIG_T::weights_row_major ? wbeats[k][n] : wbeats[n][k];
        }
    }
    typename bconf::bias_t *biases = CONFIG_T::gemm_bias();
    const unsigned PACKETS = CONFIG_T::gemm_k / data_T::size;

GemmStreamCwM:
    for (unsigned m = 0; m < CONFIG_T::gemm_m; m++) {
        typename data_T::value_type a_row[CONFIG_T::gemm_k];
#pragma hls_pipeline_init_interval 1
    GemmStreamCwGather:
        for (unsigned kp = 0; kp < PACKETS; kp++) {
            data_T beat = data_stream.read();
        #pragma hls_unroll
        GemmStreamCwGatherUnpack:
            for (unsigned k = 0; k < data_T::size; k++) {
                a_row[kp * data_T::size + k] = beat[k];
            }
        }

        typename res_T::value_type c_row[CONFIG_T::gemm_n];
        c_generic_dense::dense_resource<typename data_T::value_type, typename res_T::value_type, bconf>(
            a_row, c_row, local_w, biases);

#pragma hls_pipeline_init_interval 1
    GemmStreamCwPack:
        for (unsigned np = 0; np < CONFIG_T::gemm_n / res_T::size; np++) {
            res_T c_beat;
        #pragma hls_unroll
        GemmStreamCwPackPack:
            for (unsigned n = 0; n < res_T::size; n++) {
                c_beat[n] = c_row[np * res_T::size + n];
            }
            res_stream.write(c_beat);
        }
    }
}

} // namespace nnet
"""




def _flatten_weights(weight_matrix, k, n, weights_row_major):
    """Flatten a weight matrix into dense_resource's [n_out*n_in] order
    (index = nn*k+kk == W[kk][nn]) at EMIT time.

    ``weight_matrix`` is always the mathematical B (``K`` rows, ``N`` columns):
    ``weight_matrix[kk][nn] == B[kk][nn]``. ``weights_row_major`` records how the
    layer's ROM is described in the manifest (matching v-generic's convention) but
    never changes how *this* helper reads its own ``weight_matrix`` argument --
    callers that hold a transposed matrix (``[N][K]``) pass
    ``weights_row_major=True`` together with a matrix already given as
    ``[nn][kk]``, and this function transposes it back while flattening so the
    baked ROM is always in dense_resource order regardless of the manifest's
    layout choice.
    """
    flat = [0] * (k * n)
    if weights_row_major:
        # weight_matrix is [N][K]: weight_matrix[nn][kk] == B[kk][nn].
        for nn in range(n):
            for kk in range(k):
                flat[nn * k + kk] = weight_matrix[nn][kk]
    else:
        # weight_matrix is [K][N]: weight_matrix[kk][nn] == B[kk][nn].
        for nn in range(n):
            for kk in range(k):
                flat[nn * k + kk] = weight_matrix[kk][nn]
    return flat


def _literal(value):
    if isinstance(value, float) and value.is_integer():
        return repr(value)
    return str(value)


def _beat_rows(weight_matrix, k, n, weights_row_major):
    """Lay a weight matrix out as the beat rows the hls4ml-gemm Catapult writer's
    own ``gemm_weight_beats()``/``weight_beat_t`` contract expects (matching
    ``nnet::gemm_weight_at`` in ``nnet_gemm_ip.h``): column-major (default) is one
    K-high output column per beat, ``beats[nn][kk] == W[kk][nn]``; row-major is one
    N-wide contraction row per beat, ``beats[kk][nn] == W[kk][nn]``. Same
    ``weight_matrix`` argument convention as ``_flatten_weights`` (row-major
    callers pass the transposed ``[N][K]`` matrix)."""
    if weights_row_major:
        return [[weight_matrix[nn][kk] for nn in range(n)] for kk in range(k)]
    return [[weight_matrix[kk][nn] for kk in range(k)] for nn in range(n)]


def _beat_array_literal(ctype, array_name, beats):
    """``static nnet::array<ctype, beat_len> array_name[len(beats)] = {...};``"""
    beat_len = len(beats[0]) if beats else 0
    rows = ", ".join("{" + ", ".join(_literal(v) for v in row) + "}" for row in beats)
    return f"static nnet::array<{ctype}, {beat_len}> {array_name}[{len(beats)}] = {{ {rows} }};\n"


def _config_struct(name, m, k, n, reuse_factor, gemm_ip_id,
                   input_t, weight_t, result_t, bias_t, accum_t,
                   weights_row_major, weight_rom_values, bias_rom_values):
    """One layer's CONFIG_T + baked ROMs (weight ROM only emitted for
    const-weight layers, i.e. when ``weight_rom_values`` is not None).

    Field/method set matches the hls4ml-gemm Catapult writer's own per-layer GEMM
    config struct exactly (``gemm_m``/``gemm_k``/``gemm_n``/``gemm_ip_id``/
    ``reuse_factor``/``weight_t``/``bias_t``/``accum_t``/``product<>``, and for
    const-weight layers ``weights_row_major``/``weight_beat_t``/
    ``gemm_weight_beats()``/``gemm_bias()``) -- see ``_GEMM_IP_COMBINED_FUNCS``'s
    module comment for why this must not diverge (no private ``dense_conf`` shape).
    """
    rf = max(1, int(reuse_factor or 1))
    beat_len = k if not weights_row_major else n
    weight_beat_decl = ""
    weight_accessor = ""
    if weight_rom_values is not None:
        # Beat-layout ROM, in the same layout-agnostic contract the
        # hls4ml-gemm Catapult writer's own per-layer config struct exposes
        # (gemm_weight_beats()/weights_row_major) -- CONFIG_T here must stay
        # field-compatible with that struct (see this module's field-set
        # docstring above), so this is the only weight-ROM accessor the
        # shared kernel entry points may read; they unroll the
        # beat->dense_resource-order transpose at compile time instead of
        # requiring a second, writer-incompatible accessor.
        beats = _beat_rows(weight_rom_values, k, n, weights_row_major)
        weight_beat_decl = _beat_array_literal(weight_t, f"{name}_weight_beats", beats)
        weight_accessor = (
            f"    typedef nnet::array<weight_t, {beat_len}> weight_beat_t;\n"
            f"    static weight_beat_t *gemm_weight_beats() {{ return {name}_weight_beats; }}\n"
        )
    bias_vals = list(bias_rom_values) if bias_rom_values else [0] * n
    assert len(bias_vals) == n
    bias_body = ", ".join(_literal(v) for v in bias_vals)
    bias_rom_decl = f"static {bias_t} {name}_bias_rom[{n}] = {{ {bias_body} }};\n"

    return f"""{weight_beat_decl}{bias_rom_decl}
struct {name}_config {{
    static const unsigned gemm_m = {m};
    static const unsigned gemm_k = {k};
    static const unsigned gemm_n = {n};
    static const unsigned gemm_ip_id = {gemm_ip_id};
    static const unsigned reuse_factor = {rf};
    static const bool weights_row_major = {"true" if weights_row_major else "false"};

    typedef {input_t} input_t;
    typedef {weight_t} weight_t;
    typedef {result_t} result_t;
    typedef {bias_t} bias_t;
    typedef {accum_t} accum_t;
    template <class x_T, class w_T> using product = c_generic_dense::product::mult<x_T, w_T>;

{weight_accessor}    static bias_t *gemm_bias() {{ return {name}_bias_rom; }}
}};
"""


def _has_bias_trait(items):
    specs = []
    for item in items:
        idx = item.get("gemm_ip_index", 0)
        if item.get("has_bias") is False:
            specs.append(
                f"template <> struct gemm_ip_has_bias<{idx}> {{ static const bool value = false; }};")
    specs_txt = "\n".join(specs)
    return (
        "namespace nnet {\n"
        "// Per-layer has_bias, mirroring v-generic/hls.py's trait of the same name (kept\n"
        "// for parity/inspection -- the actual no-bias behaviour is implemented by\n"
        "// baking an all-zero bias ROM per layer, see _config_struct()).\n"
        "template <unsigned id> struct gemm_ip_has_bias { static const bool value = true; };\n"
        f"{specs_txt}\n"
        "} // namespace nnet\n"
    )


def _item_config(item, default_gemm_ip_id=0):
    """Pull one manifest item's shape/precision/ROM fields into the arguments
    ``_config_struct`` wants, with generic-matching field-name fallbacks."""
    name = item["name"]
    m = int(item.get("gemm_m", item.get("m", 0)))
    k = int(item.get("gemm_k", item.get("k", item.get("n_in", 0))))
    n = int(item.get("gemm_n", item.get("n", item.get("n_out", 0))))
    rf = int(item.get("reuse_factor", 1) or 1)
    gemm_ip_id = int(item.get("gemm_ip_index", default_gemm_ip_id) or 0)

    input_t = ac_type(item.get("input_precision") or item.get("input_t"), "ac_fixed<16,6,true>")
    weight_t = ac_type(item.get("weight_precision") or item.get("weight_t"), input_t)
    result_t = ac_type(item.get("output_precision") or item.get("result_t"), "ac_fixed<16,6,true>")
    bias_t = ac_type(item.get("bias_precision") or item.get("bias_t"), result_t)
    accum_t = ac_type(item.get("accum_precision") or item.get("accum_t"), "ac_fixed<32,12,true>")

    weights_row_major = bool(item.get("weights_row_major", False))
    weight_matrix = item.get("weight_matrix")
    weights_in_core = bool(item.get("weights_in_core", weight_matrix is not None))
    weight_rom_values = None
    if weights_in_core and weight_matrix is not None:
        weight_rom_values = weight_matrix

    bias_values = item.get("bias_values")
    if item.get("has_bias") is False:
        bias_values = [0] * n

    return dict(
        name=name, m=m, k=k, n=n, reuse_factor=rf, gemm_ip_id=gemm_ip_id,
        input_t=input_t, weight_t=weight_t, result_t=result_t, bias_t=bias_t,
        accum_t=accum_t, weights_row_major=weights_row_major,
        weight_rom_values=weight_rom_values, bias_rom_values=bias_values,
    )


def combined_header(items=None):
    """The whole-model integration header (``gemm_ip_combined.h``): per-id
    ``CONFIG_T`` structs (shape/precision/ROM) plus the four shared entry-point
    templates, mirroring ``v-generic/hls.py``'s ``combined_header()`` in shape.
    """
    items = items or []
    config_blocks = []
    for idx, item in enumerate(items):
        cfg = _item_config(item, default_gemm_ip_id=idx)
        config_blocks.append(_config_struct(
            cfg["name"], cfg["m"], cfg["k"], cfg["n"], cfg["reuse_factor"], cfg["gemm_ip_id"],
            cfg["input_t"], cfg["weight_t"], cfg["result_t"], cfg["bias_t"], cfg["accum_t"],
            cfg["weights_row_major"], cfg["weight_rom_values"], cfg["bias_rom_values"],
        ))
    configs_txt = "\n".join(config_blocks)
    has_bias_trait = _has_bias_trait(items)
    return (
        "#ifndef GEMM_IP_COMBINED_H_\n"
        "#define GEMM_IP_COMBINED_H_\n\n"
        "#include <ac_fixed.h>\n"
        "#include <ac_int.h>\n"
        "#include <ac_channel.h>\n"
        "#include <assert.h>\n\n"
        # Self-contained: define nnet::array<T,N> here (guarded) rather than
        # #include "nnet_types.h" -- the whole-model build already has its own
        # copy on the include path (hls4ml's writer emits one; this guard makes
        # a second definition here a no-op), and this keeps a single-file
        # standalone combined header (this target's own self-test packages,
        # gemm_ip_header()) buildable with no sibling file.
        "#ifndef NNET_TYPES_H_\n#define NNET_TYPES_H_\nnamespace nnet {\n"
        "template <typename T, unsigned N> struct array {\n"
        "    typedef T value_type;\n"
        "    static const unsigned size = N;\n"
        "    T data[N];\n"
        "    T &operator[](unsigned pos) { return data[pos]; }\n"
        "    const T &operator[](unsigned pos) const { return data[pos]; }\n"
        "};\n} // namespace nnet\n#endif // NNET_TYPES_H_\n\n"
        f"{_DENSE_RESOURCE_BODY}\n"
        f"{has_bias_trait}\n"
        f"{configs_txt}\n"
        f"{_GEMM_IP_COMBINED_FUNCS}\n"
        "#endif // GEMM_IP_COMBINED_H_\n"
    )


def kernel_header(name):
    """The shared kernel-only header: the copied ``dense_resource`` body plus
    the four entry-point templates, with NO per-layer ``CONFIG_T`` struct (that
    lives in ``config_header()``'s ``{name}_config.h``, which needs this
    file's ``c_generic_dense`` namespace already declared -- so package.py
    includes this file before ``{name}_config.h``, the opposite order
    ``gemm_ip_header()``'s single-file standalone package uses).
    """
    guard = f"{name.upper()}_GEMM_IP_H_"
    return (
        f"#ifndef {guard}\n"
        f"#define {guard}\n\n"
        "#include <ac_fixed.h>\n"
        "#include <ac_int.h>\n"
        "#include <ac_channel.h>\n"
        "#include <assert.h>\n\n"
        f"{_DENSE_RESOURCE_BODY}\n"
        f"{_GEMM_IP_COMBINED_FUNCS}\n"
        f"#endif // {guard}\n"
    )


def gemm_ip_header(name, m, k, n, reuse_factor=1, weight_matrix=None, bias_values=None,
                   has_bias=True, weights_row_major=False, input_precision=None,
                   weight_precision=None, output_precision=None, bias_precision=None,
                   accum_precision=None, weights_in_core=None):
    """Standalone single-layer package header: one ``CONFIG_T`` + the four
    entry points, self-contained (no dependency on any other file this target
    emits). Mirrors ``v-generic/hls.py``'s ``gemm_ip_header()``.
    """
    if weights_in_core is None:
        weights_in_core = weight_matrix is not None
    item = dict(
        name=name, gemm_m=m, gemm_k=k, gemm_n=n, reuse_factor=reuse_factor,
        gemm_ip_index=0, input_precision=input_precision, weight_precision=weight_precision,
        output_precision=output_precision, bias_precision=bias_precision,
        accum_precision=accum_precision, weights_row_major=weights_row_major,
        weight_matrix=weight_matrix if weights_in_core else None,
        weights_in_core=weights_in_core, bias_values=bias_values, has_bias=has_bias,
    )
    return combined_header([item]).replace(
        "GEMM_IP_COMBINED_H_", f"{name.upper()}_GEMM_IP_H_")


# ── Per-shape helper headers (config/bias/weights split files), matching
# v-generic/hls.py's package-file shape for callers (e.g. package.py, step 4)
# that want the config/bias/weight ROM as separate includes rather than the
# single combined_header()/gemm_ip_header() blob above. ───────────────────────

def config_header(name, m, k, n, input_precision=None, weight_precision=None,
                  output_precision=None, bias_precision=None, accum_precision=None,
                  weights_in_core=False, reuse_factor=1, weights_row_major=False):
    input_t = ac_type(input_precision, "ac_fixed<16,6,true>")
    weight_t = ac_type(weight_precision, input_t)
    result_t = ac_type(output_precision, "ac_fixed<16,6,true>")
    bias_t = ac_type(bias_precision, result_t)
    accum_t = ac_type(accum_precision, "ac_fixed<32,12,true>")
    rf = max(1, int(reuse_factor or 1))
    beat_len = k if not weights_row_major else n
    weights_inc = f'#include "{name}_weights.h"\n' if weights_in_core else ""
    weight_accessor = (
        f"    typedef nnet::array<weight_t, {beat_len}> weight_beat_t;\n"
        f"    static weight_beat_t *gemm_weight_beats() {{ return {name}_weight_beats; }}\n"
        if weights_in_core else ""
    )
    return f"""#ifndef {name.upper()}_CONFIG_H_
#define {name.upper()}_CONFIG_H_

#include <ac_fixed.h>
#include <ac_int.h>
#include "nnet_types.h"

typedef {input_t} {name}_input_t;
typedef {weight_t} {name}_weight_t;
typedef {result_t} {name}_result_t;
typedef {bias_t} {name}_bias_t;

#include "{name}_bias.h"
{weights_inc}
struct {name}_config {{
    static const unsigned gemm_m = {m};
    static const unsigned gemm_k = {k};
    static const unsigned gemm_n = {n};
    static const unsigned gemm_ip_id = 0;
    static const unsigned reuse_factor = {rf};
    static const bool weights_row_major = {"true" if weights_row_major else "false"};

    typedef {name}_input_t input_t;
    typedef {name}_weight_t weight_t;
    typedef {name}_result_t result_t;
    typedef {name}_bias_t bias_t;
    typedef {accum_t} accum_t;
    template <class x_T, class w_T> using product = c_generic_dense::product::mult<x_T, w_T>;

{weight_accessor}    static bias_t *gemm_bias() {{ return {name}_bias_rom; }}
}};

#endif // {name.upper()}_CONFIG_H_
"""


def bias_header(name, n, bias_t, bias_values):
    vals = list(bias_values) if bias_values else [0] * n
    if len(vals) != n:
        raise ValueError(f"bias length {len(vals)} != N {n}")
    body = ", ".join(_literal(v) for v in vals)
    return f"""#ifndef {name.upper()}_BIAS_H_
#define {name.upper()}_BIAS_H_

#include "{name}_config.h"

static {bias_t} {name}_bias_rom[{n}] = {{ {body} }};

#endif // {name.upper()}_BIAS_H_
"""


def weights_header(name, m, k, n, weight_matrix, weights_row_major=False):
    """Weight-stationary ROM as beat rows, in the same layout-agnostic
    ``gemm_weight_beats()`` contract the hls4ml-gemm Catapult writer's own
    per-layer config exposes (see ``_beat_rows``)."""
    beats = _beat_rows(weight_matrix, k, n, weights_row_major)
    decl = _beat_array_literal(f"{name}_weight_t", f"{name}_weight_beats", beats)
    return f"""#ifndef {name.upper()}_WEIGHTS_H_
#define {name.upper()}_WEIGHTS_H_

#include "{name}_config.h"

// Weight-stationary ROM, one beat per {"contraction row" if weights_row_major else "output column"}.
{decl}
#endif // {name.upper()}_WEIGHTS_H_
"""


def nnet_types_header():
    """Minimal self-contained beat wrapper (``nnet::array<T,N>``), the same
    interface-compatible shape ``v-generic/hls.py``'s copy has, but with no
    ``#pragma HLS`` in it (Catapult's ``ac_channel<T>`` only requires ``T`` be
    default-constructible/assignable/copyable and expose ``size`` /
    ``value_type`` / ``operator[]`` per the writer's own beat-struct contract
    -- a plain aggregate already satisfies that, no per-field unroll pragma
    needed the way Vitis's version wants one for II=1 inlining).
    """
    return """#ifndef NNET_TYPES_H_
#define NNET_TYPES_H_

namespace nnet {

// Fixed-size packed beat, interface-compatible with hls4ml's nnet::array and
// usable directly as an ac_channel<T> payload.
template <typename T, unsigned N> struct array {
    typedef T value_type;
    static const unsigned size = N;
    T data[N];
    T &operator[](unsigned pos) { return data[pos]; }
    const T &operator[](unsigned pos) const { return data[pos]; }
};

} // namespace nnet

#endif // NNET_TYPES_H_
"""


def top_cpp(name, m, k, n, interface="array", weights_in_core=False):
    """The Catapult synthesis top (``#pragma hls_design top``) that calls the
    chosen entry point. Mirrors v-generic/hls.py's ``top_cpp`` shape.

    The four entry points in ``_GEMM_IP_COMBINED_FUNCS`` take *beat* types
    (``a_row_T``/``b_col_T``/``res_row_T`` exposing ``::size`` /
    ``::value_type`` / ``operator[]``, per this module's docstring) rather
    than raw C arrays or ``ac_channel<T[K]>`` -- ``nnet::array<T,K>`` from
    ``nnet_types.h`` is that beat type for every interface, array or stream.
    """
    a_row_t = f"{name}_a_row_t"
    b_col_t = f"{name}_b_col_t"
    res_row_t = f"{name}_res_row_t"
    head = (
        "#pragma hls_design top\n"
        "#include <mc_scverify.h>\n"
        '#include "nnet_types.h"\n'
        f'#include "{name}_gemm_ip.h"\n'
        f'#include "{name}_config.h"\n\n'
        f"typedef nnet::array<{name}_input_t, {k}> {a_row_t};\n"
        f"typedef nnet::array<{name}_result_t, {n}> {res_row_t};\n"
    )
    if not weights_in_core:
        head += f"typedef nnet::array<{name}_weight_t, {k}> {b_col_t};\n"
    head += "\n"
    if interface == "array" and not weights_in_core:
        return head + f"""void CCS_BLOCK({name})(
    {a_row_t} a_rows[{m}],
    {b_col_t} b_cols[{n}],
    {res_row_t} results[{m}]
) {{
    nnet::gemm_array<{a_row_t}, {b_col_t}, {res_row_t}, {name}_config>(
        a_rows, b_cols, results);
}}
"""
    if interface == "array" and weights_in_core:
        return head + f"""void CCS_BLOCK({name})(
    {a_row_t} a_rows[{m}],
    {res_row_t} results[{m}]
) {{
    nnet::gemm_array_const_weights<{a_row_t}, {res_row_t}, {name}_config>(
        a_rows, results);
}}
"""
    if interface == "stream" and not weights_in_core:
        return head + f"""void CCS_BLOCK({name})(
    ac_channel<{a_row_t}> &a_stream,
    ac_channel<{b_col_t}> &b_stream,
    ac_channel<{res_row_t}> &res_stream
) {{
    nnet::gemm_stream<{a_row_t}, {b_col_t}, {res_row_t}, {name}_config>(
        a_stream, b_stream, res_stream);
}}
"""
    return head + f"""void CCS_BLOCK({name})(
    ac_channel<{a_row_t}> &a_stream,
    ac_channel<{res_row_t}> &res_stream
) {{
    nnet::gemm_stream_const_weights<{a_row_t}, {res_row_t}, {name}_config>(
        a_stream, res_stream);
}}
"""
