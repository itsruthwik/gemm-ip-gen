"""mvau C twin (blackbox behavioral model) + self-checking csim/cosim testbench.

Both encode the FINN MVU beat protocol for one tile, generalized from the
cosim-validated ``temp_space/mvau-spike`` (which was the SF=NF=1 case):

  * activations: ``SF`` beats per input vector (``x[sf*SIMD+simd]``); the replay
    buffer re-streams them ``NF`` times internally.
  * weights: ``NF*SF`` beats per vector, consumed in ``(nf outer, sf inner)``
    order; beat ``(nf,sf)`` holds ``W[nf*PE+pe][sf*SIMD+simd]`` packed ``[pe][simd]``
    LSB-first at ``(pe*SIMD+simd)*WEIGHT_WIDTH``.
  * output: ``NF`` beats per vector; beat ``nf`` holds accumulators for output
    rows ``nf*PE .. nf*PE+PE-1`` at ``pe*ACCU_WIDTH``.

The C twin is bit-identical to the RTL (pure integer ``Σ w·x``); csim runs it,
cosim runs the FINN RTL, and they must agree.
"""

import geometry as _geom


def _plan(shape, plan=None, **kw):
    return plan if plan is not None else _geom.fold_plan(*shape, **kw)


def _act_ctype(signed, aw):
    return f"ap_int<{aw}>" if signed else f"ap_uint<{aw}>"


def _w_matrix_literal(B, n, k):
    """C initializer for ``long W[N][K]`` from ``B`` (``[K][N]``): W[o][k]=B[k][o]."""
    rows = ["{" + ", ".join(str(int(B[kk][oo])) for kk in range(k)) + "}" for oo in range(n)]
    return "{" + ", ".join(rows) + "}"


def generate_core_twin(shape, func_name="mvau_core", plan=None, baked_weights=None, **kw):
    """C++ behavioral twin of the blackbox (the JSON ``c_files`` model).

    ``baked_weights`` (``B`` as ``[K][N]``) selects the weight-stationary twin: the
    weights are baked into the model (no ``w`` stream), matching the memstream the
    RTL bakes. Absent -> the streamed twin (``w`` stream, reserved for two-operand).
    """
    p = _plan(shape, plan=plan, **kw)
    t = p["tile"]
    if baked_weights is not None:
        if p.get("k_tiles", 1) > 1:
            return _kt_core_twin(p, t, func_name, baked_weights)
        return _ws_core_twin(p, t, func_name, baked_weights)
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    WW, AW, ACCU = t["weight_width"], t["activation_width"], t["accu_width"]
    WB, AB, PB = t["weight_stream_width_ba"], t["input_stream_width_ba"], t["output_stream_width_ba"]
    M = p["num_input_vectors"]
    actt = _act_ctype(t["signed_activations"], AW)

    return f"""#include <hls_stream.h>
#include <ap_int.h>

// Behavioral C twin of {func_name} (FINN MVU tile: PE={PE} SIMD={SIMD} SF={SF} NF={NF}).
// Bit-identical integer matmul; Vitis substitutes the RTL for csynth/cosim.
void {func_name}(hls::stream<ap_uint<{WB}> >& w,
{' ' * (len(func_name) + 6)}hls::stream<ap_uint<{AB}> >& a,
{' ' * (len(func_name) + 6)}hls::stream<ap_uint<{PB}> >& p) {{
    for (int vec = 0; vec < {M}; vec++) {{
        {actt} x[{SF}][{SIMD}];
        for (int sf = 0; sf < {SF}; sf++) {{
            ap_uint<{AB}> ab = a.read();
            for (int s = 0; s < {SIMD}; s++)
                x[sf][s] = ab.range(s * {AW} + {AW} - 1, s * {AW});
        }}
        for (int nf = 0; nf < {NF}; nf++) {{
            ap_int<{ACCU}> acc[{PE}];
            for (int pe = 0; pe < {PE}; pe++) acc[pe] = 0;
            for (int sf = 0; sf < {SF}; sf++) {{
                ap_uint<{WB}> wb = w.read();
                for (int pe = 0; pe < {PE}; pe++)
                    for (int s = 0; s < {SIMD}; s++) {{
                        ap_int<{WW}> wv = wb.range((pe * {SIMD} + s) * {WW} + {WW} - 1,
                                                   (pe * {SIMD} + s) * {WW});
                        acc[pe] += (ap_int<64>)wv * (ap_int<64>)x[sf][s];
                    }}
            }}
            ap_uint<{PB}> ob = 0;
            for (int pe = 0; pe < {PE}; pe++)
                ob.range(pe * {ACCU} + {ACCU} - 1, pe * {ACCU}) = (ap_uint<{ACCU}>)acc[pe];
            p.write(ob);
        }}
    }}
}}
"""


def _ws_core_twin(p, t, func_name, B):
    """Weight-stationary C twin: baked weights, signature ``(a, p)`` (no ``w``).

    N-tiling: the result beat concatenates the ``n_tiles`` tiles; in beat ``nf``,
    tile ``ti`` lane ``pe`` (bits ``ti*PB + pe*ACCU +: ACCU``) holds global output
    column ``ti*n_tile + nf*PE + pe`` -- matching the RTL shim's concatenation."""
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    ACCU = t["accu_width"]
    AW = t["activation_width"]
    AB, PB = t["input_stream_width_ba"], t["output_stream_width_ba"]
    NT, NTILE = p["n_tiles"], p["n_tile"]
    PB_TOTAL = NT * PB
    M, K, N = p["num_input_vectors"], p["k_pad"], p["n"]
    actt = _act_ctype(t["signed_activations"], AW)
    wlit = _w_matrix_literal(B, N, K)
    pad = ' ' * (len(func_name) + 6)
    return f"""#include <hls_stream.h>
#include <ap_int.h>

// Weight-stationary C twin of {func_name} (FINN MVU: PE={PE} SIMD={SIMD} SF={SF} NF={NF},
// N_TILES={NT}). Weights baked here match the memstreams the RTL bakes; Vitis
// substitutes the RTL (memstream(s) + mvu_vvu_axi) for csynth/cosim. Integer matmul.
static const long {func_name}_W[{N}][{K}] = {wlit};

void {func_name}(hls::stream<ap_uint<{AB}> >& a,
{pad}hls::stream<ap_uint<{PB_TOTAL}> >& p) {{
    for (int vec = 0; vec < {M}; vec++) {{
        {actt} x[{SF}][{SIMD}];
        for (int sf = 0; sf < {SF}; sf++) {{
            ap_uint<{AB}> ab = a.read();
            for (int s = 0; s < {SIMD}; s++)
                x[sf][s] = ab.range(s * {AW} + {AW} - 1, s * {AW});
        }}
        for (int nf = 0; nf < {NF}; nf++) {{
            ap_uint<{PB_TOTAL}> ob = 0;
            for (int ti = 0; ti < {NT}; ti++)
                for (int pe = 0; pe < {PE}; pe++) {{
                    int oc = ti * {NTILE} + nf * {PE} + pe;   // global output column
                    ap_int<{ACCU}> acc = 0;
                    for (int sf = 0; sf < {SF}; sf++)
                        for (int s = 0; s < {SIMD}; s++)
                            acc += (ap_int<64>){func_name}_W[oc][sf * {SIMD} + s]
                                 * (ap_int<64>)x[sf][s];
                    ob.range(ti * {PB} + pe * {ACCU} + {ACCU} - 1, ti * {PB} + pe * {ACCU})
                        = (ap_uint<{ACCU}>)acc;
                }}
            p.write(ob);
        }}
    }}
}}
"""


def _kt_core_twin(p, t, func_name, B):
    """K-tiled weight-stationary C twin (fully-spatial per-tile, SF=NF=1). Input is one
    wide beat of ``KT*AB`` (all K_pad activations, contiguous since AB=SIMD*AW); output
    is one wide beat of ``KT*PB`` holding each tile's PARTIAL for all N columns. Tile ti
    reduces K-slice rows ``ti*SIMD .. ti*SIMD+SIMD-1`` -- matching its memstream slice."""
    PE, SIMD = t["pe"], t["simd"]
    ACCU = t["accu_width"]
    AW = t["activation_width"]
    AB, PB = t["input_stream_width_ba"], t["output_stream_width_ba"]
    KT = p["k_tiles"]
    A_TOTAL, PB_TOTAL = KT * AB, KT * PB
    M, K, N = p["num_input_vectors"], p["k_pad"], p["n"]
    actt = _act_ctype(t["signed_activations"], AW)
    wlit = _w_matrix_literal(B, N, K)
    pad = ' ' * (len(func_name) + 6)
    apmax = max(1024, ((PB_TOTAL + 1023) // 1024 + 1) * 1024)
    return f"""#define AP_INT_MAX_W {apmax}   // wide K-tiled beats (KT*PB={PB_TOTAL}) exceed the 1024-bit default
#include <hls_stream.h>
#include <ap_int.h>

// K-tiled weight-stationary C twin of {func_name} (FINN MVU: KT={KT} K-tiles of
// SIMD={SIMD} each, PE={PE} full-N). Weights baked here match the KT memstreams the
// RTL bakes; Vitis substitutes the RTL for csynth/cosim. Per-tile integer matmul (partial).
static const long {func_name}_W[{N}][{K}] = {wlit};

void {func_name}(hls::stream<ap_uint<{A_TOTAL}> >& a,
{pad}hls::stream<ap_uint<{PB_TOTAL}> >& p) {{
    for (int vec = 0; vec < {M}; vec++) {{
        {actt} x[{K}];
        ap_uint<{A_TOTAL}> ab = a.read();
        for (int kk = 0; kk < {K}; kk++)
            x[kk] = ab.range(kk * {AW} + {AW} - 1, kk * {AW});   // contiguous K_pad lanes
        ap_uint<{PB_TOTAL}> ob = 0;
        for (int ti = 0; ti < {KT}; ti++)
            for (int pe = 0; pe < {PE}; pe++) {{
                ap_int<{ACCU}> acc = 0;
                for (int s = 0; s < {SIMD}; s++)
                    acc += (ap_int<64>){func_name}_W[pe][ti * {SIMD} + s]
                         * (ap_int<64>)x[ti * {SIMD} + s];
                ob.range(ti * {PB} + pe * {ACCU} + {ACCU} - 1, ti * {PB} + pe * {ACCU})
                    = (ap_uint<{ACCU}>)acc;
            }}
        p.write(ob);
    }}
}}
"""


def _kt_tb(p, t, top_name, seed, bias_codes, B):
    """K-tiled self-checking TB: baked weights, top ``(a, c)`` with one wide ``KT*AB``
    activation beat per vector. Independent full-K golden matmul + requant reference."""
    PE, SIMD = t["pe"], t["simd"]
    AW = t["activation_width"]
    AB = t["input_stream_width_ba"]
    KT = p["k_tiles"]
    A_TOTAL = KT * AB
    M, K, N = p["num_input_vectors"], p["k_pad"], p["n"]
    outW = p["output_width"]
    req_shift = p["product_frac"] - p["output_frac"]
    CB = ((N * outW) + 7) // 8 * 8
    wlit = _w_matrix_literal(B, N, K)
    if bias_codes:
        bias_arr = "    long bias[%d] = {%s};\n" % (N, ", ".join(str(c) for c in bias_codes))
        bias_add_tb = " + bias[o]"
    else:
        bias_arr, bias_add_tb = "", ""
    signed = bool(t["signed_activations"])
    arange = (1 << (AW - 1)) - 1 if signed else (1 << AW) - 1
    amod = min(7, arange + 1)
    aoff = 3 if signed else 0
    return f"""#include <hls_stream.h>
#include <ap_int.h>
#include <cstdio>

void {top_name}(hls::stream<ap_uint<{A_TOTAL}> >&,
{' ' * (len(top_name) + 1)}hls::stream<ap_uint<{CB}> >&);

// Independent requant reference: round-half-up shift by req_shift, then saturate
// to a signed {outW}-bit code (must match the drain's ap_fixed AP_RND/AP_SAT).
static long requant_ref(long acc) {{
    long shift = {req_shift};
    long q;
    if (shift > 0)      q = (acc + (1L << (shift - 1))) >> shift;   // round half up
    else if (shift < 0) q = acc << (-shift);
    else                q = acc;
    long qmax = (1L << ({outW} - 1)) - 1, qmin = -(1L << ({outW} - 1));
    if (q > qmax) q = qmax;
    if (q < qmin) q = qmin;
    return q;
}}

// K-tiled tile set: N={N} K={K} KT={KT} SIMD={SIMD} PE={PE}, M={M} vectors.
// Weights baked (must match the KT memstream inits the RTL loads); only activations fed.
static const long W[{N}][{K}] = {wlit};

int main() {{
    hls::stream<ap_uint<{A_TOTAL}> > a_in;
    hls::stream<ap_uint<{CB}> > c_out;

    long X[{M}][{K}], golden[{M}][{N}];
{bias_arr}    for (int v = 0; v < {M}; v++)
        for (int k = 0; k < {K}; k++) X[v][k] = ((v * 2 + k + {seed}) % {amod}) - {aoff};
    for (int v = 0; v < {M}; v++)
        for (int o = 0; o < {N}; o++) {{
            long acc = 0;
            for (int k = 0; k < {K}; k++) acc += W[o][k] * X[v][k];
            golden[v][o] = requant_ref(acc{bias_add_tb});
        }}

    for (int v = 0; v < {M}; v++) {{
        ap_uint<{A_TOTAL}> ab = 0;
        for (int k = 0; k < {K}; k++)
            ab.range(k * {AW} + {AW} - 1, k * {AW}) = (ap_uint<{AW}>)(ap_int<{AW}>)X[v][k];
        a_in.write(ab);
    }}

    {top_name}(a_in, c_out);

    int errors = 0;
    for (int v = 0; v < {M}; v++) {{
        ap_uint<{CB}> crow = c_out.read();
        for (int o = 0; o < {N}; o++) {{
            ap_int<{outW}> y = crow.range(o * {outW} + {outW} - 1, o * {outW});
            long got = (long)y, exp = golden[v][o];
            if (got != exp) {{ errors++; std::printf("MISMATCH v=%d o=%d got=%ld exp=%ld\\n", v, o, got, exp); }}
        }}
    }}
    if (errors == 0) std::printf("MVAU_PKG PASS\\n");
    else             std::printf("MVAU_PKG FAIL errors=%d\\n", errors);
    return errors;
}}
"""


def _ws_tb(p, t, top_name, seed, bias_codes, B):
    """Weight-stationary self-checking TB: baked weights, top ``(a, c)`` (no ``w``).
    Independent golden with the same round-half-up + saturate requant reference."""
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    AW = t["activation_width"]
    AB = t["input_stream_width_ba"]
    M, K, N = p["num_input_vectors"], p["k_pad"], p["n"]
    outW = p["output_width"]
    req_shift = p["product_frac"] - p["output_frac"]
    CB = ((N * outW) + 7) // 8 * 8
    wlit = _w_matrix_literal(B, N, K)
    if bias_codes:
        bias_arr = "    long bias[%d] = {%s};\n" % (N, ", ".join(str(c) for c in bias_codes))
        bias_add_tb = " + bias[o]"
    else:
        bias_arr, bias_add_tb = "", ""
    signed = bool(t["signed_activations"])
    arange = (1 << (AW - 1)) - 1 if signed else (1 << AW) - 1
    amod = min(7, arange + 1)
    aoff = 3 if signed else 0
    return f"""#include <hls_stream.h>
#include <ap_int.h>
#include <cstdio>

void {top_name}(hls::stream<ap_uint<{AB}> >&,
{' ' * (len(top_name) + 1)}hls::stream<ap_uint<{CB}> >&);

// Independent requant reference: round-half-up shift by req_shift, then saturate
// to a signed {outW}-bit code (must match the drain's ap_fixed AP_RND/AP_SAT).
static long requant_ref(long acc) {{
    long shift = {req_shift};
    long q;
    if (shift > 0)      q = (acc + (1L << (shift - 1))) >> shift;   // round half up
    else if (shift < 0) q = acc << (-shift);
    else                q = acc;
    long qmax = (1L << ({outW} - 1)) - 1, qmin = -(1L << ({outW} - 1));
    if (q > qmax) q = qmax;
    if (q < qmin) q = qmin;
    return q;
}}

// weight-stationary tile: N={N} K={K} PE={PE} SIMD={SIMD} SF={SF} NF={NF}, M={M} vectors.
// Weights baked (must match the memstream init the RTL loads); only activations fed.
static const long W[{N}][{K}] = {wlit};

int main() {{
    hls::stream<ap_uint<{AB}> > a_in;
    hls::stream<ap_uint<{CB}> > c_out;

    long X[{M}][{K}], golden[{M}][{N}];
{bias_arr}    for (int v = 0; v < {M}; v++)
        for (int k = 0; k < {K}; k++) X[v][k] = ((v * 2 + k + {seed}) % {amod}) - {aoff};
    for (int v = 0; v < {M}; v++)
        for (int o = 0; o < {N}; o++) {{
            long acc = 0;
            for (int k = 0; k < {K}; k++) acc += W[o][k] * X[v][k];
            golden[v][o] = requant_ref(acc{bias_add_tb});
        }}

    for (int v = 0; v < {M}; v++)
        for (int sf = 0; sf < {SF}; sf++) {{
            ap_uint<{AB}> ab = 0;
            for (int s = 0; s < {SIMD}; s++)
                ab.range(s * {AW} + {AW} - 1, s * {AW}) =
                    (ap_uint<{AW}>)(ap_int<{AW}>)X[v][sf * {SIMD} + s];
            a_in.write(ab);
        }}

    {top_name}(a_in, c_out);

    int errors = 0;
    for (int v = 0; v < {M}; v++) {{
        ap_uint<{CB}> crow = c_out.read();
        for (int o = 0; o < {N}; o++) {{
            ap_int<{outW}> y = crow.range(o * {outW} + {outW} - 1, o * {outW});
            long got = (long)y, exp = golden[v][o];
            if (got != exp) {{ errors++; std::printf("MISMATCH v=%d o=%d got=%ld exp=%ld\\n", v, o, got, exp); }}
        }}
    }}
    if (errors == 0) std::printf("MVAU_PKG PASS\\n");
    else             std::printf("MVAU_PKG FAIL errors=%d\\n", errors);
    return errors;
}}
"""


def generate_tb(shape, top_name="mvau_top", func_name="mvau_core", seed=42, plan=None,
                bias_codes=None, baked_weights=None, **kw):
    """Self-checking TB. Independent golden: full-K integer matmul, add per-column
    bias in the accumulator domain, then an independent affine requant (round-half-up
    shift + saturate) to the output ``fixed<outW,outI>`` code -- must equal the drain's
    ``ap_fixed<AP_RND,AP_SAT>``. Weights re-fed each vector (same W across vectors)."""
    p = _plan(shape, plan=plan, **kw)
    t = p["tile"]
    if baked_weights is not None:
        if p.get("k_tiles", 1) > 1:
            return _kt_tb(p, t, top_name, seed, bias_codes, baked_weights)
        return _ws_tb(p, t, top_name, seed, bias_codes, baked_weights)
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    WW, AW = t["weight_width"], t["activation_width"]
    WB, AB = t["weight_stream_width_ba"], t["input_stream_width_ba"]
    M, K, N = p["num_input_vectors"], p["k_pad"], p["n"]
    outW = p["output_width"]
    req_shift = p["product_frac"] - p["output_frac"]   # (fa+fb) - out_frac
    CB = ((N * outW) + 7) // 8 * 8
    if bias_codes:
        bias_arr = "    long bias[%d] = {%s};\n" % (N, ", ".join(str(c) for c in bias_codes))
        bias_add_tb = " + bias[o]"
    else:
        bias_arr, bias_add_tb = "", ""
    signed = bool(t["signed_activations"])
    wrange = (1 << (WW - 1)) - 1
    arange = (1 << (AW - 1)) - 1 if signed else (1 << AW) - 1
    wmod, amod = min(7, 2 * wrange + 1), min(7, arange + 1)
    aoff = 3 if signed else 0

    return f"""#include <hls_stream.h>
#include <ap_int.h>
#include <cstdio>

void {top_name}(hls::stream<ap_uint<{WB}> >&, hls::stream<ap_uint<{AB}> >&,
{' ' * (len(top_name) + 1)}hls::stream<ap_uint<{CB}> >&);

// Independent requant reference: round-half-up shift by req_shift, then saturate
// to a signed {outW}-bit code (must match the drain's ap_fixed AP_RND/AP_SAT).
static long requant_ref(long acc) {{
    long shift = {req_shift};
    long q;
    if (shift > 0)      q = (acc + (1L << (shift - 1))) >> shift;   // round half up
    else if (shift < 0) q = acc << (-shift);
    else                q = acc;
    long qmax = (1L << ({outW} - 1)) - 1, qmin = -(1L << ({outW} - 1));
    if (q > qmax) q = qmax;
    if (q < qmin) q = qmin;
    return q;
}}

// tile: N={N} K={K} PE={PE} SIMD={SIMD} SF={SF} NF={NF}, M={M} vectors, out=fixed<{outW},{p['output_int']}>
int main() {{
    hls::stream<ap_uint<{WB}> > w_in;
    hls::stream<ap_uint<{AB}> > a_in;
    hls::stream<ap_uint<{CB}> > c_out;

    long W[{N}][{K}], X[{M}][{K}], golden[{M}][{N}];
{bias_arr}    for (int o = 0; o < {N}; o++)
        for (int k = 0; k < {K}; k++) W[o][k] = ((o + k + {seed}) % {wmod}) - {wrange if wmod == 2*wrange+1 else 3};
    for (int v = 0; v < {M}; v++)
        for (int k = 0; k < {K}; k++) X[v][k] = ((v * 2 + k + {seed}) % {amod}) - {aoff};
    for (int v = 0; v < {M}; v++)
        for (int o = 0; o < {N}; o++) {{
            long acc = 0;
            for (int k = 0; k < {K}; k++) acc += W[o][k] * X[v][k];
            golden[v][o] = requant_ref(acc{bias_add_tb});
        }}

    for (int v = 0; v < {M}; v++) {{
        for (int sf = 0; sf < {SF}; sf++) {{
            ap_uint<{AB}> ab = 0;
            for (int s = 0; s < {SIMD}; s++)
                ab.range(s * {AW} + {AW} - 1, s * {AW}) =
                    (ap_uint<{AW}>)(ap_int<{AW}>)X[v][sf * {SIMD} + s];
            a_in.write(ab);
        }}
        for (int nf = 0; nf < {NF}; nf++)
            for (int sf = 0; sf < {SF}; sf++) {{
                ap_uint<{WB}> wb = 0;
                for (int pe = 0; pe < {PE}; pe++)
                    for (int s = 0; s < {SIMD}; s++)
                        wb.range((pe * {SIMD} + s) * {WW} + {WW} - 1, (pe * {SIMD} + s) * {WW}) =
                            (ap_uint<{WW}>)(ap_int<{WW}>)W[nf * {PE} + pe][sf * {SIMD} + s];
                w_in.write(wb);
            }}
    }}

    {top_name}(w_in, a_in, c_out);

    int errors = 0;
    for (int v = 0; v < {M}; v++) {{
        ap_uint<{CB}> crow = c_out.read();
        for (int o = 0; o < {N}; o++) {{
            ap_int<{outW}> y = crow.range(o * {outW} + {outW} - 1, o * {outW});
            long got = (long)y, exp = golden[v][o];
            if (got != exp) {{ errors++; std::printf("MISMATCH v=%d o=%d got=%ld exp=%ld\\n", v, o, got, exp); }}
        }}
    }}
    if (errors == 0) std::printf("MVAU_PKG PASS\\n");
    else             std::printf("MVAU_PKG FAIL errors=%d\\n", errors);
    return errors;
}}
"""
