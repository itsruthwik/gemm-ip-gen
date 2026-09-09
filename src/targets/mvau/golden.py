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
import weightpack as _wpack


def _plan(shape, plan=None, **kw):
    return plan if plan is not None else _geom.resolve_plan(shape, **kw)


def _requant_block(shift, out_width, acc_expr, q_var, indent=""):
    """Round-half-up shift by *shift* (accumulator's product_frac -> the output's
    frac), then wrap (drop high bits, no saturation) to a signed *out_width*-bit
    code -- the arithmetic the RTL requant stage and every C twin/golden reference
    share, so all three land on the identical code. ``acc_expr`` is a ``long``
    expression already at the accumulator's fixed-point scale (bias already added,
    if any); the result is bound to a fresh ``long {q_var}``, sign-extended from the
    wrapped low *out_width* bits (i.e. the two's-complement value that bit pattern
    represents)."""
    lines = [f"{indent}long {q_var};"]
    if shift > 0:
        lines.append(f"{indent}{q_var} = ({acc_expr} + (1L << {shift - 1})) >> {shift};")
    elif shift < 0:
        lines.append(f"{indent}{q_var} = {acc_expr} << {-shift};")
    else:
        lines.append(f"{indent}{q_var} = {acc_expr};")
    lines.append(f"{indent}{{ unsigned long _m = (1UL << {out_width}) - 1; {q_var} &= (long)_m; "
                 f"if ({q_var} & (1L << ({out_width} - 1))) {q_var} -= (1L << {out_width}); }}")
    return "\n".join(lines)


def _act_ctype(signed, aw):
    return f"ap_int<{aw}>" if signed else f"ap_uint<{aw}>"


def _w_matrix_literal(B, n, k):
    """C initializer for ``long W[N][K]`` from ``B`` (``[K][N]``): W[o][k]=B[k][o]."""
    rows = ["{" + ", ".join(str(int(B[kk][oo])) for kk in range(k)) + "}" for oo in range(n)]
    return "{" + ", ".join(rows) + "}"


def generate_core_twin(shape, func_name="mvau_core", plan=None, baked_weights=None,
                       bias_codes=None, **kw):
    """C++ behavioral twin of the blackbox (the JSON ``c_files`` model).

    ``baked_weights`` (``B`` as ``[K][N]``) selects the weight-stationary twin: the
    weights are baked into the model (no ``w`` stream), matching the memstream the
    RTL bakes. Absent -> the streamed twin (``w`` stream, reserved for two-operand).

    The twin now performs the *whole* per-lane pipeline the RTL requant stage does
    (K-tile sum, bias add, shift + round-half-up + wrap) and emits the already-
    narrow ``out_width``-bit-per-lane beat, matching the shim -- the drain
    downstream (``package.py``) is a pure unpack. ``bias_codes`` (from
    ``weightpack.bias_acc_codes``, same source of truth the RTL's bias ROM bakes)
    is None when the layer has no bias.
    """
    p = _plan(shape, plan=plan, **kw)
    t = p["tile"]
    if baked_weights is not None:
        if p.get("k_tiles", 1) > 1:
            return _kt_core_twin(p, t, func_name, baked_weights, bias_codes)
        return _ws_core_twin(p, t, func_name, baked_weights, bias_codes)
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


def _ws_core_twin(p, t, func_name, B, bias_codes=None):
    """Weight-stationary C twin: baked weights (+ baked bias, if any), signature
    ``(a, p)`` (no ``w``). Per lane: matmul, + bias (accumulator/product_frac
    scale), shift + round-half-up + wrap to ``out_width`` -- identical arithmetic
    to the RTL requant stage, same baked bias codes.

    N-tiling: the result beat concatenates the ``n_tiles`` tiles; in beat ``nf``,
    tile ``ti`` lane ``pe`` (bits ``ti*PB + pe*out_width +: out_width``) holds the
    already-requantized code for global output column ``ti*n_tile + nf*PE + pe``
    -- matching the RTL shim's concatenation. Padded lanes (local_oc >=
    NTILE_REAL) are left 0 (their weight column is all-zero, so their true
    requantized value is 0 too; the drain never reads them)."""
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    ACCU = t["accu_width"]
    AW = t["activation_width"]
    AB, PB = t["input_stream_width_ba"], t["output_stream_width_ba"]
    NT, NTILE = p["n_tiles"], p["n_tile"]
    PB_TOTAL = NT * PB
    M, K, N = p["num_input_vectors"], p["k_pad"], p["n"]
    NTILE_REAL = N // NT   # unpadded per-tile column count; the interface presents only these
    outW = p["output_width"]
    shift = p["product_frac"] - p["output_frac"]
    actt = _act_ctype(t["signed_activations"], AW)
    wlit = _w_matrix_literal(B, N, K)
    pad = ' ' * (len(func_name) + 6)
    if bias_codes:
        bias_decl = ("static const long %s_bias[%d] = {%s};\n"
                      % (func_name, N, ", ".join(str(c) for c in bias_codes)))
        bias_add = f" + {func_name}_bias[oc]"
    else:
        bias_decl, bias_add = "", ""
    req = _requant_block(shift, outW, f"(long)acc{bias_add}", "q", ' ' * 20)
    return f"""#include <hls_stream.h>
#include <ap_int.h>

// Weight-stationary C twin of {func_name} (FINN MVU: PE={PE} SIMD={SIMD} SF={SF} NF={NF},
// N_TILES={NT}). Weights baked here match the memstreams the RTL bakes; Vitis
// substitutes the RTL (memstream(s) + mvu_vvu_axi) for csynth/cosim. Per lane: integer
// matmul, + baked bias (if any), shift + round-half-up + wrap to out_width -- the RTL
// requant stage's arithmetic, so this beat is already the narrow, final code.
static const long {func_name}_W[{N}][{K}] = {wlit};
{bias_decl}
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
                    int local_oc = nf * {PE} + pe;   // 0..n_pad-1 within this tile
                    if (local_oc < {NTILE_REAL}) {{   // drop the N-pad tail columns
                    int oc = ti * {NTILE_REAL} + local_oc;   // global output column
                    ap_int<{ACCU}> acc = 0;
                    for (int sf = 0; sf < {SF}; sf++)
                        for (int s = 0; s < {SIMD}; s++)
                            acc += (ap_int<64>){func_name}_W[oc][sf * {SIMD} + s]
                                 * (ap_int<64>)x[sf][s];
{req}
                    ob.range(ti * {PB} + pe * {outW} + {outW} - 1, ti * {PB} + pe * {outW})
                        = (ap_uint<{outW}>)q;
                    }}
                }}
            p.write(ob);
        }}
    }}
}}
"""


def _kt_core_twin(p, t, func_name, B, bias_codes=None):
    """General tiled weight-stationary C twin (nt×gk grid): per vector ``SF_tile`` wide
    activation beats of ``KT*AB`` stream the K-bands (K-sliced, broadcast over N-slices).
    Tile (j,i) reduces K-slice i (rows ``i*MW_tile..``) for N-slice j (cols ``j*NTILE..``);
    the twin now SUMS the ``gk`` K-partials, adds bias, and shift/round/wraps to
    ``out_width`` right here (mirrors the RTL requant stage's K-sum-then-requant order),
    so the beat this twin emits is already ``nt*PB`` (one out_width lane per N-tile
    column, K-tiling summed away) -- no separate K-sum downstream."""
    PE, SIMD, MW = t["pe"], t["simd"], t["mw"]
    ACCU = t["accu_width"]
    ACCU_SUM = p["accu_sum"]
    AW = t["activation_width"]
    AB, PB, NF = t["input_stream_width_ba"], t["output_stream_width_ba"], t["nf"]
    KT, NT = p["k_tiles"], p["n_tiles"]
    SFT = MW // SIMD
    A_TOTAL, PB_TOTAL = KT * AB, NT * PB
    M, K, N = p["num_input_vectors"], p["k_pad"], p["n"]
    NTILE = N // NT
    outW = p["output_width"]
    shift = p["product_frac"] - p["output_frac"]
    actt = _act_ctype(t["signed_activations"], AW)
    wlit = _w_matrix_literal(B, N, K)
    pad = ' ' * (len(func_name) + 6)
    apmax = max(1024, ((max(A_TOTAL, PB_TOTAL) + 1023) // 1024 + 1) * 1024)
    if bias_codes:
        bias_decl = ("static const long %s_bias[%d] = {%s};\n"
                      % (func_name, N, ", ".join(str(c) for c in bias_codes)))
        bias_add = f" + {func_name}_bias[oc]"
    else:
        bias_decl, bias_add = "", ""
    req = _requant_block(shift, outW, f"(long)raw{bias_add}", "q", ' ' * 20)
    return f"""#define AP_INT_MAX_W {apmax}   // wide grid activation beat (KT*AB={A_TOTAL}) may exceed the 1024-bit default
#include <hls_stream.h>
#include <ap_int.h>

// Weight-stationary grid C twin of {func_name} (FINN MVU: nt={NT} x gk={KT}, MW_tile={MW}
// N_tile={NTILE} SIMD={SIMD} PE={PE} SF_tile={SFT} NF={NF}). Weights baked here match the grid
// memstreams the RTL bakes; Vitis substitutes the RTL for csynth/cosim. Per lane: sum the gk
// K-partials, + baked bias (if any), shift + round-half-up + wrap to out_width -- the beat is
// already the narrow, final, K-summed code.
static const long {func_name}_W[{N}][{K}] = {wlit};
{bias_decl}
void {func_name}(hls::stream<ap_uint<{A_TOTAL}> >& a,
{pad}hls::stream<ap_uint<{PB_TOTAL}> >& p) {{
    for (int vec = 0; vec < {M}; vec++) {{
        {actt} x[{K}];
        for (int sf = 0; sf < {SFT}; sf++) {{
            ap_uint<{A_TOTAL}> ab = a.read();
            for (int i = 0; i < {KT}; i++)
                for (int s = 0; s < {SIMD}; s++)
                    x[i * {MW} + sf * {SIMD} + s] =
                        ab.range(i * {AB} + s * {AW} + {AW} - 1, i * {AB} + s * {AW});
        }}
        for (int nf = 0; nf < {NF}; nf++) {{
            ap_uint<{PB_TOTAL}> ob = 0;
            for (int j = 0; j < {NT}; j++)
                for (int pe = 0; pe < {PE}; pe++) {{
                    int local_oc = nf * {PE} + pe;   // 0..n_pad-1 within this tile
                    if (local_oc < {NTILE}) {{        // drop the N-pad tail columns
                    int oc = j * {NTILE} + local_oc;
                    ap_int<{ACCU_SUM}> raw = 0;
                    for (int i = 0; i < {KT}; i++) {{
                        ap_int<{ACCU}> acc = 0;
                        for (int kk = 0; kk < {MW}; kk++)
                            acc += (ap_int<64>){func_name}_W[oc][i * {MW} + kk]
                                 * (ap_int<64>)x[i * {MW} + kk];
                        raw += acc;               // K-tile sum first, mirrors the RTL adder
                    }}
{req}
                    ob.range(j * {PB} + pe * {outW} + {outW} - 1, j * {PB} + pe * {outW})
                        = (ap_uint<{outW}>)q;
                    }}
                }}
            p.write(ob);
        }}
    }}
}}
"""


def _kt_tb(p, t, top_name, seed, bias_codes, B, n_nodes=6):
    """K-tiled self-checking TB: baked weights, top ``(a, c)`` with one wide ``KT*AB``
    activation beat per vector. Independent full-K golden matmul + requant reference.

    Weights are baked (shared across nodes); ``n_nodes`` distinct activation sets are
    queued back-to-back (all writes before any call), then the top is invoked
    ``n_nodes`` times and all ``n_nodes*M`` output rows are drained and checked --
    exercising overlapped/pipelined invocations of the blackbox core in cosim."""
    NN = n_nodes
    PE, SIMD, MW = t["pe"], t["simd"], t["mw"]
    AW = t["activation_width"]
    AB = t["input_stream_width_ba"]
    KT = p["k_tiles"]
    SFT = MW // SIMD
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

// Independent requant reference: round-half-up shift by req_shift, then wrap
// to a signed {outW}-bit code (must match the drain's ap_fixed AP_RND/AP_WRAP).
static long requant_ref(long acc) {{
    long shift = {req_shift};
    long q;
    if (shift > 0)      q = (acc + (1L << (shift - 1))) >> shift;   // round half up
    else if (shift < 0) q = acc << (-shift);
    else                q = acc;
    unsigned long mask = (1UL << {outW}) - 1;
    q &= (long)mask;
    if (q & (1L << ({outW} - 1))) q -= (1L << {outW});   // wrap: keep only the low outW bits
    return q;
}}

// K-tiled tile set: N={N} K={K} KT={KT} SIMD={SIMD} PE={PE}, M={M} vectors.
// Weights baked (must match the KT memstream inits the RTL loads); only activations fed.
static const long W[{N}][{K}] = {wlit};

#define NN {NN}

int main() {{
    hls::stream<ap_uint<{A_TOTAL}> > a_in;
    hls::stream<ap_uint<{CB}> > c_out;

    static long X[NN][{M}][{K}], golden[NN][{M}][{N}];
{bias_arr}    for (int n = 0; n < NN; n++) {{
        for (int v = 0; v < {M}; v++)
            for (int k = 0; k < {K}; k++) X[n][v][k] = ((v * 2 + k + 3 * n + {seed}) % {amod}) - {aoff};
        for (int v = 0; v < {M}; v++)
            for (int o = 0; o < {N}; o++) {{
                long acc = 0;
                for (int k = 0; k < {K}; k++) acc += W[o][k] * X[n][v][k];
                golden[n][v][o] = requant_ref(acc{bias_add_tb});
            }}

        for (int v = 0; v < {M}; v++)
            for (int sf = 0; sf < {SFT}; sf++) {{
                ap_uint<{A_TOTAL}> ab = 0;
                for (int ti = 0; ti < {KT}; ti++)
                    for (int s = 0; s < {SIMD}; s++)
                        ab.range(ti * {AB} + s * {AW} + {AW} - 1, ti * {AB} + s * {AW}) =
                            (ap_uint<{AW}>)(ap_int<{AW}>)X[n][v][ti * {MW} + sf * {SIMD} + s];
                a_in.write(ab);
            }}
    }}

    for (int n = 0; n < NN; n++) {top_name}(a_in, c_out);

    int errors = 0;
    for (int n = 0; n < NN; n++)
        for (int v = 0; v < {M}; v++) {{
            ap_uint<{CB}> crow = c_out.read();
            for (int o = 0; o < {N}; o++) {{
                ap_int<{outW}> y = crow.range(o * {outW} + {outW} - 1, o * {outW});
                long got = (long)y, exp = golden[n][v][o];
                if (got != exp) {{ errors++; std::printf("MISMATCH n=%d v=%d o=%d got=%ld exp=%ld\\n", n, v, o, got, exp); }}
            }}
        }}
    if (errors == 0) std::printf("MVAU_PKG PASS\\n");
    else             std::printf("MVAU_PKG FAIL errors=%d\\n", errors);
    return errors;
}}
"""


def _2op_core_twin(p, t, func_name):
    """Two-operand C twin (NF=1 single tile): both A and B are runtime streams. Reads B
    as ``K`` N-wide K-row beats (``b[pe]=B[k][pe]``) into residency, then per input vector
    reads ``SF`` activation beats and emits one out_width-bit-per-lane requantized output
    beat -- two-operand GEMM never has a real bias (has_bias is always False by
    construction), so the per-lane pipeline is just shift + round-half-up + wrap, matching
    the RTL requant stage."""
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    WW, AW, ACCU = t["weight_width"], t["activation_width"], t["accu_width"]
    AB, PB = t["input_stream_width_ba"], t["output_stream_width_ba"]
    N, K, M = p["n"], p["k_pad"], p["num_input_vectors"]
    BB = ((N * WW) + 7) // 8 * 8
    outW = p["output_width"]
    shift = p["product_frac"] - p["output_frac"]
    actt = _act_ctype(t["signed_activations"], AW)
    pad = ' ' * (len(func_name) + 6)
    req = _requant_block(shift, outW, "(long)acc", "q", ' ' * 16)
    return f"""#include <hls_stream.h>
#include <ap_int.h>

// Two-operand C twin of {func_name} (FINN MVU: PE={PE} SIMD={SIMD} SF={SF} NF={NF}). B is a
// runtime stream (N-wide K-row beats), buffered then replayed across M vectors; A streams
// per vector. Vitis substitutes the RTL for csynth/cosim. Per lane: integer matmul, shift +
// round-half-up + wrap to out_width (no bias -- two-operand GEMM never has one).
void {func_name}(hls::stream<ap_uint<{AB}> >& a,
{pad}hls::stream<ap_uint<{BB}> >& b,
{pad}hls::stream<ap_uint<{PB}> >& p) {{
    ap_int<{WW}> W[{N}][{K}];              // W[o][k] = B[k][o]
    for (int k = 0; k < {K}; k++) {{
        ap_uint<{BB}> bb = b.read();       // one N-wide K-row: bb[o] = B[k][o]
        for (int o = 0; o < {N}; o++)
            W[o][k] = bb.range(o * {WW} + {WW} - 1, o * {WW});
    }}
    for (int vec = 0; vec < {M}; vec++) {{
        {actt} x[{K}];
        for (int sf = 0; sf < {SF}; sf++) {{
            ap_uint<{AB}> ab = a.read();
            for (int s = 0; s < {SIMD}; s++)
                x[sf * {SIMD} + s] = ab.range(s * {AW} + {AW} - 1, s * {AW});
        }}
        for (int nf = 0; nf < {NF}; nf++) {{        // one output beat per column block
            ap_uint<{PB}> ob = 0;
            for (int pe = 0; pe < {PE}; pe++) {{
                int oc = nf * {PE} + pe;
                if (oc < {N}) {{   // drop the N-pad tail columns (untiled: n_tile == n)
                ap_int<{ACCU}> acc = 0;
                for (int k = 0; k < {K}; k++)
                    acc += (ap_int<64>)W[oc][k] * (ap_int<64>)x[k];
{req}
                ob.range(pe * {outW} + {outW} - 1, pe * {outW}) = (ap_uint<{outW}>)q;
                }}
            }}
            p.write(ob);
        }}
    }}
}}
"""


def _2op_tb(p, t, top_name, func_name, seed, n_nodes=6):
    """Two-operand self-checking TB (no bias): feed synthetic A (M×K) and B (K×N, as N-wide
    K-row beats), golden = integer matmul + affine requant, compare the requantized C rows.

    ``n_nodes`` distinct (A,B) pairs are queued back-to-back -- all writes (per node, B
    then A, in the existing per-node order) happen before any call -- then the top is
    invoked ``n_nodes`` times and all ``n_nodes*M`` output rows are drained and checked,
    exercising overlapped/pipelined invocations of the blackbox core in cosim."""
    NN = n_nodes
    PE, SIMD, SF = t["pe"], t["simd"], t["sf"]
    WW, AW = t["weight_width"], t["activation_width"]
    AB = t["input_stream_width_ba"]
    N, K, M = p["n"], p["k_pad"], p["num_input_vectors"]
    BB = ((N * WW) + 7) // 8 * 8
    outW = p["output_width"]
    req_shift = p["product_frac"] - p["output_frac"]
    CB = ((N * outW) + 7) // 8 * 8
    signed = bool(t["signed_activations"])
    wrange = (1 << (WW - 1)) - 1
    arange = (1 << (AW - 1)) - 1 if signed else (1 << AW) - 1
    wmod, amod = min(7, 2 * wrange + 1), min(7, arange + 1)
    woff = wrange if wmod == 2 * wrange + 1 else 3
    aoff = 3 if signed else 0
    return f"""#include <hls_stream.h>
#include <ap_int.h>
#include <cstdio>

void {top_name}(hls::stream<ap_uint<{AB}> >&, hls::stream<ap_uint<{BB}> >&,
{' ' * (len(top_name) + 1)}hls::stream<ap_uint<{CB}> >&);

static long requant_ref(long acc) {{
    long shift = {req_shift};
    long q;
    if (shift > 0)      q = (acc + (1L << (shift - 1))) >> shift;
    else if (shift < 0) q = acc << (-shift);
    else                q = acc;
    unsigned long mask = (1UL << {outW}) - 1;
    q &= (long)mask;
    if (q & (1L << ({outW} - 1))) q -= (1L << {outW});   // wrap: keep only the low outW bits
    return q;
}}

// two-operand tile: N={N} K={K} PE={PE} SIMD={SIMD} SF={SF} NF=1, M={M} vectors, no bias.
#define NN {NN}

int main() {{
    hls::stream<ap_uint<{AB}> > a_in;
    hls::stream<ap_uint<{BB}> > b_in;
    hls::stream<ap_uint<{CB}> > c_out;

    static long Bm[NN][{K}][{N}], X[NN][{M}][{K}], golden[NN][{M}][{N}];
    for (int n = 0; n < NN; n++) {{
        for (int k = 0; k < {K}; k++)
            for (int o = 0; o < {N}; o++) Bm[n][k][o] = ((o + k + 5 * n + {seed}) % {wmod}) - {woff};
        for (int v = 0; v < {M}; v++)
            for (int k = 0; k < {K}; k++) X[n][v][k] = ((v * 2 + k + 3 * n + {seed}) % {amod}) - {aoff};
        for (int v = 0; v < {M}; v++)
            for (int o = 0; o < {N}; o++) {{
                long acc = 0;
                for (int k = 0; k < {K}; k++) acc += Bm[n][k][o] * X[n][v][k];
                golden[n][v][o] = requant_ref(acc);
            }}

        // B first: K beats, each an N-wide K-row (b[o] = B[k][o])
        for (int k = 0; k < {K}; k++) {{
            ap_uint<{BB}> bb = 0;
            for (int o = 0; o < {N}; o++)
                bb.range(o * {WW} + {WW} - 1, o * {WW}) = (ap_uint<{WW}>)(ap_int<{WW}>)Bm[n][k][o];
            b_in.write(bb);
        }}
        // then A: M vectors, SF beats of SIMD activations each
        for (int v = 0; v < {M}; v++)
            for (int sf = 0; sf < {SF}; sf++) {{
                ap_uint<{AB}> ab = 0;
                for (int s = 0; s < {SIMD}; s++)
                    ab.range(s * {AW} + {AW} - 1, s * {AW}) =
                        (ap_uint<{AW}>)(ap_int<{AW}>)X[n][v][sf * {SIMD} + s];
                a_in.write(ab);
            }}
    }}

    for (int n = 0; n < NN; n++) {top_name}(a_in, b_in, c_out);

    int errors = 0;
    for (int n = 0; n < NN; n++)
        for (int v = 0; v < {M}; v++) {{
            ap_uint<{CB}> crow = c_out.read();
            for (int o = 0; o < {N}; o++) {{
                ap_int<{outW}> y = crow.range(o * {outW} + {outW} - 1, o * {outW});
                long got = (long)y, exp = golden[n][v][o];
                if (got != exp) {{ errors++; std::printf("MISMATCH n=%d v=%d o=%d got=%ld exp=%ld\\n", n, v, o, got, exp); }}
            }}
        }}
    if (errors == 0) std::printf("MVAU_PKG PASS\\n");
    else             std::printf("MVAU_PKG FAIL errors=%d\\n", errors);
    return errors;
}}
"""


def _2op_kt_core_twin(p, t, func_name):
    """General tiled two-operand C twin (nt×gk grid): B is a runtime stream (N-wide K-row
    beats) buffered into W[N][K]; per vector ``SF_tile`` wide activation beats of gk*AB stream
    the K-bands; per vector ``NF`` output beats carry the nt PE-wide requantized codes (tile
    (j,i) = N-slice j, K-slice i) -- the twin sums the gk K-partials, then shift/round/wraps to
    out_width right here (no bias -- two-operand GEMM never has one), so the beat this twin
    emits is already ``nt*PB`` (K-tiling summed away), mirroring the RTL requant stage."""
    PE, SIMD, MW = t["pe"], t["simd"], t["mw"]
    WW, AW, ACCU = t["weight_width"], t["activation_width"], t["accu_width"]
    AB, PB, NF = t["input_stream_width_ba"], t["output_stream_width_ba"], t["nf"]
    gk, nt = p["k_tiles"], p["n_tiles"]
    ACCU_SUM = p["accu_sum"]
    SFT = MW // SIMD
    N, K, M = p["n"], p["k_pad"], p["num_input_vectors"]
    NTILE = N // nt
    BB = ((N * WW) + 7) // 8 * 8
    A_TOTAL, PB_TOTAL = gk * AB, nt * PB
    outW = p["output_width"]
    shift = p["product_frac"] - p["output_frac"]
    actt = _act_ctype(t["signed_activations"], AW)
    apmax = max(1024, ((A_TOTAL + 1023) // 1024 + 1) * 1024)
    pad = ' ' * (len(func_name) + 6)
    req = _requant_block(shift, outW, "(long)raw", "q", ' ' * 24)
    return f"""#define AP_INT_MAX_W {apmax}   // wide grid activation beat (gk*AB={A_TOTAL}) may exceed the 1024-bit default
#include <hls_stream.h>
#include <ap_int.h>

// General tiled two-operand C twin of {func_name} (nt={nt} x gk={gk} grid; MW_tile={MW}
// N_tile={NTILE} SIMD={SIMD} PE={PE} SF_tile={SFT} NF={NF}). A SF_tile beats/vector; B replayed.
// Per lane: sum the gk K-partials, shift + round-half-up + wrap to out_width (no bias).
void {func_name}(hls::stream<ap_uint<{A_TOTAL}> >& a,
{pad}hls::stream<ap_uint<{BB}> >& b,
{pad}hls::stream<ap_uint<{PB_TOTAL}> >& p) {{
    ap_int<{WW}> W[{N}][{K}];
    for (int k = 0; k < {K}; k++) {{
        ap_uint<{BB}> bb = b.read();
        for (int o = 0; o < {N}; o++) W[o][k] = bb.range(o * {WW} + {WW} - 1, o * {WW});
    }}
    for (int vec = 0; vec < {M}; vec++) {{
        {actt} x[{K}];
        for (int sf = 0; sf < {SFT}; sf++) {{
            ap_uint<{A_TOTAL}> ab = a.read();
            for (int i = 0; i < {gk}; i++)
                for (int s = 0; s < {SIMD}; s++)
                    x[i * {MW} + sf * {SIMD} + s] =
                        ab.range(i * {AB} + s * {AW} + {AW} - 1, i * {AB} + s * {AW});
        }}
        for (int nf = 0; nf < {NF}; nf++) {{
            ap_uint<{PB_TOTAL}> ob = 0;
            for (int j = 0; j < {nt}; j++) {{
                    int local_oc = nf * {PE};   // pe added below; 0..n_pad-1 within this tile
                    for (int pe = 0; pe < {PE}; pe++) {{
                    if (local_oc + pe < {NTILE}) {{   // drop the N-pad tail columns
                    int oc = j * {NTILE} + local_oc + pe;
                    ap_int<{ACCU_SUM}> raw = 0;
                    for (int i = 0; i < {gk}; i++) {{
                        ap_int<{ACCU}> acc = 0;
                        for (int kk = 0; kk < {MW}; kk++)
                            acc += (ap_int<64>)W[oc][i * {MW} + kk] * (ap_int<64>)x[i * {MW} + kk];
                        raw += acc;
                    }}
{req}
                    ob.range(j * {PB} + pe * {outW} + {outW} - 1, j * {PB} + pe * {outW}) = (ap_uint<{outW}>)q;
                    }}
                    }}
            }}
            p.write(ob);
        }}
    }}
}}
"""


def _2op_kt_tb(p, t, top_name, func_name, seed, n_nodes=6):
    """K-tiled two-operand self-checking TB: feed A as M wide beats (gk*AB, tile-sliced) and
    B as K N-wide K-row beats; golden = full-K integer matmul + affine requant (no bias).

    ``n_nodes`` distinct (A,B) pairs are queued back-to-back before any call, then the
    top is invoked ``n_nodes`` times and all rows drained/checked (overlapped invocations)."""
    NN = n_nodes
    PE, SIMD, MW = t["pe"], t["simd"], t["mw"]
    WW, AW = t["weight_width"], t["activation_width"]
    AB = t["input_stream_width_ba"]
    gk = p["k_tiles"]
    SFT = MW // SIMD
    N, K, M = p["n"], p["k_pad"], p["num_input_vectors"]
    BB = ((N * WW) + 7) // 8 * 8
    A_TOTAL = gk * AB
    outW = p["output_width"]
    req_shift = p["product_frac"] - p["output_frac"]
    CB = ((N * outW) + 7) // 8 * 8
    signed = bool(t["signed_activations"])
    wrange = (1 << (WW - 1)) - 1
    arange = (1 << (AW - 1)) - 1 if signed else (1 << AW) - 1
    wmod, amod = min(7, 2 * wrange + 1), min(7, arange + 1)
    woff = wrange if wmod == 2 * wrange + 1 else 3
    aoff = 3 if signed else 0
    return f"""#include <hls_stream.h>
#include <ap_int.h>
#include <cstdio>

void {top_name}(hls::stream<ap_uint<{A_TOTAL}> >&, hls::stream<ap_uint<{BB}> >&,
{' ' * (len(top_name) + 1)}hls::stream<ap_uint<{CB}> >&);

static long requant_ref(long acc) {{
    long shift = {req_shift};
    long q;
    if (shift > 0)      q = (acc + (1L << (shift - 1))) >> shift;
    else if (shift < 0) q = acc << (-shift);
    else                q = acc;
    unsigned long mask = (1UL << {outW}) - 1;
    q &= (long)mask;
    if (q & (1L << ({outW} - 1))) q -= (1L << {outW});   // wrap: keep only the low outW bits
    return q;
}}

// K-tiled two-operand: N={N} K={K} gk={gk} SIMD={SIMD} PE={PE}, M={M} vectors, no bias.
#define NN {NN}

int main() {{
    hls::stream<ap_uint<{A_TOTAL}> > a_in;
    hls::stream<ap_uint<{BB}> > b_in;
    hls::stream<ap_uint<{CB}> > c_out;

    static long Bm[NN][{K}][{N}], X[NN][{M}][{K}], golden[NN][{M}][{N}];
    for (int n = 0; n < NN; n++) {{
        for (int k = 0; k < {K}; k++)
            for (int o = 0; o < {N}; o++) Bm[n][k][o] = ((o + k + 5 * n + {seed}) % {wmod}) - {woff};
        for (int v = 0; v < {M}; v++)
            for (int k = 0; k < {K}; k++) X[n][v][k] = ((v * 2 + k + 3 * n + {seed}) % {amod}) - {aoff};
        for (int v = 0; v < {M}; v++)
            for (int o = 0; o < {N}; o++) {{
                long acc = 0;
                for (int k = 0; k < {K}; k++) acc += Bm[n][k][o] * X[n][v][k];
                golden[n][v][o] = requant_ref(acc);
            }}

        for (int k = 0; k < {K}; k++) {{
            ap_uint<{BB}> bb = 0;
            for (int o = 0; o < {N}; o++)
                bb.range(o * {WW} + {WW} - 1, o * {WW}) = (ap_uint<{WW}>)(ap_int<{WW}>)Bm[n][k][o];
            b_in.write(bb);
        }}
        for (int v = 0; v < {M}; v++)
            for (int sf = 0; sf < {SFT}; sf++) {{
                ap_uint<{A_TOTAL}> ab = 0;
                for (int ti = 0; ti < {gk}; ti++)
                    for (int s = 0; s < {SIMD}; s++)
                        ab.range(ti * {AB} + s * {AW} + {AW} - 1, ti * {AB} + s * {AW}) =
                            (ap_uint<{AW}>)(ap_int<{AW}>)X[n][v][ti * {MW} + sf * {SIMD} + s];
                a_in.write(ab);
            }}
    }}

    for (int n = 0; n < NN; n++) {top_name}(a_in, b_in, c_out);

    int errors = 0;
    for (int n = 0; n < NN; n++)
        for (int v = 0; v < {M}; v++) {{
            ap_uint<{CB}> crow = c_out.read();
            for (int o = 0; o < {N}; o++) {{
                ap_int<{outW}> y = crow.range(o * {outW} + {outW} - 1, o * {outW});
                long got = (long)y, exp = golden[n][v][o];
                if (got != exp) {{ errors++; std::printf("MISMATCH n=%d v=%d o=%d got=%ld exp=%ld\\n", n, v, o, got, exp); }}
            }}
        }}
    if (errors == 0) std::printf("MVAU_PKG PASS\\n");
    else             std::printf("MVAU_PKG FAIL errors=%d\\n", errors);
    return errors;
}}
"""


def _2op_use_kt(p):
    """The K-tiled register path serves both real K-tiling (gk>1) and the fully-spatial
    single tile (SF=NF=1); the memstream single-tile path serves only the temporal fold."""
    t = p["tile"]
    return p.get("k_tiles", 1) > 1 or (t["sf"] == 1 and t["nf"] == 1)


def generate_2op_core_twin(shape, func_name="mvau_core", plan=None, **kw):
    """Public entry for the two-operand C twin. The general nt×gk grid twin covers N-tiling,
    K-tiling and the register single-tile; the single-tile memstream twin serves the untiled
    temporal fold."""
    p = _plan(shape, plan=plan, **kw)
    if p.get("n_tiles", 1) > 1 or _2op_use_kt(p):
        return _2op_kt_core_twin(p, p["tile"], func_name)
    return _2op_core_twin(p, p["tile"], func_name)


def generate_2op_tb(shape, top_name="mvau_top", func_name="mvau_core", seed=42, plan=None,
                    n_nodes=6, **kw):
    """Public entry for the two-operand self-checking TB (grid for tiled, else single-tile).

    ``n_nodes`` (default 6) queues that many independent invocations back-to-back with
    all inputs pre-queued before the first call, so cosim exercises overlapped
    invocations of the blackbox core."""
    p = _plan(shape, plan=plan, **kw)
    if p.get("n_tiles", 1) > 1 or _2op_use_kt(p):
        return _2op_kt_tb(p, p["tile"], top_name, func_name, seed, n_nodes=n_nodes)
    return _2op_tb(p, p["tile"], top_name, func_name, seed, n_nodes=n_nodes)


def _ws_tb(p, t, top_name, seed, bias_codes, B, n_nodes=6):
    """Weight-stationary self-checking TB: baked weights, top ``(a, c)`` (no ``w``).
    Independent golden with the same round-half-up + wrap requant reference.

    Weights are shared across nodes (baked); ``n_nodes`` distinct activation sets are
    queued before any call, then the top is invoked ``n_nodes`` times back-to-back."""
    NN = n_nodes
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

// Independent requant reference: round-half-up shift by req_shift, then wrap
// to a signed {outW}-bit code (must match the drain's ap_fixed AP_RND/AP_WRAP).
static long requant_ref(long acc) {{
    long shift = {req_shift};
    long q;
    if (shift > 0)      q = (acc + (1L << (shift - 1))) >> shift;   // round half up
    else if (shift < 0) q = acc << (-shift);
    else                q = acc;
    unsigned long mask = (1UL << {outW}) - 1;
    q &= (long)mask;
    if (q & (1L << ({outW} - 1))) q -= (1L << {outW});   // wrap: keep only the low outW bits
    return q;
}}

// weight-stationary tile: N={N} K={K} PE={PE} SIMD={SIMD} SF={SF} NF={NF}, M={M} vectors.
// Weights baked (must match the memstream init the RTL loads); only activations fed.
static const long W[{N}][{K}] = {wlit};
#define NN {NN}

int main() {{
    hls::stream<ap_uint<{AB}> > a_in;
    hls::stream<ap_uint<{CB}> > c_out;

    static long X[NN][{M}][{K}], golden[NN][{M}][{N}];
{bias_arr}    for (int n = 0; n < NN; n++) {{
        for (int v = 0; v < {M}; v++)
            for (int k = 0; k < {K}; k++) X[n][v][k] = ((v * 2 + k + 3 * n + {seed}) % {amod}) - {aoff};
        for (int v = 0; v < {M}; v++)
            for (int o = 0; o < {N}; o++) {{
                long acc = 0;
                for (int k = 0; k < {K}; k++) acc += W[o][k] * X[n][v][k];
                golden[n][v][o] = requant_ref(acc{bias_add_tb});
            }}

        for (int v = 0; v < {M}; v++)
            for (int sf = 0; sf < {SF}; sf++) {{
                ap_uint<{AB}> ab = 0;
                for (int s = 0; s < {SIMD}; s++)
                    ab.range(s * {AW} + {AW} - 1, s * {AW}) =
                        (ap_uint<{AW}>)(ap_int<{AW}>)X[n][v][sf * {SIMD} + s];
                a_in.write(ab);
            }}
    }}

    for (int n = 0; n < NN; n++) {top_name}(a_in, c_out);

    int errors = 0;
    for (int n = 0; n < NN; n++)
        for (int v = 0; v < {M}; v++) {{
            ap_uint<{CB}> crow = c_out.read();
            for (int o = 0; o < {N}; o++) {{
                ap_int<{outW}> y = crow.range(o * {outW} + {outW} - 1, o * {outW});
                long got = (long)y, exp = golden[n][v][o];
                if (got != exp) {{ errors++; std::printf("MISMATCH n=%d v=%d o=%d got=%ld exp=%ld\\n", n, v, o, got, exp); }}
            }}
        }}
    if (errors == 0) std::printf("MVAU_PKG PASS\\n");
    else             std::printf("MVAU_PKG FAIL errors=%d\\n", errors);
    return errors;
}}
"""


def generate_tb(shape, top_name="mvau_top", func_name="mvau_core", seed=42, plan=None,
                bias_codes=None, baked_weights=None, n_nodes=6, **kw):
    """Self-checking TB. Independent golden: full-K integer matmul, add per-column
    bias in the accumulator domain, then an independent affine requant (round-half-up
    shift + wrap) to the output ``fixed<outW,outI>`` code -- must equal the drain's
    ``ap_fixed<AP_RND,AP_WRAP>``. Weights re-fed each vector (same W across vectors)."""
    p = _plan(shape, plan=plan, **kw)
    t = p["tile"]
    if baked_weights is not None:
        if p.get("k_tiles", 1) > 1:
            return _kt_tb(p, t, top_name, seed, bias_codes, baked_weights, n_nodes=n_nodes)
        return _ws_tb(p, t, top_name, seed, bias_codes, baked_weights, n_nodes=n_nodes)
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

// Independent requant reference: round-half-up shift by req_shift, then wrap
// to a signed {outW}-bit code (must match the drain's ap_fixed AP_RND/AP_WRAP).
static long requant_ref(long acc) {{
    long shift = {req_shift};
    long q;
    if (shift > 0)      q = (acc + (1L << (shift - 1))) >> shift;   // round half up
    else if (shift < 0) q = acc << (-shift);
    else                q = acc;
    unsigned long mask = (1UL << {outW}) - 1;
    q &= (long)mask;
    if (q & (1L << ({outW} - 1))) q -= (1L << {outW});   // wrap: keep only the low outW bits
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
