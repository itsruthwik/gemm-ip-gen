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

from . import geometry as _geom
from . import weightpack as _wpack


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

    Boundary is now UNPADDED, matching the RTL wrapper and hls4ml's own row
    widths: ``a`` is one raw ``K_raw*AW``-bit beat (one hls4ml row; zero-padded to
    ``k_pad`` right here, mirroring the RTL's ``arow_reg``) and ``p`` is one raw
    ``N*out_width``-bit beat holding all ``N`` unpadded output columns in order
    (``oc = ti*NTILE_REAL + local_oc``), the N-tile stitching and pad-column drop
    already done -- matching the RTL shim's ``orow_reg``/``p_din``. Padded lanes
    (local_oc >= NTILE_REAL) are simply never written to the output beat."""
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    ACCU = t["accu_width"]
    AW = t["activation_width"]
    NT, NTILE = p["n_tiles"], p["n_tile"]
    M, KPAD, N = p["num_input_vectors"], p["k_pad"], p["n"]
    K_RAW = p["k"]   # unpadded K -- the beat this twin's boundary actually carries
    AB = K_RAW * AW   # raw, unpadded activation beat (one hls4ml row)
    NTILE_REAL = N // NT   # unpadded per-tile column count; the interface presents only these
    PB = N * p["output_width"]   # raw, unpadded result beat (one hls4ml row)
    outW = p["output_width"]
    shift = p["product_frac"] - p["output_frac"]
    actt = _act_ctype(t["signed_activations"], AW)
    wlit = _w_matrix_literal(B, N, KPAD)
    pad = ' ' * (len(func_name) + 6)
    if bias_codes:
        bias_decl = ("static const long %s_bias[%d] = {%s};\n"
                      % (func_name, N, ", ".join(str(c) for c in bias_codes)))
        bias_add = f" + {func_name}_bias[oc]"
    else:
        bias_decl, bias_add = "", ""
    req = _requant_block(shift, outW, f"(long)acc{bias_add}", "q", ' ' * 12)
    return f"""#include <hls_stream.h>
#include <ap_int.h>

// Weight-stationary C twin of {func_name} (FINN MVU: PE={PE} SIMD={SIMD} SF={SF} NF={NF},
// N_TILES={NT}). Weights baked here match the memstreams the RTL bakes; Vitis
// substitutes the RTL (memstream(s) + mvu_vvu_axi) for csynth/cosim. Per lane: integer
// matmul, + baked bias (if any), shift + round-half-up + wrap to out_width -- the RTL
// requant stage's arithmetic, so this beat is already the narrow, final code. K-padding
// (zero-fill to k_pad) and N-tile stitching/pad-drop happen here too, mirroring the RTL
// wrapper -- the boundary is hls4ml's own unpadded row width on both sides.
static const long {func_name}_W[{N}][{KPAD}] = {wlit};
{bias_decl}
void {func_name}(hls::stream<ap_uint<{AB}> >& a,
{pad}hls::stream<ap_uint<{PB}> >& p) {{
    for (int vec = 0; vec < {M}; vec++) {{
        {actt} x[{KPAD}];
        {{
            ap_uint<{AB}> ab = a.read();
            for (int k = 0; k < {K_RAW}; k++) x[k] = ab.range(k * {AW} + {AW} - 1, k * {AW});
            for (int k = {K_RAW}; k < {KPAD}; k++) x[k] = 0;   // zero-fill the K-pad lanes
        }}
        ap_uint<{PB}> ob = 0;
        for (int ti = 0; ti < {NT}; ti++)
            for (int local_oc = 0; local_oc < {NTILE_REAL}; local_oc++) {{
                int oc = ti * {NTILE_REAL} + local_oc;   // global output column
                ap_int<{ACCU}> acc = 0;
                for (int k = 0; k < {KPAD}; k++)
                    acc += (ap_int<64>){func_name}_W[oc][k] * (ap_int<64>)x[k];
{req}
                ob.range(oc * {outW} + {outW} - 1, oc * {outW}) = (ap_uint<{outW}>)q;
            }}
        p.write(ob);
    }}
}}
"""


def _kt_core_twin(p, t, func_name, B, bias_codes=None):
    """General tiled weight-stationary C twin (nt×gk grid). Boundary is now UNPADDED,
    matching the RTL wrapper and hls4ml's own row widths: ``a`` is one raw
    ``K_raw*AW``-bit beat (one hls4ml row; zero-padded to the grid's total ``k_pad``
    right here, mirroring the RTL's ``arow_reg``) and ``p`` is one raw
    ``N*out_width``-bit beat holding all ``N`` unpadded output columns in order
    (``oc = j*NTILE_REAL + local_oc``), K-tile summed, N-tile stitched, and the
    pad-column drop already done -- matching the RTL shim's ``orow_reg``/``p_din``.
    Tile (j,i) reduces K-slice i (rows ``i*MW_tile..``) for N-slice j
    (cols ``j*NTILE..``); the twin SUMS the ``gk`` K-partials, adds bias, and
    shift/round/wraps to ``out_width`` right here (mirrors the RTL requant stage's
    K-sum-then-requant order)."""
    PE, SIMD, MW = t["pe"], t["simd"], t["mw"]
    ACCU = t["accu_width"]
    ACCU_SUM = p["accu_sum"]
    AW = t["activation_width"]
    NF = t["nf"]
    KT, NT = p["k_tiles"], p["n_tiles"]
    SFT = MW // SIMD
    M, N = p["num_input_vectors"], p["n"]
    KPAD = p["k_pad"]           # grid's total padded K (MW*KT)
    K_RAW = p["k"]               # unpadded K -- the beat this twin's boundary actually carries
    AB = K_RAW * AW              # raw, unpadded activation beat (one hls4ml row)
    NTILE = N // NT
    PB = N * p["output_width"]   # raw, unpadded result beat (one hls4ml row)
    outW = p["output_width"]
    shift = p["product_frac"] - p["output_frac"]
    actt = _act_ctype(t["signed_activations"], AW)
    wlit = _w_matrix_literal(B, N, KPAD)
    pad = ' ' * (len(func_name) + 6)
    apmax = max(1024, ((max(AB, PB) + 1023) // 1024 + 1) * 1024)
    if bias_codes:
        bias_decl = ("static const long %s_bias[%d] = {%s};\n"
                      % (func_name, N, ", ".join(str(c) for c in bias_codes)))
        bias_add = f" + {func_name}_bias[oc]"
    else:
        bias_decl, bias_add = "", ""
    req = _requant_block(shift, outW, f"(long)raw{bias_add}", "q", ' ' * 12)
    return f"""#define AP_INT_MAX_W {apmax}   // raw K-wide activation row may exceed the 1024-bit default
#include <hls_stream.h>
#include <ap_int.h>

// Weight-stationary grid C twin of {func_name} (FINN MVU: nt={NT} x gk={KT}, MW_tile={MW}
// N_tile={NTILE} SIMD={SIMD} PE={PE} SF_tile={SFT} NF={NF}). Weights baked here match the grid
// memstreams the RTL bakes; Vitis substitutes the RTL for csynth/cosim. Per lane: sum the gk
// K-partials, + baked bias (if any), shift + round-half-up + wrap to out_width -- the beat is
// already the narrow, final, K-summed code. K-padding (zero-fill to k_pad) and N-tile
// stitching/pad-drop happen here too, mirroring the RTL wrapper -- the boundary is hls4ml's
// own unpadded row width on both sides.
static const long {func_name}_W[{N}][{KPAD}] = {wlit};
{bias_decl}
void {func_name}(hls::stream<ap_uint<{AB}> >& a,
{pad}hls::stream<ap_uint<{PB}> >& p) {{
    for (int vec = 0; vec < {M}; vec++) {{
        {actt} x[{KPAD}];
        {{
            ap_uint<{AB}> ab = a.read();
            for (int k = 0; k < {K_RAW}; k++) x[k] = ab.range(k * {AW} + {AW} - 1, k * {AW});
            for (int k = {K_RAW}; k < {KPAD}; k++) x[k] = 0;   // zero-fill the K-pad lanes
        }}
        ap_uint<{PB}> ob = 0;
        for (int j = 0; j < {NT}; j++)
            for (int local_oc = 0; local_oc < {NTILE}; local_oc++) {{
                int oc = j * {NTILE} + local_oc;   // global output column
                ap_int<{ACCU_SUM}> raw = 0;
                for (int i = 0; i < {KT}; i++) {{
                    ap_int<{ACCU}> acc = 0;
                    for (int kk = 0; kk < {MW}; kk++)
                        acc += (ap_int<64>){func_name}_W[oc][i * {MW} + kk]
                             * (ap_int<64>)x[i * {MW} + kk];
                    raw += acc;               // K-tile sum first, mirrors the RTL adder
                }}
{req}
                ob.range(oc * {outW} + {outW} - 1, oc * {outW}) = (ap_uint<{outW}>)q;
            }}
        p.write(ob);
    }}
}}
"""


def _kt_tb(p, t, top_name, seed, bias_codes, B, n_nodes=6):
    """K-tiled self-checking TB: baked weights, top ``(a, c)`` with one raw, unpadded
    ``K_raw*AW``-bit activation row beat and one raw, unpadded ``N*out_width``-bit
    result row beat per vector -- matches the RTL wrapper's / hls4ml's own row
    widths (no SF/NF/K-tile beat splitting at this boundary). Independent full-K
    golden matmul + requant reference.

    Weights are baked (shared across nodes); ``n_nodes`` distinct activation sets are
    queued back-to-back (all writes before any call), then the top is invoked
    ``n_nodes`` times and all ``n_nodes*M`` output rows are drained and checked --
    exercising overlapped/pipelined invocations of the blackbox core in cosim."""
    NN = n_nodes
    AW = t["activation_width"]
    M, KPAD, N = p["num_input_vectors"], p["k_pad"], p["n"]
    K = p["k"]   # unpadded K -- the beat this TB's boundary actually carries
    AB = K * AW
    outW = p["output_width"]
    req_shift = p["product_frac"] - p["output_frac"]
    CB = N * outW
    wlit = _w_matrix_literal(B, N, KPAD)
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

// K-tiled tile set: N={N} K={K} KPAD={KPAD}, M={M} vectors.
// Weights baked (must match the KT memstreams the RTL loads); only raw, unpadded
// activation rows are fed -- the RTL wrapper does the K-padding + SF_tile fan-out.
static const long W[{N}][{KPAD}] = {wlit};

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
                for (int k = 0; k < {K}; k++) acc += W[o][k] * X[n][v][k];   // pad columns of W are 0
                golden[n][v][o] = requant_ref(acc{bias_add_tb});
            }}

        for (int v = 0; v < {M}; v++) {{
            ap_uint<{AB}> ab = 0;
            for (int k = 0; k < {K}; k++)
                ab.range(k * {AW} + {AW} - 1, k * {AW}) =
                    (ap_uint<{AW}>)(ap_int<{AW}>)X[n][v][k];
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
    """General tiled two-operand C twin (nt×gk grid, memstream form: SF_tile*NF >= 2), C
    twin of ``_2op_grid_memstream_shim``'s UNPADDED boundary (mirrors ``_2op_reg_core_twin``,
    generalized to SF_tile>1 fan-out and NF>1 stitching): ``a`` is one raw ``K*AW``-bit beat
    per vector (zero-padded to the grid's total ``k_pad`` right here, mirroring the RTL's
    ``arow_pad``/``arow_reg`` fan-out), ``b`` is one raw ``N*WW``-bit K-row beat (``K`` beats
    total -- no K_pad tail; the RTL wrapper zero-fills the missing rows internally), and ``p``
    is one raw ``N*out_width``-bit beat per vector holding all ``N`` unpadded output columns
    (K-tile summed, N-tile stitched, NF column-blocks concatenated, pad tail columns dropped)."""
    PE, SIMD, MW = t["pe"], t["simd"], t["mw"]
    ACCU, ACCU_SUM, AW, WW = t["accu_width"], p["accu_sum"], t["activation_width"], t["weight_width"]
    gk, nt = p.get("k_tiles", 1), p["n_tiles"]
    NF = t["nf"]
    SFT = MW // SIMD
    M, N = p["num_input_vectors"], p["n"]
    K = p["k"]                    # raw, unpadded K -- the beats this twin's boundary carries
    KPAD = gk * MW                # grid's total padded K
    NTILE = N // nt                # unpadded per-N-tile column count
    ARAW = K * AW
    PRAW = N * t["output_width"]
    outW = t["output_width"]
    shift = p["product_frac"] - p["output_frac"]
    actt = _act_ctype(t["signed_activations"], AW)
    pad = ' ' * (len(func_name) + 6)
    apmax = max(1024, ((max(ARAW, PRAW) + 1023) // 1024 + 1) * 1024)
    req = _requant_block(shift, outW, "(long)raw", "q", ' ' * 16)
    return f"""#define AP_INT_MAX_W {apmax}   // raw K-wide activation row may exceed the 1024-bit default
#include <hls_stream.h>
#include <ap_int.h>

// General tiled two-operand C twin of {func_name} (nt={nt} x gk={gk} grid; MW_tile={MW}
// N_tile={NTILE} SIMD={SIMD} PE={PE} SF_tile={SFT} NF={NF}, no bias -- two-operand GEMM
// never has one). B is a runtime stream (N-wide K-row beats, K={K} real rows, zero-padded
// to k_pad={KPAD} right here mirroring the RTL wrapper's internal K-padding). Per lane: sum
// the gk K-partials, shift + round-half-up + wrap to out_width -- the boundary is hls4ml's
// own unpadded row width on both sides (one A/one C beat per vector).
void {func_name}(hls::stream<ap_uint<{ARAW}> >& a,
{pad}hls::stream<ap_uint<{N * WW}> >& b,
{pad}hls::stream<ap_uint<{PRAW}> >& p) {{
    ap_int<{WW}> W[{N}][{KPAD}];
    for (int k = 0; k < {K}; k++) {{
        ap_uint<{N * WW}> bb = b.read();
        for (int o = 0; o < {N}; o++) W[o][k] = bb.range(o * {WW} + {WW} - 1, o * {WW});
    }}
    for (int k = {K}; k < {KPAD}; k++)
        for (int o = 0; o < {N}; o++) W[o][k] = 0;   // zero-fill the K-pad rows
    for (int vec = 0; vec < {M}; vec++) {{
        {actt} x[{KPAD}];
        {{
            ap_uint<{ARAW}> ab = a.read();
            for (int k = 0; k < {K}; k++) x[k] = ab.range(k * {AW} + {AW} - 1, k * {AW});
            for (int k = {K}; k < {KPAD}; k++) x[k] = 0;   // zero-fill the K-pad lanes
        }}
        ap_uint<{PRAW}> ob = 0;
        for (int j = 0; j < {nt}; j++)
            for (int nf = 0; nf < {NF}; nf++)
                for (int pe = 0; pe < {PE}; pe++) {{
                    int local_oc = nf * {PE} + pe;   // 0..n_pad-1 within this N-tile
                    if (local_oc < {NTILE}) {{         // drop the N-pad tail columns
                    int oc = j * {NTILE} + local_oc;
                    ap_int<{ACCU_SUM}> raw = 0;
                    for (int i = 0; i < {gk}; i++) {{
                        ap_int<{ACCU}> acc = 0;
                        for (int kk = 0; kk < {MW}; kk++)
                            acc += (ap_int<64>)W[oc][i * {MW} + kk] * (ap_int<64>)x[i * {MW} + kk];
                        raw += acc;
                    }}
{req}
                    ob.range(oc * {outW} + {outW} - 1, oc * {outW}) = (ap_uint<{outW}>)q;
                    }}
                }}
        p.write(ob);
    }}
}}
"""


def _2op_kt_tb(p, t, top_name, func_name, seed, n_nodes=6):
    """Self-checking TB for the memstream two-operand grid (mirrors ``_2op_reg_tb``,
    generalized to SF_tile/NF>1): feed A as one raw K*AW-bit beat/vector and B as K raw
    N*WW-bit K-row beats; golden = full-K integer matmul + affine requant (no bias)."""
    NN = n_nodes
    PE, SIMD, MW = t["pe"], t["simd"], t["mw"]
    WW, AW = t["weight_width"], t["activation_width"]
    gk, nt = p.get("k_tiles", 1), p["n_tiles"]
    N, M = p["n"], p["num_input_vectors"]
    K = p["k"]
    ARAW = K * AW
    BRAW = N * WW
    outW = p["output_width"]
    req_shift = p["product_frac"] - p["output_frac"]
    CB = N * outW
    signed = bool(t["signed_activations"])
    wrange = (1 << (WW - 1)) - 1
    arange = (1 << (AW - 1)) - 1 if signed else (1 << AW) - 1
    wmod, amod = min(7, 2 * wrange + 1), min(7, arange + 1)
    woff = wrange if wmod == 2 * wrange + 1 else 3
    aoff = 3 if signed else 0
    return f"""#include <hls_stream.h>
#include <ap_int.h>
#include <cstdio>

void {top_name}(hls::stream<ap_uint<{ARAW}> >&, hls::stream<ap_uint<{BRAW}> >&,
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

// Memstream two-operand grid: N={N} K={K} nt={nt} gk={gk} SIMD={SIMD} PE={PE},
// M={M} vectors, no bias.
#define NN {NN}

int main() {{
    hls::stream<ap_uint<{ARAW}> > a_in;
    hls::stream<ap_uint<{BRAW}> > b_in;
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
            ap_uint<{BRAW}> bb = 0;
            for (int o = 0; o < {N}; o++)
                bb.range(o * {WW} + {WW} - 1, o * {WW}) = (ap_uint<{WW}>)(ap_int<{WW}>)Bm[n][k][o];
            b_in.write(bb);
        }}
        for (int v = 0; v < {M}; v++) {{
            ap_uint<{ARAW}> ab = 0;
            for (int k = 0; k < {K}; k++)
                ab.range(k * {AW} + {AW} - 1, k * {AW}) = (ap_uint<{AW}>)(ap_int<{AW}>)X[n][v][k];
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


def _2op_register_form(p):
    """True for the fully-spatial (DEPTH_tile = SF_tile*NF == 1) grid shim
    (``_2op_grid_register_shim`` in rtl.py) -- covers plain, K-tiled and N-tiled grids
    alike whenever the per-tile fold is fully spatial. This is the only two-operand
    form with an UNPADDED RTL boundary today; the memstream grid/single-tile forms
    (SF_tile*NF >= 2) keep the older, byte-aligned/padded boundary."""
    t = p["tile"]
    return t["sf"] == 1 and t["nf"] == 1


def _2op_reg_core_twin(p, t, func_name):
    """C twin of ``_2op_grid_register_shim``'s UNPADDED boundary: ``a`` is one raw
    ``K*AW``-bit beat (one hls4ml row, zero-padded to the grid's total ``k_pad`` right
    here, mirroring the RTL's ``arow_pad``), ``b`` is one raw ``N*WW``-bit K-row beat
    (``K`` beats total -- no K_pad tail beats; the RTL wrapper zero-fills the missing
    rows internally), and ``p`` is one raw ``N*out_width``-bit beat holding all ``N``
    unpadded output columns (K-tile summed, N-tile stitched -- PE == NTILE exactly in
    this fully-spatial form, so there is no pad-column drop to do)."""
    PE, SIMD, MW = t["pe"], t["simd"], t["mw"]
    ACCU, ACCU_SUM, AW, WW = t["accu_width"], p["accu_sum"], t["activation_width"], t["weight_width"]
    gk, nt = p.get("k_tiles", 1), p["n_tiles"]
    M, N = p["num_input_vectors"], p["n"]
    K = p["k"]                    # raw, unpadded K -- the beats this twin's boundary carries
    KPAD = gk * MW                # grid's total padded K
    NTILE = N // nt
    ARAW = K * AW
    PRAW = N * t["output_width"]
    outW = t["output_width"]
    shift = p["product_frac"] - p["output_frac"]
    actt = _act_ctype(t["signed_activations"], AW)
    pad = ' ' * (len(func_name) + 6)
    apmax = max(1024, ((max(ARAW, PRAW) + 1023) // 1024 + 1) * 1024)
    req = _requant_block(shift, outW, "(long)raw", "q", ' ' * 12)
    return f"""#define AP_INT_MAX_W {apmax}   // raw K-wide activation row may exceed the 1024-bit default
#include <hls_stream.h>
#include <ap_int.h>

// Fully-spatial two-operand grid C twin of {func_name} (nt={nt} x gk={gk}, MW_tile={MW}
// N_tile={NTILE} SIMD={SIMD} PE={PE}, no bias -- two-operand GEMM never has one). B is a
// runtime stream (N-wide K-row beats, K={K} real rows, zero-padded to k_pad={KPAD} right
// here mirroring the RTL wrapper's internal K-padding). Per lane: sum the gk K-partials,
// shift + round-half-up + wrap to out_width -- the boundary is hls4ml's own unpadded row
// width on both sides (no N-pad tail: PE == N_tile exactly in this fully-spatial form).
void {func_name}(hls::stream<ap_uint<{ARAW}> >& a,
{pad}hls::stream<ap_uint<{N * WW}> >& b,
{pad}hls::stream<ap_uint<{PRAW}> >& p) {{
    ap_int<{WW}> W[{N}][{KPAD}];
    for (int k = 0; k < {K}; k++) {{
        ap_uint<{N * WW}> bb = b.read();
        for (int o = 0; o < {N}; o++) W[o][k] = bb.range(o * {WW} + {WW} - 1, o * {WW});
    }}
    for (int k = {K}; k < {KPAD}; k++)
        for (int o = 0; o < {N}; o++) W[o][k] = 0;   // zero-fill the K-pad rows
    for (int vec = 0; vec < {M}; vec++) {{
        {actt} x[{KPAD}];
        {{
            ap_uint<{ARAW}> ab = a.read();
            for (int k = 0; k < {K}; k++) x[k] = ab.range(k * {AW} + {AW} - 1, k * {AW});
            for (int k = {K}; k < {KPAD}; k++) x[k] = 0;   // zero-fill the K-pad lanes
        }}
        ap_uint<{PRAW}> ob = 0;
        for (int j = 0; j < {nt}; j++)
            for (int pe = 0; pe < {PE}; pe++) {{
                int oc = j * {NTILE} + pe;
                ap_int<{ACCU_SUM}> raw = 0;
                for (int i = 0; i < {gk}; i++) {{
                    ap_int<{ACCU}> acc = 0;
                    for (int kk = 0; kk < {MW}; kk++)
                        acc += (ap_int<64>)W[oc][i * {MW} + kk] * (ap_int<64>)x[i * {MW} + kk];
                    raw += acc;
                }}
{req}
                ob.range(oc * {outW} + {outW} - 1, oc * {outW}) = (ap_uint<{outW}>)q;
            }}
        p.write(ob);
    }}
}}
"""


def _2op_reg_tb(p, t, top_name, func_name, seed, n_nodes=6):
    """Self-checking TB for the fully-spatial two-operand grid: feed A as one raw
    K*AW-bit beat/vector and B as K raw N*WW-bit K-row beats; golden = full-K integer
    matmul + affine requant (no bias). Mirrors ``_kt_tb``'s overlapped-invocation style."""
    NN = n_nodes
    PE, SIMD, MW = t["pe"], t["simd"], t["mw"]
    WW, AW = t["weight_width"], t["activation_width"]
    gk, nt = p.get("k_tiles", 1), p["n_tiles"]
    N, M = p["n"], p["num_input_vectors"]
    K = p["k"]
    ARAW = K * AW
    BRAW = N * WW
    outW = p["output_width"]
    req_shift = p["product_frac"] - p["output_frac"]
    CB = N * outW
    signed = bool(t["signed_activations"])
    wrange = (1 << (WW - 1)) - 1
    arange = (1 << (AW - 1)) - 1 if signed else (1 << AW) - 1
    wmod, amod = min(7, 2 * wrange + 1), min(7, arange + 1)
    woff = wrange if wmod == 2 * wrange + 1 else 3
    aoff = 3 if signed else 0
    return f"""#include <hls_stream.h>
#include <ap_int.h>
#include <cstdio>

void {top_name}(hls::stream<ap_uint<{ARAW}> >&, hls::stream<ap_uint<{BRAW}> >&,
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

// Fully-spatial two-operand grid: N={N} K={K} nt={nt} gk={gk} SIMD={SIMD} PE={PE},
// M={M} vectors, no bias.
#define NN {NN}

int main() {{
    hls::stream<ap_uint<{ARAW}> > a_in;
    hls::stream<ap_uint<{BRAW}> > b_in;
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
            ap_uint<{BRAW}> bb = 0;
            for (int o = 0; o < {N}; o++)
                bb.range(o * {WW} + {WW} - 1, o * {WW}) = (ap_uint<{WW}>)(ap_int<{WW}>)Bm[n][k][o];
            b_in.write(bb);
        }}
        for (int v = 0; v < {M}; v++) {{
            ap_uint<{ARAW}> ab = 0;
            for (int k = 0; k < {K}; k++)
                ab.range(k * {AW} + {AW} - 1, k * {AW}) = (ap_uint<{AW}>)(ap_int<{AW}>)X[n][v][k];
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


def generate_2op_core_twin(shape, func_name="mvau_core", plan=None, **kw):
    """Public entry for the two-operand C twin. The register twin (unpadded boundary)
    covers any grid (plain/N-tiled/K-tiled) whose per-tile fold is fully spatial; the
    older, padded grid twin serves the memstream (temporal-fold) grids, and the
    single-tile memstream twin serves the untiled temporal fold."""
    p = _plan(shape, plan=plan, **kw)
    if _2op_register_form(p):
        return _2op_reg_core_twin(p, p["tile"], func_name)
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
    if _2op_register_form(p):
        return _2op_reg_tb(p, p["tile"], top_name, func_name, seed, n_nodes=n_nodes)
    if p.get("n_tiles", 1) > 1 or _2op_use_kt(p):
        return _2op_kt_tb(p, p["tile"], top_name, func_name, seed, n_nodes=n_nodes)
    return _2op_tb(p, p["tile"], top_name, func_name, seed, n_nodes=n_nodes)


def _ws_tb(p, t, top_name, seed, bias_codes, B, n_nodes=6):
    """Weight-stationary self-checking TB: baked weights, top ``(a, c)`` (no ``w``).
    Independent golden with the same round-half-up + wrap requant reference.

    Weights are shared across nodes (baked); ``n_nodes`` distinct activation sets are
    queued before any call, then the top is invoked ``n_nodes`` times back-to-back.

    Boundary is UNPADDED (matches the RTL wrapper / hls4ml row widths): the top
    takes one raw ``K_raw*AW``-bit beat in and one raw ``N*out_width``-bit beat out,
    per vector -- no SF/NF beat splitting at this boundary any more."""
    NN = n_nodes
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    AW = t["activation_width"]
    M, KPAD, N = p["num_input_vectors"], p["k_pad"], p["n"]
    K = p["k"]   # unpadded K -- the beat this TB's boundary actually carries
    AB = K * AW
    outW = p["output_width"]
    req_shift = p["product_frac"] - p["output_frac"]
    CB = N * outW
    wlit = _w_matrix_literal(B, N, KPAD)
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

        for (int v = 0; v < {M}; v++) {{
            ap_uint<{AB}> ab = 0;
            for (int k = 0; k < {K}; k++)
                ab.range(k * {AW} + {AW} - 1, k * {AW}) =
                    (ap_uint<{AW}>)(ap_int<{AW}>)X[n][v][k];
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
