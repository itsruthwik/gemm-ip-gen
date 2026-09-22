"""Pack constant weights into a FINN ``memstream`` ``$readmemh`` init file.

The weight-stationary shim bakes B (= the GEMM's B^T mapped to the MVU weight
matrix) into a FINN ``memstream`` whose ``INIT_FILE`` this module generates. The
layout is exactly what the vendored ``mvu_vvu_axi`` expects on
``s_axis_weights_tdata``, validated byte-for-byte against a cosim-PASS
weight-stationary spike:

    memstream line (address)   wmem = nf*SF + sf        (nf outer, sf inner)
    within a WIDTH = PE*SIMD*WW word, weight W[nf*PE+pe][sf*SIMD+s] occupies
        bits [(pe*SIMD + s)*WW +: WW]     (SIMD innermost / LSB, PE outer)
    each weight a WW-bit two's-complement code.

This reduces FINN's own SIMD-flip + PE-flip + MSB-first hex packing
(``matrixvectoractivation.make_weight_file``, ``decoupled_verilog_dat``) to the
same words -- the two flips cancel against the MSB-first emission (proven and
checked numerically against the spike golden ``fd0202fd...``).

Orientation: the MVU computes y[mh] = sum_mw W[mh][mw] * x[mw] with MH=N, MW=K,
so W[mh][mw] = B[mw][mh], where ``B`` is ``[K, N]`` (hls4ml ``weight_cols[k][n]``)
as returned by ``gemm_ip.weights.load_weight_dat``.
"""


def pack_memstream_hex(B, n, k, pe, simd, weight_width, word_bits=None):
    """Return the memstream ``INIT_FILE`` contents (one hex word per line).

    ``B``: ``[K][N]`` integer weight codes (raw signed ints); indexed ``B[mw][mh]``.
    Emits ``NF*SF`` lines, each written most-significant-nibble first (``$readmemh``
    order). Each word is ``word_bits`` wide (default ``PE*SIMD*WW``, rounded up to a
    nibble); pass the shim's byte-aligned memstream ``WIDTH`` so ``$readmemh`` fills
    the full register -- the packed weights occupy the low bits, high bits zero.
    """
    if pe <= 0 or simd <= 0 or weight_width <= 0:
        raise ValueError(f"pe/simd/weight_width must be positive (got {pe}/{simd}/{weight_width})")
    if n % pe or k % simd:
        raise ValueError(f"fold mismatch: N={n} % PE={pe} or K={k} % SIMD={simd} != 0")
    packed_bits = pe * simd * weight_width
    if word_bits is None:
        word_bits = packed_bits
    elif word_bits < packed_bits:
        raise ValueError(f"word_bits {word_bits} < packed PE*SIMD*WW {packed_bits}")
    nf, sf = n // pe, k // simd
    mask = (1 << weight_width) - 1
    lo, hi = -(1 << (weight_width - 1)), (1 << (weight_width - 1)) - 1
    ndigits = (word_bits + 3) // 4
    lines = []
    for i_nf in range(nf):
        for i_sf in range(sf):
            word = 0
            for i_pe in range(pe):
                for i_s in range(simd):
                    mh, mw = i_nf * pe + i_pe, i_sf * simd + i_s
                    code = int(B[mw][mh])
                    if not (lo <= code <= hi):
                        raise ValueError(
                            f"weight code {code} at [k={mw}][n={mh}] does not fit a "
                            f"signed {weight_width}-bit lane [{lo}, {hi}] -- weight "
                            "precision exceeds the selected MVU core width")
                    word |= (code & mask) << ((i_pe * simd + i_s) * weight_width)
            lines.append(format(word, "x").zfill(ndigits))
    return "\n".join(lines) + "\n"


# ── Bias codes: one baking, two renderings (C twin static array + Verilog ROM) ─────
#
# Moved to ``gemm_ip.biasrom`` (target-agnostic; tensor_slice bakes a bias the
# same way, at its own intermediate scale) -- re-exported here so every existing
# `_wpack.bias_acc_codes` / `_wpack.bias_c_decl` / `_wpack.bias_verilog_rom` call
# site in this target keeps working unchanged.
from gemm_ip.biasrom import bias_acc_codes, bias_c_decl, bias_verilog_rom  # noqa: E402,F401


def bias_codes_for_tile(bias_codes, ti, ntile_real, ntile_pad):
    """Slice+zero-pad the N-long baked bias codes to one N-tile's own local lane
    indexing (``0..ntile_pad-1``, matching ``local_oc = nf*PE+pe``): real columns
    ``[ti*ntile_real, (ti+1)*ntile_real)`` at their local offset, padded lanes 0.
    Returns None (no bias) when *bias_codes* is None -- callers bake an all-zero
    bias in that case (there is nothing to add)."""
    if bias_codes is None:
        return [0] * ntile_pad
    lo = ti * ntile_real
    real = list(bias_codes[lo:lo + ntile_real])
    return real + [0] * (ntile_pad - len(real))


def fold_requant_constants(bias_codes, shift, n):
    """Fold the round-half-up constant (``2^(shift-1)``, when ``shift > 0``) into
    the per-column bias codes ``bias_acc_codes`` bakes, so the RTL/C requant stage
    performs one add (sum + folded constant) instead of a separate bias add and
    round add. Called once, after ``bias_acc_codes`` (and after ``package.py``'s
    own TRN pre-adjustment, if any -- that adjustment already pre-subtracts this
    same half so the two cancel and floor, rather than round, results).

    Returns an *n*-long list of folded ints, or ``None`` when there is truly
    nothing to add (no bias and ``shift <= 0``) -- callers skip the add stage
    entirely in that case.
    """
    round_c = (1 << (shift - 1)) if shift > 0 else 0
    if bias_codes is None:
        return [round_c] * n if round_c else None
    if len(bias_codes) != n:
        raise ValueError(f"bias_codes length {len(bias_codes)} != n {n}")
    return [int(b) + round_c for b in bias_codes]


def has_real_add(consts):
    """True iff *consts* (a folded-constants list, or None) has at least one
    nonzero entry -- an all-zero list (e.g. a TRN layer whose pre-subtracted
    half exactly cancels the folded-in round constant) adds nothing, exactly
    like ``None``, so callers skip the add stage (and the width/array it would
    otherwise need) in both cases."""
    return bool(consts) and any(int(c) != 0 for c in consts)


def assert_constants_fit(consts, const_w):
    """Assert every folded constant (or None/empty/all-zero) fits signed
    *const_w* bits -- the width :func:`geometry.requant_width` computed for
    them."""
    if not has_real_add(consts):
        return
    lo, hi = -(1 << (const_w - 1)), (1 << (const_w - 1)) - 1
    for c in consts:
        c = int(c)
        if not (lo <= c <= hi):
            raise ValueError(f"folded requant constant {c} does not fit signed "
                             f"const_w={const_w} bits [{lo}, {hi}]")
