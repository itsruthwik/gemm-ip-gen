"""Pack constant weights into a FINN ``memstream`` ``$readmemh`` init file.

The weight-stationary shim bakes B (= the GEMM's B^T mapped to the MVU weight
matrix) into a FINN ``memstream`` whose ``INIT_FILE`` this module generates. The
layout is exactly what the vendored ``mvu_vvu_axi`` expects on
``s_axis_weights_tdata``, validated byte-for-byte against the cosim-PASS
``temp_space/mvau-ws`` spike:

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

def bias_acc_codes(bias, product_frac, n, has_bias):
    """Scale per-column real bias to the accumulator (2^product_frac) domain.

    ``has_bias`` (the manifest's own field, computed by hls4ml from the real bias
    tensor) is the *only* gate for whether a bias is baked: when True this always
    returns an N-long list of codes -- even if every one of them rounds to zero at
    this fixed-point scale -- so the generated hardware matches what hls4ml's own
    csim expects (an add is present) rather than silently disagreeing with the
    manifest for a sub-LSB bias. When False, returns None (bake nothing) regardless
    of what ``bias`` holds. Raises if ``has_bias`` is True but ``bias`` is absent --
    that combination means the manifest is internally inconsistent, not "no bias".

    This is the single source of truth for the baked bias integers: both the C twin's
    ``static const long`` array and the RTL requant stage's bias ROM render the same
    ``codes`` list, so the two textual forms can never drift apart.
    """
    if not has_bias:
        return None
    if not bias:
        raise ValueError(
            "has_bias is True but the manifest has no bias values to bake "
            "(cfg['bias'] is missing/empty)")
    codes = [int(round(float(b) * (1 << product_frac))) for b in bias]
    if len(codes) != n:
        raise ValueError(f"bias length {len(codes)} != N {n}")
    return codes


def bias_c_decl(name, codes):
    """``static const long <name>_bias[N] = {...};`` C rendering of *codes*."""
    return (f"static const long {name}_bias[{len(codes)}] = {{"
            + ", ".join(str(c) for c in codes) + "};\n")


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


def bias_verilog_rom(reg_name, codes, width):
    """Verilog ``reg`` array + ``initial`` block rendering of *codes* (the same
    integer codes the C twin bakes as a ``static const long[]``), synthesizable as a
    small ROM indexed combinationally by the shim's per-lane output-column index.
    ``width`` must be wide enough to hold every code as a signed two's-complement
    value (the requant stage's biased-accumulator width)."""
    lines = [f"    reg signed [{width - 1}:0] {reg_name} [0:{len(codes) - 1}];",
             "    initial begin"]
    for i, c in enumerate(codes):
        c = int(c)
        if c >= 0:
            lines.append(f"        {reg_name}[{i}] = {width}'sd{c};")
        else:
            lines.append(f"        {reg_name}[{i}] = -{width}'sd{-c};")
    lines.append("    end")
    return "\n".join(lines) + "\n"
