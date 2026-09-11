"""Bias baking: one codes list, two renderings (C twin static array + Verilog ROM).

Target-agnostic (moved out of ``mvau/weightpack.py`` per
``jojo-track/open/tensor-slice-bias-in-rtl``): both mvau and tensor_slice bake a
per-column bias into their generated core at a *scale* the caller chooses (mvau:
the MAC product scale; tensor_slice: the stage-1/stage-2 intermediate scale), so
these helpers take that scale as a plain ``product_frac`` argument rather than
assuming either target's fixed-point convention.
"""


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
