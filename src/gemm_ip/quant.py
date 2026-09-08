"""Fixed-point requantization math for gemm-ip-gen (target-agnostic).

The GEMM IP feeds operands as integer *codes* (the fixed-point mantissa). These
helpers derive the fractional-bit shifts and result-lane width the wrapper uses
to requantize the raw integer dot-product back to the configured output
precision. They parse ``ac_fixed``-style precision strings and default to the
legacy integer-coded behavior when the precision is unset/unparseable.
"""

import re


def _frac_bits(precision):
    """Fractional-bit count (W - I) of a precision like 'fixed<9,5,…>'.

    The GEMM IP feeds operands as int8 *codes* = the fixed-point mantissa
    (value · 2^frac). The product of two operands therefore carries
    2^(frac_a + frac_b), which the wrapper drain divides back out. Returns 0 for
    an unset/unparseable precision (integer-coded operand ⇒ no rescale), so the
    rescale collapses to the identity for the legacy integer-input case.
    """
    if not precision:
        return 0
    m = re.search(r"u?fixed<\s*(\d+)\s*,\s*(-?\d+)", str(precision))
    if not m:
        return 0
    width, integer_bits = int(m.group(1)), int(m.group(2))
    return width - integer_bits


def _operand_bits(precision):
    """(width, signed) parsed from a precision like 'ufixed<8,2,TRN,WRAP,0>'.

    signed is False iff the precision string starts with 'u' (unsigned
    fixed-point). Returns None for an unset/unparseable precision.
    """
    if not precision:
        return None
    s = str(precision)
    m = re.search(r"(u?)fixed<\s*(\d+)", s)
    if not m:
        return None
    signed = not m.group(1)
    width = int(m.group(2))
    return (width, signed)


def _output_bits(output_precision):
    """Result-lane width (bits) from an output_precision like 'fixed<16,6,…>'.

    Drives the GEMM-IP output saturation/packing so the result honors the
    configured precision instead of the legacy hardcoded int8 clamp. Returns 8
    (legacy int8) when unset/unparseable so callers without a precision keep
    the old behavior. The first ``fixed<>`` field is the total bit width.
    """
    if not output_precision:
        return 8
    m = re.search(r"u?fixed<\s*(\d+)", str(output_precision))
    return int(m.group(1)) if m else 8
