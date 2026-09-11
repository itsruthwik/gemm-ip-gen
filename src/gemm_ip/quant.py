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
    the old behavior. The first ``fixed<>``/``ac_int<>``/``int<>`` field is the
    total bit width -- integer precisions (the standalone-package default,
    e.g. ``ac_int<16, true>``) are matched too, not just fixed<>.
    """
    if not output_precision:
        return 8
    m = re.search(r"u?(?:ac_)?(?:fixed|int)<\s*(\d+)", str(output_precision))
    return int(m.group(1)) if m else 8


def _accum_shift_bits(accum_precision, gemm_frac):
    """Bit width needed to hold ``accum_precision`` re-expressed with ``gemm_frac``
    fraction bits (sign included), i.e. the width of the GEMM-scale accumulator
    the layer's ``accum_t`` implies.

    ``accum_precision`` carries its own fractional bits (``frac_bits(accum_precision)``);
    its integer-bit count (width - frac, which already includes the sign bit for a
    signed ``fixed<>``) is preserved and re-based onto ``gemm_frac`` fraction bits:
    ``int_bits + gemm_frac``. Returns ``None`` when ``accum_precision`` is unset/
    unparseable -- callers should treat that as "assume it fits 16 bits" (S1 = 0,
    today's behavior).
    """
    bits = _operand_bits(accum_precision)
    if bits is None:
        return None
    width, _signed = bits
    frac = _frac_bits(accum_precision)
    int_bits = width - frac
    return int_bits + int(gemm_frac)
