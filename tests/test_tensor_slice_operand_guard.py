"""Tests for the tensor_slice int8-core operand-width guard.

The generated wrapper reads every operand's low 8 bits as ac_int<8,true>
({name}_to_gemm_int8). An operand wider than 8 bits gets truncated -- signed
or unsigned. Symmetric-only quantization scope: an unsigned exactly-8-bit
operand would need a zero-point offset to keep bit 7 from being misread as
the sign bit, so it is rejected; unsigned operands narrower than 8 bits
pass through unchanged. These tests confirm the guard's boundary.
"""
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gemm_ip.quant import _operand_bits  # noqa: E402
from targets.tensor_slice import package as _pkg  # noqa: E402

generate_catapult_pkg = _pkg.generate_catapult_pkg
_check_operand_fits_int8_core = _pkg._check_operand_fits_int8_core


@pytest.mark.parametrize("precision,expected", [
    ("fixed<8,2>", (8, True)),
    ("ufixed<7,0>", (7, False)),
    ("ufixed<8,2>", (8, False)),
    ("fixed<16,6>", (16, True)),
    ("ufixed<9,3>", (9, False)),
    (None, None),
])
def test_operand_bits(precision, expected):
    assert _operand_bits(precision) == expected


@pytest.mark.parametrize("precision,should_raise", [
    ("fixed<8,2>", False),      # signed width 8: still accepted
    ("ufixed<7,0>", False),     # unsigned width 7: still accepted
    ("ufixed<8,2>", True),      # unsigned width 8: rejected (would need a zero-point offset)
    ("fixed<16,6>", True),      # signed width 16: still rejected
    ("fixed<9,3>", True),       # signed width 9: still rejected
    ("ufixed<9,3>", True),      # unsigned width 9: still rejected
    (None, False),
])
def test_operand_fits_int8_core_helper(precision, should_raise):
    if should_raise:
        with pytest.raises(ValueError):
            _check_operand_fits_int8_core("t", "some_precision", precision)
    else:
        _check_operand_fits_int8_core("t", "some_precision", precision)


def test_generate_catapult_pkg_rejects_signed_9bit_input(tmp_path):
    with pytest.raises(ValueError, match="int8 core"):
        generate_catapult_pkg(
            m=1, k=8, n=8, name="t", output_dir=str(tmp_path),
            interface="stream", output_precision="fixed<10,4>",
            input_precision="fixed<9,2>", weight_precision="fixed<8,2>",
        )


def test_generate_catapult_pkg_rejects_unsigned_8bit_input(tmp_path):
    # Unsigned 8-bit is rejected under the symmetric-only quantization
    # scope: it would need a zero-point offset, which is not supported.
    with pytest.raises(ValueError, match="symmetric-only"):
        generate_catapult_pkg(
            m=1, k=8, n=8, name="t2", output_dir=str(tmp_path),
            interface="stream", output_precision="fixed<10,4>",
            input_precision="ufixed<8,2>", weight_precision="fixed<8,2>",
        )
