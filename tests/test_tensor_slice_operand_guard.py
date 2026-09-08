"""Tests for the tensor_slice int8-core operand-width guard.

The generated wrapper reads every operand's low 8 bits as ac_int<8,true>
({name}_to_gemm_int8). An operand wider than 8 bits gets truncated, and an
unsigned 8-bit operand has bit 7 misread as the sign bit. These tests confirm
the guard rejects such precisions loudly instead of letting them through.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gemm_ip.quant import _operand_bits  # noqa: E402


def _load_tensor_slice_package():
    """Import targets/tensor_slice/package.py under a unique module name.

    The tensor_slice and mvau targets both ship modules named `geometry` and
    `package` on the same sys.path trick (see gemm_ip.registry); whichever
    target's tests run first poisons sys.modules for the other. Loading by
    explicit file path with a unique name sidesteps the collision.
    """
    tdir = str(_SRC / "targets" / "tensor_slice")
    saved_path = list(sys.path)
    saved_modules = {k: sys.modules.get(k) for k in ("geometry", "package")}
    sys.path.insert(0, tdir)  # so `package.py`'s `from geometry import ...` resolves
    for stale in ("geometry", "package"):
        sys.modules.pop(stale, None)
    try:
        spec = importlib.util.spec_from_file_location("_tensor_slice_package", Path(tdir) / "package.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = saved_path
        for stale, prev in saved_modules.items():
            if prev is None:
                sys.modules.pop(stale, None)
            else:
                sys.modules[stale] = prev


_pkg = _load_tensor_slice_package()
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
    ("fixed<8,2>", False),
    ("ufixed<7,0>", False),
    ("ufixed<8,2>", True),
    ("fixed<16,6>", True),
    ("ufixed<9,3>", True),
    (None, False),
])
def test_operand_fits_int8_core_helper(precision, should_raise):
    if should_raise:
        with pytest.raises(ValueError):
            _check_operand_fits_int8_core("t", "some_precision", precision)
    else:
        _check_operand_fits_int8_core("t", "some_precision", precision)


def test_generate_catapult_pkg_rejects_unsigned_8bit_input(tmp_path):
    with pytest.raises(ValueError, match="int8 core"):
        generate_catapult_pkg(
            m=1, k=8, n=8, name="t", output_dir=str(tmp_path),
            interface="stream", output_precision="fixed<10,4>",
            input_precision="ufixed<8,2>", weight_precision="fixed<8,2>",
        )
