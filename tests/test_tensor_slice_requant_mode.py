"""The tensor_slice requant must be ONE exact step in the result type's own mode
(RND = round half up, TRN = floor). The slice rounds half-up for any non-zero
shift_amount and passes through at 0, so the generator reaches that by choosing
S1/S2: stage 2 does a TRN result's floor through the baked bias constant, and an
RND result that needs an in-slice shift takes the whole shift in the slice."""
import re
from pathlib import Path

import pytest

from gemm_ip.quant import _truncates
from test_tensor_slice_operand_guard import _pkg as _tensor_slice_pkg
from test_tensor_slice_rf import _with_tensor_slice_on_path

_generate_catapult_pkg = _tensor_slice_pkg.generate_catapult_pkg

# frac(a) + frac(b) = 12 everywhere below.
_IN = dict(input_precision="fixed<8,2>", weight_precision="fixed<8,2>")
# 8 integer bits at the 12-fraction-bit gemm scale is 20 bits: forces S1 = 4.
_WIDE_ACCUM = "fixed<20,8,RND,WRAP,0>"


def generate_catapult_pkg(*args, **kwargs):
    return _with_tensor_slice_on_path(_generate_catapult_pkg, *args, **kwargs)


def _slice_shift(pkg_dir, name):
    text = (Path(pkg_dir) / name / f"{name}_core.v").read_text()
    shifts = set(re.findall(r"\.shift_amount\(4'd(\d+)\)", text))
    assert len(shifts) == 1, f"expected one shift_amount value, got {shifts}"
    return int(shifts.pop())


def _stage2_shift(pkg_dir, name):
    text = (Path(pkg_dir) / name / f"{name}_core.v").read_text()
    m = re.search(r">>> (\d+);\s*\n\s*stage2 = ", text)
    assert m, "stage-2 shift not found in the Verilog core"
    return int(m.group(1))


def _bias_codes(pkg_dir, name):
    text = (Path(pkg_dir) / name / f"{name}_gemm_ip.h").read_text()
    m = re.search(rf"{name}_bias_codes\w*\[\d*\]\s*=\s*\{{([^}}]*)\}}", text)
    return None if m is None else [int(v) for v in m.group(1).split(",") if v.strip()]


@pytest.mark.parametrize("precision,expected", [
    ("fixed<10,5,TRN,WRAP,0>", True),
    ("ufixed<8,2,TRN,WRAP,0>", True),
    ("ac_fixed<10,5,AC_TRN,AC_WRAP>", True),
    ("fixed<15,6,RND,WRAP,0>", False),
    ("fixed<10,3>", False),   # no explicit mode keeps the round-half-up the cores always applied
    (None, False),
])
def test_truncates(precision, expected):
    assert _truncates(precision) is expected


def test_trn_result_floors_through_the_bias_constant(tmp_path):
    name = "t"
    generate_catapult_pkg(m=8, k=8, n=8, name=name, output_dir=str(tmp_path), interface="stream",
                          output_precision="fixed<10,3,TRN,WRAP,0>", **_IN)
    s2 = _stage2_shift(tmp_path, name)
    assert (_slice_shift(tmp_path, name), s2) == (0, 5)
    # floor(x / 2^S2) == round_half_up(x - 2^(S2-1), S2): a bias-free layer gets the half as its codes.
    assert _bias_codes(tmp_path, name) == [-(1 << (s2 - 1))] * 8


def test_rnd_result_keeps_a_plain_round(tmp_path):
    name = "t"
    generate_catapult_pkg(m=8, k=8, n=8, name=name, output_dir=str(tmp_path), interface="stream",
                          output_precision="fixed<10,3,RND,WRAP,0>", **_IN)
    assert (_slice_shift(tmp_path, name), _stage2_shift(tmp_path, name)) == (0, 5)
    assert _bias_codes(tmp_path, name) is None


def test_rnd_result_with_a_forced_slice_shift_rounds_once_in_the_slice(tmp_path, capsys):
    name = "t"
    generate_catapult_pkg(m=8, k=8, n=8, name=name, output_dir=str(tmp_path), interface="stream",
                          output_precision="fixed<12,5,RND,WRAP,0>", accum_precision=_WIDE_ACCUM, **_IN)
    # Total shift 12 - 7 = 5; accum_t alone would split it 4 + 1 and round twice.
    assert (_slice_shift(tmp_path, name), _stage2_shift(tmp_path, name)) == (5, 0)
    assert "S1=" not in capsys.readouterr().err


def test_trn_result_with_a_forced_slice_shift_stays_two_stage_and_warns(tmp_path, capsys):
    name = "t"
    generate_catapult_pkg(m=8, k=8, n=8, name=name, output_dir=str(tmp_path), interface="stream",
                          output_precision="fixed<12,5,TRN,WRAP,0>", accum_precision=_WIDE_ACCUM, **_IN)
    assert (_slice_shift(tmp_path, name), _stage2_shift(tmp_path, name)) == (4, 1)
    assert "truncating (TRN) result" in capsys.readouterr().err
