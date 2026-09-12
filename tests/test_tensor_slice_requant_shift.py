"""The tensor_slice total gemm->result shift must be frac(a)+frac(b)-frac(out)
for EVERY result precision, including one with no fraction bits and an unset
one. A legacy guard used to force the shift to zero in those cases and let the
drain rescale; the drain is a pure unpack now, so that would ship the raw
product-scale accumulator as the result."""
import re
from pathlib import Path

import pytest

from test_tensor_slice_operand_guard import _load_tensor_slice_package
from test_tensor_slice_rf import _with_tensor_slice_on_path

_generate_catapult_pkg = _load_tensor_slice_package().generate_catapult_pkg


def generate_catapult_pkg(*args, **kwargs):
    return _with_tensor_slice_on_path(_generate_catapult_pkg, *args, **kwargs)


def _stage2_shift(pkg_dir, name):
    text = (Path(pkg_dir) / name / f"{name}_gemm_ip.h").read_text()
    m = re.search(r"_r2 >> (\d+)\)", text)
    assert m, "stage-2 shift not found in the behavioral core"
    return int(m.group(1))


def _core_stage2_shift(pkg_dir, name):
    text = (Path(pkg_dir) / name / f"{name}_core.v").read_text()
    m = re.search(r">>> (\d+);\s*\n\s*stage2 = ", text)
    assert m, "stage-2 shift not found in the Verilog core"
    return int(m.group(1))


@pytest.mark.parametrize("output_precision,expected_shift", [
    ("fixed<10,3>", 5),   # 12 - 7
    ("fixed<8,8>", 12),   # zero result fraction: the whole product fraction
    (None, 12),           # unset result precision: same
])
def test_total_shift_covers_zero_fraction_results(tmp_path, output_precision, expected_shift):
    name = "t"
    generate_catapult_pkg(
        m=8, k=8, n=8, name=name, output_dir=str(tmp_path),
        interface="stream", output_precision=output_precision,
        input_precision="fixed<8,2>", weight_precision="fixed<8,2>",
    )
    assert _stage2_shift(tmp_path, name) == expected_shift
    assert _core_stage2_shift(tmp_path, name) == expected_shift


def test_result_finer_than_product_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="left shift"):
        generate_catapult_pkg(
            m=8, k=8, n=8, name="t", output_dir=str(tmp_path),
            interface="stream", output_precision="fixed<16,2>",
            input_precision="fixed<8,2>", weight_precision="fixed<8,2>",
        )
