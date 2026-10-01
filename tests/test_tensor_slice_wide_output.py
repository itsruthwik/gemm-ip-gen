"""The tensor_slice output lane is 32 bits: a K contraction whose sum needs more
than 16 bits must reach stage 2 intact with S1 = 0, not wrap at the slice pin.
Near-full-scale int8 operands over K = 64 drive every sum well past 2^15."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from targets.tensor_slice import golden as _golden  # noqa: E402
from targets.tensor_slice import rtl as _rtl  # noqa: E402

_MODEL = _SRC / "targets" / "tensor_slice" / "tensor_slice_int8_atlas.v"


def test_slice_port_is_eight_32_bit_lanes():
    text = _MODEL.read_text()
    assert "output wire [255:0] c_data_out" in text
    assert "function signed [31:0] stage1" in text


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not on PATH")
def test_sum_beyond_16_bits_survives_the_slice(tmp_path):
    m, k, n, s2, out_width = 8, 64, 8, 8, 16
    rng = np.random.default_rng(3)
    sign_a = rng.choice([-1, 1], size=(m, 1))
    sign_b = rng.choice([-1, 1], size=(1, n))
    A = (sign_a * rng.integers(100, 128, size=(m, k))).astype(np.int64)
    B = (sign_b * rng.integers(100, 128, size=(k, n))).astype(np.int64)
    assert np.abs(A @ B).min() >= (1 << 15)
    bias = [37, -37, 1000, -1000, 0, 5, -5, 12345]

    C = _golden.two_stage_reference(A, B, bias, s1=0, s2=s2, out_width=out_width)
    mod = "wide_out_wrapper"
    rtl_path = tmp_path / "core.v"
    tb_path = tmp_path / "tb.v"
    out_path = tmp_path / "sim.out"
    rtl_path.write_text(_rtl.generate_combined_core_verilog(
        m, k, n, module_name=mod, s1=0, s2=s2, out_width=out_width, bias_codes=bias))
    tb_path.write_text(_golden.generate_tb_with_data(
        m, k, n, mod, 0, "catapult", A, B, bias, C, out_width=out_width))

    comp = subprocess.run(["iverilog", "-g2012", "-DSYNTHESIS", "-o", str(out_path),
                           str(tb_path), str(rtl_path), str(_MODEL)],
                          text=True, capture_output=True)
    assert comp.returncode == 0, comp.stdout + comp.stderr
    sim = subprocess.run(["vvp", str(out_path)], text=True, capture_output=True)
    log = sim.stdout + sim.stderr
    assert "ALL_PASS" in log and "FAILURES=" not in log, log
