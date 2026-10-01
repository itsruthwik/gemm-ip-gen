"""A bias finer than the tensor_slice intermediate scale (input frac + weight frac - S1).

It cannot be a whole code there. Rounding it to the nearest code flips results that sit
near a rounding tie; floor is exact when stage 2 then rounds half-up by >= 1 bit or
truncates (floor(b) plus the rounding half crosses the same cut sum + b does), and
round-half-up is exact when no stage-2 shift follows."""
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gemm_ip.biasrom import bias_acc_codes  # noqa: E402
from targets.tensor_slice import golden as _golden  # noqa: E402
from test_tensor_slice_requant_mode import _IN, _bias_codes, generate_catapult_pkg  # noqa: E402

# frac(a) + frac(b) = 12 (see _IN); these biases carry 14 fractional bits, and each one's
# nearest 2^-12 code differs from its floor.
FINE_BIAS = [k / (1 << 14) for k in (3, -3, 7, -1, 11, 2050, -2051, 1)]


def _exact(A, B, bias, s2, rnd, out_width):
    """Exact rational result: (A @ B) at 2^-12 plus the real bias, one round to 2^-(12-s2)."""
    raw = A.astype(np.int64) @ B.astype(np.int64)
    out = np.zeros(raw.shape, dtype=np.int64)
    for (r, c), v in np.ndenumerate(raw):
        x = (Fraction(int(v)) + Fraction(bias[c]) * (1 << 12)) / (1 << s2)
        q = int(np.floor(float(x + Fraction(1, 2)))) if rnd else int(np.floor(float(x)))
        q = (q + (1 << (out_width - 1))) % (1 << out_width) - (1 << (out_width - 1))
        out[r, c] = q
    return out


@pytest.mark.parametrize("output_precision,s2,rnd", [
    ("fixed<10,3,RND,WRAP,0>", 5, True),
    ("fixed<10,3,TRN,WRAP,0>", 5, False),
    ("fixed<16,4,RND,WRAP,0>", 0, True),
])
def test_fine_bias_codes_are_exact(tmp_path, output_precision, s2, rnd):
    name = "fb"
    generate_catapult_pkg(m=8, k=8, n=8, name=name, output_dir=str(tmp_path), interface="stream",
                          output_precision=output_precision, has_bias=True, bias=FINE_BIAS,
                          bias_precision="fixed<16,2,TRN,WRAP,0>", **_IN)
    codes = _bias_codes(tmp_path, name)
    expected = bias_acc_codes(FINE_BIAS, 12, 8, True, rounding="half_up" if (s2 == 0 and rnd) else "floor")
    if not rnd and s2 > 0:
        expected = [c - (1 << (s2 - 1)) for c in expected]   # TRN folds the half out
    assert codes == expected
    assert codes != bias_acc_codes(FINE_BIAS, 12, 8, True) or not rnd  # not today's nearest

    out_width = 10 if s2 else 16
    rng = np.random.default_rng(7)
    A = rng.integers(-128, 128, (32, 8))
    B = rng.integers(-128, 128, (8, 8))
    got = _golden.two_stage_reference(A, B, codes, s1=0, s2=s2, out_width=out_width)
    np.testing.assert_array_equal(got, _exact(A, B, FINE_BIAS, s2, rnd, out_width))


def test_nearest_rounding_of_a_fine_bias_is_not_exact():
    # The failure being fixed: the historical nearest-code bias misses ties.
    rng = np.random.default_rng(3)
    A = rng.integers(-128, 128, (64, 8))
    B = rng.integers(-128, 128, (8, 8))
    near = bias_acc_codes(FINE_BIAS, 12, 8, True)
    got = _golden.two_stage_reference(A, B, near, s1=0, s2=5, out_width=10)
    assert (got != _exact(A, B, FINE_BIAS, 5, True, 10)).any()
