"""Unit tests for the mvau memstream weight packer.

Locks the cosim-PASS spike golden (single tile) and cross-checks the multi-tile
(nf, sf, pe, s) ordering against a numpy reimplementation of FINN's authoritative
packing (matrixvectoractivation.make_weight_file, decoupled_verilog_dat).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_MVAU = str(Path(__file__).resolve().parent.parent / "src" / "targets" / "mvau")
if _MVAU not in sys.path:
    sys.path.insert(0, _MVAU)
import weightpack as wp  # noqa: E402


def _finn_reference(B, n, k, pe, simd, ww):
    """FINN-faithful reference: reshape/transpose/flip pipeline + MSB-first hex.

    B is [K, N]. Mirrors get_hw_compatible_weight_tensor + the decoupled .dat
    branch of make_weight_file (SIMD flip, PE flip, MSB-first pack).
    """
    orig = np.asarray(B, dtype=np.int64)          # (MW, MH) = (K, N)
    mw, mh = k, n
    wmem = mw * mh // (pe * simd)
    ret = orig.T                                   # (MH, MW) = (N, K)
    # interleave_matrix_outer_dim_from_partitions(ret, pe): reshape+transpose
    ret = ret.reshape(-1, pe, mw).transpose((1, 0, 2))   # (PE, NF, MW)
    ret = ret.reshape(1, pe, wmem, simd)
    ret = np.flip(ret, axis=-1)                    # SIMD flip
    t = np.transpose(ret, (0, 2, 1, 3))            # (1, WMEM, PE, SIMD)
    t = np.flip(t, axis=-2)                        # PE flip
    t = t.reshape(1, -1, pe * simd)                # (1, WMEM, PE*SIMD)
    ndigits = (pe * simd * ww + 3) // 4
    mask = (1 << ww) - 1
    lines = []
    for row in t[0]:
        word = 0
        for val in row:                            # first element -> MSB
            word = (word << ww) | (int(val) & mask)
        lines.append(format(word, "x").zfill(ndigits))
    return "\n".join(lines) + "\n"


def test_pack_matches_cosim_spike_golden():
    # temp_space/mvau-ws spike (cosim PASS). WS_W[pe][s] used as W[pe][s];
    # B is [K][N] = W^T, i.e. B[k][n] = WS_W[n][k].
    WS_W = [[1, -2, 2, -4], [-3, 4, -3, 1], [-4, 4, -1, -4], [-3, 2, 2, -3]]
    B = [[WS_W[nn][kk] for nn in range(4)] for kk in range(4)]   # B[k][n]
    out = wp.pack_memstream_hex(B, n=4, k=4, pe=4, simd=4, weight_width=8)
    assert out.strip() == "fd0202fdfcff04fc01fd04fdfc02fe01"


@pytest.mark.parametrize("n,k,pe,simd,ww", [
    (4, 4, 4, 4, 8),     # single tile (spike shape)
    (8, 8, 4, 4, 8),     # NF=2, SF=2 -> exercises nf-outer/sf-inner line order
    (16, 8, 8, 2, 4),    # NF=2, SF=4, 4-bit codes
    (6, 6, 3, 2, 8),     # non-power-of-two folds
    (8, 4, 2, 4, 4),     # NF=4, SF=1
])
def test_pack_matches_finn_reference(n, k, pe, simd, ww):
    lo, hi = -(1 << (ww - 1)), (1 << (ww - 1)) - 1
    rng = np.random.default_rng(1234 + n * 100 + k * 10 + pe)
    B = rng.integers(lo, hi + 1, size=(k, n)).tolist()   # [K][N]
    assert wp.pack_memstream_hex(B, n, k, pe, simd, ww) == _finn_reference(B, n, k, pe, simd, ww)


def test_pack_rejects_out_of_range_code():
    B = [[200, 0], [0, 0]]      # 200 does not fit signed 8-bit
    with pytest.raises(ValueError):
        wp.pack_memstream_hex(B, n=2, k=2, pe=2, simd=2, weight_width=8)


def test_pack_rejects_fold_mismatch():
    B = [[1, 1, 1], [1, 1, 1]]   # K=2, N=3
    with pytest.raises(ValueError):
        wp.pack_memstream_hex(B, n=3, k=2, pe=2, simd=2, weight_width=8)
