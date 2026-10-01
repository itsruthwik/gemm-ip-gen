"""Unit tests for the cmvu golden oracle and packers."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Package-qualified import (see test_cmvu_geometry.py) to avoid a bare
# `geometry`/`golden` sys.modules collision with the mvau tests.
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from targets.cmvu import geometry as g  # noqa: E402
from targets.cmvu import golden as gold  # noqa: E402


# ── requant ───────────────────────────────────────────────────────────────────

def test_requant_identity_and_wrap():
    assert gold.requant(100, 0) == 100
    assert gold.requant(300, 0) == gold.wrap_int8(300) == 44
    assert gold.requant(-300, 0) == -44


def test_requant_floor_trn():
    # The block only truncates (floor/TRN) now; rounding lives in the bias.
    assert gold.requant(3, 1) == 1
    assert gold.requant(-3, 1) == -2
    assert gold.requant(1, 1) == 0
    assert gold.requant(8, 4) == 0
    assert gold.requant(0, 4) == 0


@pytest.mark.parametrize("bad", [-1, g.MAX_SHIFT + 1])
def test_requant_shift_range(bad):
    with pytest.raises(ValueError, match="shift must be"):
        gold.requant(0, bad)


def test_requant_w_floor_shift_wrap():
    # floor (truncating) shift then two's-complement wrap to W bits
    assert gold.requant_w(3, 1, 8) == 1
    assert gold.requant_w(-3, 1, 8) == -2
    assert gold.requant_w(100, 0, 8) == 100
    assert gold.requant_w(200, 0, 8) == gold.wrap_to_w(200, 8) == -56
    assert gold.requant_w(200, 0, 16) == 200
    assert gold.requant_w(40000, 0, 16) == gold.wrap_to_w(40000, 16)


def test_requant_w_width_range():
    with pytest.raises(ValueError, match="width must be"):
        gold.requant_w(0, 0, 0)
    with pytest.raises(ValueError, match="width must be"):
        gold.requant_w(0, 0, g.RESULT_WIDTH + 1)


@pytest.mark.parametrize("width", [8, 12, 16])
def test_pack_unpack_res_roundtrip_width(width):
    lanes = [1, -2, 3, -4, 5, -6, 7, -8]
    bits = gold.pack_res_row(lanes, width=width)
    assert gold.unpack_res_row(bits, 1, width=width) == lanes


# ── oracle ────────────────────────────────────────────────────────────────────

def test_reference_matches_naive():
    rng = np.random.default_rng(7)
    A = rng.integers(-128, 128, size=(5, 8))
    B = rng.integers(-128, 128, size=(8, 6))
    Y = gold.reference_rows(A, B, shift=3)
    for m in range(5):
        for j in range(6):
            assert Y[m, j] == gold.requant(int(A[m] @ B[:, j]), 3)


def test_reference_with_bias_codes():
    A = np.array([[1, 2]])
    B = np.array([[3], [4]])
    Y = gold.reference_rows(A, B, bias_codes=[5], shift=0)
    assert int(Y[0, 0]) == 1 * 3 + 2 * 4 + 5


def test_reference_shape_errors():
    with pytest.raises(ValueError, match="shape mismatch"):
        gold.reference_rows(np.zeros((2, 3)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="bias_codes"):
        gold.reference_rows(np.zeros((1, 2)), np.zeros((2, 3)), bias_codes=[1])


# ── bias baking ───────────────────────────────────────────────────────────────

def test_bias_codes_exact_no_round():
    # requant_shift=0 -> no rounding constant regardless of rnd
    codes, warns = gold.bias_codes([1.0, -1.0, 0.5, -0.5], 1, requant_shift=0)
    assert list(codes) == [2, -2, 1, -1]
    assert warns == []


def test_bias_codes_folds_round_constant():
    # rnd=True (default) folds 1 << (requant_shift-1) into every lane
    codes, warns = gold.bias_codes([1.0, -1.0], 1, requant_shift=3)
    assert list(codes) == [2 + 4, -2 + 4]
    assert warns == []


def test_bias_codes_trn_no_round_constant():
    codes, warns = gold.bias_codes([1.0, -1.0], 1, rnd=False, requant_shift=3)
    assert list(codes) == [2, -2]
    assert warns == []


def test_bias_codes_no_bias_rnd_only_round_constant():
    codes, warns = gold.bias_codes(None, 0, rnd=True, requant_shift=4)
    assert all(c == (1 << 3) for c in codes)
    assert warns == []


def test_bias_codes_wrap_warns():
    codes, warns = gold.bias_codes([1.0, -2.0], 31, requant_shift=0)
    assert list(codes) == [gold.wrap_int32(1 << 31), gold.wrap_int32(-(1 << 32))]
    assert len(warns) == 2
    assert "bias lane 0" in warns[0] and "wrapped" in warns[0]


# ── packers ───────────────────────────────────────────────────────────────────

def test_pack_a_row_lsb_first():
    val = gold.pack_a_row([1, 2, -1], 1)
    assert val == 0x00FF0201  # lane0=0x01, lane1=0x02, lane2=0xFF, pad 0
    assert gold.unpack_res_row(val, 1) == [1, 2, -1, 0, 0, 0, 0, 0]


def test_pack_unpack_res_roundtrip():
    lanes = [0, 1, -1, 127, -128, 42, -42, 7]
    assert gold.unpack_res_row(gold.pack_res_row(lanes), 1) == lanes


def test_pack_a_row_width_error():
    with pytest.raises(ValueError, match="width is"):
        gold.pack_a_row([1, 2, 3, 4, 5], 1)


def test_pad_row():
    assert gold.pad_row([1, 2], 1) == [1, 2, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="width is"):
        gold.pad_row(list(range(9)), 1)


# ── tiles ─────────────────────────────────────────────────────────────────────

def test_weight_tiles_tail_padding():
    B = np.arange(60).reshape(6, 10)
    tiles = gold.weight_tiles(B, 6, 10, 2, 2)
    assert tiles[0][0].shape == (4, 8)
    assert tiles[0][0][0, 0] == 0 and tiles[0][0][3, 7] == 37
    # kt=1 covers K rows 4..7 but K=6 -> last two rows are zero padding
    assert tiles[1][0][1, 0] == 50 and tiles[1][0][2, 0] == 0
    # nt=1 covers N cols 8..15 but N=10 -> last six columns are zero padding
    assert tiles[0][1][0, 1] == 9 and tiles[0][1][0, 2] == 0


def test_pack_tile_layout():
    tile = np.zeros((4, 8), dtype=np.int64)
    tile[0, 1] = 5
    tile[1, 0] = 7
    tile[3, 7] = -1
    val = gold.pack_tile(tile)
    assert (val >> 8) & 0xFF == 5          # index i*8+j = 1
    assert (val >> 64) & 0xFF == 7         # index 8
    assert (val >> 248) & 0xFF == 0xFF     # index 31
    assert gold.hex_literal(val, 32)[:2] == "ff"


def test_hex_literal_zero_padded():
    assert gold.hex_literal(0x1, 32) == "0" * 63 + "1"


# ── runtime-B beat packer ─────────────────────────────────────────────────────

def _B(k, n, seed=3):
    return np.random.default_rng(seed).integers(-128, 128, size=(k, n))


def test_b_load_beats_order_and_padding():
    B = _B(16, 8)
    beats = gold.b_load_beats(B, 16, 8, 4, 1)
    assert len(beats) == 4 * 1 * 4          # ktiles * ntiles * K_PHYS
    tiles = gold.weight_tiles(B, 16, 8, 4, 1)
    # first beat is tile(0,0) row 0 = low 64 bits of the packed tile
    assert beats[0] == (gold.pack_tile(tiles[0][0]) & ((1 << 64) - 1))
    # k=4/n=8 with k_chunks_pad=n_chunks_pad=2: only tile(0,0) is real, so the
    # last tile (kt=1,nt=1) is entirely zero-padding.
    beats_pad = gold.b_load_beats(_B(4, 8), 4, 8, 2, 2)
    assert len(beats_pad) == 2 * 2 * 4
    assert all(b == 0 for b in beats_pad[-4:])


def test_runtime_b_tb_loads_then_streams():
    B = _B(8, 16)
    tb = gold.generate_runtime_b_tb(4, 8, 16, 1, 1, B, shift=4,
                                    module_name="cmvu_core", name="rb")
    assert "b_mem" in tb and "b_valid = 1'b1" in tb
    assert "@(negedge clk)" in tb
    assert "ALL_PASS" in tb


# ── exactness: cmvu golden vs. an independent pure-numpy hls4ml reference ──────
#
# hls4ml semantics, reimplemented independently of golden.py: exact int
# products at "code" scale (no fixed-point objects), accumulate, add the raw
# (unbaked) bias code, then requant with either TRN (floor) or RND
# (round-half-up, shift>0) and wrap to W bits signed. Compared against
# golden.reference_rows fed the SAME shift/bias via golden.bias_codes (which
# folds the round constant into the bias at accumulator scale), i.e. the path
# the generator actually drives cmvu_mode1 with.

def _np_hls4ml_reference(A, B, bias_code, shift, width, rnd):
    total = A.astype(np.int64) @ B.astype(np.int64)
    if bias_code is not None:
        total = total + np.asarray(bias_code, dtype=np.int64)[None, :]
    if rnd and shift > 0:
        total = total + (1 << (shift - 1))
    q = total >> shift  # Python >> on int64 arrays is arithmetic per-lane... (see below)
    return np.array([[gold.wrap_to_w(int(v), width) for v in row] for row in q])


@pytest.mark.parametrize("rnd", [True, False])
@pytest.mark.parametrize("with_bias", [True, False])
def test_exactness_vs_independent_numpy_reference(rnd, with_bias):
    rng = np.random.default_rng(1234 + int(rnd) * 2 + int(with_bias))
    for trial in range(8):
        m, k, n = 3, rng.integers(1, 9), rng.integers(1, 9)
        shift = int(rng.integers(0, 6))
        width = int(rng.choice([8, 12, 16]))
        A = rng.integers(-128, 128, size=(m, k))
        B = rng.integers(-128, 128, size=(k, n))
        if with_bias:
            raw_bias = rng.integers(-64, 65, size=n) / 4.0
            bias_code_raw = np.round(raw_bias * (1 << shift)).astype(np.int64)
        else:
            bias_code_raw = np.zeros(n, dtype=np.int64)

        # cmvu's actual generator path: bias baked (round const folded in).
        baked_codes, _ = gold.bias_codes(
            bias_code_raw.astype(np.float64) / (1 << shift), shift, rnd=rnd,
            requant_shift=shift)
        got = gold.reference_rows(A, B, bias_codes=baked_codes, shift=shift,
                                  result_width=width)

        want = _np_hls4ml_reference(A, B, bias_code_raw, shift, width, rnd)
        assert np.array_equal(got, want), (
            f"trial {trial}: m={m} k={k} n={n} shift={shift} width={width} "
            f"rnd={rnd} with_bias={with_bias}")


@pytest.mark.parametrize("shift,rnd", [(5, True), (5, False), (0, True)])
def test_bias_finer_than_products_is_exact(shift, rnd):
    # Bias with 2 fractional bits below the product scale (pf=8): not a whole code, but
    # the floor-baked code (round-half-up with no shift) keeps the requant exact.
    from fractions import Fraction
    pf, W = 8, 16
    bias = [k / (1 << (pf + 2)) for k in (3, -3, 1, -1, 2050, -2051, 7, 0)]
    codes, _ = gold.bias_codes(bias, pf, rnd=rnd, requant_shift=shift)
    rng = np.random.default_rng(11)
    A = rng.integers(-128, 128, (48, 16))
    B = rng.integers(-128, 128, (16, 8))
    got = gold.reference_rows(A, B, codes, shift=shift, result_width=W)
    raw = A @ B
    for (r, c), v in np.ndenumerate(raw):
        x = (Fraction(int(v)) + Fraction(bias[c]) * (1 << pf)) / (1 << shift)
        q = (x + Fraction(1, 2)).__floor__() if rnd else x.__floor__()
        q = (q + (1 << (W - 1))) % (1 << W) - (1 << (W - 1))
        assert got[r, c] == q, (r, c)
