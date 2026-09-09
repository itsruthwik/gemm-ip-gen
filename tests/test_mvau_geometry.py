"""Unit tests for the mvau target's geometry (RTL param resolution)."""
import sys
from pathlib import Path

import pytest

# geometry.py is self-contained; import it with the target dir on sys.path
# (same mechanism the registry uses to load a target's modules).
_MVAU = str(Path(__file__).resolve().parent.parent / "src" / "targets" / "mvau")
if _MVAU not in sys.path:
    sys.path.insert(0, _MVAU)
import geometry as g  # noqa: E402

VERSAL = "xcvc1902-vsva2197-2MP-e-S"


# ── precision parsing ─────────────────────────────────────────────────────────

def test_parse_width_and_signed():
    assert g.parse_width("fixed<8,4>") == 8
    assert g.parse_width("fixed<16,6,AP_RND>") == 16
    assert g.parse_width(None) == 8            # default
    assert g.parse_width("garbage") == 8
    assert g.parse_signed("fixed<8,4>") is True
    assert g.parse_signed("ufixed<8,4>") is False
    assert g.parse_signed(None) is True


def test_parse_int_bits_and_frac():
    assert g.parse_int_bits("fixed<8,4>") == 4
    assert g.parse_int_bits("fixed<8,-2>") == -2
    assert g.parse_int_bits(None, default=3) == 3
    assert g.parse_frac("fixed<8,4>") == 4
    assert g.parse_frac(None) == 0


# ── Versal-only gate ───────────────────────────────────────────────────────────

def test_require_versal_accepts_versal_parts():
    for part in ("xcvc1902", "xcvp1202", "xcvm1802", "xcve2802", "xcvh1522"):
        g.require_versal(part)   # no raise


def test_require_versal_rejects_non_versal():
    with pytest.raises(ValueError):
        g.require_versal("xcvu13p")   # US+ Virtex, not Versal
    with pytest.raises(ValueError):
        g.require_versal("xc7z020")
    with pytest.raises(ValueError):
        g.require_versal(None)


# ── envelope rejection ────────────────────────────────────────────────────────

def test_envelope_reject_wide_weight():
    with pytest.raises(ValueError):
        g.check_envelope(16, 8, True)


def test_envelope_reject_wide_act():
    with pytest.raises(ValueError):
        g.check_envelope(8, 16, True)
    # 9b signed activation is allowed (the only >8b case the MVU envelope permits)
    g.check_envelope(8, 9, True)   # ok, no raise
    with pytest.raises(ValueError):
        g.check_envelope(8, 9, False)   # unsigned 9b is not allowed


# ── accumulator sizing ────────────────────────────────────────────────────────

def test_accu_width():
    assert g.accu_width(4, 8, 8) == 2 + 8 + 8 + 1      # ceil(log2 4)=2
    assert g.accu_width(1, 8, 8) == 0 + 8 + 8 + 1
    assert g.accu_width(256, 8, 8) == 8 + 8 + 8 + 1
    assert g.accu_width(6, 4, 4) == 3 + 4 + 4 + 1      # ceil(log2 6)=3


# ── narrow weights ────────────────────────────────────────────────────────────

def test_narrow_weights():
    assert g.narrow_weights(None, 4) == 0
    assert g.narrow_weights([[-8, 7], [1, 2]], 4, signed=True) == 0   # uses min (-8)
    assert g.narrow_weights([[-7, 7], [1, 2]], 4, signed=True) == 1   # excludes -8


# ── resolve_fold: fold-N (the default) ─────────────────────────────────────────

def test_resolve_fold_n_exact():
    r = g.resolve_fold(k=6, n=8, reuse_factor=4, fold_axis="n")
    assert r["pe"] == 2 and r["simd"] == 6
    assert r["k_pad"] == 6 and r["n_pad"] == 8
    assert r["reuse_factor"] == 4
    assert r["warnings"] == []


def test_resolve_fold_n_pads():
    # N=10, RF=4 -> n_pad = ceil(10/4)*4 = 12, PE = 3
    r = g.resolve_fold(k=6, n=10, reuse_factor=4, fold_axis="n")
    assert r["n_pad"] == 12 and r["pe"] == 3 and r["simd"] == 6
    assert r["k_pad"] == 6
    assert r["warnings"] == []   # RF itself was honored exactly; only N padded


def test_resolve_fold_n_rf_exceeds_n_legalizes():
    # RF > N cannot be honored (PE would be < 1) -> legalizes to RF=N (PE=1), warns.
    r = g.resolve_fold(k=4, n=3, reuse_factor=8, fold_axis="n", name="foo")
    assert r["reuse_factor"] == 3 and r["pe"] == 1 and r["n_pad"] == 3
    assert len(r["warnings"]) == 1
    assert 'layer "foo"' in r["warnings"][0]


# ── resolve_fold: fold-K ────────────────────────────────────────────────────────

def test_resolve_fold_k_exact():
    r = g.resolve_fold(k=8, n=4, reuse_factor=2, fold_axis="k")
    assert r["pe"] == 4 and r["simd"] == 4
    assert r["k_pad"] == 8 and r["n_pad"] == 4
    assert r["warnings"] == []


def test_resolve_fold_k_below_3_never_folds():
    # K < 3: SIMD=K is the only legal point; any RF != 1 legalizes to 1.
    r = g.resolve_fold(k=2, n=4, reuse_factor=5, fold_axis="k")
    assert r["simd"] == 2 and r["k_pad"] == 2 and r["reuse_factor"] == 1
    assert len(r["warnings"]) == 1


def test_resolve_fold_k_rf_exceeds_floor_legalizes():
    # K=8: SIMD floor is 3 -> max legal RF is ceil(8/3)=3. RF=5 legalizes to 3.
    r = g.resolve_fold(k=8, n=4, reuse_factor=5, fold_axis="k")
    assert r["reuse_factor"] == 3
    assert r["k_pad"] == 9 and r["simd"] == 3   # ceil(8/3)*3 = 9
    assert len(r["warnings"]) == 1


# ── resolve_fold: fold-KN ────────────────────────────────────────────────────────

def test_resolve_fold_kn_pads_both():
    # K=10, N=10, RF=4: N-side -> n_pad=12, PE=3; K-side -> SIMD floor=ceil(10/3)=4,
    # RF=4 is exactly at the floor -> k_pad = ceil(10/4)*4 = 12, SIMD=3.
    r = g.resolve_fold(k=10, n=10, reuse_factor=4, fold_axis="kn")
    assert r["n_pad"] == 12 and r["pe"] == 3
    assert r["k_pad"] == 12 and r["simd"] == 3
    assert r["reuse_factor"] == 4


def test_resolve_fold_invalid_axis():
    with pytest.raises(ValueError):
        g.resolve_fold(k=4, n=4, reuse_factor=1, fold_axis="bogus")


# ── the full plan (fold_plan) ────────────────────────────────────────────────────

def test_fold_plan_matches_validated_spike():
    # Must reproduce temp_space/mvau-spike geometry (the cosim-PASS config).
    p = g.fold_plan(1, 4, 4, weight_precision="fixed<8,4>",
                    input_precision="fixed<8,4>", output_precision="fixed<16,6>",
                    part=VERSAL, reuse_factor=1)
    t = p["tile"]
    assert t["compute_core"] == "mvu_vvu_8sx9_dsp58"
    assert (t["pe"], t["simd"], t["sf"], t["nf"]) == (4, 4, 1, 1)
    assert p["k_pad"] == 4 and p["n_pad"] == 4


def test_fold_plan_n_tiling():
    p = g.fold_plan(1, 8, 8, weight_precision="fixed<8,4>",
                    input_precision="fixed<8,4>", output_precision="fixed<16,6>",
                    part=VERSAL, n_tiles=2, reuse_factor=1)
    assert p["n_tiles"] == 2 and p["n_tile"] == 4
    assert p["tile"]["mh"] == 4
    with pytest.raises(ValueError):
        g.fold_plan(1, 8, 8, part=VERSAL, n_tiles=3)   # 3 does not divide 8


def test_fold_plan_rejects_non_versal():
    with pytest.raises(ValueError):
        g.fold_plan(1, 8, 8, weight_precision="fixed<8,4>",
                    input_precision="fixed<8,4>", part="xcvu13p")


def test_fold_plan_reject_wide_precision():
    with pytest.raises(ValueError):
        g.fold_plan(1, 8, 8, weight_precision="fixed<16,6>",
                    input_precision="fixed<8,4>", part=VERSAL)


def test_fold_plan_n_pad_reported_and_manifest_fields():
    # (4, 6, 10) RF=4, fold-N: n_pad=12, PE=3.
    p = g.fold_plan(4, 6, 10, weight_precision="fixed<8,4>",
                    input_precision="fixed<8,4>", output_precision="fixed<16,6>",
                    part=VERSAL, reuse_factor=4, fold_axis="n")
    assert p["n"] == 10 and p["n_pad"] == 12
    assert p["tile"]["pe"] == 3
    assert p["reuse_factor"] == 4
    assert p["effective_reuse"] == (p["k_pad"] * p["n_pad"]) // (p["tile"]["pe"] * p["tile"]["simd"])


def test_fold_plan_kn_effective_reuse_is_squared():
    # fold-kn: each MAC is reused RF**2 times per input vector (not RF).
    p = g.fold_plan(4, 10, 10, weight_precision="fixed<8,4>",
                    input_precision="fixed<8,4>", output_precision="fixed<16,6>",
                    part=VERSAL, reuse_factor=4, fold_axis="kn")
    assert p["reuse_factor"] == 4
    assert p["effective_reuse"] == 16


# ── ap_uint beat-width limit ────────────────────────────────────────────────────

def test_fold_plan_rejects_output_beat_over_ap_uint_limit():
    # fc1-like (K=784, N=128), fold-K RF=262: PE=128, accu=23b -> a 2944b output
    # beat, well past the 1024b ap_uint cap the hls4ml-facing header must fit.
    with pytest.raises(ValueError, match="2944"):
        g.fold_plan(1, 784, 128, weight_precision="fixed<8,4>",
                    input_precision="fixed<4,2>", output_precision="fixed<16,6>",
                    part=VERSAL, reuse_factor=262, fold_axis="k", name="gemm_fc1")


def test_fold_plan_fold_axis_n_within_ap_uint_limit_is_ok():
    # Same (K, N) shape, fold-N RF=3: PE=43, and a narrow enough activation that
    # every declared beat (weight/input/output) stays under the 1024b cap.
    p = g.fold_plan(1, 784, 128, weight_precision="fixed<8,4>",
                    input_precision="ufixed<1,0>", output_precision="fixed<16,6>",
                    part=VERSAL, reuse_factor=3, fold_axis="n", name="gemm_fc1")
    assert p["tile"]["pe"] == 43
    assert p["tile"]["simd"] == 784


def test_fold_plan_manual_pe_simd_rejects_wide_weight_stream_beat():
    # Manual (pe, simd) debug path, weights_in_core=False (weight matrix is a C++
    # stream): PE*SIMD*weight_width = 128*3*8 = 3072b exceeds the ap_uint cap.
    with pytest.raises(ValueError, match="weight beat"):
        g.fold_plan(1, 784, 128, weight_precision="fixed<8,4>",
                    input_precision="fixed<4,2>", part=VERSAL,
                    pe=128, simd=3, weights_in_core=False, name="gemm_fc1")
