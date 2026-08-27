"""Unit tests for the mvau target's folding/geometry search."""
import sys
from pathlib import Path

import pytest

# geometry.py is self-contained; import it with the target dir on sys.path
# (same mechanism the registry uses to load a target's modules).
_MVAU = str(Path(__file__).resolve().parent.parent / "src" / "targets" / "mvau")
if _MVAU not in sys.path:
    sys.path.insert(0, _MVAU)
import geometry as g  # noqa: E402


# ── precision parsing ─────────────────────────────────────────────────────────

def test_parse_width_and_signed():
    assert g.parse_width("fixed<8,4>") == 8
    assert g.parse_width("fixed<16,6,AP_RND>") == 16
    assert g.parse_width(None) == 8            # default
    assert g.parse_width("garbage") == 8
    assert g.parse_signed("fixed<8,4>") is True
    assert g.parse_signed("ufixed<8,4>") is False
    assert g.parse_signed(None) is True


# ── core selection by device + precision ──────────────────────────────────────

def test_dsp_block_for_part():
    assert g.dsp_block_for_part("xcvc1902") == "DSP58"     # Versal
    assert g.dsp_block_for_part("xcvu13p") == "DSP48E2"    # US+ Virtex
    assert g.dsp_block_for_part("xczu7ev") == "DSP48E2"    # Zynq US+
    assert g.dsp_block_for_part("xc7z020") == "DSP48E1"    # 7-series
    assert g.dsp_block_for_part(None) == "DSP48E2"         # default


def test_select_core():
    assert g.select_core("DSP58", 8, 8) == "mvu_vvu_8sx9_dsp58"
    assert g.select_core("DSP58", 4, 4) == "mvu_vvu_8sx9_dsp58"
    assert g.select_core("DSP48E2", 8, 8) == "mvu_8sx8u_dsp48"
    assert g.select_core("DSP48E2", 4, 4) == "mvu_4sx4u_dsp48e2"
    assert g.select_core("DSP48E1", 4, 4) == "mvu_4sx4u_dsp48e1"
    assert g.select_core("DSP48E1", 5, 4) == "mvu_8sx8u_dsp48"  # >4b weight


# ── envelope rejection ────────────────────────────────────────────────────────

def test_envelope_reject_wide_weight():
    with pytest.raises(ValueError):
        g.check_envelope("DSP48E2", 16, 8, True)


def test_envelope_reject_wide_act():
    with pytest.raises(ValueError):
        g.check_envelope("DSP48E2", 8, 16, True)
    # 9b signed activation is allowed only on DSP58
    with pytest.raises(ValueError):
        g.check_envelope("DSP48E2", 8, 9, True)
    g.check_envelope("DSP58", 8, 9, True)   # ok, no raise


# ── accumulator sizing ────────────────────────────────────────────────────────

def test_accu_width():
    assert g.accu_width(4, 8, 8) == 2 + 8 + 8 + 1      # ceil(log2 4)=2
    assert g.accu_width(1, 8, 8) == 0 + 8 + 8 + 1
    assert g.accu_width(256, 8, 8) == 8 + 8 + 8 + 1
    assert g.accu_width(6, 4, 4) == 3 + 4 + 4 + 1      # ceil(log2 6)=3


# ── folding search ────────────────────────────────────────────────────────────

def test_fold_full_unroll_small():
    # RF=1 -> full unroll bounded by the weight-width cap (8*SIMD<=36 -> SIMD<=4)
    pe, simd = g.fold(k=4, n=4, weight_width=8, target_cycles=1)
    assert (pe, simd) == (4, 4)


def test_fold_wwidth_cap_bounds_simd():
    # K=8, weight_width=8: 8*SIMD<=36 -> SIMD<=4, so SIMD snaps to 4 (divisor of 8)
    pe, simd = g.fold(k=8, n=8, weight_width=8, target_cycles=1)
    assert simd == 4
    assert 8 * simd <= g.WWIDTH_MAX


def test_fold_divisor_constraints():
    for k, n in [(12, 10), (16, 8), (6, 9)]:
        pe, simd = g.fold(k, n, weight_width=4, target_cycles=1)
        assert k % simd == 0
        assert n % pe == 0


def test_fold_reuse_factor_folds():
    # A higher cycle target should not fully unroll: II = SF*NF should be larger.
    pe_hi, simd_hi = g.fold(k=8, n=8, weight_width=4, target_cycles=1)
    pe_lo, simd_lo = g.fold(k=8, n=8, weight_width=4, target_cycles=32)
    ii_hi = g.exp_cycles(8, 8, pe_hi, simd_hi)
    ii_lo = g.exp_cycles(8, 8, pe_lo, simd_lo)
    assert ii_lo >= ii_hi        # looser target -> more folding -> higher II


def test_target_from_knobs():
    assert g.target_from_knobs(1, reuse_factor=8)[0] == 8
    assert g.target_from_knobs(1, reuse_factor=1)[0] == 1
    # TargetCycles is a whole-frame budget -> per-vector divides by M
    t, src = g.target_from_knobs(4, target_cycles=40)
    assert t == 10 and src == "target_cycles"


# ── segment length (DSP58) ────────────────────────────────────────────────────

def test_segment_len():
    assert g.segment_len(6, 5.0, "DSP48E2") == 0            # non-DSP58 -> unused
    s = g.segment_len(9, 5.0, "DSP58")
    assert 1 <= s <= 3                                       # <= ceil(9/3)=3


# ── narrow weights ────────────────────────────────────────────────────────────

def test_narrow_weights():
    assert g.narrow_weights(None, 4) == 0
    assert g.narrow_weights([[-8, 7], [1, 2]], 4, signed=True) == 0   # uses min (-8)
    assert g.narrow_weights([[-7, 7], [1, 2]], 4, signed=True) == 1   # excludes -8


# ── the full plan ─────────────────────────────────────────────────────────────

def test_plan_matches_validated_spike():
    # Must reproduce temp_space/mvau-spike geometry (the cosim-PASS config).
    p = g.fold_plan(1, 4, 4, weight_precision="fixed<8,4>",
                    input_precision="fixed<8,4>", output_precision="fixed<16,6>",
                    part="xcvu13p")
    t = p["tile"]
    assert t["compute_core"] == "mvu_8sx8u_dsp48"
    assert (t["pe"], t["simd"], t["sf"], t["nf"]) == (4, 4, 1, 1)
    assert (t["weight_stream_width_ba"], t["input_stream_width_ba"],
            t["output_stream_width_ba"]) == (128, 32, 80)
    assert t["wmem"] == 1


def test_plan_n_tiling():
    p = g.fold_plan(1, 8, 8, weight_precision="fixed<8,4>",
                    input_precision="fixed<8,4>", output_precision="fixed<16,6>",
                    part="xcvu13p", n_tiles=2)
    assert p["n_tiles"] == 2 and p["n_tile"] == 4
    assert p["tile"]["mh"] == 4
    with pytest.raises(ValueError):
        g.fold_plan(1, 8, 8, part="xcvu13p", n_tiles=3)   # 3 does not divide 8


def test_plan_dsp48e1_narrow_required():
    # 4-bit on a 7-series part with unknown weights must reject (NARROW required).
    with pytest.raises(ValueError):
        g.fold_plan(1, 8, 8, weight_precision="fixed<4,2>",
                    input_precision="fixed<4,2>", part="xc7z020")
    # ...but ok when weights exclude the most-negative code.
    p = g.fold_plan(1, 8, 8, weight_precision="fixed<4,2>",
                    input_precision="fixed<4,2>", part="xc7z020",
                    weights=[[-7, 7, 1, 1, 1, 1, 1, 1]] * 8)
    assert p["tile"]["narrow_weights"] == 1


def test_latency_and_ii_deterministic():
    # II == SF (output-beat interval), verified in XSIM (temp_space/mvau-lat).
    assert g.output_ii(2) == 2 and g.output_ii(4) == 4
    # DSP58: SF + ceil(CHAINLEN/SEG) + 2  (SIMD=4 -> CHAINLEN=2, SEG=2 -> +1)
    assert g.latency_cycles("mvu_vvu_8sx9_dsp58", 2, 4, 2) == 5   # measured 5
    assert g.latency_cycles("mvu_vvu_8sx9_dsp58", 4, 4, 2) == 7   # measured 7
    # DSP48 cores: SF + 5 fixed core stages
    assert g.latency_cycles("mvu_8sx8u_dsp48", 2, 4, 0) == 7      # measured 7
    assert g.latency_cycles("mvu_8sx8u_dsp48", 4, 4, 0) == 9      # measured 9


def test_plan_carries_deterministic_perf():
    # fc on Versal: PE=16 SIMD=4 SF=2 DSP58 -> lat=5, II=2, DSP=32.
    t = g.fold_plan(1, 8, 16, weight_precision="fixed<8,4>", input_precision="fixed<8,4>",
                    output_precision="fixed<16,6>", part="xcve2802-vsvh1760-2MP-e-S")["tile"]
    assert t["latency_cycles"] == 5 and t["ii"] == 2
    assert t["dsp_estimate"] == t["pe"] * 2   # PE*ceil(SIMD/3)


def test_plan_reject_wide_precision():
    with pytest.raises(ValueError):
        g.fold_plan(1, 8, 8, weight_precision="fixed<16,6>",
                    input_precision="fixed<8,4>", part="xcvu13p")
