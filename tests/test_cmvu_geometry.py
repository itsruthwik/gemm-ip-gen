"""Unit tests for the cmvu target's geometry (fold legalization, slot map)."""
import sys
from pathlib import Path

import pytest

# Import package-qualified so this test never plants a bare `geometry` (or
# `golden`/`rtl`) in sys.modules -- the mvau tests import their own top-level
# `geometry`, and a shared name collides across targets.
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from targets.cmvu import geometry as g  # noqa: E402


# ── axis fold legalization ────────────────────────────────────────────────────

def test_fully_spatial():
    r = g.resolve_geometry(4, 8, 16, 1, 1)
    assert (r["k_spatial"], r["k_passes"]) == (2, 1)
    assert (r["n_spatial"], r["n_passes"]) == (2, 1)
    assert r["blocks"] == 4
    assert r["multipliers"] == 128
    assert r["slots_per_block"] == 1
    assert r["warnings"] == []


def test_temporal_k():
    r = g.resolve_geometry(4, 8, 8, 2, 1)
    assert (r["k_spatial"], r["k_passes"]) == (1, 2)
    assert (r["n_spatial"], r["n_passes"]) == (1, 1)
    assert r["blocks"] == 1
    assert r["slots_per_block"] == 2
    assert r["first_out_latency"] == g.L


def test_mixed_fold():
    r = g.resolve_geometry(1, 16, 16, 2, 2)
    assert (r["k_spatial"], r["k_passes"]) == (2, 2)
    assert (r["n_spatial"], r["n_passes"]) == (1, 2)
    assert r["blocks"] == 2
    assert r["slots_per_block"] == 4
    assert r["first_out_latency"] == g.L + 1  # + (k_spatial-1) cascade hops


def test_slot_budget_at_cap():
    r = g.resolve_geometry(1, 16, 16, 4, 2)
    assert (r["k_passes"], r["n_passes"]) == (4, 2)
    assert r["slots_per_block"] == 8
    assert r["blocks"] == 1


def test_spatial_latency_law():
    r = g.resolve_geometry(1, 16, 16, 1, 1)
    assert (r["k_spatial"], r["n_spatial"]) == (4, 2)
    assert r["first_out_latency"] == g.L + 1 + 3


def test_non_divisible_fold_warns_and_legalizes():
    # K=19 -> 5 K tiles; KFold=4 -> spatial=2, passes=3 (not 4).
    r = g.resolve_geometry(1, 19, 8, 4, 1)
    assert (r["k_spatial"], r["k_passes"]) == (2, 3)
    assert r["kfold_requested"] == 4
    assert r["kfold_effective"] == 3
    assert len(r["warnings"]) == 1
    assert "KFold=4" in r["warnings"][0] and "3" in r["warnings"][0]


# ── knob validation ───────────────────────────────────────────────────────────

def test_missing_knob_is_error():
    with pytest.raises(ValueError, match="required"):
        g.resolve_geometry(1, 8, 8, None, 1)
    with pytest.raises(ValueError, match="required"):
        g.resolve_geometry(1, 8, 8, 1, None)


@pytest.mark.parametrize("bad", [0, -1, True, 2])
def test_invalid_kfold_is_error(bad):
    # K=8 -> 2 chunks, so 2 is legal; use K=4 for the range check instead.
    with pytest.raises(ValueError):
        g.resolve_geometry(1, 4, 8, bad, 1)


def test_out_of_range_fold_is_error():
    with pytest.raises(ValueError, match=r"1\.\.4"):
        g.resolve_geometry(1, 16, 8, 5, 1)


def test_invalid_m_is_error():
    with pytest.raises(ValueError, match="m must be"):
        g.resolve_geometry(0, 8, 8, 1, 1)


# ── oversubscription ──────────────────────────────────────────────────────────

def test_oversubscribed_slots_is_error():
    with pytest.raises(ValueError) as ei:
        g.resolve_geometry(1, 32, 32, 8, 4)
    msg = str(ei.value)
    assert "32 resident weight tiles" in msg
    assert "only 8" in msg
    assert "lower KFold/NFold" in msg


# ── slot map / schedule ───────────────────────────────────────────────────────

def test_slot_sequence_is_n_outer_k_inner():
    assert g.slot_sequence(2, 2) == ((0, 0), (1, 0), (0, 1), (1, 1))
    assert g.slot_of(0, 0, 2) == 0
    assert g.slot_of(1, 0, 2) == 1
    assert g.slot_of(0, 1, 2) == 2
    assert g.slot_of(1, 1, 2) == 3


def test_tile_index_maps():
    # pass 1 of a 2-column cascade covers K tiles 2 and 3
    assert g.ktile_of(0, 0, 2) == 0
    assert g.ktile_of(1, 1, 2) == 3
    assert g.ntile_of(0, 1, 2) == 1
    assert g.ntile_of(1, 0, 2) == 2


def test_slot_sequence_covers_every_pass():
    seq = g.slot_sequence(3, 2)
    assert len(seq) == 6
    assert set(seq) == {(kp, np) for kp in range(2) for np in range(3)}


# ── tails and widths ──────────────────────────────────────────────────────────

def test_tail_lanes():
    assert tuple(g.k_tail_lanes(10, i) for i in range(3)) == (4, 4, 2)
    assert tuple(g.k_tail_lanes(8, i) for i in range(2)) == (4, 4)
    assert tuple(g.n_tail_lanes(10, i) for i in range(2)) == (8, 2)
    assert g.k_tail_lanes(4, 5) == 0
    assert g.n_tail_lanes(8, 3) == 0


def test_stream_widths():
    r = g.resolve_geometry(1, 16, 16, 1, 1)
    assert r["a_row_bits"] == 4 * g.K_PHYS * g.IN_WIDTH      # 128
    assert r["res_row_bits"] == 2 * g.N_PHYS * g.RESULT_WIDTH  # 256
    assert r["weight_tiles"] == 8
    assert r["weight_tile_bits"] == g.K_PHYS * g.N_PHYS * g.COEF_WIDTH


def test_result_width_default_and_effective():
    # default W is the physical RESULT_WIDTH (16); out_w = W-1
    r = g.resolve_geometry(1, 16, 16, 1, 1)
    assert r["result_width"] == g.RESULT_WIDTH == 16
    assert r["out_w"] == 15
    # a narrower effective width shrinks the emitted res_row only
    r8 = g.resolve_geometry(1, 16, 16, 1, 1, result_width=8)
    assert r8["result_width"] == 8 and r8["out_w"] == 7
    assert r8["res_row_bits"] == 2 * g.N_PHYS * 8
    assert r8["a_row_bits"] == r["a_row_bits"]


@pytest.mark.parametrize("bad", [0, -1, 17, 32])
def test_result_width_out_of_range_is_error(bad):
    with pytest.raises(ValueError, match="result_width"):
        g.resolve_geometry(1, 16, 16, 1, 1, result_width=bad)
