"""Unit tests for the cmvu Target wiring (flow.py, registry, CLI, package)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from targets.cmvu.flow import CmvuTarget  # noqa: E402
from targets.cmvu import geometry as g  # noqa: E402

T = CmvuTarget()


def _cfg(**over):
    cfg = {
        "name": "layer_x",
        "output_dir": "/tmp/does-not-matter",
        "kfold": 1,
        "nfold": 1,
        "interface": "stream",
        "input_precision": "fixed<8,4>",
        "weight_precision": "fixed<8,4>",
        "output_precision": "fixed<8,4>",
    }
    cfg.update(over)
    return cfg


# ── registry / target identity ────────────────────────────────────────────────

def test_registry_resolves_cmvu_to_catapult():
    from gemm_ip.registry import load_target, supported_tools, TARGETS
    assert "cmvu" in TARGETS
    assert supported_tools("cmvu") == ("catapult",)
    t = load_target("cmvu")
    assert t.name == "cmvu"
    assert t.tool == "catapult"


def test_knobs_are_kfold_nfold_only():
    assert {k["key"] for k in T.knobs} == {"kfold", "nfold"}


def test_weight_layouts_accept_both():
    assert set(T.weight_layouts) == {"column_major", "row_major"}


# ── normalize_config ──────────────────────────────────────────────────────────

def test_normalize_list_reports_geometry():
    items = T.normalize_config([
        {"name": "a", "m": 4, "k": 16, "n": 16, "kfold": 2, "nfold": 2},
    ])
    it = items[0]
    assert (it["kfold"], it["nfold"]) == (2, 2)
    assert it["k_spatial"] == 2 and it["k_passes"] == 2
    assert it["n_spatial"] == 1 and it["n_passes"] == 2
    assert it["slots_per_block"] == 4


def test_normalize_named_dict_reattaches_knobs():
    items = T.normalize_config({
        "a": {"gemm_m": 4, "gemm_k": 8, "gemm_n": 8, "n_in": 8, "n_out": 8,
              "kfold": 2, "nfold": 1},
    })
    it = items[0]
    assert it["name"] == "a"
    assert (it["kfold"], it["nfold"]) == (2, 1)
    assert it["k_passes"] == 2


@pytest.mark.parametrize("legacy", [
    {"fold_axis": "k"}, {"m_reuse_factor": 2},
    {"k_reuse_factor": 2}, {"n_reuse_factor": 2},
])
def test_normalize_rejects_tensor_slice_knobs(legacy):
    cfg = [dict({"name": "a", "m": 4, "k": 8, "n": 8, "kfold": 1,
                 "nfold": 1}, **legacy)]
    with pytest.raises(ValueError, match="layer 'a'.*tensor_slice knob"):
        T.normalize_config(cfg)


@pytest.mark.parametrize("rf", [1, 4])
def test_normalize_ignores_reuse_factor(rf, capsys):
    # the hls4ml manifest carries hls4ml's ReuseFactor on every GEMM layer
    cfg = [{"name": "a", "m": 4, "k": 8, "n": 8, "kfold": 2, "nfold": 1,
            "reuse_factor": rf}]
    it = T.normalize_config(cfg)[0]
    assert (it["kfold"], it["nfold"]) == (2, 1)
    warned = "cmvu ignores reuse_factor" in capsys.readouterr().err
    assert warned == (rf != 1)


@pytest.mark.parametrize("missing", [
    {"kfold": None}, {"nfold": None}, {"kfold": None, "nfold": None},
])
def test_normalize_requires_both_folds(missing):
    cfg = [dict({"name": "a", "m": 4, "k": 8, "n": 8, "kfold": 1,
                 "nfold": 1}, **missing)]
    with pytest.raises(ValueError, match="requires both KFold and NFold"):
        T.normalize_config(cfg)


def test_normalize_rejects_array_interface():
    cfg = [{"name": "a", "m": 4, "k": 8, "n": 8, "kfold": 1, "nfold": 1,
            "interface": "array"}]
    with pytest.raises(ValueError, match="interface='stream' only"):
        T.normalize_config(cfg)


def test_normalize_rejects_oversubscribed_slots():
    # K=32 -> 8 chunks; kfold=1 -> spatial=8, passes=1; N=64 -> 8 chunks;
    # nfold=1 -> spatial=8, passes=1 => 1*1 slots (fine). Use temporal folds
    # that multiply past 8: K=64 (16 chunks) kfold=16 -> passes 16 -> >8.
    cfg = [{"name": "a", "m": 4, "k": 64, "n": 8, "kfold": 16, "nfold": 1}]
    with pytest.raises(ValueError, match="resident weight"):
        T.normalize_config(cfg)


# ── package() validation ──────────────────────────────────────────────────────

def test_package_rejects_array_interface(tmp_path):
    with pytest.raises(ValueError, match="interface='stream' only"):
        T.package((4, 4, 8), _cfg(interface="array", output_dir=str(tmp_path)))


def test_package_requires_folds(tmp_path):
    with pytest.raises(ValueError, match="requires both KFold and NFold"):
        T.package((4, 4, 8), _cfg(kfold=None, output_dir=str(tmp_path)))


def test_package_accepts_trn_output(tmp_path):
    # TRN is now accepted (no rounding constant folded into the bias)
    pkg = T.package((4, 4, 8),
                    _cfg(name="trn", output_precision="fixed<8,0,TRN,WRAP,0>",
                         output_dir=str(tmp_path)))
    assert (pkg / "trn_core.sv").is_file()


def test_package_rejects_bad_round_mode(tmp_path):
    with pytest.raises(ValueError, match="RND or TRN"):
        T.package((4, 4, 8),
                  _cfg(output_precision="fixed<8,0,RND_CONV,WRAP,0>",
                       output_dir=str(tmp_path)))


@pytest.mark.parametrize("ovf", ["SAT", "SAT_SYM"])
def test_package_rejects_saturating_output(tmp_path, ovf):
    with pytest.raises(ValueError, match="SAT"):
        T.package((4, 4, 8),
                  _cfg(output_precision=f"fixed<8,0,RND,{ovf},0>",
                       output_dir=str(tmp_path)))


def test_package_warns_on_narrow_accum_precision(tmp_path, capsys):
    # cmvu accumulates exact products in int32; a narrower accum only warns
    T.package((4, 4, 8),
              _cfg(accum_precision="fixed<8,4>",
                   input_precision="fixed<8,4>",
                   weight_precision="fixed<8,4>",
                   output_dir=str(tmp_path)))
    assert "narrower than the product frac" in capsys.readouterr().err


def test_package_rejects_inexact_bias(tmp_path):
    with pytest.raises(ValueError, match="not exactly representable"):
        T.package((4, 4, 8),
                  _cfg(name="bad_bias", has_bias=True, bias=[0.03],
                       output_dir=str(tmp_path)))


def test_package_rejects_shift_out_of_range(tmp_path):
    # frac_a + frac_b - frac_out = 20 + 20 - 0 = 40 > 31
    with pytest.raises(ValueError, match="outside 0..31"):
        T.package((4, 4, 8),
                  _cfg(input_precision="fixed<24,4>",
                       weight_precision="fixed<24,4>",
                       output_precision="fixed<8,0>",
                       output_dir=str(tmp_path)))


def test_package_rejects_non_int8_operand(tmp_path):
    with pytest.raises(ValueError, match="int8"):
        T.package((4, 4, 8),
                  _cfg(weight_precision="fixed<16,4>",
                       output_dir=str(tmp_path)))


def test_package_accepts_narrow_operand(tmp_path):
    # a 7-bit signed weight code fits (sign-extends into) the int8 lane
    pkg = T.package((4, 4, 8),
                    _cfg(name="layer_n", weight_precision="fixed<7,0>",
                         output_dir=str(tmp_path)))
    assert (pkg / "layer_n_core.sv").is_file()


def test_normalize_rejects_cmvu_blocked_layer():
    cfg = [{"name": "b", "m": 4, "k": 8, "n": 8, "kfold": 1, "nfold": 1,
            "cmvu_blocked": True, "cmvu_block_reason": "output 15-bit > int8"}]
    with pytest.raises(ValueError, match="not representable"):
        T.normalize_config(cfg)


# ── packaging + verify ────────────────────────────────────────────────────────

def test_package_and_verify(tmp_path):
    pkg = T.package((4, 16, 16), _cfg(name="layer_y", kfold=2, nfold=2,
                                      output_dir=str(tmp_path)))
    # finalize is a batch step; call it as the CLI does
    T.finalize([], str(tmp_path))
    assert T.verify(pkg) is True
    assert (pkg / "layer_y_core.sv").is_file()
    assert (tmp_path / "cmvu_mode1.sv").is_file()


def test_verify_rejects_missing(tmp_path):
    pkg = T.package((4, 4, 8), _cfg(name="layer_z", output_dir=str(tmp_path)))
    (pkg / "layer_z_tb.cpp").unlink()
    with pytest.raises(RuntimeError, match="missing/empty"):
        T.verify(pkg)


# ── CLI surface ───────────────────────────────────────────────────────────────

def test_cli_list_targets_and_describe(capsys):
    from gemm_ip import cli
    argv = sys.argv
    try:
        sys.argv = ["gemm_ip", "--list-targets"]
        cli.main()
        out = json.loads(capsys.readouterr().out)
        assert "cmvu" in out["targets"]
        sys.argv = ["gemm_ip", "--describe", "cmvu"]
        cli.main()
        desc = json.loads(capsys.readouterr().out)
        assert desc["name"] == "cmvu" and desc["tool"] == "catapult"
    finally:
        sys.argv = argv


# ── runtime-B (two-operand) packaging ─────────────────────────────────────────

def test_package_runtime_b_two_stream(tmp_path):
    pkg = T.package((4, 8, 8),
                    _cfg(name="rb", kfold=1, nfold=1, weights_in_core=False,
                         output_dir=str(tmp_path)))
    core = (pkg / "rb_core.sv").read_text()
    assert "b_beat" in core and "b_valid" in core
    # no baked weight-tile init (force/release) in runtime-B mode
    assert "force " not in core
    inst = (pkg / "rb_inst.cpp").read_text()
    assert "b_stream" in inst
    hdr = (pkg / "rb_gemm_ip.h").read_text()
    assert "b_beat" in hdr and "b_stream" in hdr
    tcl = (pkg / "run_catapult.tcl").read_text()
    assert "b_stream:rsc" in tcl


def test_package_const_weights_still_default(tmp_path):
    pkg = T.package((4, 8, 8), _cfg(name="cw", kfold=1, nfold=1,
                                    output_dir=str(tmp_path)))
    core = (pkg / "cw_core.sv").read_text()
    assert "b_beat" not in core
    assert "256'h" in core


# ── effective output width ─────────────────────────────────────────────────────

def test_output_width_derivation():
    from targets.cmvu.flow import _output_width
    # unset/unparseable -> physical max
    assert _output_width(None) == 16
    assert _output_width("garbage") == 16
    assert _output_width("fixed<16,6,TRN,WRAP,0>") == 16
    assert _output_width("fixed<8,2>") == 8
    assert _output_width("ac_int<12,true>") == 12


def test_output_width_over_cap_is_error():
    from targets.cmvu.flow import _output_width
    with pytest.raises(ValueError, match="result width"):
        _output_width("fixed<24,8>")


@pytest.mark.parametrize("prec,w", [("fixed<16,8>", 16), ("fixed<8,2>", 8),
                                    ("ac_int<12,true>", 12)])
def test_package_threads_output_width(tmp_path, prec, w):
    pkg = T.package((4, 8, 8), _cfg(name=f"ow{w}", kfold=1, nfold=1,
                                    output_precision=prec,
                                    output_dir=str(tmp_path)))
    inst = (pkg / f"ow{w}_inst.cpp").read_text()
    hdr = (pkg / f"ow{w}_gemm_ip.h").read_text()
    assert f"ac_int<{w}, true>, 8> res_t" in inst
    assert f"slc<{w}>(j * {w})" in hdr


# ── unsupported-layer errors: fail at package/normalize time, name the layer ──

# Each entry: (kwargs override for _cfg, extra shape override or None).
# Every case must raise ValueError (never NotImplementedError) whose message
# contains the layer name -- checked before any RTL is generated.
_UNSUPPORTED_CASES = [
    ("array_interface", dict(interface="array"), None),
    ("output_sat", dict(output_precision="fixed<8,4,RND,SAT,0>"), None),
    ("output_sat_sym", dict(output_precision="fixed<8,4,RND,SAT_SYM,0>"), None),
    ("output_bad_round", dict(output_precision="fixed<8,4,RND_CONV,WRAP,0>"), None),
    ("output_width_over_cap", dict(output_precision="fixed<20,12>",
                                   input_precision="fixed<8,0>",
                                   weight_precision="fixed<8,0>"), None),
    ("shift_out_of_range", dict(input_precision="fixed<24,4>",
                                weight_precision="fixed<24,4>",
                                output_precision="fixed<8,0>"), None),
    ("inexact_bias", dict(has_bias=True, bias=[0.03]), None),
    ("kfold_illegal", dict(kfold=100, input_precision="fixed<8,0>",
                           weight_precision="fixed<8,0>",
                           output_precision="fixed<8,0>"), (4, 32, 64)),
    ("slot_budget_exceeded", dict(kfold=8, nfold=8,
                                  input_precision="fixed<8,0>",
                                  weight_precision="fixed<8,0>",
                                  output_precision="fixed<8,0>"), (4, 64, 64)),
]


@pytest.mark.parametrize("case_id,overrides,shape", _UNSUPPORTED_CASES,
                        ids=[c[0] for c in _UNSUPPORTED_CASES])
def test_unsupported_layer_errors_name_the_layer(tmp_path, case_id, overrides,
                                                 shape):
    name = f"unsup_{case_id}"
    cfg = _cfg(name=name, output_dir=str(tmp_path), **overrides)
    shape = shape or (4, 4, 8)
    with pytest.raises(ValueError) as excinfo:
        T.package(shape, cfg)
    assert not isinstance(excinfo.value, NotImplementedError)
    assert name in str(excinfo.value)


def test_col_major_runtime_b_packages_with_more_than_two_k_passes(tmp_path):
    # Passes beyond the live pair are staged in the wrapper's tile store, so
    # deep K folding is legal for column-major runtime-B (K_PASSES=8 here).
    cfg = _cfg(name="deep_k", output_dir=str(tmp_path), kfold=8, nfold=1,
               weights_in_core=False, weight_layout="column_major",
               input_precision="fixed<8,0>", weight_precision="fixed<8,0>",
               output_precision="fixed<8,0>")
    T.package((4, 64, 8), cfg)
    core = next(tmp_path.rglob("deep_k_core.sv")).read_text()
    assert "K_PASSES  = 8" in core and "stg_c0_j5" in core


def test_vitis_tool_error_for_cmvu_is_clear():
    from gemm_ip.registry import load_target
    with pytest.raises(ValueError) as excinfo:
        load_target("cmvu", "vitis")
    msg = str(excinfo.value)
    assert "cmvu" in msg and "vitis" in msg and "catapult" in msg
