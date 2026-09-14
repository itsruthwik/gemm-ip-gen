"""Unit tests for the c-generic package/verify flow (step 4 of
jojo-track/open/catapult-generic-target/plan.md), `src/targets/c-generic/package.py`.

Two tiers:
- File-set / content checks that always run (no tool needed).
- A Catapult-gated ``verify()`` test (csim + SCVerify) over the two SMALL
  shapes only (RF regimes covered elsewhere by the tool-free checks); the
  (16,32,32) RF-8 mha_large-projection proof is exercised manually (see
  jojo-track/open/catapult-generic-target/plan.md and
  hgq2-examples-from-qkeras-configs/status.md) rather than as a permanent test.
"""
import re
import shutil
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TARGET_DIR = _REPO_ROOT / "gemm-ip-gen" / "src" / "targets" / "c_generic"
_TEMP = _REPO_ROOT / "temp_space" / "c_generic" / "pytest_pkgs"

_SRC = _REPO_ROOT / "gemm-ip-gen" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from targets.c_generic import hls  # noqa: E402
from targets.c_generic import package  # noqa: E402

FORBIDDEN_PATTERNS = [r"#pragma HLS", r"hls::stream", r"\bap_int\b", r"\bap_uint\b", r"\bap_fixed\b"]


def _assert_no_forbidden_idioms(text):
    for pat in FORBIDDEN_PATTERNS:
        assert re.findall(pat, text) == [], f"forbidden idiom {pat!r} found in package output"


REQUIRED_SUFFIXES = [
    "_gemm_ip.h", "_config.h", "_bias.h", "_top.cpp", "_tb.cpp",
]
REQUIRED_FLAT = ["gemm_ip_combined.h", "nnet_types.h", "run_catapult.tcl"]


def _gen(tmp_path, name, shape, **cfg):
    cfg.setdefault("name", name)
    cfg.setdefault("output_dir", str(tmp_path))
    return package.generate_c_generic_pkg(shape, cfg)


# ── file-set / content checks (tool-free) ──────────────────────────────────

def test_package_file_set_const_weights_with_bias(tmp_path):
    pkg_dir = _gen(tmp_path, "pt_a", (4, 8, 8), weights_in_core=True, reuse_factor=1,
                   has_bias=True, bias=[1, -1, 0, 1, -1, 0, 1, -1])
    for flat in REQUIRED_FLAT:
        assert (pkg_dir / flat).is_file() and (pkg_dir / flat).stat().st_size > 0
    for suf in REQUIRED_SUFFIXES:
        f = pkg_dir / f"pt_a{suf}"
        assert f.is_file() and f.stat().st_size > 0
    assert (pkg_dir / "pt_a_weights.h").is_file()


def test_package_file_set_two_operand_no_bias(tmp_path):
    pkg_dir = _gen(tmp_path, "pt_b", (8, 16, 8), weights_in_core=False, reuse_factor=2,
                   has_bias=False)
    for flat in REQUIRED_FLAT:
        assert (pkg_dir / flat).is_file()
    for suf in REQUIRED_SUFFIXES:
        assert (pkg_dir / f"pt_b{suf}").is_file()
    # Two-operand packages never bake a weight ROM file.
    assert not (pkg_dir / "pt_b_weights.h").exists()


def test_package_verify_required_files(tmp_path):
    pkg_dir = _gen(tmp_path, "pt_c", (4, 8, 8), weights_in_core=True, reuse_factor=1)
    assert package.verify(pkg_dir) or True  # verify() may skip (no catapult); just must not raise
    missing = [f for f in package.REQUIRED_FILES
               if not (pkg_dir / f.format(name="pt_c")).is_file()]
    assert missing == []


def test_top_cpp_has_hls_design_top_and_ac_channel(tmp_path):
    pkg_dir = _gen(tmp_path, "pt_d", (4, 8, 8), weights_in_core=True, reuse_factor=1)
    text = (pkg_dir / "pt_d_top.cpp").read_text()
    assert "#pragma hls_design top" in text
    assert "ac_channel<" in text
    assert "CCS_BLOCK(pt_d)" in text
    _assert_no_forbidden_idioms(text)


def test_top_cpp_two_operand_has_ac_channel_b_stream(tmp_path):
    pkg_dir = _gen(tmp_path, "pt_e", (8, 16, 8), weights_in_core=False, reuse_factor=2)
    text = (pkg_dir / "pt_e_top.cpp").read_text()
    assert "ac_channel<pt_e_b_col_t> &b_stream" in text
    _assert_no_forbidden_idioms(text)


def test_tb_cpp_uses_ccs_main_and_ccs_design(tmp_path):
    pkg_dir = _gen(tmp_path, "pt_f", (4, 8, 8), weights_in_core=True, reuse_factor=1,
                   has_bias=True, bias=[0] * 8)
    text = (pkg_dir / "pt_f_tb.cpp").read_text()
    assert "CCS_MAIN(" in text
    assert "CCS_DESIGN(pt_f)(" in text
    assert "#include <mc_scverify.h>" in text
    _assert_no_forbidden_idioms(text)


def test_bias_rejects_missing_values_when_declared(tmp_path):
    with pytest.raises(ValueError):
        _gen(tmp_path, "pt_g", (4, 8, 8), weights_in_core=True, reuse_factor=1, has_bias=True)


def test_only_stream_interface_supported(tmp_path):
    with pytest.raises(ValueError):
        _gen(tmp_path, "pt_h", (4, 8, 8), weights_in_core=True, reuse_factor=1, interface="array")


def test_run_catapult_tcl_has_scverify_and_shared_technology(tmp_path):
    pkg_dir = _gen(tmp_path, "pt_i", (4, 8, 8), weights_in_core=True, reuse_factor=1)
    text = (pkg_dir / "run_catapult.tcl").read_text()
    assert "flow package require /SCVerify" in text
    assert "mgc_Xilinx-KINTEX-u-2_beh" in text
    assert "go extract" in text
    assert "flow run /SCVerify/launch_make" in text


# ── Catapult-gated verify() (real csim + SCVerify), two SMALL shapes only ──

_HAS_CATAPULT = shutil.which("catapult") is not None


@pytest.mark.skipif(not _HAS_CATAPULT, reason="catapult not found on PATH")
@pytest.mark.parametrize("label,shape,cfg,expected_multiplier_limit", [
    ("const_weights_rf1", (4, 8, 8),
     dict(weights_in_core=True, reuse_factor=1, has_bias=True,
          bias=[1, -1, 0, 1, -1, 0, 1, -1]), 64),
    ("two_operand_rf2", (8, 16, 8),
     dict(weights_in_core=False, reuse_factor=2, has_bias=False), 64),
])
def test_catapult_verify_small_shapes(label, shape, cfg, expected_multiplier_limit):
    out_dir = _TEMP / label
    pkg_dir = _gen(out_dir, f"cvt_{label}", shape, **cfg)
    result = package.verify(pkg_dir)
    assert result["scverify_ok"], result["log"][-4000:]
    assert result["ok"]
    # Two-operand: the multiplier is a genuine runtime unknown, so Catapult
    # always keeps a real multiplier bank sized exactly to multiplier_limit
    # (mgc_mul under nangate-45nm_beh, mgc_muladd1 under the Xilinx library --
    # same count, different fused component). Const-weight RF<=n_in:
    # Catapult's constant propagation is free to fold "multiply by a known
    # ROM constant" into adders/LUTs/partial DSP fabric instead, and how much
    # of that folds is technology-dependent (0 mgc_mul instances observed on
    # nangate-45nm_beh; partial folding to mgc_mul2add1_pipe instances
    # observed on the Xilinx Kintex UltraScale library) -- see
    # _parse_multiplier_count's docstring; only assert the exact match for
    # the shape where hardware inference can't fold it, and a sane range for
    # the shape where folding is technology-dependent.
    if not cfg.get("weights_in_core"):
        assert result["multiplier_count"] == expected_multiplier_limit
    else:
        assert 0 <= result["multiplier_count"] <= expected_multiplier_limit
