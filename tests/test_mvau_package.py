"""Codegen tests for the mvau target: package assembly, shim, JSON, C twin/TB.

These are tool-free (no Vitis). The end-to-end cosim of the generated packages is
validated separately (temp_space/mvau-gen; jojo-track status): fully-unrolled,
NF-folded, and SF/NF-folded shapes all cosim PASS on 2025.2.
"""
import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from gemm_ip.registry import load_target  # noqa: E402

_CFG = dict(weight_precision="fixed<8,4>", input_precision="fixed<8,4>",
            output_precision="fixed<16,6>", part="xcvu13p-flga2577-2-e",
            clock_period_ns=5)


def _gen(tmp_path, shape, name, **extra):
    t = load_target("mvau")
    cfg = dict(_CFG, name=name, output_dir=str(tmp_path), **extra)
    pkg = Path(t.package(shape, cfg))
    t.verify(pkg)   # raises if incomplete
    return pkg


def test_package_files_and_verify(tmp_path):
    pkg = _gen(tmp_path, (4, 4, 4), "gemm_4x4x4")
    for f in ["gemm_4x4x4_core.v", "gemm_4x4x4_core.cpp", "gemm_4x4x4_top.cpp",
              "gemm_4x4x4.json", "gemm_4x4x4_tb.cpp", "gemm_4x4x4_gemm_ip.h",
              "run_vitis.tcl"]:
        assert (pkg / f).is_file() and (pkg / f).stat().st_size > 0
    for s in ["mvu_vvu_axi.sv", "replay_buffer.sv", "mvu_8sx8u_dsp48.sv"]:
        assert (pkg / "rtl_static" / s).is_file()


def test_blackbox_json_contract(tmp_path):
    pkg = _gen(tmp_path, (4, 4, 4), "gemm_4x4x4")
    j = json.loads((pkg / "gemm_4x4x4.json").read_text())
    # module name must equal the C function name (Vitis cosim gotcha)
    assert j["c_function_name"] == j["rtl_top_module_name"] == "gemm_4x4x4_core"
    rcs = j["rtl_common_signal"]
    # CE mandatory + all five ap_ctrl keys present (2025.2 requirements)
    assert rcs["module_clock_enable"] == "ap_ce"
    for k in ("idle", "start", "ready", "done", "continue"):
        assert f"ap_ctrl_chain_protocol_{k}" in rcs
    # FIFO port map, not AXIS. Weight-stationary: weights are baked in the memstream,
    # so there is no weight port -- only the activation input and the result output.
    pnames = {p["c_name"]: p["rtl_ports"] for p in j["c_parameters"]}
    assert "w" not in pnames
    assert pnames["a"]["FIFO_data_read_in"] == "a_dout"
    assert pnames["p"]["FIFO_data_write_out"] == "p_din"


def test_shim_module_name_matches_c_function(tmp_path):
    pkg = _gen(tmp_path, (4, 4, 4), "gemm_4x4x4")
    v = (pkg / "gemm_4x4x4_core.v").read_text()
    assert "module gemm_4x4x4_core (" in v
    assert 'COMPUTE_CORE("mvu_8sx8u_dsp48")' in v
    assert "ap_ce" in v and "~ap_rst" in v   # active-high reset + CE stall


def test_folded_top_beat_counts(tmp_path):
    # (3,16,8) rf=16 folds to PE=4 SIMD=4 SF=4 NF=2, M=3.
    pkg = _gen(tmp_path, (3, 16, 8), "gemm_3x16x8_f", reuse_factor=16)
    top = (pkg / "gemm_3x16x8_f_top.cpp").read_text()
    # weight-stationary: weights baked in the memstream (no feed_w). acts M*SF = 3*4 = 12 ;
    # requant loops M=3 vectors.
    assert "i < 12;" in top and "vec < 3;" in top
    assert "feed_w" not in top and "i < 24;" not in top


def test_weight_stationary_package(tmp_path):
    pkg = _gen(tmp_path, (4, 4, 4), "gemm_4x4x4")
    # the packed memstream init lives alongside the vendored RTL
    dat = pkg / "rtl_static" / "gemm_4x4x4_weights.dat"
    assert dat.is_file() and dat.stat().st_size > 0
    assert (pkg / "rtl_static" / "memstream.sv").is_file()
    v = (pkg / "gemm_4x4x4_core.v").read_text()
    assert "memstream #(" in v and "$readmemh" not in v      # instantiated, path via INIT_FILE
    assert f'.INIT_FILE("{dat.resolve()}")' in v             # absolute init path
    assert "w_dout" not in v                                 # no external weight port
    # C twin + TB bake the weights (no w stream)
    twin = (pkg / "gemm_4x4x4_core.cpp").read_text()
    assert "gemm_4x4x4_core_W[4][4]" in twin
    tb = (pkg / "gemm_4x4x4_tb.cpp").read_text()
    assert "static const long W[4][4]" in tb and "w_in" not in tb


def test_baked_weights_match_packer(tmp_path):
    # a real weight_matrix ([K][N]) must be packed byte-identically by the package
    import numpy as np
    sys.path.insert(0, str(_SRC / "targets" / "mvau"))
    import weightpack as wp
    K, N = 4, 4
    B = [[((k * 3 + n) % 7) - 3 for n in range(N)] for k in range(K)]   # [K][N]
    pkg = _gen(tmp_path, (1, K, N), "gemm_wm", weight_matrix=np.asarray(B))
    got = (pkg / "rtl_static" / "gemm_wm_weights.dat").read_text()
    # PE=SIMD=4 single tile at this shape; word = byte-aligned weight-stream width
    exp = wp.pack_memstream_hex(B, N, K, 4, 4, 8, word_bits=128)
    assert got == exp


def test_affine_drain_present(tmp_path):
    pkg = _gen(tmp_path, (3, 16, 8), "gemm_3x16x8_f", reuse_factor=16)
    top = (pkg / "gemm_3x16x8_f_top.cpp").read_text()
    # the drain: ap_fixed output with round-half-up + saturate, and a requant stage
    assert "ap_fixed<16, 6, AP_RND, AP_SAT>" in top
    assert "static void requant(" in top and "rescale + round + saturate" in top


def test_reject_wide_precision(tmp_path):
    with pytest.raises(ValueError):
        _gen(tmp_path, (2, 8, 8), "bad", weight_precision="fixed<16,6>")


def test_bias_scaled_and_added(tmp_path):
    # per-column bias, scaled to the accumulator (2^(fa+fb)=2^8) domain and added.
    bias = [0.5, -0.25, 0.0, 0.25, 1.0, -1.0, 0.75, -0.5]
    pkg = _gen(tmp_path, (2, 8, 8), "gb", reuse_factor=1, bias=bias)
    top = (pkg / "gb_top.cpp").read_text()
    assert "static const long gb_bias[8] = {128, -64, 0, 64, 256, -256, 192, -128}" in top
    assert "+ gb_bias[oc]" in top


def test_no_bias_omits_array(tmp_path):
    pkg = _gen(tmp_path, (2, 8, 8), "gnb")           # no bias
    top = (pkg / "gnb_top.cpp").read_text()
    assert "gnb_bias" not in top
    assert "(ap_int<64>)raw;" in top                 # bias-free add path
