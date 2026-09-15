"""Codegen tests for the mvau target: package assembly, shim, JSON, C twin/TB.

These are tool-free (no Vitis). Simulation (csim/cosim) is validated separately
(temp_space; jojo-track status).
"""
import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from gemm_ip.registry import load_target  # noqa: E402

VERSAL = "xcvc1902-vsva2197-2MP-e-S"

_CFG = dict(weight_precision="fixed<8,4>", input_precision="fixed<8,4>",
            output_precision="fixed<16,6>", part=VERSAL,
            clock_period_ns=5)


def _gen(tmp_path, shape, name, **extra):
    t = load_target("mvau")
    cfg = dict(_CFG, name=name, output_dir=str(tmp_path), **extra)
    pkg = Path(t.package(shape, cfg))
    t.verify(pkg)   # raises if incomplete
    return pkg


def test_package_files_and_verify(tmp_path):
    pkg = _gen(tmp_path, (4, 4, 4), "gemm_4x4x4", reuse_factor=1)
    for f in ["gemm_4x4x4_core.v", "gemm_4x4x4_core.cpp", "gemm_4x4x4_top.cpp",
              "gemm_4x4x4.json", "gemm_4x4x4_tb.cpp", "gemm_4x4x4_gemm_ip.h",
              "run_vitis.tcl"]:
        assert (pkg / f).is_file() and (pkg / f).stat().st_size > 0
    for s in ["mvu_vvu_axi.sv", "replay_buffer.sv", "mvu_vvu_8sx9_dsp58.sv"]:
        assert (pkg / "rtl_static" / s).is_file()


def test_blackbox_json_contract(tmp_path):
    pkg = _gen(tmp_path, (4, 4, 4), "gemm_4x4x4", reuse_factor=1)
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
    assert "not synthesis-accurate" in j["_comment"]
    pnames = {p["c_name"]: p["rtl_ports"] for p in j["c_parameters"]}
    assert "w" not in pnames
    assert pnames["a"]["FIFO_data_read_in"] == "a_dout"
    assert pnames["p"]["FIFO_data_write_out"] == "p_din"


def test_shim_module_name_matches_c_function(tmp_path):
    pkg = _gen(tmp_path, (4, 4, 4), "gemm_4x4x4", reuse_factor=1)
    v = (pkg / "gemm_4x4x4_core.v").read_text()
    assert "module gemm_4x4x4_core (" in v
    assert ".VERSION(3)" in v
    assert "ap_ce" in v and "~ap_rst" in v   # active-high reset + CE stall


def test_folded_top_beat_counts(tmp_path):
    # (3,16,8) rf=16, fold-k: SIMD floor=ceil(16/3)=6 -> RF legalizes to 6, PE=8,
    # SIMD=3, SF=6 (k_pad=18). The K-padding + SF fan-out now live in the RTL
    # wrapper (_generate_ws_shim), not this top -- the top is a pure passthrough:
    # one raw K*AW-bit beat/vector in, one raw N*out_width-bit beat/vector out.
    pkg = _gen(tmp_path, (3, 16, 8), "gemm_3x16x8_f", reuse_factor=16, fold_axis="k")
    top = (pkg / "gemm_3x16x8_f_top.cpp").read_text()
    assert "feed_w" not in top and "feed_a" not in top
    assert "a_in" in top and "ap_uint<128> >& a_in" in top    # K*AW = 16*8, unpadded
    assert "ap_uint<128> >& c_out" in top                     # N*out_width = 8*16, unpadded


def test_weight_stationary_package(tmp_path):
    pkg = _gen(tmp_path, (4, 4, 4), "gemm_4x4x4", reuse_factor=1)
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
    from targets.mvau import weightpack as wp
    K, N = 4, 4
    B = [[((k * 3 + n) % 7) - 3 for n in range(N)] for k in range(K)]   # [K][N]
    pkg = _gen(tmp_path, (1, K, N), "gemm_wm", reuse_factor=1, weight_matrix=np.asarray(B))
    got = (pkg / "rtl_static" / "gemm_wm_weights.dat").read_text()
    # PE=SIMD=4 single tile at this shape; word = byte-aligned weight-stream width
    exp = wp.pack_memstream_hex(B, N, K, 4, 4, 8, word_bits=128)
    assert got == exp


def test_n_tiling_stitched_in_rtl(tmp_path):
    # N split into 2 column slices, each an independent MVU tile stitched in the
    # RTL shim: shared activation broadcast in, tile outputs concatenated out.
    import numpy as np
    from targets.mvau import weightpack as wp
    K, N, NT = 8, 8, 2
    NTILE = N // NT
    B = [[((k * 5 + n) % 7) - 3 for n in range(N)] for k in range(K)]   # [K][N]
    pkg = _gen(tmp_path, (1, K, N), "gemm_nt", n_tiles=NT, reuse_factor=1,
               weight_matrix=np.asarray(B))

    # one memstream init (.dat) per N-column slice, packed from that slice of B
    for ti in range(NT):
        dat = pkg / "rtl_static" / f"gemm_nt_weights_t{ti}.dat"
        assert dat.is_file() and dat.stat().st_size > 0
        B_ti = [[B[kk][ti * NTILE + oo] for oo in range(NTILE)] for kk in range(K)]
        assert dat.read_text() == wp.pack_memstream_hex(B_ti, NTILE, K, 4, 8, 8, word_bits=256)

    v = (pkg / "gemm_nt_core.v").read_text()
    assert v.count("memstream #(") == NT and v.count("mvu_vvu_axi #(") == NT
    assert all(f"inst_{ti}" in v and f"wmem_{ti}" in v for ti in range(NT))
    assert "(&in_tready)" in v and "(&out_tvalid)" in v      # lockstep broadcast + fan-in
    for ti in range(NT):
        dat = (pkg / "rtl_static" / f"gemm_nt_weights_t{ti}.dat").resolve()
        assert f'.INIT_FILE("{dat}")' in v

    # C twin does the N-tile stitching + pad-column drop (mirroring the RTL); the
    # top is now a pure passthrough (that logic moved into the RTL wrapper).
    twin = (pkg / "gemm_nt_core.cpp").read_text()
    assert "ti < 2" in twin and "gemm_nt_core_W[8][8]" in twin
    top = (pkg / "gemm_nt_top.cpp").read_text()
    assert "for (int ti" not in top
    assert "ap_uint<64> >& a_in" in top and "ap_uint<128> >& c_out" in top


def test_k_tiling_unpadded_boundary_and_grid_stitched_in_rtl(tmp_path):
    # K=16 folded (fold_axis="k", RF=8) -> SIMD=2, SF=8 (k_pad=16); k_tiles=2 splits
    # the 8 SF-folds into 2 MVU cores (SF_tile=4) reducing MW=8 each, summed in RTL.
    # Boundary is UNPADDED: a raw K=16*AW-bit row in, N=4*out_width-bit row out.
    pkg = _gen(tmp_path, (1, 16, 4), "gemm_kt", k_tiles=2, reuse_factor=8, fold_axis="k")
    v = (pkg / "gemm_kt_core.v").read_text()
    assert v.count("memstream #(") == 2 and v.count("mvu_vvu_axi #(") == 2
    assert "input  wire [127:0] a_dout" in v      # K*AW = 16*8, unpadded
    assert "output wire [63:0] p_din" in v        # N*out_width = 4*16, unpadded
    assert "arow_reg" in v and "orow_reg" in v
    assert "sf_cnt" in v   # per-tile SF fan-out now lives in the wrapper

    top = (pkg / "gemm_kt_top.cpp").read_text()
    assert "feed_a" not in top and "static void unpack(" not in top   # pure passthrough
    assert "ap_uint<128> >& a_in" in top and "ap_uint<64> >& c_out" in top

    core = (pkg / "gemm_kt_core.cpp").read_text()
    assert "gemm_kt_core_W[4][" in core   # W[N][k_pad] (k_pad may exceed raw K=16)

    ip_hdr = (pkg / "gemm_kt_gemm_ip.h").read_text()
    assert "static_assert" in ip_hdr and "_repack_a" in ip_hdr and "_drain" in ip_hdr
    assert "data_T::size == 16" in ip_hdr


def test_k_tiling_csim_matches_golden(tmp_path):
    # Compile the C twin (behavioral stand-in for the RTL) against its own TB and
    # confirm the independent golden matmul + requant reference matches exactly --
    # the K-tiled counterpart of the plain-path bit-exactness check.
    import subprocess
    pkg = _gen(tmp_path, (1, 16, 4), "gemm_kt", k_tiles=2, reuse_factor=8, fold_axis="k")
    vitis_inc = "/mnt/vault1/tools/AMD/Vitis_HLS/2024.1/include"
    if not Path(vitis_inc).is_dir():
        pytest.skip("Vitis HLS headers not available in this environment")
    exe = tmp_path / "kt_tb"
    subprocess.run(["g++", "-std=c++14", f"-I{vitis_inc}", "-o", str(exe),
                    str(pkg / "gemm_kt_tb.cpp"), str(pkg / "gemm_kt_top.cpp"),
                    str(pkg / "gemm_kt_core.cpp")], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    assert "MVAU_PKG PASS" in out


def test_k_and_n_tiling_combined_with_bias_and_nf(tmp_path):
    # Combined N-tiling + K-tiling grid with NF>1 (exercises the orow_reg latch /
    # nf_cnt path on the K-tiled shim, not just the plain one) and a real bias.
    bias = [0.5, -0.25, 0.0, 0.25, 1.0, -1.0, 0.75, -0.5]
    pkg = _gen(tmp_path, (1, 16, 8), "gemm_kt2", k_tiles=2, n_tiles=2,
               reuse_factor=8, fold_axis="kn", bias=bias)
    v = (pkg / "gemm_kt2_core.v").read_text()
    assert v.count("memstream #(") == 4 and v.count("mvu_vvu_axi #(") == 4   # 2x2 grid
    assert "nf_cnt" in v and "orow_reg" in v
    core = (pkg / "gemm_kt2_core.cpp").read_text()
    assert "gemm_kt2_core_bias[8]" in core


def test_n_tiling_manifest_lists_all_weight_files(tmp_path):
    from targets.mvau import package as pkgmod
    items = [{"name": "gemm_nt", "m": 1, "k": 8, "n": 8, "n_tiles": 2},
             {"name": "gemm_1", "m": 1, "k": 4, "n": 4}]   # default single tile
    man = json.loads(pkgmod.gen_integration_manifest(items))
    cores = {c["name"]: c for c in man["cores"]}
    assert cores["gemm_nt"]["weight_data"] == [
        "gemm_nt/rtl_static/gemm_nt_weights_t0.dat",
        "gemm_nt/rtl_static/gemm_nt_weights_t1.dat"]
    assert cores["gemm_1"]["weight_data"] == ["gemm_1/rtl_static/gemm_1_weights.dat"]


def test_affine_drain_present(tmp_path):
    pkg = _gen(tmp_path, (3, 16, 8), "gemm_3x16x8_f", reuse_factor=16, fold_axis="k")
    # requant (shift + round-half-up + wrap) lives in the core twin (and the RTL
    # requant stage), emitting an already-narrow, already-unpadded beat; the top
    # is now a pure passthrough into the blackbox (no repack/unpack of its own).
    core = (pkg / "gemm_3x16x8_f_core.cpp").read_text()
    assert "round-half-up" in core and "wrap" in core
    top = (pkg / "gemm_3x16x8_f_top.cpp").read_text()
    assert "static void unpack(" not in top and "static void feed" not in top
    # the dedicated hls4ml-facing IP is where the pure bit-reinterpretation lives
    ip_hdr = (pkg / "gemm_3x16x8_f_gemm_ip.h").read_text()
    assert "static_assert" in ip_hdr and "_repack_a" in ip_hdr and "_drain" in ip_hdr


def test_reject_wide_precision(tmp_path):
    with pytest.raises(ValueError):
        _gen(tmp_path, (2, 8, 8), "bad", weight_precision="fixed<16,6>")


def test_reject_non_versal_part(tmp_path):
    with pytest.raises(ValueError):
        _gen(tmp_path, (2, 8, 8), "notversal", part="xcvu13p-flga2577-2-e")


def test_bias_scaled_and_added(tmp_path):
    # per-column bias, scaled to the accumulator (2^(fa+fb)=2^8) domain and added.
    # Bias now lives in the core twin (matches the RTL requant stage's bias ROM),
    # not the top-level drain.
    bias = [0.5, -0.25, 0.0, 0.25, 1.0, -1.0, 0.75, -0.5]
    pkg = _gen(tmp_path, (2, 8, 8), "gb", reuse_factor=1, bias=bias)
    core = (pkg / "gb_core.cpp").read_text()
    assert "static const long gb_core_bias[8] = {128, -64, 0, 64, 256, -256, 192, -128}" in core
    assert "gb_core_bias[oc]" in core
    top = (pkg / "gb_top.cpp").read_text()
    assert "gb_core_bias" not in top


def test_no_bias_omits_array(tmp_path):
    # has_bias=False (the manifest's own field) is the gate for "no bias" -- not
    # whether a bias value happens to be given.
    pkg = _gen(tmp_path, (2, 8, 8), "gnb", reuse_factor=1, has_bias=False)
    core = (pkg / "gnb_core.cpp").read_text()
    assert "gnb_bias" not in core
    top = (pkg / "gnb_top.cpp").read_text()
    assert "gnb_bias" not in top


def test_has_bias_false_ignores_nonzero_bias_values(tmp_path):
    # has_bias is the single source of truth: even a real, non-zero bias array
    # must not be baked (and no add emitted) when the manifest says has_bias=False.
    bias = [0.5, -0.25, 0.0, 0.25, 1.0, -1.0, 0.75, -0.5]
    pkg = _gen(tmp_path, (2, 8, 8), "gfb", reuse_factor=1, has_bias=False, bias=bias)
    core = (pkg / "gfb_core.cpp").read_text()
    assert "gfb_bias" not in core


def test_has_bias_true_bakes_sublsb_bias(tmp_path):
    # A bias that scales to all-zero codes at this fixed-point precision must
    # still be baked and added when has_bias is True -- hardware has to match the
    # manifest (and hls4ml's csim expectation), not silently re-derive presence
    # from the (here, sub-LSB) scaled values.
    tiny = 1.0 / (1 << 20)   # far below the 2^8 accumulator scale -> rounds to 0
    bias = [tiny] * 8
    pkg = _gen(tmp_path, (2, 8, 8), "gsl", reuse_factor=1, has_bias=True, bias=bias)
    core = (pkg / "gsl_core.cpp").read_text()
    assert "static const long gsl_core_bias[8] = {0, 0, 0, 0, 0, 0, 0, 0}" in core
    assert "gsl_core_bias[oc]" in core


def test_has_bias_true_without_bias_raises(tmp_path):
    with pytest.raises(ValueError):
        _gen(tmp_path, (2, 8, 8), "gmissing", reuse_factor=1, has_bias=True)


def test_manifest_reuse_factor_legalized(tmp_path, capsys):
    from targets.mvau import package as pkgmod
    # (4, 6, 8), fold-n, RF=16 > N=8 -- cannot be honored (PE would be < 1);
    # legalizes to RF=N=8 (PE=1).
    items = [{"name": "gemm_rf16", "m": 4, "k": 6, "n": 8, "reuse_factor": 16,
              "weights_in_core": True, "weight_precision": "fixed<8,4>",
              "input_precision": "fixed<8,4>", "part": VERSAL}]
    man = json.loads(pkgmod.gen_integration_manifest(items))
    core = man["cores"][0]
    assert core["reuse_factor_requested"] == 16
    assert core["reuse_factor"] == 8
    assert core["effective_reuse"] == 8
    assert core["n_pad"] == 8
    assert core["fold_axis"] == "n"
    assert core["pe"] == 1 and core["simd"] == 6
    assert core["ii_per_vector"] == 8

    out = capsys.readouterr().out
    assert 'WARNING: Invalid ReuseFactor=16 in layer "gemm_rf16".' in out
    assert "Using ReuseFactor=8 instead." in out


def test_manifest_reuse_factor_not_legalized(tmp_path, capsys):
    from targets.mvau import package as pkgmod
    items = [{"name": "gemm_rf1", "m": 4, "k": 4, "n": 4, "reuse_factor": 1,
              "weights_in_core": True, "weight_precision": "fixed<8,4>",
              "input_precision": "fixed<8,4>", "part": VERSAL}]
    man = json.loads(pkgmod.gen_integration_manifest(items))
    core = man["cores"][0]
    assert core["reuse_factor_requested"] == 1
    assert core["reuse_factor"] == 1
    assert core["n_pad"] == 4

    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_manifest_n_pad_reported_for_fold_n(tmp_path):
    from targets.mvau import package as pkgmod
    # (4, 6, 10) RF=4, fold-n: n_pad = 12, PE = 3.
    items = [{"name": "gemm_npad", "m": 4, "k": 6, "n": 10, "reuse_factor": 4,
              "fold_axis": "n", "weights_in_core": True,
              "weight_precision": "fixed<8,4>", "input_precision": "fixed<8,4>",
              "part": VERSAL}]
    man = json.loads(pkgmod.gen_integration_manifest(items))
    core = man["cores"][0]
    assert core["n_pad"] == 12
    assert core["pe"] == 3
    assert core["fold_axis"] == "n"


# ── N-padding: package generation is bit-exact end to end (see test plan step 2
# for the simulated verification of these same shapes) ─────────────────────────

@pytest.mark.parametrize("shape,extra", [
    ((4, 6, 10), dict(reuse_factor=4, fold_axis="n")),   # n_pad=12, PE=3
    ((2, 9, 5), dict(reuse_factor=2, fold_axis="n")),    # n_pad=6, PE=3
    ((4, 10, 10), dict(reuse_factor=4, fold_axis="kn")), # n_pad=12, k_pad=12
    ((4, 8, 8), dict(reuse_factor=4, fold_axis="n")),    # regression: RF divides N
])
def test_n_padded_package_generates_and_verifies(tmp_path, shape, extra):
    name = "gemm_np_" + "x".join(str(s) for s in shape)
    pkg = _gen(tmp_path, shape, name, **extra)
    assert (pkg / f"{name}_top.cpp").is_file()
    # the interface presents exactly N output columns, not n_pad
    n = shape[2]
    cb = (pkg / f"{name}_top.cpp").read_text()
    assert f"ap_uint<{n * 16}>" in cb or f"crow" in cb   # requant drain present


# ── Two-operand (gemm_stream) fully-spatial grid: UNPADDED RTL boundary ─────────

def _gen_2op(tmp_path, shape, name, **extra):
    t = load_target("mvau")
    cfg = dict(_CFG, name=name, output_dir=str(tmp_path),
               weights_in_core=False, second_operand_row_major=True)
    cfg.update(extra)
    pkg = Path(t.package(shape, cfg))
    t.verify(pkg)
    return pkg


def test_two_operand_depth1_unpadded_boundary(tmp_path):
    # M=4 K=4 N=4, RF=1 -> fully-spatial (PE=SIMD=4, SF=NF=1, DEPTH=NF*SF=1): the
    # single-tile dynamic_load_2op shim now serves this depth too (the old
    # register-form grid shim is gone -- see the "Cleanup: collapse 2-op to a single
    # dynamic_load_2op tile" plan). No padding needed at all, so the boundary widths
    # are unpadded -- exactly the geometry mha_tiny's QK^T / A.V two-operand GEMMs
    # route through in practice.
    pkg = _gen_2op(tmp_path, (4, 4, 4), "gemm_2op_d1", reuse_factor=1)
    v = (pkg / "gemm_2op_d1_core.v").read_text()
    assert "dynamic_load_2op" in v
    assert "input  wire [31:0] a_dout" in v    # SIMD*AW = 4*8, unpadded (SF=1)
    assert "input  wire [31:0] b_dout" in v    # PE*WEIGHT_WIDTH = 4*8 (loader's narrow beat)
    assert "output wire [63:0] p_din" in v     # PE*ACCU raw (post-requant it's PE*out_width)

    top = (pkg / "gemm_2op_d1_top.cpp").read_text()
    assert "feed_a" in top and "feed_b" in top

    ip_hdr = (pkg / "gemm_2op_d1_gemm_ip.h").read_text()
    assert "static_assert" in ip_hdr and "_repack_a" in ip_hdr and "_repack_b" in ip_hdr
    assert "data1_T::size == 4" in ip_hdr   # repack_b's row-major (Mode A) static_assert

    _csim_check_2op(tmp_path, pkg, "gemm_2op_d1")


def test_two_operand_depth1_col_major_csim_matches_golden(tmp_path):
    # Same geometry (DEPTH==1, SF=NF=1) but Mode B (col-major B) -- closes the RF=1
    # Mode-B gap the DEPTH==1 port onto dynamic_load_2op was meant to close.
    pkg = _gen_2op(tmp_path, (4, 4, 4), "gemm_2op_d1b", reuse_factor=1,
                   second_operand_row_major=False)
    v = (pkg / "gemm_2op_d1b_core.v").read_text()
    assert "dynamic_load_2op" in v and ".MODE(1)" in v

    ip_hdr = (pkg / "gemm_2op_d1b_gemm_ip.h").read_text()
    assert "data0_T::size == 4" not in ip_hdr   # repack_a has no static_assert
    assert "data1_T::size == 4" in ip_hdr        # repack_b's col-major static_assert (data1_T::size==K)

    _csim_check_2op(tmp_path, pkg, "gemm_2op_d1b")


# ── Two-operand single-tile shim (dynamic_load_2op), fold chosen by resolve_fold ──
#
# VERIFY-ONLY: cases e-i in run_rtl_tests.py and the _gen_2op tests above all drive
# generate_two_operand_shim with hand-picked pe=/simd=, bypassing resolve_fold. These
# cases instead specify fold_axis + reuse_factor (no pe/simd, k_tiles=1 default) so
# the plan is resolved the way a real manifest would drive it, landing on the
# single-tile untiled shim (nt=1, k_tiles=1, DEPTH=NF*SF>=2) for each fold axis and
# both B-layout modes. Geometries were chosen (see geometry.fold_plan) so:
#   fold-n RF=2:  (m,k,n)=(4,4,8)  -> pe=4 simd=4 sf=1 nf=2 (DEPTH=NF=2)
#   fold-k RF=2:  (m,k,n)=(4,9,4)  -> pe=4 simd=5 sf=2 nf=1 (DEPTH=SF=2, k_pad=10)
#   fold-kn RF=2: (m,k,n)=(4,9,8)  -> pe=4 simd=5 sf=2 nf=2 (DEPTH=SF*NF=4, k_pad=10)
# all confirmed single_tile=True (use_kt=False, nt=1) at collection time.

from targets.mvau import geometry as _geom_check  # noqa: E402


def _csim_check_2op(tmp_path, pkg, name):
    import subprocess
    vitis_inc = "/mnt/vault1/tools/AMD/Vitis_HLS/2024.1/include"
    if not Path(vitis_inc).is_dir():
        pytest.skip("Vitis HLS headers not available in this environment")
    exe = tmp_path / f"{name}_exe"
    subprocess.run(["g++", "-std=c++14", f"-I{vitis_inc}", "-o", str(exe),
                    str(pkg / f"{name}_tb.cpp"), str(pkg / f"{name}_top.cpp"),
                    str(pkg / f"{name}_core.cpp")], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    assert "MVAU_PKG PASS" in out


@pytest.mark.parametrize("mode_kw,mode_tag", [
    (dict(second_operand_row_major=True), "rowmajor"),
    (dict(second_operand_row_major=False), "colmajor"),
])
@pytest.mark.parametrize("shape,extra", [
    ((4, 4, 8), dict(reuse_factor=2, fold_axis="n")),   # PE=4 SIMD=4 SF=1 NF=2 (DEPTH=NF=2)
    ((4, 9, 4), dict(reuse_factor=2, fold_axis="k")),   # PE=4 SIMD=5 SF=2 NF=1 (DEPTH=SF=2)
    ((4, 9, 8), dict(reuse_factor=2, fold_axis="kn")),  # PE=4 SIMD=5 SF=2 NF=2 (DEPTH=SF*NF=4)
])
def test_two_operand_single_tile_resolve_fold_csim_matches_golden(
        tmp_path, shape, extra, mode_kw, mode_tag):
    axis = extra["fold_axis"]
    name = f"gemm_2op_rf_{axis}_{mode_tag}"
    plan = _geom_check.fold_plan(*shape, **_CFG, weights_in_core=False,
                                  **extra, **mode_kw)
    tile = plan["tile"]
    # Confirm this geometry actually resolved (via resolve_fold, no explicit pe/simd)
    # onto the single-tile untiled shim (DEPTH=NF*SF>=2), not the register/grid form.
    assert plan["n_tiles"] == 1 and plan.get("k_tiles", 1) == 1
    assert tile["sf"] * tile["nf"] >= 2
    assert not (tile["sf"] == 1 and tile["nf"] == 1)

    t = load_target("mvau")
    cfg = dict(_CFG, name=name, output_dir=str(tmp_path), weights_in_core=False,
               **extra, **mode_kw)
    pkg = Path(t.package(shape, cfg))
    t.verify(pkg)
    v = (pkg / f"{name}_core.v").read_text()
    assert "dynamic_load_2op" in v
    _csim_check_2op(tmp_path, pkg, name)
