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
    assert 'COMPUTE_CORE("mvu_vvu_8sx9_dsp58")' in v
    assert "ap_ce" in v and "~ap_rst" in v   # active-high reset + CE stall


def test_folded_top_beat_counts(tmp_path):
    # (3,16,8) rf=16, fold-k: SIMD floor=ceil(16/3)=6 -> RF legalizes to 6, PE=8,
    # SIMD=3, SF=6 (k_pad=18).
    pkg = _gen(tmp_path, (3, 16, 8), "gemm_3x16x8_f", reuse_factor=16, fold_axis="k")
    top = (pkg / "gemm_3x16x8_f_top.cpp").read_text()
    # weight-stationary: weights baked in the memstream (no feed_w). acts M*SF = 3*6 = 18;
    # requant loops M=3 vectors.
    assert "i < 18;" in top and "vec < 3;" in top
    assert "feed_w" not in top


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
    sys.path.insert(0, str(_SRC / "targets" / "mvau"))
    import weightpack as wp
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
    sys.path.insert(0, str(_SRC / "targets" / "mvau"))
    import weightpack as wp
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

    # C twin + drain widen to the concatenated beat and index the global columns
    twin = (pkg / "gemm_nt_core.cpp").read_text()
    assert "ti < 2" in twin and "gemm_nt_core_W[8][8]" in twin
    top = (pkg / "gemm_nt_top.cpp").read_text()
    assert "ti < 2" in top


def test_n_tiling_manifest_lists_all_weight_files(tmp_path):
    sys.path.insert(0, str(_SRC / "targets" / "mvau"))
    import package as pkgmod
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
    # requant (shift + round-half-up + wrap) now lives in the core twin, emitting an
    # already-narrow beat; the top is a pure unpack.
    core = (pkg / "gemm_3x16x8_f_core.cpp").read_text()
    assert "round-half-up" in core and "wrap" in core
    top = (pkg / "gemm_3x16x8_f_top.cpp").read_text()
    assert "static void unpack(" in top


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
    sys.path.insert(0, str(_SRC / "targets" / "mvau"))
    import package as pkgmod
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
    sys.path.insert(0, str(_SRC / "targets" / "mvau"))
    import package as pkgmod
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
    sys.path.insert(0, str(_SRC / "targets" / "mvau"))
    import package as pkgmod
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
