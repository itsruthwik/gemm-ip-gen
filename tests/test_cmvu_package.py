"""Unit tests for cmvu's gen_combined_header dispatch layer.

Covers the missing integration between cmvu's per-layer headers (which define
``<name>_gemm_stream_const_weights`` / ``<name>_gemm_stream_runtime_b``) and
the four nnet:: entry points the Catapult hls4ml backend actually calls
(nnet::gemm_stream, nnet::gemm_stream_const_weights, nnet::gemm_array,
nnet::gemm_array_const_weights -- see nnet_gemm_ip.h / nnet_gemm_stream.h).
"""
import re
import pytest
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from targets.cmvu import package as pkg  # noqa: E402


def _const_weights_item(name="gemm_conv1", m=100, k=36, n=4, gemm_ip_index=16):
    return {"name": name, "m": m, "k": k, "n": n,
           "weights_in_core": True, "gemm_ip_index": gemm_ip_index}


def _runtime_b_item(name="gemm_attn", m=8, k=16, n=8, gemm_ip_index=42):
    return {"name": name, "m": m, "k": k, "n": n,
           "weights_in_core": False, "gemm_ip_index": gemm_ip_index}


def test_combined_header_includes_per_layer_headers():
    items = [_const_weights_item(), _runtime_b_item()]
    header = pkg.gen_combined_header(items)
    assert '#include "gemm_conv1/gemm_conv1_gemm_ip.h"' in header
    assert '#include "gemm_attn/gemm_attn_gemm_ip.h"' in header


def test_combined_header_defines_all_four_primaries():
    items = [_const_weights_item(), _runtime_b_item()]
    header = pkg.gen_combined_header(items)
    for sig in [
        "void gemm_stream_const_weights(",
        "void gemm_stream(",
        "void gemm_array(",
        "void gemm_array_const_weights(",
    ]:
        assert sig in header, f"missing {sig!r}"
    # all four live under namespace nnet
    assert re.search(r"namespace nnet\s*\{", header)


def test_const_weights_item_routes_under_its_gemm_ip_id():
    item = _const_weights_item()
    header = pkg.gen_combined_header([item])
    assert "CONFIG_T::gemm_ip_id == 16" in header
    assert "CONFIG_T::gemm_m == 100" in header
    assert "CONFIG_T::gemm_k == 36" in header
    assert "CONFIG_T::gemm_n == 4" in header
    assert "nnet::gemm_conv1_gemm_stream_const_weights<data_T, res_T, CONFIG_T>(" in header
    # weight-stationary item must NOT appear in the runtime-B (gemm_stream) body
    stream_body = header.split("void gemm_stream(")[1].split("void gemm_array(")[0]
    assert "gemm_conv1_gemm_stream_const_weights" not in stream_body


def test_runtime_b_item_routes_to_its_runtime_b_function():
    item = _runtime_b_item()
    header = pkg.gen_combined_header([item])
    assert "CONFIG_T::gemm_ip_id == 42" in header
    assert "nnet::gemm_attn_gemm_stream_runtime_b<data0_T, data1_T, res_T, CONFIG_T>(" in header
    const_weights_body = header.split("gemm_stream_const_weights(")[1].split(
        "void gemm_stream(")[0]
    assert "gemm_attn" not in const_weights_body


def test_mixed_items_each_route_to_the_right_dispatcher():
    items = [_const_weights_item(), _runtime_b_item()]
    header = pkg.gen_combined_header(items)
    const_weights_body = header.split("gemm_stream_const_weights(")[1].split(
        "void gemm_stream(")[0]
    stream_body = header.split("void gemm_stream(")[1].split("void gemm_array(")[0]
    assert "gemm_conv1_gemm_stream_const_weights" in const_weights_body
    assert "gemm_attn_gemm_stream_runtime_b" not in const_weights_body
    assert "gemm_attn_gemm_stream_runtime_b" in stream_body
    assert "gemm_conv1_gemm_stream_const_weights" not in stream_body


def test_empty_dispatchers_are_static_assert_only():
    header = pkg.gen_combined_header([])
    const_weights_body = header.split("gemm_stream_const_weights(")[1].split(
        "void gemm_stream(")[0]
    stream_body = header.split("void gemm_stream(")[1].split("void gemm_array(")[0]
    assert "static_assert(CONFIG_T::gemm_m == 0" in const_weights_body
    assert "static_assert(CONFIG_T::gemm_m == 0" in stream_body


def test_array_entries_are_static_assert_only_no_io_parallel():
    header = pkg.gen_combined_header([_const_weights_item(), _runtime_b_item()])
    array_body = header.split("void gemm_array(")[1].split(
        "void gemm_array_const_weights(")[0]
    array_cw_body = header.split("void gemm_array_const_weights(")[1].split(
        "} // namespace nnet")[0]
    for body in (array_body, array_cw_body):
        assert "static_assert(CONFIG_T::gemm_m == 0" in body
        assert "io_parallel" in body
        # no per-layer call is ever routed into the array dispatchers
        assert "gemm_conv1" not in body
        assert "gemm_attn" not in body


def test_shape_only_fallback_when_gemm_ip_index_absent():
    item = _const_weights_item(gemm_ip_index=None)
    header = pkg.gen_combined_header([item])
    body = header.split("gemm_stream_const_weights(")[1].split("void gemm_stream(")[0]
    assert "CONFIG_T::gemm_ip_id" not in body
    assert "CONFIG_T::gemm_m == 100" in body


# ── VTR-facing cmvu_mode1 blackbox model ────────────────────────────────────


def test_cmvu_mode1_vtr_model_has_blackbox_and_matches_wrapper_ports():
    from targets.cmvu import geometry as g
    model = pkg.gen_cmvu_mode1_vtr_model()
    assert "(* blackbox *)" in model
    assert "module cmvu_mode1 #(" in model
    assert "endmodule" in model
    # no body: only port/parameter declarations between the port list and endmodule
    body = model.split(");", 1)[1].split("endmodule", 1)[0]
    assert body.strip() == ""
    # exact port list the wrapper (rtl.py) instantiates against
    for port in ("clk", "rst", "valid", "acc_first", "acc_last", "a_in", "b_in",
                "w_we", "w_load_start", "w_col_major", "w_dual_tile", "tile_sel",
                "w_tile_sel", "a_signed", "b_signed", "shift_amt", "out_w",
                "cascade_in", "bias_in", "y_out", "cascade_out", "y_valid", "done"):
        assert port in model, f"missing port {port}"
    # parameter defaults come from geometry.py's physical constants (the
    # single source the wrapper itself instantiates against)
    assert f"IN_WIDTH        = {g.IN_WIDTH}" in model
    assert f"COEF_WIDTH      = {g.COEF_WIDTH}" in model
    assert f"ACC_WIDTH       = {g.ACC_WIDTH}" in model
    assert f"BIAS_WIDTH      = {g.BIAS_WIDTH}" in model  # 32-bit per lane
    assert f"RESULT_WIDTH    = {g.RESULT_WIDTH}" in model
    assert f"SHIFT_WIDTH     = {g.SHIFT_WIDTH}" in model
    assert f"K               = {g.K_PHYS}" in model
    assert f"N               = {g.N_PHYS}" in model
    assert f"M_MEM_TILES     = {g.MEM_TILES}" in model
    assert "CASCADE_EN       = 1'b1" in model


def test_cmvu_mode1_vtr_model_compiles_with_iverilog(tmp_path):
    import shutil
    import subprocess
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog not on PATH")
    model = pkg.gen_cmvu_mode1_vtr_model()
    src = tmp_path / "cmvu_mode1_vtr_blackbox.sv"
    src.write_text(model)
    r = subprocess.run(["iverilog", "-g2012", "-t", "null", str(src)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_finalize_ships_the_vtr_model_outside_the_sources_tcl(tmp_path):
    from targets.cmvu.flow import CmvuTarget
    from targets.cmvu import geometry as g
    t = CmvuTarget()
    tcl = t.sources_tcl([])
    # The stub declares `module cmvu_mode1` too; listed here, SCVerify would
    # compile it after the real block and replace it.
    assert g.CMVU_MODE1_VTR_MODEL not in tcl
    assert "cmvu_mode1.sv" in tcl and "-exclude true" in tcl
    t.finalize([], str(tmp_path))
    model_path = tmp_path / g.CMVU_MODE1_VTR_MODEL
    assert model_path.is_file()
    assert "(* blackbox *)" in model_path.read_text()
    # distinct from the real vendored file of a similar name
    assert model_path.name != "cmvu_mode1.sv"
    assert (tmp_path / "cmvu_mode1.sv").is_file()


def test_const_weights_entry_is_a_free_running_block():
    # The entry pipelines its own main loop (no Catapult Tcl directive, no
    # hls4ml change): synthesized, it is one wrapper clock per call with state
    # kept across calls, so frames overlap; the C model stays one frame/call.
    from targets.cmvu import geometry as g
    m, k, n, kf, nf = 4, 16, 16, 2, 2
    geo = g.resolve_geometry(m, k, n, kf, nf)
    W = [[(i * n + j) % 7 - 3 for j in range(n)] for i in range(k)]
    header = pkg.gen_public_header("gemm_t", m, k, n, W, [0] * n, 4, geo)
    entry = header[header.index("#pragma hls_design block"):]
    assert entry.startswith("#pragma hls_design block\n"
                            "#pragma hls_pipeline_init_interval 1\n"
                            "template <class data_T, class res_T, typename CONFIG_T>\n"
                            "void gemm_t_gemm_stream_const_weights(")
    synth, c_model = entry.split("#else", 1)
    assert "a_stream.nb_read(beat)" in synth
    assert "static ac_int<4, false> gap" in synth
    period = geo["k_passes"] * geo["n_passes"]
    assert f"gap = {period} - 1;" in synth
    assert "RUN: for" not in synth
    assert "reset_state()" in c_model and "RUN: for" in c_model
