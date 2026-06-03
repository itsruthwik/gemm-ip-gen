"""Tests for the Vitis GEMM blackbox generator."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Ensure the package is importable
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in os.sys.path:
    os.sys.path.insert(0, _src_dir)

from gemm_ip.metadata import (
    normalize_gemm_config,
    a_stream_width,
    b_stream_width,
    c_stream_width,
    bias_stream_width,
    grid_rows,
    grid_cols,
    k_steps,
    tail_mask_hex,
)
from gemm_ip.vitis import (
    generate_vitis_pkg,
    gen_combined_header,
    gen_integration_manifest,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def _gen_item(**overrides):
    cfg = {"name": "test_gemm", "m": 8, "k": 8, "n": 8, "backend": "vitis"}
    cfg.update(overrides)
    return normalize_gemm_config(cfg)[0]


# ─── 1. Config normalization ───────────────────────────────────────────────────


class TestConfigNormalization:
    def test_backend_vitis_accepted(self):
        item = _gen_item()
        assert item["backend"] == "vitis"
        assert item["m"] == 8 and item["k"] == 8 and item["n"] == 8

    def test_hls4ml_style_config(self):
        items = normalize_gemm_config({
            "dense1": {"type": "Dense", "n_in": 16, "n_out": 8, "gemm_m": 2, "backend": "vitis"},
        })
        assert items[0]["m"] == 2
        assert items[0]["k"] == 16
        assert items[0]["n"] == 8
        assert items[0]["backend"] == "vitis"

    def test_grid_derivation(self):
        item = _gen_item(m=14, n=6, k=6)
        assert item["grid_rows"] == 2  # ceil(14/8)
        assert item["grid_cols"] == 1  # ceil(6/8)
        assert item["k_steps"] == 1    # ceil(6/8)

    def test_emit_name_sanitized(self):
        item = _gen_item(name="my.layer/1")
        assert "_" in item["emit_name"]
        assert item["emit_name"] != "my.layer/1"


# ─── 2. File presence ─────────────────────────────────────────────────────────


PREFIXED_FILES = ["_wrapper.cpp", ".v", "_wrapper.json", "_gemm_ip.h", "_tb.cpp", "_design.cpp"]
UNPREFIXED_FILES = ["run_vitis.tcl", "run_vitis.py", "hls_config.cfg", "tensor_slice_int8.v"]


class TestFilePresence:
    def test_standard_8x8x8(self, tmp_output):
        item = _gen_item(m=8, k=8, n=8)
        generate_vitis_pkg(item, tmp_output)
        pkg_dir = Path(tmp_output) / item["emit_name"]
        files = os.listdir(pkg_dir)
        name = item["emit_name"]
        for suffix in PREFIXED_FILES:
            expected = f"{name}{suffix}"
            assert expected in files, f"Missing {expected} in {files}"
        for fname in UNPREFIXED_FILES:
            assert fname in files, f"Missing {fname} in {files}"

    def test_tail_dimensions_14x6x6(self, tmp_output):
        item = _gen_item(m=14, k=6, n=6)
        generate_vitis_pkg(item, tmp_output)
        pkg_dir = Path(tmp_output) / item["emit_name"]
        files = os.listdir(pkg_dir)
        name = item["emit_name"]
        assert f"{name}_wrapper.cpp" in files
        assert f"{name}.v" in files

    def test_non_power_of_two_5x5x5(self, tmp_output):
        item = _gen_item(m=5, k=5, n=5)
        generate_vitis_pkg(item, tmp_output)
        pkg_dir = Path(tmp_output) / item["emit_name"]
        files = os.listdir(pkg_dir)
        assert f"{item['emit_name']}_wrapper.cpp" in files

    def test_run_vitis_uses_blackbox_json(self, tmp_output):
        item = _gen_item()
        generate_vitis_pkg(item, tmp_output)
        script_path = Path(tmp_output) / item["emit_name"] / "run_vitis.tcl"
        content = script_path.read_text()
        assert f"add_files -blackbox {item['emit_name']}/{item['emit_name']}_wrapper.json" in content
        assert f"add_files {item['emit_name']}/{item['emit_name']}.v" not in content

    def test_python_hls_config_uses_blackbox_json(self, tmp_output):
        item = _gen_item()
        generate_vitis_pkg(item, tmp_output)
        cfg_path = Path(tmp_output) / item["emit_name"] / "hls_config.cfg"
        content = cfg_path.read_text()
        assert f"syn.top={item['emit_name']}_design" in content
        assert f"syn.file={item['emit_name']}/{item['emit_name']}_design.cpp" in content
        assert f"tb.file={item['emit_name']}/{item['emit_name']}_tb.cpp" in content
        assert f"syn.blackbox.file={item['emit_name']}/{item['emit_name']}_wrapper.json" in content

    def test_python_runner_creates_hls_component(self, tmp_output):
        item = _gen_item()
        generate_vitis_pkg(item, tmp_output)
        script_path = Path(tmp_output) / item["emit_name"] / "run_vitis.py"
        content = script_path.read_text()
        assert "client.create_hls_component" in content
        assert "comp.run(operation)" in content


# ─── 3. JSON descriptor validity ──────────────────────────────────────────────


class TestJsonDescriptor:
    def test_valid_json(self, tmp_output):
        item = _gen_item()
        generate_vitis_pkg(item, tmp_output)
        json_path = Path(tmp_output) / item["emit_name"] / f"{item['emit_name']}_wrapper.json"
        with open(json_path) as f:
            desc = json.load(f)
        assert desc["c_function_name"] == f"{item['emit_name']}_wrapper"
        assert desc["rtl_top_module_name"] == f"{item['emit_name']}_wrapper"

    def test_stream_parameters(self, tmp_output):
        item = _gen_item()
        generate_vitis_pkg(item, tmp_output)
        json_path = Path(tmp_output) / item["emit_name"] / f"{item['emit_name']}_wrapper.json"
        with open(json_path) as f:
            desc = json.load(f)
        params = {p["c_name"]: p for p in desc["c_parameters"]}
        assert "a_stream" in params
        assert "b_stream" in params
        assert "bias_stream" in params
        assert "c_stream" in params
        assert params["a_stream"]["c_port_direction"] == "in"
        assert params["c_stream"]["c_port_direction"] == "out"

    def test_blackbox_metadata(self, tmp_output):
        item = _gen_item(m=14, k=6, n=6)
        generate_vitis_pkg(item, tmp_output)
        json_path = Path(tmp_output) / item["emit_name"] / f"{item['emit_name']}_wrapper.json"
        with open(json_path) as f:
            desc = json.load(f)
        assert "rtl_common_signal" in desc
        assert desc["rtl_common_signal"]["module_clock"] == "ap_clk"
        assert desc["rtl_common_signal"]["module_reset"] == "ap_rst"
        assert desc["rtl_common_signal"]["module_clock_enable"] == "ap_ce"
        assert desc["rtl_common_signal"]["ap_ctrl_chain_protocol_idle"] == ""
        assert desc["rtl_common_signal"]["ap_ctrl_chain_protocol_start"] == ""
        assert desc["rtl_common_signal"]["ap_ctrl_chain_protocol_ready"] == ""
        assert desc["rtl_common_signal"]["ap_ctrl_chain_protocol_done"] == ""
        assert desc["rtl_common_signal"]["ap_ctrl_chain_protocol_continue"] == ""

    def test_fifo_map(self, tmp_output):
        item = _gen_item()
        generate_vitis_pkg(item, tmp_output)
        json_path = Path(tmp_output) / item["emit_name"] / f"{item['emit_name']}_wrapper.json"
        with open(json_path) as f:
            desc = json.load(f)
        params = {p["c_name"]: p for p in desc["c_parameters"]}
        assert params["a_stream"]["rtl_ports"]["FIFO_data_read_in"] == "a_tdata"
        assert params["a_stream"]["rtl_ports"]["FIFO_empty_flag"] == "a_tvalid"
        assert params["c_stream"]["rtl_ports"]["FIFO_data_write_out"] == "c_tdata"

    def test_rtl_files_listed(self, tmp_output):
        item = _gen_item()
        generate_vitis_pkg(item, tmp_output)
        json_path = Path(tmp_output) / item["emit_name"] / f"{item['emit_name']}_wrapper.json"
        with open(json_path) as f:
            desc = json.load(f)
        assert any(".v" in f for f in desc["rtl_files"])


# ─── 4. Stream width derivation ────────────────────────────────────────────────


class TestStreamWidths:
    def test_8x8x8_widths(self):
        assert a_stream_width(8) == 64
        assert b_stream_width(8) == 64
        assert c_stream_width(8) == 64
        assert bias_stream_width(8) == 64

    def test_16x8x8_widths(self):
        assert a_stream_width(16) == 128  # 2 tiles * 64
        assert b_stream_width(8) == 64
        assert c_stream_width(8) == 64

    def test_8x16x8_widths(self):
        assert a_stream_width(8) == 64
        assert b_stream_width(16) == 128  # 2 tiles * 64
        assert c_stream_width(16) == 128

    def test_14x6x6_widths(self):
        assert a_stream_width(14) == 128  # ceil(14/8)=2 tiles
        assert b_stream_width(6) == 64    # ceil(6/8)=1 tile
        assert c_stream_width(6) == 64

    def test_non_power_of_two_5x5_5(self):
        assert a_stream_width(5) == 64
        assert b_stream_width(5) == 64
        assert c_stream_width(5) == 64
        assert grid_rows(5) == 1
        assert grid_cols(5) == 1
        assert k_steps(5) == 1  # ceil(5/8)


# ─── 5. Tail dimension masks ───────────────────────────────────────────────────


class TestTailMasks:
    def test_exact_multiple(self):
        assert tail_mask_hex(8, 0) == 0xFF  # all 8 lanes valid
        assert tail_mask_hex(16, 0) == 0xFF
        assert tail_mask_hex(16, 1) == 0xFF

    def test_partial_last_tile(self):
        assert tail_mask_hex(14, 0) == 0xFF  # tile 0: all valid
        assert tail_mask_hex(14, 1) == 0x3F  # tile 1: 6 valid lanes (bits 0-5)
        assert tail_mask_hex(6, 0) == 0x3F   # single tile: 6 valid

    def test_zero_after_total(self):
        assert tail_mask_hex(8, 1) == 0x00  # tile index beyond total → 0
        assert tail_mask_hex(5, 1) == 0x00


# ─── 6. Combined header dispatch ──────────────────────────────────────────────


class TestCombinedHeader:
    def test_single_item_dispatch(self):
        items = normalize_gemm_config({
            "gemm_a": {"m": 8, "k": 8, "n": 8, "backend": "vitis"},
        })
        header = gen_combined_header(items)
        assert "gemm_ip_stream" in header
        assert "gemm_a_gemm_ip_stream" in header

    def test_multi_item_dispatch(self):
        items = normalize_gemm_config({
            "gemm_a": {"m": 8, "k": 8, "n": 8, "backend": "vitis"},
            "gemm_b": {"m": 16, "k": 8, "n": 8, "backend": "vitis"},
        })
        header = gen_combined_header(items)
        assert "gemm_a_gemm_ip_stream" in header
        assert "gemm_b_gemm_ip_stream" in header

    def test_gemm_ip_index_dispatch(self):
        items = normalize_gemm_config({
            "gemm_a": {"m": 8, "k": 8, "n": 8, "backend": "vitis", "gemm_ip_index": 0},
            "gemm_b": {"m": 8, "k": 8, "n": 8, "backend": "vitis", "gemm_ip_index": 1},
        })
        header = gen_combined_header(items)
        assert "gemm_ip_id == 0" in header
        assert "gemm_ip_id == 1" in header

    def test_header_includes_simulation_helper(self):
        items = normalize_gemm_config({
            "gemm_a": {"m": 8, "k": 8, "n": 8, "backend": "vitis"},
        })
        header = gen_combined_header(items)
        assert "gemm_ip_stream_sim" in header
        assert "nnet::array" in header


# ─── 7. Integration manifest ───────────────────────────────────────────────────


class TestIntegrationManifest:
    def test_manifest_backend(self):
        items = normalize_gemm_config({
            "gemm_a": {"m": 8, "k": 8, "n": 8, "backend": "vitis"},
        })
        manifest = json.loads(gen_integration_manifest(items))
        assert manifest["backend"] == "vitis"
        assert manifest["package_format"] == "vitis_blackboxes_v1"

    def test_manifest_core_count(self):
        items = normalize_gemm_config({
            "gemm_a": {"m": 8, "k": 8, "n": 8, "backend": "vitis"},
            "gemm_b": {"m": 16, "k": 8, "n": 8, "backend": "vitis"},
        })
        manifest = json.loads(gen_integration_manifest(items))
        assert len(manifest["cores"]) == 2
        assert manifest["cores"][0]["m"] == 8
        assert manifest["cores"][1]["m"] == 16

    def test_manifest_has_combined_header(self):
        items = normalize_gemm_config({
            "gemm_a": {"m": 8, "k": 8, "n": 8, "backend": "vitis"},
        })
        manifest = json.loads(gen_integration_manifest(items))
        assert "combined_header" in manifest
        assert "gemm_ip_combined.h" in manifest["combined_header"]


# ─── 8. C++ wrapper signature ─────────────────────────────────────────────────


class TestWrapperSignature:
    def test_wrapper_cpp_has_correct_signature(self, tmp_output):
        item = _gen_item(m=8, k=8, n=8)
        generate_vitis_pkg(item, tmp_output)
        cpp_path = Path(tmp_output) / item["emit_name"] / f"{item['emit_name']}_wrapper.cpp"
        content = cpp_path.read_text()
        assert "hls::stream<ap_uint<64>>" in content
        assert "void" in content
        assert "_wrapper(" in content

    def test_wrapper_cpp_tail_dimensions(self, tmp_output):
        item = _gen_item(m=14, k=6, n=6)
        generate_vitis_pkg(item, tmp_output)
        cpp_path = Path(tmp_output) / item["emit_name"] / f"{item['emit_name']}_wrapper.cpp"
        content = cpp_path.read_text()
        assert "saturated_int8" in content
        assert "M=14" in content
        assert "N=6" in content


# ─── 9. RTL port names ────────────────────────────────────────────────────────


class TestRtlPorts:
    def test_has_ap_ctrl_signals(self, tmp_output):
        item = _gen_item()
        generate_vitis_pkg(item, tmp_output)
        v_path = Path(tmp_output) / item["emit_name"] / f"{item['emit_name']}.v"
        content = v_path.read_text()
        for signal in ["ap_clk", "ap_rst", "ap_ce"]:
            assert signal in content, f"Missing {signal}"

    def test_has_fifo_ports(self, tmp_output):
        item = _gen_item()
        generate_vitis_pkg(item, tmp_output)
        v_path = Path(tmp_output) / item["emit_name"] / f"{item['emit_name']}.v"
        content = v_path.read_text()
        for port in ["a_tdata", "a_tvalid", "a_tready",
                     "bias_tdata", "bias_tvalid", "bias_tready",
                     "b_tdata", "b_tvalid", "b_tready",
                     "c_tdata", "c_tvalid", "c_tready"]:
            assert port in content, f"Missing port {port}"
        assert "c_tlast" not in content

    def test_grid_core_instantiated(self, tmp_output):
        """Single .v file contains tensor_slice instances + Vitis ctrl."""
        item = _gen_item()
        generate_vitis_pkg(item, tmp_output)
        v_path = Path(tmp_output) / item["emit_name"] / f"{item['emit_name']}.v"
        content = v_path.read_text()
        assert "tensor_slice" in content
        assert "a_tdata" in content


# ─── 10. C++ model arithmetic validation ──────────────────────────────────────


def _pack_a_beat(activations, kk, grid_rows):
    """Pack one K-position slice of A into an integer (matches C++ model)."""
    m = activations.shape[0]
    val = 0
    for r in range(grid_rows):
        for rl in range(8):
            actual_row = r * 8 + rl
            byte_val = int(activations[actual_row, kk]) & 0xFF if actual_row < m else 0
            val |= byte_val << ((r * 64 + rl * 8))
    return val


def _pack_b_beat(weights, kk, grid_cols):
    """Pack one K-position slice of B into an integer (matches C++ model)."""
    n = weights.shape[0]
    val = 0
    for c in range(grid_cols):
        for cl in range(8):
            actual_col = c * 8 + cl
            byte_val = int(weights[actual_col, kk]) & 0xFF if actual_col < n else 0
            val |= byte_val << ((c * 64 + cl * 8))
    return val


def _pack_bias(biases, grid_cols):
    """Pack bias into an integer (matches C++ model)."""
    n = len(biases)
    val = 0
    for c in range(grid_cols):
        for lane in range(8):
            actual_col = c * 8 + lane
            byte_val = int(biases[actual_col]) & 0xFF if actual_col < n else 0
            val |= byte_val << ((c * 64 + lane * 8))
    return val


def _unpack_a_beat(pkt, r, rl):
    """Extract A byte from packed beat (matches C++ model unpack)."""
    return (pkt >> (r * 64 + rl * 8)) & 0xFF


def _unpack_b_beat(pkt, c, cl):
    """Extract B byte from packed beat (matches C++ model unpack)."""
    return (pkt >> (c * 64 + cl * 8)) & 0xFF


def _to_int8(byte_val):
    """Interpret an unpacked byte as a signed int8 value."""
    return byte_val - 256 if byte_val & 0x80 else byte_val


def _cpp_model_gemm(m, k, n, activations, weights, biases):
    """Simulate the C++ wrapper model arithmetic (fixed protocol).

    Returns (m, n) int8 result array, saturated from int32 accumulators.
    """
    import numpy as np

    gr = (m + 7) // 8
    gc = (n + 7) // 8

    # Bias unpack (with col*64 offset)
    bias_pkt = _pack_bias(biases, gc)
    bias_vals = np.zeros(n, dtype=np.int32)
    for col in range(gc):
        for lane in range(8):
            ac = col * 8 + lane
            if ac < n:
                bias_vals[ac] = np.array((bias_pkt >> (col * 64 + lane * 8)) & 0xFF, dtype=np.uint8).view(np.int8)

    # Accumulator buffer
    acc = np.zeros((gr, gc, 8, 8), dtype=np.int32)

    # Feed K cycles — outer product per cycle
    for kk in range(k):
        a_pkt = _pack_a_beat(activations, kk, gr)
        b_pkt = _pack_b_beat(weights, kk, gc)

        for r in range(gr):
            for c in range(gc):
                for rl in range(8):
                    actual_row = r * 8 + rl
                    a_val = _to_int8(_unpack_a_beat(a_pkt, r, rl)) if actual_row < m else 0
                    for cl in range(8):
                        actual_col = c * 8 + cl
                        b_val = _to_int8(_unpack_b_beat(b_pkt, c, cl)) if actual_col < n else 0
                        acc[r, c, rl, cl] += a_val * b_val

    # Add bias + saturate
    result = np.zeros((m, n), dtype=np.int8)
    for actual_row in range(m):
        r = actual_row // 8
        rl = actual_row % 8
        for actual_col in range(n):
            c = actual_col // 8
            cl = actual_col % 8
            val = acc[r, c, rl, cl] + bias_vals[actual_col]
            result[actual_row, actual_col] = np.clip(val, -128, 127).astype(np.int8)

    return result


class TestCppModelArithmetic:
    """Validate the C++ model arithmetic against numpy golden reference."""

    CASES = [
        (8, 8, 8),
        (16, 8, 8),
        (8, 8, 16),
        (16, 16, 16),
        (5, 5, 5),
        (14, 6, 6),
        (12, 10, 10),
    ]

    @pytest.mark.parametrize("m,k,n", CASES)
    def test_cpp_model_matches_golden(self, m, k, n):
        import numpy as np
        rng = np.random.default_rng(42)
        # Small values to avoid saturation in most cases
        max_val = max(1, int((127 / max(k, 1)) ** 0.5))
        A = rng.integers(-max_val, max_val + 1, size=(m, k), dtype=np.int8)
        # weights stored as [n, k] (matches C++ testbench layout)
        W = rng.integers(-max_val, max_val + 1, size=(n, k), dtype=np.int8)
        bias = rng.integers(-4, 5, size=(n,), dtype=np.int8)

        # Golden: standard GEMM + bias + saturate
        golden = A.astype(np.int32) @ W.astype(np.int32).T + bias.astype(np.int32)
        golden = np.clip(golden, -128, 127).astype(np.int8)

        # C++ model simulation
        result = _cpp_model_gemm(m, k, n, A, W, bias)

        mismatches = np.sum(result != golden)
        assert mismatches == 0, f"{m}x{k}x{n}: {mismatches} mismatches"

    def test_saturation_boundaries(self):
        """Verify saturation at int8 boundaries."""
        import numpy as np
        m, k, n = 2, 1, 2
        # A * W^T will overflow int8
        A = np.array([[100], [100]], dtype=np.int8)
        W = np.array([[100], [100]], dtype=np.int8)
        bias = np.array([0, 0], dtype=np.int8)

        golden = np.clip(A.astype(np.int32) @ W.astype(np.int32).T + bias.astype(np.int32), -128, 127).astype(np.int8)
        result = _cpp_model_gemm(m, k, n, A, W, bias)

        assert np.array_equal(result, golden)
        # 100*100 = 10000 → clamped to 127
        assert result[0, 0] == 127

    def test_bias_only_no_weights(self):
        """K=0 edge case: result should be saturated bias only."""
        import numpy as np
        m, k, n = 2, 0, 3
        A = np.zeros((m, k), dtype=np.int8)
        W = np.zeros((n, k), dtype=np.int8)
        bias = np.array([-2, 5, 127], dtype=np.int8)

        golden = np.tile(bias.astype(np.int32), (m, 1))
        golden = np.clip(golden, -128, 127).astype(np.int8)

        result = _cpp_model_gemm(m, k, n, A, W, bias)
        assert np.array_equal(result, golden)

    def test_multi_tile_bias_correctness(self):
        """Bias unpack uses col*64 offset — verify for multi-column grids."""
        import numpy as np
        m, k, n = 2, 1, 10  # gc=2
        A = np.zeros((m, k), dtype=np.int8)
        W = np.zeros((n, k), dtype=np.int8)
        bias = np.arange(n, dtype=np.int8)

        golden = np.tile(bias.astype(np.int32), (m, 1))
        golden = np.clip(golden, -128, 127).astype(np.int8)

        result = _cpp_model_gemm(m, k, n, A, W, bias)
        assert np.array_equal(result, golden)
        # Verify bias columns 8 and 9 are correct (these are in tile 1)
        assert result[0, 8] == 8
        assert result[0, 9] == 9
