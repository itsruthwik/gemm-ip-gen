import sys
import shutil
import subprocess
from pathlib import Path
import pytest

# Ensure the package is importable
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from gemm_ip.catapult import (
    _assert_core_port_widths,
    _is_ac_integer_type,
    _normalize_config_items,
    gen_combined_header,
    gen_public_header,
    generate_catapult_pkg,
)

# Ensure tensor-slice generators are importable
_ts_dir = str(Path(__file__).resolve().parent.parent / "src" / "tensor-slice")
if _ts_dir not in sys.path:
    sys.path.insert(0, _ts_dir)
from generate_catapult_rtl import generate_grid_verilog


def test_normalize_config_preserves_protocol_and_defaults_to_stream():
    cfg = {
        "dense1": {
            "type": "Dense",
            "n_in": 8,
            "n_out": 4,
            "gemm_m": 1,
            "gemm_k": 8,
            "gemm_n": 4,
        },
        "query": {
            "type": "EinsumDense",
            "n_in": 8,
            "n_out": 4,
            "gemm_m": 4,
            "gemm_k": 8,
            "gemm_n": 4,
            "interface": "array",
            "protocol": {"kind": "catapult_ccore_array"},
            "gemm_ip_index": 17,
        },
    }

    items = _normalize_config_items(cfg)
    by_name = {item["name"]: item for item in items}

    assert by_name["dense1"]["interface"] == "stream"
    assert by_name["dense1"]["gemm_k_spatial"] == 1
    assert by_name["dense1"]["k"] == 8
    assert by_name["dense1"]["n"] == 4
    assert by_name["query"]["interface"] == "array"
    assert by_name["query"]["protocol"]["kind"] == "catapult_ccore_array"
    assert by_name["query"]["gemm_ip_index"] == 17


def test_normalize_config_defaults_gemm_k_spatial_to_full_k_chunks():
    cfg = {
        "conv": {
            "n_in": 72,
            "n_out": 8,
            "gemm_m": 16,
            "gemm_k": 72,
            "gemm_n": 8,
        },
    }

    items = _normalize_config_items(cfg)

    assert items[0]["gemm_k_spatial"] == 9


def test_normalize_config_preserves_gemm_k_spatial():
    cfg = {
        "conv": {
            "n_in": 72,
            "n_out": 8,
            "gemm_m": 16,
            "gemm_k": 72,
            "gemm_n": 8,
            "gemm_k_spatial": 3,
        },
    }

    items = _normalize_config_items(cfg)

    assert items[0]["gemm_k_spatial"] == 3


def test_normalize_config_rejects_invalid_gemm_k_spatial():
    with pytest.raises(ValueError, match="gemm_k_spatial must be >= 1"):
        _normalize_config_items({"name": "bad", "m": 8, "k": 16, "n": 8, "gemm_k_spatial": 0})

    with pytest.raises(ValueError, match="exceeds K_CHUNKS"):
        _normalize_config_items({"name": "bad", "m": 8, "k": 16, "n": 8, "gemm_k_spatial": 3})


def test_combined_header_emits_stream_array_and_layer_id_dispatch():
    items = [
        {"name": "dense1", "m": 1, "k": 8, "n": 4, "interface": "stream", "gemm_ip_index": 3},
        {"name": "query", "m": 4, "k": 8, "n": 4, "interface": "array", "gemm_ip_index": 7},
    ]

    header = gen_combined_header(items)

    assert "void gemm_stream(" in header
    assert "void gemm_ip_stream_const_weights(" in header
    assert "void gemm_array(" in header
    assert "CONFIG_T::gemm_ip_id == 3" in header
    assert "CONFIG_T::gemm_ip_id == 7" in header
    assert "dense1_gemm_ip_stream" in header
    assert "dense1_gemm_ip_stream_const_weights" in header
    assert "query_gemm_ip_array" in header


def test_generate_array_package_uses_array_top(tmp_path):
    generate_catapult_pkg(4, 8, 4, "query", tmp_path, interface="array")

    inst_cpp = (tmp_path / "query" / "query_inst.cpp").read_text()
    tb_cpp = (tmp_path / "query" / "query_tb.cpp").read_text()
    header = (tmp_path / "query" / "query_gemm_ip.h").read_text()

    assert "void query_gemm_ip_array" in header
    assert "nnet::query_gemm_ip_array" in inst_cpp
    assert "res_t results[4]" in inst_cpp
    assert "nnet::query_gemm_ip_array" in tb_cpp


def test_generate_weightless_array_package(tmp_path):
    """io_parallel const-weight GEMM: array-in/out weightless entry, direct feed
    (no channel — a channel bridge to the stream entry hits Catapult HIER-11)."""
    import numpy as np
    B = np.zeros((8, 4), dtype=np.int8)  # baked weights [K, N]
    generate_catapult_pkg(4, 8, 4, "wq", tmp_path, interface="array", weight_matrix=B)

    header = (tmp_path / "wq" / "wq_gemm_ip.h").read_text()
    inst_cpp = (tmp_path / "wq" / "wq_inst.cpp").read_text()
    tb_cpp = (tmp_path / "wq" / "wq_tb.cpp").read_text()

    # The weightless array entry is emitted and takes A + bias only (no weight port).
    assert "void wq_gemm_ip_array_weightless" in header
    # Direct feed: gemm.run gets a_rows + bias, no b_cols packed operand.
    assert "gemm.run(a_rows_packed, bias_packed" in header
    # The standalone top and TB route to the weightless array entry with array ports.
    assert "nnet::wq_gemm_ip_array_weightless" in inst_cpp
    assert "res_t results[4]" in inst_cpp
    assert "nnet::wq_gemm_ip_array_weightless" in tb_cpp


def test_generate_weightless_stream_package(tmp_path):
    """io_stream const-weight GEMM: A-channel-in / res-channel-out weightless entry,
    no external weight port (weights live in the csim B_ROM / RTL wrapper ROM).
    Structural mirror of test_generate_weightless_array_package."""
    import numpy as np
    B = np.zeros((8, 4), dtype=np.int8)  # baked weights [K, N]
    generate_catapult_pkg(4, 8, 4, "sw", tmp_path, interface="stream", weight_matrix=B)

    header = (tmp_path / "sw" / "sw_gemm_ip.h").read_text()
    inst_cpp = (tmp_path / "sw" / "sw_inst.cpp").read_text()
    tb_cpp = (tmp_path / "sw" / "sw_tb.cpp").read_text()

    # The weightless stream entry is emitted and carries no external B beat / weight port.
    assert "void sw_gemm_ip_stream_weightless(" in header
    assert "b_beat" not in header
    assert "weight_cols" not in header
    # Direct feed: gemm.run gets a_rows + bias, no b_cols packed operand.
    assert "gemm.run(a_rows, bias_packed" in header
    # The standalone top and TB route to the weightless stream entry with channel ports.
    assert "nnet::sw_gemm_ip_stream_weightless" in inst_cpp
    assert "a_stream" in inst_cpp and "res_stream" in inst_cpp
    assert "nnet::sw_gemm_ip_stream_weightless" in tb_cpp


def test_combined_header_emits_array_weightless_dispatch():
    """A weightless array item routes to the gemm_ip_array_weightless dispatcher that
    hls4ml's nnet::gemm_array_weightless calls under GEMM_IP_HEADER."""
    items = [
        {"name": "wq", "m": 4, "k": 8, "n": 4, "interface": "array",
         "weights_in_core": True, "gemm_ip_index": 5},
    ]
    header = gen_combined_header(items)
    assert "void gemm_array_weightless(" in header
    assert "wq_gemm_ip_array_weightless<a_beat_T, bias_T, res_T, CONFIG_T>" in header


# ---------------------------------------------------------------------------
# Output precision / type-aware assignment tests
# ---------------------------------------------------------------------------

def test_is_ac_integer_type():
    """The helper correctly classifies integer vs fixed-point type strings."""
    assert _is_ac_integer_type("int<8>") is True
    assert _is_ac_integer_type("uint<8>") is True
    assert _is_ac_integer_type("ac_int<8,true>") is True
    assert _is_ac_integer_type("ac_uint<8>") is True
    assert _is_ac_integer_type("int<8,true>") is True
    assert _is_ac_integer_type(" int<8> ") is True  # whitespace tolerance

    assert _is_ac_integer_type("fixed<16,6,TRN,WRAP,0>") is False
    assert _is_ac_integer_type("ufixed<16,6,TRN,WRAP,0>") is False
    assert _is_ac_integer_type("ac_fixed<16,6,true>") is False
    assert _is_ac_integer_type("float<25,2,8,TRN>") is False
    assert _is_ac_integer_type(None) is False
    assert _is_ac_integer_type("") is False
    assert _is_ac_integer_type("ap_int<8>") is False  # not handled


def test_output_assignment_uses_to_int_for_integer_result(tmp_path):
    """Integer ``output_precision`` produces ``value.to_int()`` in the header."""
    generate_catapult_pkg(4, 8, 4, "test_int", tmp_path, output_precision="int<8>")
    header = (tmp_path / "test_int" / "test_int_gemm_ip.h").read_text()

    # Both the stream and array output assignment sites must use to_int()
    stream_matches = header.count("value.to_int()")
    assert stream_matches >= 2, (
        f"Expected at least 2 occurrences of 'value.to_int()' "
        f"in integer-result header, found {stream_matches}"
    )


def test_output_assignment_omits_to_int_for_fixed_result(tmp_path):
    """Fixed-point ``output_precision`` omits ``.to_int()`` from the output assignment."""
    generate_catapult_pkg(4, 8, 4, "test_fixed", tmp_path,
                          output_precision="fixed<16,6,TRN,WRAP,0>")
    header = (tmp_path / "test_fixed" / "test_fixed_gemm_ip.h").read_text()
    # The output assignment should be `>(value)` not `>(value.to_int())`
    # We check for the pattern `value_type>(value)` immediately before the semicolon
    # to distinguish from `{name}_to_gemm_int8(const src_T &value)`
    lines = header.split('\n')
    found_value_to_int_in_output = False
    for line in lines:
        if '>(value.to_int())' in line and 'out_pack' in line:
            found_value_to_int_in_output = True
    assert not found_value_to_int_in_output, (
        "Fixed-point result must not use value.to_int() in output assignment"
    )


def test_output_assignment_uses_to_int_when_no_precision(tmp_path):
    """When ``output_precision`` is not provided the result lane defaults to the
    INTEGER ``ac_int<16, true>`` typedef, so the output assignment must use
    ``value.to_int()`` — casting the ac_fixed accumulator straight to an ac_int
    lane does not compile under the AC datatypes (Catapult CRD-312)."""
    generate_catapult_pkg(4, 8, 4, "test_default", tmp_path)
    header = (tmp_path / "test_default" / "test_default_gemm_ip.h").read_text()
    out_assign_sites = [
        line for line in header.split('\n')
        if 'out_pack' in line and 'value' in line and 'static_cast' in line
    ]
    assert out_assign_sites, "expected output-assignment sites in the header"
    assert all('value.to_int()' in line for line in out_assign_sites), (
        "Default (integer result lane) header must use value.to_int() in every "
        "output assignment"
    )


def test_bias_cols_in_rtl_and_cpp_header(tmp_path):
    """Generated RTL and C++ header contain bias_cols; no + biases[col]; no preload_data."""
    generate_catapult_pkg(4, 8, 4, "test_bias", tmp_path)
    rtl_path = tmp_path / "test_bias" / "test_bias_core.v"
    header_path = tmp_path / "test_bias" / "test_bias_gemm_ip.h"

    rtl = rtl_path.read_text()
    header = header_path.read_text()

    # RTL: bias_cols/preload_valid ports must exist
    assert "bias_cols" in rtl, "RTL must contain bias_cols port"
    assert "preload_valid" in rtl, "RTL must contain preload_valid port"

    # RTL: must NOT contain preload_data
    assert ".preload_data" not in rtl, "RTL must not contain .preload_data"

    # C++ header: bias_cols in ccore run signature
    assert "bias_cols" in header, "C++ header must contain bias_cols"
    # C++ header: no + biases[col] post-processing
    assert "+ biases[col]" not in header, "C++ header must not contain + biases[col]"
    # C++ header: bias_packed packing loop exists
    assert "bias_packed" in header, "C++ header must contain bias_packed"


def test_stream_and_array_pass_bias_cols(tmp_path):
    """Both stream and array wrappers pass bias_cols into ccore run calls."""
    # Stream wrapper
    generate_catapult_pkg(4, 8, 4, "test_s", tmp_path, interface="stream")
    h = (tmp_path / "test_s" / "test_s_gemm_ip.h").read_text()
    # Check that bias_packed is passed to all gemm.run() calls
    assert "void test_s_gemm_ip_stream_const_weights" in h, "Stream wrapper must contain const-weight entry point"
    assert "bias_packed" in h, "Stream wrapper must contain bias_packed"
    # Back-to-back wrapper: bias is still added in the drain capture (never loaded
    # into the core), but preload_valid must NOT be a compile-time constant 0 — that
    # lets VTR/parmys prove the core's transaction_active (hence the tensor_slice
    # result path) dead and prune every slice. It pulses on each frame's leading beat
    # (p==0), a live per-frame signal that arms the core; this rides the per-frame
    # idle beat so it costs no extra cycle and keeps bias in the drain (bias_packed=0).
    assert "frame_preload = (in_feed && p == 0) ? 1 : 0" in h, \
        "Stream wrapper must derive a live per-frame preload pulse (not a constant)"
    assert "gemm.run(a_rows, b_cols, bias_packed, frame_preload, feed_valid, c_row, v, l)" in h, \
        "Stream wrapper feed call must pass bias_packed with the live frame_preload"

    # Array wrapper
    generate_catapult_pkg(4, 8, 4, "test_a", tmp_path, interface="array")
    h = (tmp_path / "test_a" / "test_a_gemm_ip.h").read_text()
    assert "bias_packed" in h, "Array wrapper must contain bias_packed"
    assert "feed_preload_valid = (step == 0) ? 1 : 0" in h, \
        "Array wrapper feed call must use conditional preload on step 0"
    assert "gemm.run(a_rows_packed, b_cols_packed, bias_packed, feed_preload_valid, feed_valid, c_row, v, l)" in h, \
        "Array wrapper feed call must pass bias_packed"
    assert "gemm.run(last_a_rows, last_b_cols, bias_packed, drain_preload_valid, drain_valid, c_row, v, l)" in h, \
        "Array wrapper drain call must pass bias_packed"


def test_stream_const_weights_prepacks_a_replay_without_read_a_rows(tmp_path):
    """Const-weight stream wrapper reads A rows in FEED and replays packed K chunks."""
    generate_catapult_pkg(16, 72, 8, "test_areplay", tmp_path, interface="stream", gemm_k_spatial=3)
    h = (tmp_path / "test_areplay" / "test_areplay_gemm_ip.h").read_text()

    assert "void test_areplay_gemm_ip_stream_const_weights" in h
    assert "READ_A_ROWS" not in h
    assert "a_rows_arr" not in h
    assert "ac_int<128, false> a_replay[9][16]" in h
    assert "if (kc == 0)" in h
    assert "a_beat_T a_beat = a_stream.read()" in h
    assert "ROW_PACK_DIRECT" in h
    assert "PREPACK_REPLAY" in h
    assert "ROW_PACK_REPLAY" in h
    assert "a_replay[replay_kc][t] = replay_rows" in h
    assert "a_rows = a_replay[kc][t]" in h
    assert "b_beat_T b_beat = weight_cols[t]" in h


def test_stream_const_weights_full_k_spatial_has_no_a_replay(tmp_path):
    generate_catapult_pkg(16, 72, 8, "test_fullk", tmp_path, interface="stream")
    h = (tmp_path / "test_fullk" / "test_fullk_gemm_ip.h").read_text()

    assert "void test_fullk_gemm_ip_stream_const_weights" in h
    assert "a_replay" not in h
    assert "ROW_PACK_FULL_KC" in h
    assert "COL_PACK_FULL_KC" in h
    # Back-to-back loop bound (n_frames=1): first_out + mr + 6 = 80 + 16 + 6 = 102.
    assert "RUN: for (int step = 0; step < 102; step++)" in h
    # Narrow full-K word: 64*k_chunks bits (one tile, all K chunks), independent of
    # grid_rows/grid_cols.  The wrapper RTL re-inserts the tile offset by beat index.
    assert "ac_int<576, false>  a_rows" in h
    assert "ac_int<576, false>  b_cols" in h


def test_array_full_k_spatial_feeds_logical_rows_once(tmp_path):
    generate_catapult_pkg(24, 24, 24, "test_array_fullk", tmp_path, interface="array")
    h = (tmp_path / "test_array_fullk" / "test_array_fullk_gemm_ip.h").read_text()

    assert "void test_array_fullk_gemm_ip_array" in h
    assert "RUN_ARRAY: for (int step = 0; step < 78; step++)" in h  # first_out(48) + m(24) + 6
    assert "ROW_PACK_ARRAY_FULL_KC" in h
    assert "COL_PACK_ARRAY_FULL_KC" in h
    assert "k_chunks * input_beats" not in h


def test_array_partial_k_spatial_keeps_time_tiled_feed(tmp_path):
    generate_catapult_pkg(24, 24, 24, "test_array_partialk", tmp_path, interface="array", gemm_k_spatial=1)
    h = (tmp_path / "test_array_partialk" / "test_array_partialk_gemm_ip.h").read_text()

    assert "void test_array_partialk_gemm_ip_array" in h
    assert "RUN_ARRAY: for (int step = 0; step < 102; step++)" in h  # first_out(72) + m(24) + 6
    assert "int kc = eff_step / 24" in h
    assert "ROW_PACK_ARRAY_FULL_KC" not in h


def test_single_k_chunk_wide_designs_use_chunked_layout(tmp_path):
    """k<=8 (k_chunks==1) must generate the CHUNKED layout end-to-end. The old
    full-mode selection emitted 64-bit wrapper words against the chunked grid's
    grid_{rows,cols}*64-bit ports; the simulator X-padded the gap and every
    row/col tile beyond the first was corrupted (multi-tile chained cosim bug)."""
    import re as _re

    # B-side multi-tile: the mlp3/cnn5 dense1 shape (8, 8, 16).
    generate_catapult_pkg(8, 8, 16, "test_n16k8", tmp_path)
    h = (tmp_path / "test_n16k8" / "test_n16k8_gemm_ip.h").read_text()
    v = (tmp_path / "test_n16k8" / "test_n16k8_core.v").read_text()
    assert _re.search(r"ac_int<128, false>\s*b_cols", h)   # grid_cols*64, not 64
    assert "ROW_PACK_FULL_KC" not in h                     # chunked feed loops
    assert _re.search(r"\[127:0\]\s*b_cols", v)

    # A-side multi-tile: (16, 8, 8) has the same hazard on a_rows.
    generate_catapult_pkg(16, 8, 8, "test_m16k8", tmp_path)
    h2 = (tmp_path / "test_m16k8" / "test_m16k8_gemm_ip.h").read_text()
    v2 = (tmp_path / "test_m16k8" / "test_m16k8_core.v").read_text()
    assert _re.search(r"ac_int<128, false>\s*a_rows", h2)  # grid_rows*64, not 64
    assert _re.search(r"\[127:0\]\s*a_rows", v2)


def test_assert_core_port_widths_rejects_layout_mismatch():
    """The generation-time cross-check must hard-fail on header/grid width drift
    instead of leaving a silent X-padded port for the simulator to find."""
    hdr = (
        "void run(ac_int<64, false>  a_rows, ac_int<64, false>  b_cols,\n"
        "         ac_int<128, false>  bias_cols, ac_int<256, false>& c_row);"
    )
    rtl = (
        "input  wire [63:0]   a_rows,\n"
        "input  wire [127:0]   b_cols,\n"
        "input  wire [127:0]   bias_cols,\n"
        "output reg  [255:0]   c_row"
    )
    with pytest.raises(RuntimeError, match="b_cols"):
        _assert_core_port_widths("bad", hdr, rtl)


def test_catapult_header_reads_int16_result_lanes(tmp_path):
    generate_catapult_pkg(4, 8, 4, "test_i16lane", tmp_path)
    h = (tmp_path / "test_i16lane" / "test_i16lane_gemm_ip.h").read_text()

    assert "c_row.template slc<16>(col_tile * 128 + col_local * 16)" in h
    assert "c_row.template slc<8>(col_tile * 128 + col_local * 8)" not in h


def test_generate_k_spatial_package_warns_and_emits_partitions(tmp_path, capsys):
    generate_catapult_pkg(16, 72, 8, "test_ksp", tmp_path, gemm_k_spatial=3)
    captured = capsys.readouterr()
    rtl = (tmp_path / "test_ksp" / "test_ksp_core.v").read_text()

    assert "gemm_k_spatial=3 is experimental" in captured.err
    assert "K_SPATIAL=3" in rtl
    assert "K_SPATIAL_PARTITION 0: chunks 0..2" in rtl
    assert "K_SPATIAL_PARTITION 1: chunks 3..5" in rtl
    assert "K_SPATIAL_PARTITION 2: chunks 6..8" in rtl
    assert rtl.count('(* black_box = "true" *) (* keep = "true" *) tensor_slice_int8') == 6
    assert "partial outputs are INT16" in rtl
    assert "sat_int8_to_i16" in rtl

def test_full_k_first_out_drops_for_deep_k():
    """Full-K-spatial feeds every K chunk in one max(M,N)-beat pass, so its
    first_out must fall below the chunked schedule wherever the feed (not the
    systolic K+N wave) dominates — the deep-K conv shapes. Wave-dominated
    shapes stay equal, and full-K may never be later than chunked."""
    from gemm_ip.catapult import latency_cycles

    def pair(m, k, n):
        gr, gc = (m + 7) // 8, (n + 7) // 8
        return (latency_cycles(m, k, n, gr, gc, full_k_spatial=False),
                latency_cycles(m, k, n, gr, gc, full_k_spatial=True))

    # Deep-K: feed-dominated, full-K strictly earlier.
    assert pair(16, 72, 8) == (144, 80)
    assert pair(25, 81, 10) == (275, 91)
    assert pair(10, 27, 10) == (40, 37)

    # Wave-dominated: equal, never worse.
    for shape in [(16, 16, 16), (10, 10, 10), (8, 24, 8)]:
        chunked, fullk = pair(*shape)
        assert fullk == chunked

    # k_chunks == 1: the branches coincide (fc/attn/mlp3 packages unchanged).
    for shape in [(8, 8, 8), (16, 8, 8), (8, 8, 16)]:
        chunked, fullk = pair(*shape)
        assert fullk == chunked


def test_full_k_first_out_not_below_feed_beats():
    """first_out >= total feed beats by construction: the first output row must
    land inside the DRAIN window, never inside FEED (FEED ignores out_valid)."""
    from gemm_ip.catapult import latency_cycles

    for (m, k, n) in [(16, 16, 16), (10, 10, 10), (8, 24, 8), (10, 27, 10),
                      (16, 72, 8), (25, 81, 10), (8, 8, 8), (64, 16, 8)]:
        gr, gc = (m + 7) // 8, (n + 7) // 8
        for fk in (False, True):
            beats = max(m, n) if fk else ((k + 7) // 8) * max(m, n)
            assert latency_cycles(m, k, n, gr, gc, full_k_spatial=fk) >= beats


def test_full_k_package_uses_full_k_drain_timing(tmp_path):
    """The generated full-K package must carry the full-K first_out coherently:
    behavioral FIRST_OUT localparam, C++ clk_cnt emit gate, and the DRAIN trip
    count all derive from the same value."""
    import re as _re

    # conv2d shape, full-K (k_chunks == 9): first_out = 16 + (72+8-16) = 80.
    generate_catapult_pkg(16, 72, 8, "test_fullk_t", tmp_path, gemm_k_spatial=9)
    h = (tmp_path / "test_fullk_t" / "test_fullk_t_gemm_ip.h").read_text()
    v = (tmp_path / "test_fullk_t" / "test_fullk_t_core.v").read_text()
    assert "localparam integer FIRST_OUT         = 80;" in v
    assert "scc >= 81" in h                           # first_out + 1
    # Back-to-back loop bound (n_frames=1): first_out + mr + 6 = 80 + 16 + 6 = 102.
    assert "RUN: for (int step = 0; step < 102; step++)" in h

    # Same shape, chunked: first_out = 144 (9 chunks x 16 beats, wave = 0).
    generate_catapult_pkg(16, 72, 8, "test_chk_t", tmp_path, gemm_k_spatial=1)
    h1 = (tmp_path / "test_chk_t" / "test_chk_t_gemm_ip.h").read_text()
    v1 = (tmp_path / "test_chk_t" / "test_chk_t_core.v").read_text()
    assert "localparam integer FIRST_OUT         = 144;" in v1
    assert "scc >= 145" in h1
    # Back-to-back loop bound (n_frames=1): first_out(144) + mr(16) + 6 = 166.
    assert "RUN: for (int step = 0; step < 166; step++)" in h1

    # k_chunks == 1 control keeps the legacy value on both paths.
    generate_catapult_pkg(8, 8, 8, "test_k1_t", tmp_path)
    v2 = (tmp_path / "test_k1_t" / "test_k1_t_core.v").read_text()
    assert "localparam integer FIRST_OUT         = 16;" in v2


def test_assert_core_first_out_rejects_mismatch():
    """The generation-time cross-check must hard-fail if the behavioral grid's
    FIRST_OUT drifts from latency_cycles (sim model vs wrapper drain window)."""
    from gemm_ip.catapult import _assert_core_first_out

    good = "localparam integer FIRST_OUT = 80;"
    _assert_core_first_out("ok", 16, 72, 8, 9, good)

    bad = "localparam integer FIRST_OUT = 144;"
    with pytest.raises(RuntimeError, match="FIRST_OUT"):
        _assert_core_first_out("bad", 16, 72, 8, 9, bad)
