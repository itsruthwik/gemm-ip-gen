import sys
import shutil
import subprocess
from pathlib import Path

from generate_catapult_pkg import (
    _is_ac_integer_type,
    _normalize_config_items,
    gen_combined_header,
    gen_public_header,
    generate_catapult_pkg,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "tensor-slice"))
from generate_verilog_grid import generate_grid_verilog


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
    assert by_name["dense1"]["k"] == 8
    assert by_name["dense1"]["n"] == 4
    assert by_name["query"]["interface"] == "array"
    assert by_name["query"]["protocol"]["kind"] == "catapult_ccore_array"
    assert by_name["query"]["gemm_ip_index"] == 17


def test_combined_header_emits_stream_array_and_layer_id_dispatch():
    items = [
        {"name": "dense1", "m": 1, "k": 8, "n": 4, "interface": "stream", "gemm_ip_index": 3},
        {"name": "query", "m": 4, "k": 8, "n": 4, "interface": "array", "gemm_ip_index": 7},
    ]

    header = gen_combined_header(items)

    assert "void gemm_ip_stream(" in header
    assert "void gemm_ip_array(" in header
    assert "CONFIG_T::gemm_ip_id == 3" in header
    assert "CONFIG_T::gemm_ip_id == 7" in header
    assert "dense1_gemm_ip_stream" in header
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


def test_generated_rtl_uses_fifo_hold_protocol():
    rtl = generate_grid_verilog(4, 8, 4, "query_core")

    assert "wire output_take = en & (out_count != 0);" in rtl
    assert "out_fifo[out_wr_ptr] <= row_mux;" in rtl
    assert "if (out_count != 0) begin" in rtl
    assert "out_rd_ptr <= out_rd_ptr + 1;" in rtl
    assert "endcase" in rtl


def test_generated_rtl_holds_output_until_en(tmp_path):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        return

    rtl_path = tmp_path / "query_core.v"
    stub_path = tmp_path / "tensor_slice_stub.v"
    tb_path = tmp_path / "tb_sparse_en.v"
    sim_path = tmp_path / "sim.out"

    rtl_path.write_text(generate_grid_verilog(4, 8, 4, "query_core"))
    stub_path.write_text(
        r"""
module tensor_slice_int8(
    input clk, input reset, input pe_reset,
    input start_mat_mul, output done_mat_mul,
    input [63:0] a_data,
    input [63:0] b_data,
    input [63:0] a_data_in,
    input [63:0] b_data_in,
    output [63:0] a_data_out,
    output [63:0] b_data_out,
    output reg [127:0] c_data_out,
    output reg c_data_available,
    input [7:0] validity_mask_a_rows,
    input [7:0] validity_mask_a_cols_b_rows,
    input [7:0] validity_mask_b_cols,
    input [1:0] slice_dtype,
    input slice_mode,
    input [2:0] op,
    input preload,
    input no_rounding,
    input [7:0] final_mat_mul_size,
    input [4:0] a_loc,
    input [4:0] b_loc
);
    reg active;
    reg [7:0] cycle;
    reg [7:0] row;

    assign a_data_out = 64'd0;
    assign b_data_out = 64'd0;
    assign done_mat_mul = active && row == 8;

    always @(posedge clk) begin
        if (reset || pe_reset) begin
            active <= 1'b0;
            cycle <= 8'd0;
            row <= 8'd0;
            c_data_available <= 1'b0;
            c_data_out <= 128'd0;
        end else if (start_mat_mul) begin
            active <= 1'b1;
            cycle <= 8'd0;
            row <= 8'd0;
            c_data_available <= 1'b0;
            c_data_out <= 128'd0;
        end else if (active) begin
            cycle <= cycle + 1'b1;
            if (cycle >= 3 && row < 8) begin
                c_data_available <= 1'b1;
                c_data_out <= {120'd0, row};
                row <= row + 1'b1;
            end else begin
                c_data_available <= 1'b0;
            end
            if (row == 8) begin
                active <= 1'b0;
            end
        end else begin
            c_data_available <= 1'b0;
        end
    end
endmodule
"""
    )
    tb_path.write_text(
        r"""
`timescale 1ns/1ps

module tb_sparse_en;
    reg clk = 0;
    reg rst = 1;
    reg en = 1;
    reg in_valid = 0;
    reg [63:0] a_rows = 0;
    reg [63:0] b_cols = 0;
    reg [63:0] bias_cols = 0;
    reg preload_valid = 0;
    wire [127:0] c_row;
    wire out_valid;
    wire out_last;

    integer cycle;
    integer take_count;
    integer fail_count;
    reg [127:0] held_row;
    reg holding;

    query_core dut(
        .clk(clk),
        .rst(rst),
        .en(en),
        .a_rows(a_rows),
        .b_cols(b_cols),
        .bias_cols(bias_cols),
        .preload_valid(preload_valid),
        .in_valid(in_valid),
        .c_row(c_row),
        .out_valid(out_valid),
        .out_last(out_last)
    );

    always #5 clk = ~clk;

    initial begin
        cycle = 0;
        take_count = 0;
        fail_count = 0;
        holding = 0;
        repeat (2) @(posedge clk);
        rst = 0;
        in_valid = 1;
        en = 1;
        repeat (8) begin
            a_rows = a_rows + 64'd1;
            b_cols = b_cols + 64'd3;
            @(posedge clk);
        end
        in_valid = 0;
        repeat (80) @(posedge clk);
        if (take_count != 8) begin
            $display("FAIL take_count got %0d expected 8", take_count);
            fail_count = fail_count + 1;
        end
        if (fail_count == 0) begin
            $display("SPARSE_EN_PROTOCOL_PASSED");
        end else begin
            $display("SPARSE_EN_PROTOCOL_FAILED");
        end
        $finish;
    end

    always @(posedge clk) begin
        if (rst) begin
            cycle <= 0;
        end else begin
            cycle <= cycle + 1;
            if (take_count >= 8) begin
                en <= 1'b0;
            end else if (cycle > 11) begin
                en <= (cycle % 3) != 0;
            end

            if (out_valid && !en) begin
                if (!holding) begin
                    held_row <= c_row;
                    holding <= 1'b1;
                end else if (c_row !== held_row) begin
                    $display("FAIL c_row changed while en low at cycle %0d", cycle);
                    fail_count = fail_count + 1;
                end
            end

            if (out_valid && en) begin
                holding <= 1'b0;
                if (c_row[7:0] !== take_count[7:0]) begin
                    $display("FAIL row got %0d expected %0d", c_row[7:0], take_count);
                    fail_count = fail_count + 1;
                end
                if (out_last !== (take_count == 7)) begin
                    $display("FAIL out_last got %0d at take %0d", out_last, take_count);
                    fail_count = fail_count + 1;
                end
                take_count = take_count + 1;
            end
        end
    end
endmodule
"""
    )

    compile_res = subprocess.run(
        ["iverilog", "-g2012", "-o", str(sim_path), str(tb_path), str(rtl_path), str(stub_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert compile_res.returncode == 0, compile_res.stdout + compile_res.stderr

    sim_res = subprocess.run(["vvp", str(sim_path)], cwd=tmp_path, text=True, capture_output=True)
    assert sim_res.returncode == 0, sim_res.stdout + sim_res.stderr
    assert "SPARSE_EN_PROTOCOL_PASSED" in sim_res.stdout


def test_generated_rtl_explicit_bias_preload(tmp_path):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        return

    rtl_path = tmp_path / "bias_core.v"
    tb_path = tmp_path / "tb_bias_core.v"
    sim_path = tmp_path / "sim_bias.out"

    rtl_path.write_text(generate_grid_verilog(4, 8, 4, "bias_core"))
    tb_path.write_text(
        r"""
`timescale 1ns/1ps

module tb_bias_core;
    reg clk = 0;
    reg rst = 1;
    reg en = 1;
    reg in_valid = 0;
    reg preload_valid = 0;
    reg [63:0] a_rows = 0;
    reg [63:0] b_cols = 0;
    reg [63:0] bias_cols = 0;
    wire [127:0] c_row;
    wire out_valid;
    wire out_last;

    integer rows = 0;
    integer fail_count = 0;

    bias_core dut(
        .clk(clk),
        .rst(rst),
        .en(en),
        .a_rows(a_rows),
        .b_cols(b_cols),
        .bias_cols(bias_cols),
        .preload_valid(preload_valid),
        .in_valid(in_valid),
        .c_row(c_row),
        .out_valid(out_valid),
        .out_last(out_last)
    );

    always #5 clk = ~clk;

    initial begin
        repeat (2) @(posedge clk);
        rst = 0;

        // Preload +5 into output columns 0..3 for K cycles.
        bias_cols = 64'h0000000005050505;
        preload_valid = 1;
        repeat (8) @(posedge clk);
        preload_valid = 0;

        // Feed K cycles: valid rows 0..3 are all 1, valid cols 0..3 are all 2.
        // Expected valid result: 8 * (1 * 2) + 5 = 21 = 0x15.
        in_valid = 1;
        a_rows = 64'h0000000001010101;
        b_cols = 64'h0000000002020202;
        repeat (8) @(posedge clk);
        in_valid = 0;
        a_rows = 0;
        b_cols = 0;

        repeat (80) @(posedge clk);
        if (fail_count == 0 && rows == 8) begin
            $display("GENERATED_RTL_BIAS_PRELOAD_PASSED");
        end else begin
            $display("GENERATED_RTL_BIAS_PRELOAD_FAILED rows=%0d fail_count=%0d", rows, fail_count);
        end
        $finish;
    end

    always @(posedge clk) begin
        if (out_valid) begin
            if (rows < 4) begin
                if (c_row[31:0] !== 32'h15151515) begin
                    $display("FAIL valid row %0d got %h", rows, c_row[31:0]);
                    fail_count = fail_count + 1;
                end
            end else begin
                if (c_row[31:0] !== 32'h00000000) begin
                    $display("FAIL padded row %0d got %h", rows, c_row[31:0]);
                    fail_count = fail_count + 1;
                end
            end
            rows = rows + 1;
        end
    end
endmodule
"""
    )

    compile_res = subprocess.run(
        ["iverilog", "-g2012", "-o", str(sim_path), str(tb_path), str(rtl_path), "tensor-slice/tensor_slice_int8.v"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
    )
    assert compile_res.returncode == 0, compile_res.stdout + compile_res.stderr

    sim_res = subprocess.run(["vvp", str(sim_path)], text=True, capture_output=True)
    assert sim_res.returncode == 0, sim_res.stdout + sim_res.stderr
    assert "GENERATED_RTL_BIAS_PRELOAD_PASSED" in sim_res.stdout


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


def test_output_assignment_defaults_to_value_when_no_precision(tmp_path):
    """When ``output_precision`` is not provided the header uses ``value`` (no .to_int())."""
    generate_catapult_pkg(4, 8, 4, "test_default", tmp_path)
    header = (tmp_path / "test_default" / "test_default_gemm_ip.h").read_text()
    lines = header.split('\n')
    found_value_to_int_in_output = False
    for line in lines:
        if '>(value.to_int())' in line and 'out_pack' in line:
            found_value_to_int_in_output = True
    assert not found_value_to_int_in_output, (
        "Default (no precision) header should not use value.to_int() in output assignment"
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
    assert "bias_packed" in h, "Stream wrapper must contain bias_packed"
    assert "gemm.run(preload_a_rows, preload_b_cols, bias_packed, preload_valid, preload_in_valid, c_row, v, l)" in h, \
        "Stream wrapper must issue explicit preload calls"
    assert "gemm.run(last_a_rows, last_b_cols, bias_packed, feed_preload_valid, feed_valid, c_row, v, l)" in h, \
        "Stream wrapper feed call must pass bias_packed"
    assert "gemm.run(last_a_rows, last_b_cols, bias_packed, drain_preload_valid, drain_valid, c_row, v, l)" in h, \
        "Stream wrapper drain call must pass bias_packed"

    # Array wrapper
    generate_catapult_pkg(4, 8, 4, "test_a", tmp_path, interface="array")
    h = (tmp_path / "test_a" / "test_a_gemm_ip.h").read_text()
    assert "bias_packed" in h, "Array wrapper must contain bias_packed"
    assert "gemm.run(preload_a_rows, preload_b_cols, bias_packed, preload_valid, preload_in_valid, c_row, v, l)" in h, \
        "Array wrapper must issue explicit preload calls"
    assert "gemm.run(last_a_rows, last_b_cols, bias_packed, feed_preload_valid, feed_valid, c_row, v, l)" in h, \
        "Array wrapper feed call must pass bias_packed"
    assert "gemm.run(last_a_rows, last_b_cols, bias_packed, drain_preload_valid, drain_valid, c_row, v, l)" in h, \
        "Array wrapper drain call must pass bias_packed"

