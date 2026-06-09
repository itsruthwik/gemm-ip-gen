"""RTL simulation tests for Catapult and Vitis GEMM RTL using iverilog.

These tests generate RTL + self-checking testbenches, compile with iverilog,
and run vvp simulation.  Tests are skipped if iverilog is not available.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_src_dir = str(Path(__file__).resolve().parent.parent / "src")
_ts_dir = _src_dir + "/tensor-slice"
for p in [_src_dir, _ts_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_catapult_rtl import (
    generate_sim_verilog,
    generate_synth_verilog,
    generate_combined_core_verilog,
    generate_k_spatial_combined_core_verilog,
)
from generate_vitis_rtl import generate_vitis_sim_rtl, generate_vitis_synth_rtl
from generate_verilog_tb import generate_tb


def _run_sim_combined(rtl_src, tb_src, rtl_file, tb_file, extra_v=None, timeout=900):
    """Like _run_sim but passes -DSYNTHESIS to force the synth path."""
    rtl_file.write_text(rtl_src)
    tb_file.write_text(tb_src)
    sim_out = tb_file.with_suffix(".out")
    cmd = ["iverilog", "-g2012", "-o", str(sim_out), str(tb_file), str(rtl_file), "-DSYNTHESIS"]
    if extra_v:
        cmd.append(str(extra_v))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"iverilog timeout for {rtl_file.stem}")
    if r.returncode != 0:
        pytest.fail(f"iverilog failed: {r.stderr[:300]}")
    # No vvp needed — structural smoke only


def _have_iverilog():
    try:
        r = subprocess.run(["iverilog", "-V"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _have_vvp():
    try:
        r = subprocess.run(["vvp", "-V"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


HAVE_SIM = _have_iverilog() and _have_vvp()

# Chunked RTL contract with K>8 support.
DEFAULT_CASES = [
    pytest.param( 8,  8,  8, id="8x8x8"),
    pytest.param(16,  8,  8, id="16x8x8"),
    pytest.param( 8,  8, 16, id="8x8x16"),
    pytest.param(16,  8, 16, id="16x8x16"),
    pytest.param( 5,  8, 13, id="5x8x13"),
    pytest.param( 9,  8, 10, id="9x8x10"),
    # K > 8
    pytest.param( 8, 16,  8, id="8x16x8"),
    pytest.param( 8,  9,  8, id="8x9x8"),
    pytest.param(16, 16, 16, id="16x16x16"),
    pytest.param( 9, 17, 10, id="9x17x10"),
]


def _run_sim(rtl_src, tb_src, rtl_file, tb_file, extra_v=None, timeout=900):
    """Write RTL + TB, compile with iverilog, simulate with vvp."""
    rtl_file.write_text(rtl_src)
    tb_file.write_text(tb_src)
    sim_out = tb_file.with_suffix(".out")

    cmd = ["iverilog", "-g2012", "-o", str(sim_out), str(tb_file), str(rtl_file)]
    if extra_v:
        cmd.append(str(extra_v))

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"iverilog timeout for {rtl_file.stem} — compilation hung or exceeded budget")
    if r.returncode != 0:
        pytest.fail(f"iverilog failed: {r.stderr[:300]}")

    try:
        r = subprocess.run(["vvp", str(sim_out)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"vvp timeout for {rtl_file.stem} — simulation hung or exceeded budget")
    if r.returncode != 0:
        pytest.fail(f"vvp exit {r.returncode}: {r.stdout[-300:]}")
    if "ALL_PASS" not in r.stdout:
        pytest.fail(f"simulation did not report ALL_PASS: {r.stdout[-500:]}")
    if "FAILURES=" in r.stdout:
        for line in r.stdout.splitlines():
            if "FAILURES=" in line:
                pytest.fail(f"Simulation had failures: {line.strip()}")

    # Extract behavioral latency diagnostics.
    bh_starts, bh_dones, bh_iis = [], [], []
    for line in r.stdout.splitlines():
        if line.startswith("BEH_II="):
            bh_iis.append(int(line.split("=")[-1]))
        elif line.startswith("BEH_START"):
            bh_starts.append(int(line.split("=")[-1]))
        elif line.startswith("BEH_DONE"):
            bh_dones.append(int(line.split("=")[-1]))
    if bh_starts and len(bh_starts) == len(bh_dones):
        bh_lat = [d - s + 1 for d, s in zip(bh_dones, bh_starts)]
    else:
        bh_lat = []
    if bh_lat:
        parts = []
        parts.append(f"beh={bh_lat[0]}")
        if bh_iis:
            steady_ii = [x for x in bh_iis[1:] if x > 0]
            if steady_ii: parts.append(f"II={steady_ii[0]}")
        n = len(bh_lat)
        consistent = all(l == bh_lat[0] for l in bh_lat)
        print(f"\n    {rtl_file.stem}  {' '.join(parts)} cycles  ({n} vectors{', all consistent' if consistent else ''})")
    steady_ii = [x for x in bh_iis[1:] if x > 0]
    return {
        "latencies": bh_lat,
        "iis": bh_iis,
        "steady_ii": steady_ii,
        "stdout": r.stdout,
    }


# ── Catapult RTL simulation ──────────────────────────────────────────────────


@pytest.mark.skipif(not HAVE_SIM, reason="iverilog/vvp not available")
class TestCatapultRtlSim:
    @pytest.mark.parametrize("m,k,n", DEFAULT_CASES)
    def test_catapult_core_sim(self, m, k, n, tmp_path):
        mod = f"gemm_{m}x{k}x{n}_core"
        _run_sim(
            generate_combined_core_verilog(m, k, n, module_name=mod),
            generate_tb(m, k, n, module_name=mod, protocol="catapult", num_vectors=10),
            tmp_path / f"{mod}.v",
            tmp_path / f"tb_{mod}.v",
        )

    @pytest.mark.parametrize("m,k,n", DEFAULT_CASES)
    def test_catapult_back2back(self, m, k, n, tmp_path):
        mod = f"cat_b2b_{m}x{k}x{n}"
        _run_sim(
            generate_combined_core_verilog(m, k, n, module_name=mod),
            generate_tb(m, k, n, module_name=mod, protocol="catapult", num_vectors=2, back2back=True),
            tmp_path / f"{mod}.v", tmp_path / f"tb_{mod}.v",
        )

    @pytest.mark.parametrize("m,k,n,k_spatial", [
        pytest.param(8, 16, 8, 2, id="8x16x8-p2"),
        pytest.param(8, 24, 8, 3, id="8x24x8-p3"),
        pytest.param(16, 72, 8, 3, id="16x72x8-p3"),
        pytest.param(25, 81, 10, 3, id="25x81x10-p3"),
    ])
    def test_catapult_k_spatial_core_sim(self, m, k, n, k_spatial, tmp_path):
        mod = f"ksp_{m}x{k}x{n}_p{k_spatial}"
        full_k_spatial = k_spatial == (k + 7) // 8
        _run_sim(
            generate_k_spatial_combined_core_verilog(m, k, n, module_name=mod, k_spatial=k_spatial),
            generate_tb(m, k, n, module_name=mod, protocol="catapult", num_vectors=4, full_k_spatial=full_k_spatial),
            tmp_path / f"{mod}.v",
            tmp_path / f"tb_{mod}.v",
        )


TENSOR_SLICE_STUB = r"""
module tensor_slice_int8(
    input clk, input reset, input pe_reset,
    input start_mat_mul, output done_mat_mul,
    input [63:0] a_data,
    input [63:0] b_data,
    input [63:0] a_data_in,
    input [63:0] b_data_in,
    output [63:0] a_data_out,
    output [63:0] b_data_out,
    output [127:0] c_data_out,
    output c_data_available,
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
    assign done_mat_mul = start_mat_mul;
    assign a_data_out = a_data | a_data_in;
    assign b_data_out = b_data | b_data_in;
    assign c_data_out = 128'd0;
    assign c_data_available = 1'b0;
endmodule
"""


# ── Vitis RTL simulation ─────────────────────────────────────────────────────


@pytest.mark.skipif(not HAVE_SIM, reason="iverilog/vvp not available")
class TestVitisRtlSim:
    @pytest.mark.parametrize("m,k,n", DEFAULT_CASES)
    def test_vitis_wrapper_sim(self, m, k, n, tmp_path):
        mod = f"gemm_{m}x{k}x{n}"
        _run_sim(
            generate_vitis_sim_rtl(m, k, n, module_name=mod),
            generate_tb(m, k, n, module_name=mod, protocol="vitis", num_vectors=10),
            tmp_path / f"{mod}.v",
            tmp_path / f"tb_vitis_{m}x{k}x{n}.v",
        )

    @pytest.mark.parametrize("m,k,n", DEFAULT_CASES)
    def test_vitis_back2back(self, m, k, n, tmp_path):
        mod = f"vit_b2b_{m}x{k}x{n}"
        _run_sim(
            generate_vitis_sim_rtl(m, k, n, module_name=mod),
            generate_tb(m, k, n, module_name=mod, protocol="vitis", num_vectors=2),
            tmp_path / f"{mod}.v", tmp_path / f"tb_{mod}.v",
            timeout=1200,
        )

    @pytest.mark.parametrize("m,k,n,expected_latency,expected_ii", [
        pytest.param(8, 8, 8, 25, 8, id="8x8x8"),
        pytest.param(14, 6, 6, 29, 14, id="14x6x6-tail"),
        pytest.param(8, 16, 8, 33, 16, id="8x16x8-kgt8"),
    ])
    def test_vitis_behavioral_reports_lowered_ii(self, m, k, n, expected_latency, expected_ii, tmp_path):
        mod = f"vit_lowered_ii_{m}x{k}x{n}"
        result = _run_sim(
            generate_vitis_sim_rtl(m, k, n, module_name=mod),
            generate_tb(m, k, n, module_name=mod, protocol="vitis", num_vectors=3),
            tmp_path / f"{mod}.v",
            tmp_path / f"tb_{mod}.v",
            timeout=1200,
        )
        assert result["latencies"]
        assert result["latencies"][0] == expected_latency
        assert result["steady_ii"]
        assert all(ii == expected_ii for ii in result["steady_ii"])


# ── Synth RTL structural smoke only ───────────────────────────────────────────

@pytest.mark.skipif(not HAVE_SIM, reason="iverilog not available")
class TestSynthStructuralSmoke:
    @pytest.mark.parametrize("m,k,n", [
        pytest.param(8, 8, 8, id="synth-8x8x8"),
        pytest.param(16, 16, 16, id="synth-16x16x16"),
        pytest.param(9, 17, 10, id="synth-9x17x10"),
    ])
    def test_catapult_synth_structure(self, m, k, n, tmp_path):
        mod = f"cat_synth_{m}x{k}x{n}"
        rtl = generate_combined_core_verilog(m, k, n, module_name=mod)
        self._check_synth_rtl(rtl, m, k, n)
        self._compile_with_stub(rtl, mod, tmp_path, define_synth=True)

    def test_catapult_k_spatial_synth_structure(self, tmp_path):
        m, k, n, k_spatial = 16, 72, 8, 3
        mod = "cat_ksp_synth_16x72x8_p3"
        rtl = generate_k_spatial_combined_core_verilog(m, k, n, module_name=mod, k_spatial=k_spatial)
        assert "K_SPATIAL=3" in rtl
        assert "K_SPATIAL_PARTITION 0: chunks 0..2" in rtl
        assert "K_SPATIAL_PARTITION 1: chunks 3..5" in rtl
        assert "K_SPATIAL_PARTITION 2: chunks 6..8" in rtl
        assert rtl.count('(* black_box = "true" *) (* keep = "true" *) tensor_slice_int8') == 6
        assert "sat_int8_to_i16" in rtl
        self._compile_with_stub(rtl, mod, tmp_path, define_synth=True)

    @pytest.mark.parametrize("m,k,n", [
        pytest.param(8, 8, 8, id="synth-8x8x8"),
        pytest.param(16, 16, 16, id="synth-16x16x16"),
        pytest.param(9, 17, 10, id="synth-9x17x10"),
    ])
    def test_vitis_synth_structure(self, m, k, n, tmp_path):
        mod = f"vit_synth_{m}x{k}x{n}"
        rtl = generate_vitis_synth_rtl(m, k, n, module_name=mod)
        self._check_synth_rtl(rtl, m, k, n)
        self._compile_with_stub(rtl, mod, tmp_path)

    def test_vitis_partial_k_spatial_synth_structure(self, tmp_path):
        m, k, n, k_spatial = 16, 72, 8, 3
        mod = "vit_ksp_synth_16x72x8_p3"
        rtl = generate_vitis_synth_rtl(m, k, n, module_name=mod, gemm_k_spatial=k_spatial)
        assert "K_SPATIAL=3" in rtl
        assert "K_SPATIAL_PARTITION 0: chunks 0..2" in rtl
        assert "K_SPATIAL_PARTITION 1: chunks 3..5" in rtl
        assert "K_SPATIAL_PARTITION 2: chunks 6..8" in rtl
        assert rtl.count('(* black_box = "true" *) (* keep = "true" *) tensor_slice_int8') == 6
        assert "partial outputs are INT16" in rtl
        assert "sat_int8_to_i16" in rtl

    @staticmethod
    def _check_synth_rtl(rtl, m, k, n):
        grid_rows = (m + 7) // 8
        grid_cols = (n + 7) // 8
        k_chunks = (k + 7) // 8
        assert rtl.count('(* black_box = "true" *) (* keep = "true" *) tensor_slice_int8') == grid_rows * grid_cols
        assert "module tensor_slice_int8" not in rtl
        assert f"localparam integer K_CHUNKS = {k_chunks};" in rtl
        assert "current_k_mask" in rtl
        assert "current_k_size" in rtl
        assert "slice_pe_reset = slice_start && (chunk_idx == 16'd0)" in rtl
        assert ".validity_mask_a_cols_b_rows(current_k_mask)" in rtl

    @staticmethod
    def _compile_with_stub(rtl, mod, tmp_path, define_synth=False):
        rtl_file = tmp_path / f"{mod}.v"
        stub_file = tmp_path / "tensor_slice_stub.v"
        out_file = tmp_path / f"{mod}.out"
        rtl_file.write_text(rtl)
        stub_file.write_text(TENSOR_SLICE_STUB)
        cmd = ["iverilog", "-g2012", "-s", mod, "-o", str(out_file), str(rtl_file), str(stub_file)]
        if define_synth:
            cmd.insert(2, "-DSYNTHESIS")
        r = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert r.returncode == 0, r.stdout + r.stderr
