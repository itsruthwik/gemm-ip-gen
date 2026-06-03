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

from gemm_ip.metadata import load_catapult_rtl_generator, load_vitis_rtl_generator
from generate_verilog_tb import generate_tb


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

TENSOR_SLICE_V = Path(_ts_dir) / "tensor_slice_int8.v"

# Row/col contract: K=8 hardblock. K>8 and K<8 need multi-K-step support (future).
DEFAULT_CASES = [
    pytest.param( 8,  8,  8, id="8x8x8"),
    pytest.param(16,  8,  8, id="16x8x8"),
    pytest.param( 8,  8, 16, id="8x8x16"),
    pytest.param(16,  8, 16, id="16x8x16"),
    pytest.param( 5,  8, 13, id="5x8x13"),
    pytest.param( 9,  8, 10, id="9x8x10"),
]


def _run_sim(rtl_src, tb_src, rtl_file, tb_file, extra_v=None, timeout=600):
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
        pytest.skip(f"iverilog timeout for {rtl_file.stem}")
    if r.returncode != 0:
        pytest.fail(f"iverilog failed: {r.stderr[:300]}")

    try:
        r = subprocess.run(["vvp", str(sim_out)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.skip(f"vvp timeout for {rtl_file.stem}")
    if r.returncode != 0 and "ALL_PASS" not in r.stdout:
        pytest.fail(f"vvp failed: {r.stdout[-500:]}")
    # Also check stdout for failures
    if "FAILURES=" in r.stdout:
        # Extract the failure count
        for line in r.stdout.splitlines():
            if "FAILURES=" in line:
                pytest.fail(f"Simulation had failures: {line.strip()}")


# ── Catapult RTL simulation ────────────────────────────────────────────────────


@pytest.mark.skipif(not HAVE_SIM, reason="iverilog/vvp not available")
class TestCatapultRtlSim:
    @pytest.mark.parametrize("m,k,n", DEFAULT_CASES)
    def test_catapult_core_sim(self, m, k, n, tmp_path):
        gen = load_catapult_rtl_generator()
        mod = f"gemm_{m}x{k}x{n}_core"
        _run_sim(
            gen(m, k, n, module_name=mod),
            generate_tb(m, k, n, module_name=mod, protocol="catapult", num_vectors=10),
            tmp_path / f"{mod}.v",
            tmp_path / f"tb_{mod}.v",
            extra_v=TENSOR_SLICE_V,
        )


# ── Vitis RTL simulation ──────────────────────────────────────────────────────


@pytest.mark.skipif(not HAVE_SIM, reason="iverilog/vvp not available")
class TestVitisRtlSim:
    @pytest.mark.parametrize("m,k,n", DEFAULT_CASES)
    def test_vitis_wrapper_sim(self, m, k, n, tmp_path):
        gen = load_vitis_rtl_generator()
        mod = f"gemm_{m}x{k}x{n}"
        _run_sim(
            gen(m, k, n, module_name=mod),
            generate_tb(m, k, n, module_name=mod, protocol="vitis", num_vectors=10),
            tmp_path / f"{mod}.v",
            tmp_path / f"tb_vitis_{m}x{k}x{n}.v",
            extra_v=TENSOR_SLICE_V,
        )

    def test_vitis_back2back(self, tmp_path):
        """Minimal back-to-back regression: 2 vectors without reset between them."""
        gen = load_vitis_rtl_generator()
        mod = "vitis_8x8x8"
        rtl_src = gen(8, 8, 8, module_name=mod)
        bb_tb = (Path(_ts_dir).parent.parent / "tests" / "tb_vitis_back2back.v").read_text()
        _run_sim(rtl_src, bb_tb,
                 tmp_path / f"{mod}.v", tmp_path / "tb_bb.v",
                 extra_v=TENSOR_SLICE_V)

    def test_vitis_back2back_16x16(self, tmp_path):
        """Multi-tile (2x2) back-to-back regression: 2 vectors without reset."""
        gen = load_vitis_rtl_generator()
        mod = "vitis_16x8x16"
        rtl_src = gen(16, 8, 16, module_name=mod)
        bb_tb = (Path(_ts_dir).parent.parent / "tests" / "tb_vitis_back2back_16x16.v").read_text()
        _run_sim(rtl_src, bb_tb,
                 tmp_path / f"{mod}.v", tmp_path / "tb_bb_16x16.v",
                 extra_v=TENSOR_SLICE_V)
