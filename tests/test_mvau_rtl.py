"""RTL-level (xsim) regression for the mvau target -- see
src/targets/mvau/run_rtl_tests.py for the harness itself (SV testbench
generation + xvlog/xelab/xsim invocation). Skips cleanly when xvlog/xelab/xsim
are not available (on PATH, or via sourcing Vitis's settings64.sh)."""
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from targets.mvau import run_rtl_tests as _rt  # noqa: E402


@pytest.fixture(scope="module")
def xsim_env():
    env = _rt._tools_env()
    if env is None:
        pytest.skip("xvlog/xelab/xsim not on PATH and Vitis settings64.sh unavailable")
    return env


@pytest.mark.parametrize("case_name", list(_rt.CASES))
def test_mvau_rtl_case(xsim_env, case_name):
    ok, detail = _rt.run_case(case_name, seed=1, keep=False, env=xsim_env)
    assert ok, f"{case_name}: {detail.get('result_line', detail.get('log', ''))[-2000:]}"
