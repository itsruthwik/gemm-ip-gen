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

# Shape/plan-kwargs for each weight-stationary case, mirroring run_rtl_tests.CASES'
# lambdas -- needed here (independently of xsim) to compute M*SF*NF, the bound on
# the decoupled handshake's steady-state per-node interval with no idle gap
# between overlapped nodes.
_WS_CASE_PARAMS = {
    "a_plain_ws": ((4, 4, 4), {}),
    "b_padded_ws": ((2, 7, 5), dict(reuse_factor=2, fold_axis="kn")),
    "c_sf2_ws": ((2, 8, 4), dict(pe=4, simd=4)),
    "d_ktiled_ws": ((2, 16, 4), dict(pe=4, simd=4, k_tiles=2)),
}

# Same idea for the two-operand (dynamic_load_2op) cases: the bound also has to
# account for the loader's own beats/node (k_pad row-major / n col-major, one
# whole B row/column per cycle -- see generate_two_operand_shim), since a node
# can't finish before its B is fully loaded.
_2OP_CASE_PARAMS = {
    "e_2op_register": ((2, 4, 4), 0, dict(pe=4, simd=4)),
    "e2_2op_register_col_major": ((2, 4, 4), 1, dict(pe=4, simd=4)),
    "f_2op_memstream": ((2, 8, 4), 0, dict(pe=4, simd=4)),
    "h_2op_col_major_memstream": ((2, 8, 4), 1, dict(pe=4, simd=4)),
    "w_2op_nf2_row_major_av": ((8, 8, 16), 0, dict(pe=8, simd=8)),
    "x_2op_sf2_col_major_qk": ((8, 16, 8), 1, dict(pe=4, simd=16)),
}


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


@pytest.mark.parametrize("case_name", list(_WS_CASE_PARAMS))
def test_mvau_rtl_ws_steady_state_interval(xsim_env, case_name):
    """The decoupled handshake should overlap consecutive nodes with no idle
    gap: with backpressure off, the steady-state per-node interval (the gap
    between consecutive ap_ready pulses, once the pipeline has filled) should
    be at most M*SF*NF + 2 (the core's own back-to-back throughput plus a
    couple of cycles of handshake/registration slack) -- not the old
    IDLE/RUN FSM's M*SF*NF + ~7 (a full re-arm gap after every node)."""
    shape, plan_kw = _WS_CASE_PARAMS[case_name]
    plan = _rt._find_case_plan(shape, **plan_kw)
    t = plan["tile"]
    m, sf, nf = shape[0], t["sf"], t["nf"]
    bound = m * sf * nf + 2

    ok, detail = _rt.run_case(case_name, seed=1, keep=False, env=xsim_env, backpressure=False)
    assert ok, f"{case_name}: {detail.get('result_line', detail.get('log', ''))[-2000:]}"
    interval = detail["cycles"]["steady_state_interval"]
    print(f"{case_name}: M={m} SF={sf} NF={nf} steady_state_interval={interval} bound={bound}")
    assert interval <= bound, (
        f"{case_name}: steady-state per-node interval {interval} exceeds "
        f"M*SF*NF+2={bound} (gaps={detail['cycles']['steady_state_gaps']})")


@pytest.mark.parametrize("case_name", list(_2OP_CASE_PARAMS))
def test_mvau_rtl_2op_steady_state_interval(xsim_env, case_name):
    """Mirrors test_mvau_rtl_ws_steady_state_interval for the two-operand
    (dynamic_load_2op) shim: with backpressure off, the steady-state per-node
    interval should be at most max(M*SF*NF, load_beats) + 2, where load_beats
    is the loader's beats/node (k_pad row-major / n col-major -- one whole B
    row/column per cycle, not the old k_pad*NF / n*SF narrow-sub-beat cost)."""
    shape, mode, plan_kw = _2OP_CASE_PARAMS[case_name]
    plan = _rt._find_case_plan(shape, **plan_kw)
    t = plan["tile"]
    m, sf, nf = shape[0], t["sf"], t["nf"]
    load_beats = plan["k_pad"] if mode == 0 else plan["n"]
    bound = max(m * sf * nf, load_beats) + 2

    ok, detail = _rt.run_case(case_name, seed=1, keep=False, env=xsim_env, backpressure=False)
    assert ok, f"{case_name}: {detail.get('result_line', detail.get('log', ''))[-2000:]}"
    interval = detail["cycles"]["steady_state_interval"]
    print(f"{case_name}: M={m} SF={sf} NF={nf} load_beats={load_beats} "
          f"steady_state_interval={interval} bound={bound}")
    assert interval <= bound, (
        f"{case_name}: steady-state per-node interval {interval} exceeds "
        f"max(M*SF*NF, load_beats)+2={bound} (gaps={detail['cycles']['steady_state_gaps']})")
