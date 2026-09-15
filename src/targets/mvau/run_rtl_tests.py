#!/usr/bin/env python3
"""RTL-level (xsim) regression for the mvau target's generated shim RTL.

Modeled on tensor_slice/run_rtl_tests.py, but the mvau shim instantiates FINN's
``mvu_vvu_axi`` (which pulls in Vivado unisim primitives when not forced
behavioral, and its cosim was previously only ever exercised via Vitis's own
UVM-heavy ``vitis-run --tcl run_vitis.tcl`` cosim flow -- see package.py's
``run_vitis_smoke``). This harness instead drives the raw shim RTL with a
small hand-written self-checking SV testbench (rtl_sv_tb.py) and Xilinx's own
xvlog/xelab/xsim directly, the way tensor_slice drives its combined core with
iverilog -- much cheaper than a full HLS csim+csynth+cosim round trip per case.

xvlog/xelab/xsim invocation flags below were harvested from a real
``vitis-run --tcl run_vitis.tcl --mode hls`` cosim run kept under
temp_space/mvau-spike (MVAU_SPIKE PASS) and a hand probe under
temp_space/mvau-rtlsim: xvlog analyzes glbl.v (and our sources) into the
``work`` library; xelab elaborates ``tb glbl`` with
``--timescale 1ns/1ps -relax -L unisims_ver -L unimacro_ver -L secureip``;
xsim runs the elaborated snapshot in batch mode with ``-R``.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETS = HERE.parent
SRC = TARGETS.parent

from . import geometry as _geom
from . import rtl as _rtl
from . import package as _pkg
from . import rtl_sv_tb as _tb

REPO_ROOT = SRC.parent.parent  # ./gemm-ip-gen/src -> gemm-ip-gen -> atlas/
WORK_ROOT = REPO_ROOT / "temp_space" / "mvau_rtl_tests"

_COMMON = dict(weight_precision="fixed<8,4>", input_precision="fixed<8,4>",
              output_precision="fixed<16,6>", accum_precision="fixed<20,8>",
              part="xcve2802-vsvh1760-2MP-e-S", clock_period_ns=5)

VITIS_SETTINGS = "/mnt/vault1/tools/AMD/Vitis/2024.1/settings64.sh"

N_NODES = 4  # invocations/nodes per case (keeps xsim runtime small)


def _tools_env():
    """Return an environ dict with xvlog/xelab/xsim on PATH, sourcing Vitis's
    settings64.sh in a subshell if they are not already available."""
    if shutil.which("xvlog") and shutil.which("xelab") and shutil.which("xsim"):
        return os.environ.copy()
    if not Path(VITIS_SETTINGS).is_file():
        return None
    probe = subprocess.run(
        f"source {VITIS_SETTINGS} >/dev/null 2>&1 && env",
        shell=True, executable="/bin/bash", text=True, capture_output=True)
    if probe.returncode != 0:
        return None
    env = {}
    for line in probe.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    if shutil.which("xvlog", path=env.get("PATH")) is None:
        return None
    return env


def _find_case_plan(shape, **kw):
    return _geom.resolve_plan(shape, **_COMMON, **kw)


def _round_bits(n):
    return max(1, (n - 1).bit_length())


# ── case builders: each returns (workdir, module_name, sv_files, kind, tb_args) ──

def _build_ws_case(work, name, shape, seed, backpressure=True, **plan_kw):
    plan = _find_case_plan(shape, **plan_kw)
    t = plan["tile"]
    N, K, K_pad = plan["n"], plan["k"], plan["k_pad"]
    WW, AW, outW = t["weight_width"], t["activation_width"], t["output_width"]
    signed = bool(t["signed_activations"])
    shift = plan["product_frac"] - plan["output_frac"]

    B = _tb.synth_weights(N, K_pad, WW)   # [K_pad][N]; K_pad>=K, extra rows unused (K real)
    NTILE = plan["n_tile"]                # padded per-tile width (>= N; PE must divide it)
    k_tiles = int(plan.get("k_tiles", 1) or 1)
    MW = t["mw"]                          # per-K-tile row count (K_pad / k_tiles)
    init_files = []
    if k_tiles > 1:
        for i in range(k_tiles):
            dat_path = work / f"{name}_w{i}.dat"
            dat_path.write_text(_pkg._wpack.pack_memstream_hex(
                [[(B[i * MW + kk][oo] if oo < N else 0) for oo in range(NTILE)]
                 for kk in range(MW)],
                NTILE, MW, t["pe"], t["simd"], WW, word_bits=t["weight_stream_width_ba"]))
            init_files.append(str(dat_path))
    else:
        dat_path = work / f"{name}_w.dat"
        dat_path.write_text(_pkg._wpack.pack_memstream_hex(
            [[(B[kk][oo] if oo < N else 0) for oo in range(NTILE)] for kk in range(K_pad)],
            NTILE, K_pad, t["pe"], t["simd"], WW, word_bits=t["weight_stream_width_ba"]))
        init_files.append(str(dat_path))

    module_name = f"{name}_core"
    core_v = _rtl.generate_shim(shape, module_name=module_name, force_behavioral=True,
                                tile=t, weights_in_core=True, init_files=init_files,
                                n_tiles=1, k_tiles=k_tiles,
                                bias_codes=None, raw_k=K, raw_n=N)
    (work / f"{module_name}.v").write_text(core_v)

    a_words, exp_words = [], []
    for node in range(N_NODES):
        X = _tb.synth_activations(plan["num_input_vectors"], K, AW, signed, seed, node)
        for v in range(plan["num_input_vectors"]):
            a_words.append(_tb.pack_beat(X[v], AW))
            row = []
            for o in range(N):
                acc = sum(B[kk][o] * X[v][kk] for kk in range(K))
                row.append(_tb.requant_ref(acc, shift, outW))
            exp_words.append(_tb.pack_beat(row, outW))

    ab, pb = K * AW, N * outW
    a_dat, exp_dat = work / f"{name}_a.dat", work / f"{name}_exp.dat"
    _tb.write_dat(a_dat, a_words, (ab + 3) // 4)
    _tb.write_dat(exp_dat, exp_words, (pb + 3) // 4)

    sv_tb = _tb.generate_sv_tb("ws", module_name, ab, pb, len(a_words), len(exp_words),
                               N_NODES, str(a_dat), str(exp_dat), backpressure=backpressure)
    (work / "tb.sv").write_text(sv_tb)
    return module_name, [work / f"{module_name}.v"]


def _build_2op_case(work, name, shape, seed, kind, backpressure=True, fsm_debug=False,
                    mode=0, **plan_kw):
    """kind: 'reg' (SF=NF=1) or 'ms' (SF*NF>=2, untiled). ``mode`` (ms only): 0 = row-major
    (dynamic_load_2op MODE=0), 1 = col-major (MODE=1) -- selects the loader's narrow B
    beat layout/order (see rtl.py's generate_two_operand_shim)."""
    plan = _find_case_plan(shape, **plan_kw)
    t = plan["tile"]
    module_name = f"{name}_core"

    if kind == "reg":
        N, K = plan["n"], plan["k"]
        WW, AW, outW = t["weight_width"], t["activation_width"], t["output_width"]
        signed = bool(t["signed_activations"])
        shift = plan["product_frac"] - plan["output_frac"]
        core_v = _rtl.generate_two_operand_kt_shim(shape, module_name=module_name,
                                                    force_behavioral=True, tile=t, plan=plan,
                                                    k_tiles=1)
        (work / f"{module_name}.v").write_text(core_v)

        a_words, b_words, exp_words = [], [], []
        for node in range(N_NODES):
            Bm = _tb.synth_b_stream(K, N, WW, seed, node)
            X = _tb.synth_activations(plan["num_input_vectors"], K, AW, signed, seed, node)
            for kk in range(K):
                b_words.append(_tb.pack_beat(Bm[kk], WW))
            for v in range(plan["num_input_vectors"]):
                a_words.append(_tb.pack_beat(X[v], AW))
                row = []
                for o in range(N):
                    acc = sum(Bm[kk][o] * X[v][kk] for kk in range(K))
                    row.append(_tb.requant_ref(acc, shift, outW))
                exp_words.append(_tb.pack_beat(row, outW))
        ab, bb, pb = K * AW, N * WW, N * outW
    else:
        N, K_pad = plan["n"], plan["k_pad"]
        PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
        WW, AW, outW, ACCU = t["weight_width"], t["activation_width"], t["output_width"], t["accu_width"]
        signed = bool(t["signed_activations"])
        shift = plan["product_frac"] - plan["output_frac"]
        core_v = _rtl.generate_two_operand_shim(shape, module_name=module_name,
                                                force_behavioral=True, tile=t, plan=plan,
                                                mode=mode)
        (work / f"{module_name}.v").write_text(core_v)

        ab_bits = t["input_stream_width_ba"]
        # dynamic_load_2op's own NARROW input beat -- PE-wide (mode 0) or SIMD-wide
        # (mode 1) -- fed directly (no wide-beat gearbox at this RTL-level harness; that
        # lives in the HLS feed_b process for the real package, see package.py).
        lanes_raw = PE if mode == 0 else SIMD
        bb_bits = ((lanes_raw * WW) + 7) // 8 * 8
        pb_bits = t["output_stream_width_ba"]
        a_words, b_words, exp_words = [], [], []
        for node in range(N_NODES):
            Bm = _tb.synth_b_stream(K_pad, N, WW, seed, node)   # [K_pad][N]
            X = _tb.synth_activations(plan["num_input_vectors"], K_pad, AW, signed, seed, node)
            # B beats in the loader's own writer order (see rtl_static/dynamic_load_2op.sv):
            #   mode 0 (row-major): nf-fast/simd-mid/sf-slow, PE-wide beat = Bm[sf*SIMD+simd][nf*PE+pe]
            #   mode 1 (col-major, transposed): sf-fast/pe-mid/nf-slow, SIMD-wide beat = Bm[sf*SIMD+s][nf*PE+pe]
            if mode == 0:
                for sf in range(SF):
                    for simd in range(SIMD):
                        kk = sf * SIMD + simd
                        for nf in range(NF):
                            lane = [Bm[kk][nf * PE + pe] if nf * PE + pe < N else 0
                                    for pe in range(PE)]
                            b_words.append(_tb.pack_beat(lane, WW))
            else:
                for nf in range(NF):
                    for pe in range(PE):
                        oc = nf * PE + pe
                        for sf in range(SF):
                            lane = [Bm[sf * SIMD + s][oc] if oc < N else 0
                                    for s in range(SIMD)]
                            b_words.append(_tb.pack_beat(lane, WW))
            for v in range(plan["num_input_vectors"]):
                for sf in range(SF):
                    lane = X[v][sf * SIMD:(sf + 1) * SIMD]
                    a_words.append(_tb.pack_beat(lane, AW))
                for nf in range(NF):
                    row = []
                    for pe in range(PE):
                        oc = nf * PE + pe
                        if oc < N:
                            acc = sum(Bm[kk][oc] * X[v][kk] for kk in range(K_pad))
                            row.append(_tb.requant_ref(acc, shift, outW))
                        else:
                            row.append(0)
                    exp_words.append(_tb.pack_beat(row, outW))
        ab, bb, pb = ab_bits, bb_bits, pb_bits

    a_dat, b_dat, exp_dat = work / f"{name}_a.dat", work / f"{name}_b.dat", work / f"{name}_exp.dat"
    _tb.write_dat(a_dat, a_words, (ab + 3) // 4)
    _tb.write_dat(b_dat, b_words, (bb + 3) // 4)
    _tb.write_dat(exp_dat, exp_words, (pb + 3) // 4)

    sv_tb = _tb.generate_sv_tb("2op", module_name, ab, pb, len(a_words), len(exp_words),
                               N_NODES, str(a_dat), str(exp_dat), bb=bb,
                               b_beats=len(b_words), b_dat=str(b_dat),
                               backpressure=backpressure, fsm_debug=fsm_debug)
    (work / "tb.sv").write_text(sv_tb)
    return module_name, [work / f"{module_name}.v"]


CASES = {
    "a_plain_ws": lambda work, seed, **kw: _build_ws_case(
        work, "a", (4, 4, 4), seed, backpressure=kw.get("backpressure", True)),
    "b_padded_ws": lambda work, seed, **kw: _build_ws_case(
        work, "b", (2, 7, 5), seed, reuse_factor=2, fold_axis="kn",
        backpressure=kw.get("backpressure", True)),
    "c_sf2_ws": lambda work, seed, **kw: _build_ws_case(
        work, "c", (2, 8, 4), seed, pe=4, simd=4, backpressure=kw.get("backpressure", True)),
    "d_ktiled_ws": lambda work, seed, **kw: _build_ws_case(
        work, "d", (2, 16, 4), seed, pe=4, simd=4, k_tiles=2,
        backpressure=kw.get("backpressure", True)),
    "e_2op_register": lambda work, seed, **kw: _build_2op_case(
        work, "e", (2, 4, 4), seed, "reg", pe=4, simd=4,
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
    "f_2op_memstream": lambda work, seed, **kw: _build_2op_case(
        work, "f", (2, 8, 4), seed, "ms", pe=4, simd=4,
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
    "g_2op_qk_memstream": lambda work, seed, **kw: _build_2op_case(
        work, "g", (16, 12, 16), seed, "ms", pe=16, simd=6,
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
    "h_2op_col_major_memstream": lambda work, seed, **kw: _build_2op_case(
        work, "h", (2, 8, 4), seed, "ms", pe=4, simd=4, mode=1,
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
    "i_2op_col_major_qk_memstream": lambda work, seed, **kw: _build_2op_case(
        work, "i", (16, 12, 16), seed, "ms", pe=16, simd=6, mode=1,
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
    # ── resolve_fold-driven single-tile cases (VERIFY-ONLY additions) ──
    # Unlike e-i above (hand-picked pe=/simd=, bypassing resolve_fold), these specify
    # fold_axis + reuse_factor so geometry.resolve_fold actually derives (PE, SIMD),
    # the way a real manifest drives it. Each lands on the single-tile untiled
    # dynamic_load_2op shim (DEPTH=NF*SF>=2, k_tiles=1 default, n_tiles=1) -- see
    # geometry.fold_plan for the resolved numbers noted per case.
    "j_2op_foldn_memstream": lambda work, seed, **kw: _build_2op_case(
        # (m,k,n)=(4,4,8) RF=2 fold_axis=n -> PE=4 SIMD=4 SF=1 NF=2 (DEPTH=NF=2)
        work, "j", (4, 4, 8), seed, "ms", reuse_factor=2, fold_axis="n",
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
    "k_2op_foldn_col_major_memstream": lambda work, seed, **kw: _build_2op_case(
        work, "k", (4, 4, 8), seed, "ms", reuse_factor=2, fold_axis="n", mode=1,
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
    "l_2op_foldk_memstream": lambda work, seed, **kw: _build_2op_case(
        # (m,k,n)=(4,9,4) RF=2 fold_axis=k -> PE=4 SIMD=5 SF=2 NF=1 (DEPTH=SF=2, k_pad=10)
        work, "l", (4, 9, 4), seed, "ms", reuse_factor=2, fold_axis="k",
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
    "m_2op_foldk_col_major_memstream": lambda work, seed, **kw: _build_2op_case(
        work, "m", (4, 9, 4), seed, "ms", reuse_factor=2, fold_axis="k", mode=1,
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
    "n_2op_foldkn_memstream": lambda work, seed, **kw: _build_2op_case(
        # (m,k,n)=(4,9,8) RF=2 fold_axis=kn -> PE=4 SIMD=5 SF=2 NF=2 (DEPTH=4, k_pad=10)
        work, "n", (4, 9, 8), seed, "ms", reuse_factor=2, fold_axis="kn",
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
    "o_2op_foldkn_col_major_memstream": lambda work, seed, **kw: _build_2op_case(
        work, "o", (4, 9, 8), seed, "ms", reuse_factor=2, fold_axis="kn", mode=1,
        backpressure=kw.get("backpressure", True), fsm_debug=kw.get("fsm_debug", False)),
}

RTL_STATIC_DIR = HERE / "rtl_static"
STATIC_SOURCES = [
    "mvu_vvu_axi.sv", "replay_buffer.sv", "memstream.sv",
    "mvu_pkg.sv", "mvu.sv", "add_multi.sv", "mvu_vvu_8sx9_dsp58.sv",
    "dynamic_load_2op.sv",
]


def _run(cmd, cwd, env):
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)


def run_case(case_name, seed=1, keep=False, env=None, backpressure=True, fsm_debug=False):
    """Generate + xvlog/xelab/xsim one case. Returns (ok, detail_dict)."""
    work = WORK_ROOT / f"{case_name}_s{seed}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for s in STATIC_SOURCES:
        shutil.copy(RTL_STATIC_DIR / s, work / s)

    module_name, core_files = CASES[case_name](
        work, seed, backpressure=backpressure, fsm_debug=fsm_debug)

    glbl = None
    for cand in Path(env["XILINX_VIVADO"]).glob("data/verilog/src/glbl.v") if env.get("XILINX_VIVADO") else []:
        glbl = cand
        break
    if glbl is None:
        # search common install roots
        for root in ("/mnt/vault1/tools/AMD/2025.2", "/mnt/vault1/tools/AMD/Vivado/2024.1"):
            cand = Path(root) / "data" / "verilog" / "src" / "glbl.v"
            if cand.is_file():
                glbl = cand
                break
    sv_sources = [str(work / s) for s in STATIC_SOURCES] + [str(f) for f in core_files] + [str(work / "tb.sv")]
    if glbl:
        sv_sources.append(str(glbl))

    xvlog_cmd = ["xvlog", "-sv"] + sv_sources
    r = _run(xvlog_cmd, work, env)
    (work / "xvlog.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        return False, {"stage": "xvlog", "log": r.stdout + r.stderr}

    xelab_cmd = ["xelab", "tb"] + (["glbl"] if glbl else []) + [
        "--timescale", "1ns/1ps", "-relax",
        "-L", "unisims_ver", "-L", "unimacro_ver", "-L", "secureip", "-s", "tbsim"]
    r = _run(xelab_cmd, work, env)
    (work / "xelab.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        return False, {"stage": "xelab", "log": r.stdout + r.stderr}

    xsim_cmd = ["xsim", "tbsim", "-R"]
    r = _run(xsim_cmd, work, env)
    log = r.stdout + r.stderr
    (work / "xsim.log").write_text(log)

    result = {"stage": "xsim", "log": log}
    ok = False
    ready_by_node, done_by_node = {}, {}
    fsm_write, fsm_run, a_beats_log, p_beats_log = [], [], [], []
    for line in log.splitlines():
        if line.startswith("NODE ") and "ready=" in line:
            # NODE <i> ready=<c> -- the pipelined driver's ready stamp (per-node
            # input-side completion; may fire many cycles before that node's
            # own done stamp, since nodes are now allowed to overlap)
            parts = line.split()
            ready_by_node[int(parts[1])] = int(parts[2].split("=", 1)[1])
        elif line.startswith("NODE ") and "done=" in line:
            # NODE <i> done=<c> -- the ap_continue-ack process's done stamp
            parts = line.split()
            done_by_node[int(parts[1])] = int(parts[2].split("=", 1)[1])
        elif "CYCLES nodes=" in line:
            fields = dict(tok.split("=", 1) for tok in line.split()[1:])
            result["cycles"] = {
                "nodes": int(fields["nodes"]),
                "first_start": int(fields["first_start"]),
                "last_done": int(fields["last_done"]),
                "per_node": float(fields["per_node"]),
            }
        elif line.startswith("FSM_WRITE cyc="):
            fsm_write.append(int(line.split("cyc=", 1)[1]))
        elif line.startswith("FSM_RUN cyc="):
            fsm_run.append(int(line.split("cyc=", 1)[1]))
        elif line.startswith("A_BEAT cyc="):
            a_beats_log.append(int(line.split("cyc=", 1)[1]))
        elif line.startswith("P_BEAT cyc="):
            p_beats_log.append(int(line.split("cyc=", 1)[1]))
        if "TEST_RESULT:" in line:
            result["result_line"] = line.strip()
            ok = "PASS" in line and r.returncode == 0
            break
    nodes = [{"node": i, "ready": ready_by_node[i], "done": done_by_node.get(i)}
             for i in sorted(ready_by_node)]
    if nodes:
        result["node_cycles"] = nodes
        # steady-state per-node interval: ready[i+1]-ready[i] over the last few
        # nodes (skips the pipeline's fill-up transient on the first couple of
        # nodes, once MAX_INFLIGHT nodes are overlapped back-to-back).
        readys = [n["ready"] for n in nodes]
        if len(readys) >= 2:
            tail = readys[-min(3, len(readys)):]
            gaps = [b - a for a, b in zip(tail, tail[1:])]
            if "cycles" not in result:
                result["cycles"] = {}
            result["cycles"]["steady_state_interval"] = sum(gaps) / len(gaps)
            result["cycles"]["steady_state_gaps"] = gaps
    if nodes and (fsm_write or fsm_run or a_beats_log or p_beats_log):
        # fsm_debug: bucket each stamp by the [ready(node), ready(node+1)) interval
        # it falls in, so per-node WRITE/RUN entry + last A/P beat can be reported.
        breakdown = []
        for idx, n in enumerate(nodes):
            lo = n["ready"]
            hi = nodes[idx + 1]["ready"] if idx + 1 < len(nodes) else float("inf")
            def _first(lst, lo=lo, hi=hi):
                xs = [c for c in lst if lo <= c < hi]
                return xs[0] if xs else None
            def _last(lst, lo=lo, hi=hi):
                xs = [c for c in lst if lo <= c < hi]
                return xs[-1] if xs else None
            breakdown.append({
                "node": n["node"],
                "write_enter": _first(fsm_write),
                "run_enter": _first(fsm_run),
                "last_a_beat": _last(a_beats_log),
                "last_p_beat": _last(p_beats_log),
            })
        result["fsm_breakdown"] = breakdown
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return ok, result


def run(cases=None, seeds=None, keep=False):
    env = _tools_env()
    if env is None:
        print("ERROR: xvlog/xelab/xsim not found on PATH and could not source "
              f"{VITIS_SETTINGS}", file=sys.stderr)
        return 2
    cases = cases or list(CASES)
    seeds = seeds or [1]
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    failures = []
    for c in cases:
        for seed in seeds:
            ok, detail = run_case(c, seed=seed, keep=keep, env=env)
            tag = f"{c} seed={seed}"
            print(f"{'PASS' if ok else 'FAIL'} {tag}: {detail.get('result_line', detail.get('stage'))}")
            if not ok:
                print(detail["log"][-4000:])
                failures.append(tag)
    if failures:
        print(f"\n{len(failures)}/{len(cases) * len(seeds)} FAILED:")
        for t in failures:
            print(f"  {t}")
        return 1
    print(f"\nALL PASSED: {len(cases)} cases x {len(seeds)} seeds")
    return 0


def main():
    ap = argparse.ArgumentParser(description="RTL-level xsim test for the mvau shim")
    ap.add_argument("--cases", nargs="*", choices=list(CASES), help="subset of cases")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--keep", action="store_true", help="keep temp_space/mvau_rtl_tests/* dirs")
    args = ap.parse_args()
    return run(cases=args.cases, seeds=args.seeds, keep=args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
