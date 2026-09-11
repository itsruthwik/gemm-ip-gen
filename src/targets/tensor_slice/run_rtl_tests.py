#!/usr/bin/env python3
"""
RTL-level test for the tensor-slice GEMM wrapper.

For a spread of GEMM sizes (tiled and non-8-multiple), this generates the
behavioral combined-core wrapper RTL (``rtl``) and a
self-checking testbench (``golden``), writes them under
``tb/generated/``, then compiles + runs each with Icarus Verilog and checks the
testbench's ``ALL_PASS`` marker.

This is the RTL-level regression: it exercises the wrapper protocol + the
behavioral GEMM datapath end-to-end. No external ``tensor_slice_int8`` IP is
needed — iverilog compiles without ``-DSYNTHESIS``, so the combined core's
``ifndef SYNTHESIS`` behavioral branch (which does the matmul directly) is the
one simulated.

Run it via the repo venv (numpy is needed by the generators):

    ./run_rtl_tests.sh                          # default size set x default seeds
    ./run_rtl_tests.sh --cases 8x8x8 16x16x16   # specific sizes
    ./run_rtl_tests.sh --seeds 1 7              # specific seeds
    ./run_rtl_tests.sh --keep                   # keep tb/generated/*.v for inspection
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
SRC = HERE.parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rtl import generate_combined_core_verilog, generate_k_spatial_combined_core_verilog
from golden import generate_tb
from geometry import k_chunks as _k_chunks, resolve_reuse_factor, resolve_fold_m
from gemm_ip.weights import build_weight_rom_k_spatial

GEN_DIR = HERE / "tb" / "generated"

# A spread of shapes: single-tile, multi-row-tile, multi-col-tile, multi-tile,
# multi-K-chunk, and non-8-multiple tails. Each case is (m, k, n, rf); rf=None
# means "rf = k_chunks", i.e. today's chunked endpoint (k_spatial=1) -- the
# original 10 shapes, unchanged generator path. The extra cases below exercise
# the general ReuseFactor fold: (8,24,8,2)/(9,17,10,2)/(16,40,16,2) are
# multi-pass K-spatial (9x17x10 pads its last pass); (8,16,8,1) and
# (24,16,8,1) legalize to full-K (rf=1 -> k_spatial=k_chunks); (16,40,16,5)
# legalizes to chunked (rf=k_chunks -> k_spatial=1), same generator path as
# the default cases.
DEFAULT_CASES = [
    (8, 8, 8, None),
    (16, 8, 8, None),
    (8, 8, 16, None),
    (16, 16, 16, None),
    (8, 16, 8, None),
    (24, 16, 8, None),
    (14, 6, 6, None),
    (9, 17, 10, None),
    (5, 5, 5, None),
    (12, 10, 10, None),
    (8, 24, 8, 2),
    (9, 17, 10, 2),
    (16, 40, 16, 2),
    (8, 16, 8, 1),
    (16, 40, 16, 5),
    (24, 16, 8, 1),
]
DEFAULT_SEEDS = [1, 7, 42]

# Weight-stationary (const-weight ROM) cases, m > n: the ROM now holds only
# passes*n entries (see gemm_ip/weights.py + rtl.py _weight_rom_block), so beats
# t >= n of each pass — which only exist here because input_beats = max(m,n) = m
# > n — must read back zero via the beat_ctr/rom_base mux. One rf=1 (full-K, one
# pass) and one rf>1 (multi-pass) case, both m > n, exercise that tail.
ROM_CASES = [
    (16, 8, 8, 1),   # m > n, full-K (rf=1 -> k_spatial=k_chunks=1 pass)
    (16, 24, 8, 2),  # m > n, multi-pass K-spatial (rf=2 -> passes=k_chunks/2)
]

# Fold-M regression: (m, k, n, rf). The core is generated for M_g = 8*mg rows
# (K and N fully spatial, one K pass); the golden iverilog testbench's
# back-to-back multi-frame mode (back2back=True, num_vectors=m_passes) is the
# regression hook -- it feeds m_passes distinct frames of M_g rows each and
# checks every output row against golden in frame order, exactly how the
# wrapper RUN loop replays the same rf=1 core (proven in
# temp_space/multiframe-probe/). (9,17,10,2) pads the last pass; (16,40,16,2)
# is an exact-division fold; (8,16,8,1) is the RF=1 (m_passes=1, single-frame)
# identity case; (16,40,16,5) drives RF to its grid_rows bound (5 -> legalizes
# down since grid_rows(16)=2).
FOLD_M_CASES = [
    (9, 17, 10, 2),
    (16, 40, 16, 2),
    (8, 16, 8, 1),
    (16, 40, 16, 5),
]


def _run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True)


def run_case(m, k, n, seed, rf=None, weights_in_core=False, fold_axis="k"):
    """Generate core RTL + TB for one shape/seed/ReuseFactor/axis, simulate, return (ok, log).

    Under fold_axis "m" the core is the rf=1 (full-K, full-N) core for M_g rows and
    the testbench's back-to-back multi-frame mode feeds m_passes frames -- the
    RTL-level equivalent of the wrapper's multi-frame RUN loop (see FOLD_M_CASES).
    """
    if fold_axis == "m":
        fm = resolve_fold_m(m, rf if rf is not None else 1)
        for w in fm["warnings"]:
            print(w, file=sys.stderr)
        rf_use = fm["reuse_factor"]
        core_m = fm["mg"] * 8
        num_vectors = fm["m_passes"]
        k_spatial = _k_chunks(k)
    else:
        rf_use = rf if rf is not None else _k_chunks(k)
        resolved = resolve_reuse_factor(k, rf_use)
        for w in resolved["warnings"]:
            print(w, file=sys.stderr)
        core_m = m
        num_vectors = 10
        k_spatial = resolved["k_spatial"]

    stem = f"gemm_{m}x{k}x{n}_rf{rf_use}_s{seed}"
    if fold_axis != "k":
        stem += f"_fold{fold_axis}"
    if weights_in_core:
        stem += "_rom"
    mod = f"{stem}_wrapper"
    rtl = GEN_DIR / f"{stem}.v"
    tb = GEN_DIR / f"tb_{stem}.v"
    out = GEN_DIR / f"{stem}.out"

    weight_rom = None
    fixed_B = None
    if weights_in_core:
        import numpy as np
        rng = np.random.default_rng(seed)
        max_val = max(1, int((127 / max(k, 1)) ** 0.5))
        fixed_B = rng.integers(-max_val, max_val + 1, size=(k, n), dtype=np.int8)
        weight_rom = build_weight_rom_k_spatial(fixed_B, core_m, n, k, k_spatial)

    if k_spatial == 1:
        rtl.write_text(generate_combined_core_verilog(core_m, k, n, module_name=mod, weight_rom=weight_rom))
    else:
        rtl.write_text(generate_k_spatial_combined_core_verilog(
            core_m, k, n, module_name=mod, k_spatial=k_spatial, out_bits=8, weight_rom=weight_rom))
    tb.write_text(generate_tb(core_m, k, n, module_name=mod, seed=seed, k_spatial=k_spatial,
                              num_vectors=num_vectors, back2back=(fold_axis == "m"),
                              weights_in_core=weights_in_core, fixed_B=fixed_B))

    comp = _run(["iverilog", "-g2012", "-o", str(out), str(tb), str(rtl)])
    if comp.returncode != 0:
        return False, "iverilog compile failed\n" + comp.stdout + comp.stderr
    sim = _run(["vvp", str(out)])
    log = sim.stdout + sim.stderr
    if sim.returncode != 0:
        return False, "vvp failed\n" + log
    if "ALL_PASS" not in log or "FAILURES=" in log:
        return False, log
    return True, log


def run(cases=None, seeds=None, keep=False):
    """Generate + simulate the wrapper for each (shape, seed); return 0 if all pass.

    ``cases`` is a list of ``(m, k, n)`` tuples (default: DEFAULT_CASES); ``seeds``
    a list of ints (default: DEFAULT_SEEDS). Returns 2 if iverilog/vvp are absent.
    """
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        print("ERROR: iverilog/vvp not found on PATH", file=sys.stderr)
        return 2

    rom_cases = list(ROM_CASES) if not cases else []
    fold_m_cases = list(FOLD_M_CASES) if not cases else []
    cases = list(cases) if cases else list(DEFAULT_CASES)
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for case in cases:
        m, k, n, rf = case if len(case) == 4 else (*case, None)
        for seed in seeds:
            ok, log = run_case(m, k, n, seed, rf=rf)
            rf_tag = f" rf={rf}" if rf is not None else ""
            tag = f"{m}x{k}x{n}{rf_tag} seed={seed}"
            print(f"{'PASS' if ok else 'FAIL'} {tag}")
            if not ok:
                print(log)
                failures.append(tag)

    for case in rom_cases:
        m, k, n, rf = case if len(case) == 4 else (*case, None)
        for seed in seeds:
            ok, log = run_case(m, k, n, seed, rf=rf, weights_in_core=True)
            rf_tag = f" rf={rf}" if rf is not None else ""
            tag = f"{m}x{k}x{n}{rf_tag} seed={seed} rom"
            print(f"{'PASS' if ok else 'FAIL'} {tag}")
            if not ok:
                print(log)
                failures.append(tag)

    for case in fold_m_cases:
        m, k, n, rf = case
        for seed in seeds:
            for rom in (False, True):
                ok, log = run_case(m, k, n, seed, rf=rf, weights_in_core=rom, fold_axis="m")
                tag = f"{m}x{k}x{n} rf={rf} seed={seed} fold_axis=m{' rom' if rom else ''}"
                print(f"{'PASS' if ok else 'FAIL'} {tag}")
                if not ok:
                    print(log)
                    failures.append(tag)

    if not keep:
        shutil.rmtree(GEN_DIR, ignore_errors=True)

    total = len(cases) * len(seeds) + len(rom_cases) * len(seeds) + len(fold_m_cases) * len(seeds)
    if failures:
        print(f"\n{len(failures)}/{total} FAILED:")
        for t in failures:
            print(f"  {t}")
        return 1
    print(f"\nALL PASSED: {len(cases)} shapes x {len(seeds)} seeds = {total} runs")
    return 0


def main():
    ap = argparse.ArgumentParser(description="RTL-level iverilog test for the GEMM wrapper")
    ap.add_argument("--cases", nargs="*", help="GEMM shapes as MxKxN, e.g. 8x8x8 16x16x16")
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS,
                    help="Random seeds for the self-checking testbenches")
    ap.add_argument("--keep", action="store_true",
                    help="Keep the generated RTL/TB under tb/generated/ for inspection")
    args = ap.parse_args()

    cases = []
    for c in (args.cases or []):
        try:
            m, k, n = (int(x) for x in c.lower().split("x"))
            cases.append((m, k, n))
        except Exception:
            print(f"ERROR: invalid case '{c}', expected MxKxN", file=sys.stderr)
            return 2

    return run(cases=cases or None, seeds=args.seeds, keep=args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
