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

from rtl import generate_combined_core_verilog
from golden import generate_tb

GEN_DIR = HERE / "tb" / "generated"

# A spread of shapes: single-tile, multi-row-tile, multi-col-tile, multi-tile,
# multi-K-chunk, and non-8-multiple tails.
DEFAULT_CASES = [
    (8, 8, 8),
    (16, 8, 8),
    (8, 8, 16),
    (16, 16, 16),
    (8, 16, 8),
    (24, 16, 8),
    (14, 6, 6),
    (9, 17, 10),
    (5, 5, 5),
    (12, 10, 10),
]
DEFAULT_SEEDS = [1, 7, 42]


def _run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True)


def run_case(m, k, n, seed):
    """Generate wrapper RTL + TB for one shape/seed, simulate, return (ok, log)."""
    stem = f"gemm_{m}x{k}x{n}_s{seed}"
    mod = f"{stem}_wrapper"
    rtl = GEN_DIR / f"{stem}.v"
    tb = GEN_DIR / f"tb_{stem}.v"
    out = GEN_DIR / f"{stem}.out"

    rtl.write_text(generate_combined_core_verilog(m, k, n, module_name=mod))
    tb.write_text(generate_tb(m, k, n, module_name=mod, seed=seed))

    # No -DSYNTHESIS: the behavioral branch is compiled, so no external slice IP.
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

    cases = list(cases) if cases else list(DEFAULT_CASES)
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for (m, k, n) in cases:
        for seed in seeds:
            ok, log = run_case(m, k, n, seed)
            tag = f"{m}x{k}x{n} seed={seed}"
            print(f"{'PASS' if ok else 'FAIL'} {tag}")
            if not ok:
                print(log)
                failures.append(tag)

    if not keep:
        shutil.rmtree(GEN_DIR, ignore_errors=True)

    total = len(cases) * len(seeds)
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
