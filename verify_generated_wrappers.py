#!/usr/bin/env python3
"""
Generate RTL wrappers/testbenches from the Python generators and verify them
with Icarus Verilog across a mix of tiled and non-8-multiple GEMM sizes.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _load_tensor_slice_generators():
    root = Path(__file__).resolve().parent
    ts_dir = root / "tensor-slice"
    ts_dir_str = str(ts_dir)
    if ts_dir_str not in sys.path:
        sys.path.insert(0, ts_dir_str)
    from generate_verilog_grid import generate_grid_verilog
    from generate_verilog_tb import generate_tb
    return generate_grid_verilog, generate_tb


generate_grid_verilog, generate_tb = _load_tensor_slice_generators()


DEFAULT_CASES = [
    (8, 8, 8),
    (16, 8, 8),
    (8, 8, 16),
    (16, 8, 16),
    (16, 16, 16),
    (5, 5, 5),
    (12, 10, 10),
    (9, 13, 7),
    (7, 9, 15),
]

DEFAULT_SEEDS = [1, 7, 42]


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def verify_case(workdir: Path, m: int, k: int, n: int, seed: int) -> tuple[bool, str]:
    stem = f"gemm_{m}x{k}x{n}_s{seed}"
    mod_name = f"{stem}_wrapper"
    rtl_path = workdir / f"{stem}.v"
    tb_path = workdir / f"tb_{stem}.v"
    sim_path = workdir / f"{stem}.out"
    slice_path = Path(__file__).resolve().parent / "tensor-slice" / "tensor_slice_int8.v"

    rtl_path.write_text(generate_grid_verilog(m, k, n, mod_name))
    tb_path.write_text(generate_tb(m, k, n, mod_name, seed=seed))

    compile_res = run(
        ["iverilog", "-g2012", "-o", str(sim_path), str(tb_path), str(rtl_path), str(slice_path)],
        cwd=workdir,
    )
    if compile_res.returncode != 0:
        return False, f"compile failed\n{compile_res.stdout}{compile_res.stderr}"

    sim_res = run(["vvp", str(sim_path)], cwd=workdir)
    combined = sim_res.stdout + sim_res.stderr
    if sim_res.returncode != 0:
        return False, f"simulation failed\n{combined}"
    if "SIMULATION PASSED" not in combined:
        return False, combined
    return True, combined


def main():
    parser = argparse.ArgumentParser(description="Verify generated GEMM RTL wrappers with iverilog")
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Case list as MxKxN, e.g. 8x8x8 16x8x16",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds for generated self-checking TBs",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the generated temporary workdir for inspection",
    )
    args = parser.parse_args()

    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        print("ERROR: iverilog/vvp not found on PATH", file=sys.stderr)
        return 2

    cases = []
    for item in (args.cases or []):
        try:
            m_str, k_str, n_str = item.lower().split("x")
            cases.append((int(m_str), int(k_str), int(n_str)))
        except Exception:
            print(f"ERROR: invalid case '{item}', expected MxKxN", file=sys.stderr)
            return 2
    if not cases:
        cases = list(DEFAULT_CASES)

    failures = []
    with tempfile.TemporaryDirectory(prefix="gemm_wrapper_verify_") as tmp:
        workdir = Path(tmp)
        print(f"Working directory: {workdir}")
        for m, k, n in cases:
            for seed in args.seeds:
                ok, output = verify_case(workdir, m, k, n, seed)
                tag = f"{m}x{k}x{n} seed={seed}"
                if ok:
                    print(f"PASS {tag}")
                else:
                    print(f"FAIL {tag}")
                    print(output)
                    failures.append(tag)
        if args.keep_workdir:
            keep_path = Path.cwd() / "gemm_wrapper_verify_workdir"
            if keep_path.exists():
                shutil.rmtree(keep_path)
            shutil.copytree(workdir, keep_path)
            print(f"Copied workdir to {keep_path}")

    if failures:
        print("\nFAILED CASES:")
        for item in failures:
            print(f"  {item}")
        return 1

    print(f"\nAll wrapper regressions passed: {len(cases)} cases x {len(args.seeds)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
