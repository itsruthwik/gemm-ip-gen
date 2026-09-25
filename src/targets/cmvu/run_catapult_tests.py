#!/usr/bin/env python3
"""cmvu Catapult SCVerify acceptance matrix (the gate).

For each config, generate a full package and run the Catapult flow
(``run_catapult.tcl``), which ends in SCVerify RTL cosim. A config passes only
when SCVerify reports ``error count = 0`` and the csim TB prints PASS.

Tiered matrix: M in {1,4,8,16} on temporal 1/1, spatial 2x2, and mixed 2/2
(12 runs), plus all 11 base configs at M=4 (11 runs), plus 5 key configs
re-run at W=8 (width sweep, exercising the out_w slicing path under SCVerify).
Icarus/VCS (``run_rtl_tests``) is a cheap pre-filter, not the gate.

    python -m targets.cmvu.run_catapult_tests --quick      # 3 configs
    python -m targets.cmvu.run_catapult_tests              # full matrix
    python -m targets.cmvu.run_catapult_tests --list
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import geometry as _geom
from . import golden as _golden
from . import package as _pkg

# name -> (m, k, n, kfold, nfold, shift, bias, seed, width)
BASE_CONFIGS = {
    "t_1p1":      (4,  4,  8, 1, 1, 0, False, 0, 16),
    "t_2p1":      (4,  8,  8, 2, 1, 3, True,  1, 16),
    "t_2p2":      (4,  8, 16, 2, 2, 5, True,  2, 16),
    "t_4p2cap":   (4, 16, 16, 4, 2, 2, True,  3, 16),
    "t_2p2tails": (4,  6, 10, 2, 2, 4, True,  4, 16),
    "s_2x1":      (4,  8,  8, 1, 1, 3, True,  5, 16),
    "s_1x2":      (4,  4, 16, 1, 1, 5, False, 6, 16),
    "s_2x2":      (4,  8, 16, 1, 1, 0, True,  7, 16),
    "s_2x2tails": (4,  6, 10, 1, 1, 4, True,  8, 16),
    "m_2p2":      (4, 16, 16, 2, 2, 4, True,  9, 16),
    "m_2p1":      (4, 16, 16, 2, 1, 3, False, 10, 16),
}

# M sweep: base name -> (k, n, kfold, nfold, shift, bias, seed, width)
SWEEP_BASES = {
    "sw_t_1p1": (8,  8, 1, 1, 0, False, 20, 16),
    "sw_s_2x2": (8, 16, 1, 1, 0, True,  21, 16),
    "sw_m_2p2": (16, 16, 2, 2, 4, True,  22, 16),
}
SWEEP_M = [1, 4, 8, 16]

# W sweep: key configs re-run at W=8 to exercise the out_w slicing path
# end-to-end under SCVerify, not just the Icarus pre-filter.
# name -> (m, k, n, kfold, nfold, shift, bias, seed, width)
WIDTH_SWEEP = {
    "w8_t_2p2":      (4,  8, 16, 2, 2, 5, True,  2, 8),
    "w8_s_2x2":      (4,  8, 16, 1, 1, 0, True,  7, 8),
    "w8_s_2x2tails": (4,  6, 10, 1, 1, 4, True,  8, 8),
    "w8_m_2p2":      (4, 16, 16, 2, 2, 4, True,  9, 8),
    "w8_m_2p1":      (4, 16, 16, 2, 1, 3, False, 10, 8),
}

# Three representative configs for a fast smoke (one per fold category).
QUICK = ["t_2p2", "s_2x2", "m_2p2"]


def all_runs():
    runs = dict(BASE_CONFIGS)
    for base, (k, n, kf, nf, sh, bias, seed, width) in SWEEP_BASES.items():
        for m in SWEEP_M:
            runs[f"{base}_m{m}"] = (m, k, n, kf, nf, sh, bias, seed, width)
    runs.update(WIDTH_SWEEP)
    return runs


def _make_pkg(name, m, k, n, kf, nf, shift, bias, seed, workdir, width):
    rtl_dir = _geom.vendored_rtl_dir()
    if rtl_dir is None:
        raise FileNotFoundError(
            f"cmvu vendored block RTL not found; expected "
            f"{', '.join(_geom.VENDORED_SV)} under {_geom.CMVU_RTL_DIR}.")
    out_dir = workdir / name
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    for sv in _geom.VENDORED_SV:
        shutil.copy(rtl_dir / sv, out_dir / sv)
    rng = np.random.default_rng(seed)
    W = rng.integers(-128, 128, size=(k, n))
    codes = None
    if bias:
        raw = rng.integers(-64, 65, size=n) / 4.0
        codes, _ = _golden.bias_codes(raw, shift)
    return _pkg.generate_catapult_pkg(
        m, k, n, name, str(out_dir), kf, nf, W, bias_codes=codes, shift=shift,
        result_width=width)


def run(cases=None, workdir=None, resume=True, keep=False, timeout=3600):
    runs = all_runs()
    names = list(cases) if cases else list(runs)
    for nm in names:
        if nm not in runs:
            raise ValueError(f"unknown cmvu matrix config {nm!r}")
    workdir = Path(workdir) if workdir else Path.cwd() / "cmvu_catapult_matrix"
    workdir.mkdir(parents=True, exist_ok=True)
    results_path = workdir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    failures = []
    for name in names:
        if resume and results.get(name, {}).get("ok"):
            print(f"=== {name}: already PASS, skip ===")
            continue
        m, k, n, kf, nf, shift, bias, seed, width = runs[name]
        pkg = _make_pkg(name, m, k, n, kf, nf, shift, bias, seed, workdir, width)
        t0 = time.time()
        try:
            p = subprocess.run(["catapult", "-shell", "-f", "run_catapult.tcl"],
                               cwd=pkg, capture_output=True, text=True,
                               timeout=timeout)
            out = (p.stdout or "") + (p.stderr or "")
            rc = p.returncode
        except FileNotFoundError:
            print("catapult not found on PATH")
            return 1
        except subprocess.TimeoutExpired:
            out, rc = "TIMEOUT", 124
        errs = [ln.strip() for ln in out.splitlines() if "error count" in ln]
        ok = ("CMVU CSIM: PASS" in out and "Incorrect Data Detected" not in out
              and rc == 0 and bool(errs)
              and all("error count          = 0" in ln for ln in errs))
        results[name] = {"ok": bool(ok), "rc": rc,
                         "seconds": round(time.time() - t0, 1), "errors": errs}
        results_path.write_text(json.dumps(results, indent=2))
        print(f"=== {name}: {'PASS' if ok else 'FAIL'} "
              f"({results[name]['seconds']}s) {errs}")
        if not ok:
            failures.append(name)
            for ln in out.splitlines():
                if ("Error" in ln or "rror count" in ln or "CSIM" in ln
                        or "FAIL" in ln):
                    print("   ", ln.strip())
        if not keep:
            shutil.rmtree(pkg, ignore_errors=True)

    print(f"\nCMVU CATAPULT {'PASS' if not failures else 'FAIL'}: "
          f"{len(names) - len(failures)}/{len(names)} ok; failed={failures}")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="config names (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help=f"run the smoke subset {QUICK}")
    parser.add_argument("--list", action="store_true", help="list configs")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep", action="store_true",
                        help="keep generated packages after the run")
    args = parser.parse_args(argv)

    if args.list:
        for name, (m, k, n, kf, nf, sh, bias, seed, width) in all_runs().items():
            geo = _geom.resolve_geometry(m, k, n, kf, nf, name, width)
            print(f"{name:16s} m={m:2d} k={k:2d} n={n:2d} kf={kf} nf={nf} "
                  f"w={width:2d} -> ks={geo['k_spatial']} kp={geo['k_passes']} "
                  f"ns={geo['n_spatial']} np={geo['n_passes']} bias={bias}")
        return 0

    cases = args.cases or (QUICK if args.quick else None)
    return run(cases=cases, workdir=args.workdir, resume=not args.no_resume,
               keep=args.keep)


if __name__ == "__main__":
    sys.exit(main())
