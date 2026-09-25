#!/usr/bin/env python3
"""cmvu RTL-level pre-filter regression.

Generates the wrapper + self-checking TB for a spread of shapes/folds and runs
them under Icarus (and VCS when available). This is a cheap pre-filter: the
mandatory acceptance gate is the Catapult SCVerify matrix, not this runner.

Run via the repo venv (numpy is needed by the generators):

    python -m targets.cmvu.run_rtl_tests
    python -m targets.cmvu.run_rtl_tests --cases 4x8x8x2x1 --seeds 1 7
    python -m targets.cmvu.run_rtl_tests --keep
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import geometry as _geom
from . import golden as _golden
from . import rtl as _rtl

GEN_DIR = Path(__file__).resolve().parent / "tb" / "generated"

# (m, k, n, kfold, nfold, shift, bias, width, rnd). ``width`` is the effective
# result width W: the physical max (16) unless the case exercises a narrower
# lane (the block slices each RESULT_WIDTH lane down to W). ``rnd`` selects
# whether a case's bias word carries the round-half-up constant (RND,
# default -- matches the block's historical default before rounding moved
# into the bias) or not (TRN, a plain floor with no rounding constant).
DEFAULT_CASES = [
    (4, 4, 8, 1, 1, 0, False, 16, True),   # single 4x8 tile
    (4, 8, 8, 2, 1, 3, True, 16, True),    # temporal K
    (4, 8, 16, 2, 2, 5, True, 8, True),    # temporal K+N, narrow W
    (4, 16, 16, 4, 2, 2, True, 16, True),  # slot cap (8 resident tiles)
    (4, 6, 10, 2, 2, 4, True, 12, True),   # non-multiple tails, mid W
    (4, 8, 8, 1, 1, 3, True, 16, True),    # spatial 2x1
    (4, 4, 16, 1, 1, 5, False, 8, True),   # spatial 1x2, narrow W
    (4, 8, 16, 1, 1, 0, True, 16, True),   # spatial 2x2
    (4, 6, 10, 1, 1, 4, True, 16, True),   # spatial 2x2 + tails
    (4, 16, 16, 2, 2, 4, True, 16, True),  # mixed spatial+temporal
    (4, 16, 16, 2, 1, 3, False, 16, True), # mixed, K temporal only
    (4, 8, 8, 2, 1, 3, True, 16, False),   # TRN: bias present, no round const
    (4, 8, 16, 1, 1, 0, True, 16, False),  # TRN: spatial 2x2, no round const
]
DEFAULT_SEEDS = [1, 7]

# Runtime-B cases: (m, k, n, kfold, nfold, shift, bias, width, rnd, layout).
# ``layout`` is "col" (hls4ml column-major b_beat, the mha_small QK shape and
# K_PASSES=2 paired-tile loads) or "row" (hls4ml row-major b_beat, the
# mha_small aV shape; N_PASSES>1 exercises the row-major wrapper's per-tile
# fill/drain residual buffer). Column and row tails (K/N neither a multiple
# of 4/8) hit each format's self-generated zero-padding.
RUNTIME_B_CASES = [
    (8, 16, 8, 2, 1, 3, True, 16, True, "col"),   # mha_small QK: K_PASSES=2 (paired)
    (8, 8, 16, 2, 1, 0, True, 16, True, "row"),   # mha_small aV: row-major, K_PASSES=2
    (8, 6, 10, 1, 1, 4, True, 16, True, "col"),   # column tail, K_PASSES=1
    (8, 6, 10, 1, 1, 4, True, 16, True, "row"),   # row tail, K_PASSES=1
    (4, 8, 16, 1, 2, 5, True, 16, True, "row"),   # row-major, N_PASSES=2 (residual buffer)
]


def _run(cmd, cwd):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=600)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def _tail(text, n=8):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def _parse_case(text):
    """Parse ``MxKxNxKFoldxNFold`` (optionally ``:shift``, ``:bias``, ``:wN``, ``:trn``)."""
    body, _, rest = text.partition(":")
    parts = body.lower().split("x")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"case {text!r}: expected MxKxNxKFoldxNFold")
    m, k, n, kf, nf = (int(p) for p in parts)
    shift = 0
    bias = False
    width = _geom.RESULT_WIDTH
    rnd = True
    if rest:
        for tok in rest.split(":"):
            if tok in ("bias", "b"):
                bias = True
            elif tok == "trn":
                rnd = False
            elif tok.startswith("w"):
                width = int(tok[1:])
            elif tok:
                shift = int(tok)
    return (m, k, n, kf, nf, shift, bias, width, rnd)


def run(cases=None, seeds=None, keep=False, use_vcs=True, en_gaps=False,
        backpressure=False):
    if en_gaps and backpressure:
        raise ValueError("--en-gaps and --backpressure are mutually exclusive")
    rtl_dir = _geom.vendored_rtl_dir()
    if rtl_dir is None:
        raise FileNotFoundError(
            f"cmvu vendored block RTL not found; expected "
            f"{', '.join(_geom.VENDORED_SV)} under {_geom.CMVU_RTL_DIR}.")
    rtl_files = [str((rtl_dir / f).resolve()) for f in _geom.VENDORED_SV]

    cases = list(cases) if cases else list(DEFAULT_CASES)
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
    if GEN_DIR.exists() and not keep:
        shutil.rmtree(GEN_DIR, ignore_errors=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    have_vcs = use_vcs and shutil.which("vcs") is not None
    failures = 0
    total = 0
    for (m, k, n, kf, nf, shift, bias, width, rnd) in cases:
        for seed in seeds:
            total += 1
            tag = (f"m{m}_k{k}_n{n}_kf{kf}_nf{nf}_s{shift}"
                   f"_{'b' if bias else 'nb'}_{'rnd' if rnd else 'trn'}"
                   f"_w{width}_seed{seed}" + ("_engap" if en_gaps else "") + ("_bp" if backpressure else ""))
            d = GEN_DIR / tag
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            rng = np.random.default_rng(seed)
            W = rng.integers(-128, 128, size=(k, n))
            if bias:
                raw = rng.integers(-64, 65, size=n) / 4.0
                codes, warns = _golden.bias_codes(raw, shift, rnd=rnd)
                for w in warns:
                    print("   ", w)
            elif rnd:
                # No static bias, but RND output still needs the rounding
                # constant folded into the (all-zero-mean) bias word.
                codes, warns = _golden.bias_codes(
                    np.zeros(n), shift, rnd=True)
                for w in warns:
                    print("   ", w)
            else:
                codes = None

            (d / "cmvu_core.v").write_text(_rtl.generate_core(
                m, k, n, kf, nf, W, bias_codes=codes, shift=shift,
                module_name="cmvu_core", name=tag, result_width=width))
            (d / "cmvu_core_tb.v").write_text(_golden.generate_tb(
                m, k, n, kf, nf, W, bias_codes=codes, shift=shift,
                module_name="cmvu_core", seed=seed + 100, name=tag,
                result_width=width, en_gaps=en_gaps, backpressure=backpressure))

            rc, out = _run(["iverilog", "-g2012", "-o", "simv_icarus",
                            "-s", "cmvu_core_tb", "cmvu_core_tb.v",
                            "cmvu_core.v", *rtl_files], d)
            if rc != 0:
                print(f"  {tag} ICARUS BUILD FAIL\n{_tail(out)}")
                failures += 1
            else:
                rc, out = _run(["vvp", "simv_icarus"], d)
                if not ("ALL_PASS" in out and rc == 0):
                    print(f"  {tag} ICARUS FAIL\n{_tail(out)}")
                    failures += 1

            if have_vcs:
                rc, out = _run(["vcs", "-full64", "-sverilog",
                                "-timescale=1ns/1ps", "-o", "simv_vcs",
                                "-Mdir=csrc", "cmvu_core_tb.v", "cmvu_core.v",
                                *rtl_files], d)
                if rc != 0:
                    print(f"  {tag} VCS BUILD FAIL\n{_tail(out)}")
                    failures += 1
                else:
                    rc, out = _run(["./simv_vcs"], d)
                    if not ("ALL_PASS" in out and rc == 0):
                        print(f"  {tag} VCS FAIL\n{_tail(out)}")
                        failures += 1

    rb_cases = list(RUNTIME_B_CASES)
    for (m, k, n, kf, nf, shift, bias, width, rnd, layout) in rb_cases:
        row_major = (layout == "row")
        for seed in seeds:
            total += 1
            tag = (f"rb_{layout}_m{m}_k{k}_n{n}_kf{kf}_nf{nf}_s{shift}"
                   f"_{'b' if bias else 'nb'}_{'rnd' if rnd else 'trn'}"
                   f"_w{width}_seed{seed}" + ("_engap" if en_gaps else "") + ("_bp" if backpressure else ""))
            d = GEN_DIR / tag
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            rng = np.random.default_rng(seed)
            W = rng.integers(-128, 128, size=(k, n))
            if bias:
                raw = rng.integers(-64, 65, size=n) / 4.0
                codes, warns = _golden.bias_codes(raw, shift, rnd=rnd)
                for w in warns:
                    print("   ", w)
            elif rnd:
                codes, warns = _golden.bias_codes(
                    np.zeros(n), shift, rnd=True)
                for w in warns:
                    print("   ", w)
            else:
                codes = None

            (d / "cmvu_core.v").write_text(_rtl.generate_core(
                m, k, n, kf, nf, W, bias_codes=codes, shift=shift,
                module_name="cmvu_core", name=tag, runtime_b=True,
                result_width=width, b_row_major=row_major))
            (d / "cmvu_core_tb.v").write_text(_golden.generate_runtime_b_tb(
                m, k, n, kf, nf, W, bias_codes=codes, shift=shift,
                module_name="cmvu_core", seed=seed + 100, name=tag,
                result_width=width, en_gaps=en_gaps, b_row_major=row_major,
                    backpressure=backpressure))

            rc, out = _run(["iverilog", "-g2012", "-o", "simv_icarus",
                            "-s", "cmvu_core_tb", "cmvu_core_tb.v",
                            "cmvu_core.v", *rtl_files], d)
            if rc != 0:
                print(f"  {tag} ICARUS BUILD FAIL\n{_tail(out)}")
                failures += 1
            else:
                rc, out = _run(["vvp", "simv_icarus"], d)
                if not ("ALL_PASS" in out and rc == 0):
                    print(f"  {tag} ICARUS FAIL\n{_tail(out)}")
                    failures += 1

            if have_vcs:
                rc, out = _run(["vcs", "-full64", "-sverilog",
                                "-timescale=1ns/1ps", "-o", "simv_vcs",
                                "-Mdir=csrc", "cmvu_core_tb.v", "cmvu_core.v",
                                *rtl_files], d)
                if rc != 0:
                    print(f"  {tag} VCS BUILD FAIL\n{_tail(out)}")
                    failures += 1
                else:
                    rc, out = _run(["./simv_vcs"], d)
                    if not ("ALL_PASS" in out and rc == 0):
                        print(f"  {tag} VCS FAIL\n{_tail(out)}")
                        failures += 1

    # Multi-call runtime-B: hls4ml streams a NEW B ahead of every call's A
    # rows (a fresh K for QK, a fresh V for aV), but the wrapper is
    # persistent hardware -- a single-call TB can't catch a load FSM that
    # only ever re-arms at reset. Replays each shape's B-load + M-row
    # transaction 3x back-to-back against the SAME dut instance.
    for (m, k, n, kf, nf, shift, bias, width, rnd, layout) in rb_cases:
        row_major = (layout == "row")
        for seed in seeds:
            total += 1
            tag = (f"rbmc_{layout}_m{m}_k{k}_n{n}_kf{kf}_nf{nf}_s{shift}"
                   f"_{'b' if bias else 'nb'}_{'rnd' if rnd else 'trn'}"
                   f"_w{width}_seed{seed}" + ("_engap" if en_gaps else "") + ("_bp" if backpressure else ""))
            d = GEN_DIR / tag
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            rng = np.random.default_rng(seed)
            if bias:
                raw = rng.integers(-64, 65, size=n) / 4.0
                codes, warns = _golden.bias_codes(raw, shift, rnd=rnd)
                for w in warns:
                    print("   ", w)
            elif rnd:
                codes, warns = _golden.bias_codes(
                    np.zeros(n), shift, rnd=True)
                for w in warns:
                    print("   ", w)
            else:
                codes = None

            (d / "cmvu_core.v").write_text(_rtl.generate_core(
                m, k, n, kf, nf, None, bias_codes=codes, shift=shift,
                module_name="cmvu_core", name=tag, runtime_b=True,
                result_width=width, b_row_major=row_major))
            (d / "cmvu_core_tb.v").write_text(
                _golden.generate_runtime_b_multi_call_tb(
                    m, k, n, kf, nf, bias_codes=codes, shift=shift,
                    module_name="cmvu_core", seed=seed + 200, name=tag,
                    result_width=width, en_gaps=en_gaps, b_row_major=row_major,
                    n_calls=3, backpressure=backpressure))

            rc, out = _run(["iverilog", "-g2012", "-o", "simv_icarus",
                            "-s", "cmvu_core_tb", "cmvu_core_tb.v",
                            "cmvu_core.v", *rtl_files], d)
            if rc != 0:
                print(f"  {tag} ICARUS BUILD FAIL\n{_tail(out)}")
                failures += 1
            else:
                rc, out = _run(["vvp", "simv_icarus"], d)
                if not ("ALL_PASS" in out and rc == 0):
                    print(f"  {tag} ICARUS FAIL\n{_tail(out)}")
                    failures += 1

            if have_vcs:
                rc, out = _run(["vcs", "-full64", "-sverilog",
                                "-timescale=1ns/1ps", "-o", "simv_vcs",
                                "-Mdir=csrc", "cmvu_core_tb.v", "cmvu_core.v",
                                *rtl_files], d)
                if rc != 0:
                    print(f"  {tag} VCS BUILD FAIL\n{_tail(out)}")
                    failures += 1
                else:
                    rc, out = _run(["./simv_vcs"], d)
                    if not ("ALL_PASS" in out and rc == 0):
                        print(f"  {tag} VCS FAIL\n{_tail(out)}")
                        failures += 1

    if not keep:
        shutil.rmtree(GEN_DIR, ignore_errors=True)
    tools = "iverilog+vcs" if have_vcs else "iverilog"
    print(f"\nCMVU RTL ({tools}): {total - failures}/{total} pass")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=_parse_case,
                        help="MxKxNxKFoldxNFold[:shift][:bias] cases")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--keep", action="store_true",
                        help="keep tb/generated/* for inspection")
    parser.add_argument("--no-vcs", action="store_true",
                        help="skip the VCS pass even if vcs is on PATH")
    parser.add_argument("--en-gaps", action="store_true",
                        help="drive en with randomized gaps instead of "
                             "holding it at 1 (clock-gating regression)")
    parser.add_argument("--backpressure", action="store_true",
                        help="drive en with long (20-50 cycle) disabled "
                             "bursts instead of holding it at 1 (sustained-"
                             "backpressure regression; mutually exclusive "
                             "with --en-gaps)")
    args = parser.parse_args(argv)
    return run(cases=args.cases, seeds=args.seeds, keep=args.keep,
               use_vcs=not args.no_vcs, en_gaps=args.en_gaps,
               backpressure=args.backpressure)


if __name__ == "__main__":
    sys.exit(main())
