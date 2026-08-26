#!/usr/bin/env python3
"""Backward-compat shim — delegates to ``gemm_ip.cli``.

The hardblock is now selected with ``--target`` (default: tensor_slice). This
shim tolerates the retired ``--backend [catapult]`` invocation by dropping it.
"""
import sys
sys.path.insert(0, "src")
from gemm_ip.cli import main  # noqa: E402

if __name__ == "__main__":
    # Drop the retired `--backend`/`-b` flag (and a following value) and any bare
    # `catapult` token so old callers keep working against the --target CLI.
    argv, out, skip = sys.argv[1:], [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in ("--backend", "-b"):
            skip = True
            continue
        if a == "catapult":
            continue
        out.append(a)
    sys.argv = [sys.argv[0]] + out
    main()
