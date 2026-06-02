#!/usr/bin/env python3
"""Backward-compat shim — delegates to ``gemm_ip.cli``."""
import sys
sys.path.insert(0, "src")
from gemm_ip.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + ["--backend", "vitis"] + [a for a in sys.argv[1:]]
    main()
