#!/usr/bin/env python3
"""Backward-compat shim — delegates to ``gemm_ip.cli``."""
import sys
sys.path.insert(0, "src")
from gemm_ip.cli import main  # noqa: E402

if __name__ == "__main__":
    # Replace the script name so argparse sees `--backend catapult` by default
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "catapult"]
    if "--backend" not in sys.argv and "-b" not in sys.argv:
        sys.argv.insert(1, "--backend")
        sys.argv.insert(2, "catapult")
    main()
