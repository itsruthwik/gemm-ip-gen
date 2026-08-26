#!/usr/bin/env bash
# RTL-level test runner for the tensor-slice GEMM wrapper.
#
# Generates the behavioral wrapper RTL + a self-checking testbench for a spread
# of GEMM shapes and runs them under Icarus Verilog. Uses the repo venv, which
# has the numpy the Python generators need. All arguments are forwarded to
# run_rtl_tests.py (--cases, --seeds, --keep).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/../../../.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3)"
fi
exec "$PY" "$HERE/run_rtl_tests.py" "$@"
