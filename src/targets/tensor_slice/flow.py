"""tensor_slice target: binds the hardblock's modules to the Target contract.

Delegates each contract method to the sibling modules — ``geometry`` (tiling),
``rtl`` (synth/behavioral Verilog), ``golden`` (self-checking testbench),
``package`` (the full blackbox package), and ``run_rtl_tests`` (the RTL
regression). The heavy modules (``package`` pulls in ``gemm_ip.quant``;
``run_rtl_tests`` shells out to iverilog) are imported lazily so importing this
module stays cheap and free of core dependencies.
"""

import sys
from pathlib import Path

# base.py is one level up (src/targets/); the sibling modules are alongside.
_targets_root = str(Path(__file__).resolve().parent.parent)
if _targets_root not in sys.path:
    sys.path.insert(0, _targets_root)
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from base import Target  # noqa: E402
import geometry as _geom  # noqa: E402
import rtl as _rtl  # noqa: E402
import golden as _golden  # noqa: E402


def _package():
    import package
    return package


def _run_rtl_tests():
    import run_rtl_tests
    return run_rtl_tests


class TensorSliceTarget(Target):
    name = "tensor_slice"
    tool = "catapult"

    def geometry(self, shape):
        m, k, n = shape
        gr, gc = _geom.grid_rows(m), _geom.grid_cols(n)
        return {
            "grid_rows": gr,
            "grid_cols": gc,
            "k_chunks": _geom.k_chunks(k),
            "a_stream_width": _geom.a_stream_width(m),
            "b_stream_width": _geom.b_stream_width(n),
            "c_stream_width": _geom.c_stream_width(n),
            "latency_cycles": _geom.latency_cycles(k, gr, gc, m=m, n=n, k=k),
        }

    def emit_rtl(self, shape, **kwargs):
        return _rtl.generate_synth_verilog(*shape, **kwargs)

    def emit_behavioral(self, shape, **kwargs):
        return _rtl.generate_sim_verilog(*shape, **kwargs)

    def golden(self, shape, seed=42, **kwargs):
        return _golden.generate_tb(*shape, seed=seed, **kwargs)

    def package(self, shape, cfg):
        cfg = dict(cfg)
        name = cfg.pop("name")
        self.validate_knobs(cfg, name)
        output_dir = cfg.pop("output_dir")
        return _package().generate_catapult_pkg(*shape, name, output_dir, **cfg)

    def verify(self, package):
        """Structural check that packaging produced a well-formed package.

        Deeper functional verification (the Catapult SCVerify cosim) is the
        tool-level bar and is run separately.
        """
        pkg = Path(package)
        name = pkg.name
        required = [f"{name}_core.v", "nnet_types.h", f"{name}_gemm_ip.h",
                    f"{name}_inst.cpp", f"{name}_tb.cpp", "run_catapult.tcl"]
        missing = [f for f in required
                   if not (pkg / f).is_file() or (pkg / f).stat().st_size == 0]
        if missing:
            raise RuntimeError(f"{name}: package incomplete, missing/empty: {missing}")
        return True

    def rtl_test(self, cases=None, seeds=None, keep=False):
        return _run_rtl_tests().run(cases=cases, seeds=seeds, keep=keep)

    # ── batch orchestration (multi-package; used by the CLI) ─────────────────
    def normalize_config(self, cfg):
        from gemm_ip.config import _normalize_config_items
        return _normalize_config_items(cfg)

    def combined_header(self, items):
        return _package().gen_combined_header(items)

    def integration_manifest(self, items):
        return _package().gen_integration_manifest(items)

    def blackbox_tcl(self, items):
        return _package().gen_blackbox_tcl(items)


TARGET = TensorSliceTarget()
