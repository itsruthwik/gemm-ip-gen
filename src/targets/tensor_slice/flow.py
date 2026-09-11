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

    knobs = [
        {"name": "FoldAxis", "key": "fold_axis", "type": "enum",
         "choices": ("k", "m", "n"), "default": "k",
         "description": "which dimension ReuseFactor folds"},
    ]

    def geometry(self, shape, reuse_factor=1, fold_axis="k"):
        m, k, n = shape
        gr, gc = _geom.grid_rows(m), _geom.grid_cols(n)
        resolved = _geom.resolve_reuse_factor(k, reuse_factor, fold_axis=fold_axis, m=m, n=n)
        for w in resolved["warnings"]:
            print(w, file=sys.stderr)
        ks = resolved["k_spatial"]
        mg = resolved.get("mg", gr)
        m_passes = resolved.get("m_passes", 1)
        cg = resolved.get("cg", gc)
        n_passes = resolved.get("n_passes", 1)
        if fold_axis == "m":
            mult = 64 * mg * gc * resolved["k_chunks"]
        elif fold_axis == "n":
            mult = 64 * gr * cg * resolved["k_chunks"]
        else:
            mult = _geom.multipliers(m, n, ks)
        return {
            "grid_rows": gr,
            "grid_cols": gc,
            "k_chunks": resolved["k_chunks"],
            "k_chunks_pad": resolved["k_chunks_pad"],
            "k_spatial": ks,
            "k_passes": resolved["passes"],
            "reuse_factor_requested": resolved["reuse_factor_requested"],
            "reuse_factor": resolved["reuse_factor"],
            "effective_reuse": resolved["effective_reuse"],
            "fold_axis": fold_axis,
            "m_groups": mg,
            "m_passes": m_passes,
            "grid_rows_pad": resolved.get("grid_rows_pad", gr),
            "n_groups": cg if fold_axis == "n" else gc,
            "n_passes": n_passes,
            "grid_cols_pad": resolved.get("grid_cols_pad", gc),
            "core_cols": n if n_passes == 1 else 8 * cg,
            "multipliers": mult,
            "a_stream_width": _geom.a_stream_width(m, ks),
            "b_stream_width": _geom.b_stream_width(n, ks),
            "c_stream_width": _geom.c_stream_width(n),
            "latency_cycles": _geom.latency_first_out(m, k, n, ks),
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
        items = _normalize_config_items(cfg)
        for item in items:
            self.validate_knobs(item, item.get("name"))
            fold_axis = str(item.get("fold_axis") or "k").lower()
            resolved = _geom.resolve_reuse_factor(
                item["k"], item.get("reuse_factor", 1), item.get("name"),
                fold_axis=fold_axis, m=item["m"], n=item["n"])
            for w in resolved["warnings"]:
                print(w, file=sys.stderr)
            gr = _geom.grid_rows(item["m"])
            gc = _geom.grid_cols(item["n"])
            mg = resolved.get("mg", gr)
            m_passes = resolved.get("m_passes", 1)
            cg = resolved.get("cg", gc)
            n_passes = resolved.get("n_passes", 1)
            item["k_spatial"] = resolved["k_spatial"]
            item["k_passes"] = resolved["passes"]
            item["k_chunks_pad"] = resolved["k_chunks_pad"]
            item["reuse_factor_requested"] = resolved["reuse_factor_requested"]
            item["reuse_factor"] = resolved["reuse_factor"]
            item["effective_reuse"] = resolved["effective_reuse"]
            item["fold_axis"] = fold_axis
            item["m_groups"] = mg
            item["m_passes"] = m_passes
            item["grid_rows_pad"] = resolved.get("grid_rows_pad", gr)
            item["n_groups"] = cg
            item["n_passes"] = n_passes
            item["grid_cols_pad"] = resolved.get("grid_cols_pad", gc)
            item["core_cols"] = item["n"] if n_passes == 1 else 8 * cg
            if fold_axis == "m":
                item["multipliers"] = 64 * mg * gc * resolved["k_chunks"]
            elif fold_axis == "n":
                item["multipliers"] = 64 * gr * cg * resolved["k_chunks"]
            else:
                item["multipliers"] = _geom.multipliers(item["m"], item["n"], resolved["k_spatial"])
        return items

    def combined_header(self, items):
        return _package().gen_combined_header(items)

    def integration_manifest(self, items):
        return _package().gen_integration_manifest(items)

    def blackbox_tcl(self, items):
        return _package().gen_blackbox_tcl(items)


TARGET = TensorSliceTarget()
