"""tensor_slice target: binds the hardblock's modules to the Target contract.

Delegates each contract method to the sibling modules — ``geometry`` (tiling),
``rtl`` (synth/behavioral Verilog), ``golden`` (self-checking testbench),
``package`` (the full blackbox package), and ``run_rtl_tests`` (the RTL
regression). The heavy modules (``package`` pulls in ``gemm_ip.quant``;
``run_rtl_tests`` shells out to iverilog) are imported lazily so importing this
module stays cheap and free of core dependencies.
"""

import sys

from ..base import Target
from . import geometry as _geom
from . import rtl as _rtl
from . import golden as _golden


def _axis_reuse_factors(fold_axis, reuse_factor, m_rf, k_rf, n_rf):
    """Resolve the (m_rf, k_rf, n_rf) pass-count triple from the legacy
    ``fold_axis``/``reuse_factor`` pair and the explicit per-axis
    ``MFold``/``KFold``/``NFold`` knobs, with the same precedence
    ``package._resolve_axis_reuse_factors`` applies to the RTL: any explicit
    per-axis knob wins outright over the legacy pair. Kept in sync with that
    function so the manifest geometry matches the generated core; the
    user-facing precedence-collision warning is emitted there, not here, to
    avoid printing it twice per build."""
    explicit = {ax: v for ax, v in (("m", m_rf), ("k", k_rf), ("n", n_rf))
                if v is not None}
    fold_axis = str(fold_axis or "k").lower()
    if explicit:
        return explicit.get("m", 1), explicit.get("k", 1), explicit.get("n", 1)
    if fold_axis == "m":
        return reuse_factor, 1, 1
    if fold_axis == "n":
        return 1, 1, reuse_factor
    return 1, reuse_factor, 1


def _package():
    from . import package
    return package


def _run_rtl_tests():
    from . import run_rtl_tests
    return run_rtl_tests


class TensorSliceTarget(Target):
    name = "tensor_slice"
    tool = "catapult"

    knobs = [
        {"name": "FoldAxis", "key": "fold_axis", "type": "enum",
         "choices": ("k", "m", "n"), "default": "k",
         "description": "legacy single-axis knob: which dimension ReuseFactor folds. "
                         "Back-compat alias for MFold/KFold/NFold=1 elsewhere; ignored "
                         "(with a warning) if any of MFold/KFold/NFold is set explicitly."},
        {"name": "MFold", "key": "m_reuse_factor", "type": "int", "default": None,
         "description": "independent M-axis fold pass count (legal 1..ceil(m/8)); "
                         "1 (or unset) means M is fully spatial. Explicit MFold/KFold/"
                         "NFold take precedence over legacy FoldAxis+ReuseFactor. Setting "
                         "2+ axes to a value >1 at once (combined folding) is csim-valid "
                         "but not yet synth-valid -- see KFold."},
        {"name": "KFold", "key": "k_reuse_factor", "type": "int", "default": None,
         "description": "independent K-axis fold pass count (legal 1..ceil(k/8)); "
                         "1 (or unset) means K is fully spatial (today's default/legacy "
                         "ReuseFactor axis). See MFold for precedence and the combined-"
                         "folding synth-grid limitation (a hard error is raised if 2+ of "
                         "MFold/KFold/NFold resolve to a pass count >1 and a synth package "
                         "is requested; the general synth grid is a pending roadmap item)."},
        {"name": "NFold", "key": "n_reuse_factor", "type": "int", "default": None,
         "description": "independent N-axis fold pass count (legal 1..ceil(n/8)); "
                         "1 (or unset) means N is fully spatial. See MFold."},
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
            # Resolve via the same independent M/K/N geometry the RTL uses
            # (``package.generate_catapult_pkg`` -> ``resolve_mkn_geometry``),
            # honouring the explicit MFold/KFold/NFold knobs. The legacy
            # single-axis ``fold_axis``/``reuse_factor`` path is the degenerate
            # case of this (one axis folded), so single-axis runs are
            # unchanged; combined folds now report the geometry actually built
            # instead of a single-axis approximation.
            m_rf, k_rf, n_rf = _axis_reuse_factors(
                fold_axis, item.get("reuse_factor", 1),
                item.get("m_reuse_factor"), item.get("k_reuse_factor"),
                item.get("n_reuse_factor"))
            resolved = _geom.resolve_mkn_geometry(
                item["m"], item["k"], item["n"],
                m_reuse_factor=m_rf, k_reuse_factor=k_rf, n_reuse_factor=n_rf,
                name=item.get("name"))
            for w in resolved["warnings"]:
                print(w, file=sys.stderr)
            n_passes = resolved["n_passes"]
            item["k_spatial"] = resolved["k_spatial"]
            item["k_passes"] = resolved["k_passes"]
            item["k_chunks_pad"] = resolved["k_chunks_pad"]
            item["reuse_factor_requested"] = resolved["k_reuse_factor_requested"]
            item["reuse_factor"] = resolved["reuse_factor"]
            item["effective_reuse"] = resolved["effective_reuse"]
            item["fold_axis"] = fold_axis
            item["m_groups"] = resolved["m_spatial"]
            item["m_passes"] = resolved["m_passes"]
            item["grid_rows_pad"] = resolved["grid_rows_pad"]
            item["n_groups"] = resolved["n_spatial"]
            item["n_passes"] = n_passes
            item["grid_cols_pad"] = resolved["grid_cols_pad"]
            item["core_cols"] = item["n"] if n_passes == 1 else 8 * resolved["n_spatial"]
            item["multipliers"] = resolved["multipliers"]
        return items

    def combined_header(self, items):
        return _package().gen_combined_header(items)

    def integration_manifest(self, items):
        return _package().gen_integration_manifest(items)

    def blackbox_tcl(self, items):
        return _package().gen_blackbox_tcl(items)


TARGET = TensorSliceTarget()
