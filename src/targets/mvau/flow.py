"""mvau target: FINN's RTL MVU (mvu_vvu_axi + DSP-packing cores) blackboxed into
a Vitis HLS dataflow top.

Tool: Vitis HLS. The DSP-dense (2-4 MACs/DSP) counterpart to the soft-logic
``generic`` Vitis target, and the Vitis analogue of Catapult's ``tensor_slice``.
Delegates the Target contract to sibling modules: ``geometry`` (folding search),
``rtl`` (wrapper fill + shim), ``golden`` (C twin + testbench), ``package`` (the
blackbox package: JSON + dataflow top + drain + run_vitis.tcl). The vendored FINN
RTL lives in ``rtl_static/`` (see FINN_COMMIT.txt).

Interface, shim and packaging are validated end-to-end on Vitis 2025.2 -- see
jojo-track/open/mvau-vitis-target and temp_space/mvau-spike (cosim PASS).
"""

import sys
from pathlib import Path

# base.py is one level up (src/targets/); sibling modules are alongside.
_targets_root = str(Path(__file__).resolve().parent.parent)
if _targets_root not in sys.path:
    sys.path.insert(0, _targets_root)
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from base import Target  # noqa: E402
import geometry as _geom  # noqa: E402

#: The vendored FINN static RTL compiled alongside every generated wrapper/shim.
RTL_STATIC = Path(__file__).resolve().parent / "rtl_static"
STATIC_SOURCES = [
    "mvu_vvu_axi.sv", "replay_buffer.sv", "memstream.sv",
    "mvu_4sx4u.sv", "mvu_8sx8u_dsp48.sv", "mvu_vvu_8sx9_dsp58.sv",
]


def _normalize_mvau_items(cfg):
    """Normalize a gemm_config (hls4ml-named dict / list / single dict) into mvau
    items, self-contained (no core-config import). Carries the fields mvau needs:
    precisions, part, folding knobs, id, weights_in_core."""
    def one(name, it):
        m = it.get("gemm_m", it.get("m", 1))
        k = it.get("gemm_k", it.get("k", it.get("n_in", 8)))
        n = it.get("gemm_n", it.get("n", it.get("n_out", 8)))
        return {
            "name": name, "emit_name": name,
            "m": int(m), "k": int(k), "n": int(n),
            "interface": it.get("interface", "stream"),
            "protocol": it.get("protocol", {}),
            "gemm_ip_id": it.get("gemm_ip_id", name),
            "gemm_ip_index": it.get("gemm_ip_index"),
            "strategy": it.get("strategy", "latency"),
            "reuse_factor": it.get("reuse_factor", 1),
            "parallelization_factor": it.get("parallelization_factor", 1),
            "target_cycles": it.get("target_cycles"),
            "input_precision": it.get("input_precision"),
            "weight_precision": it.get("weight_precision"),
            "output_precision": it.get("output_precision"),
            "accum_precision": it.get("accum_precision"),
            "bias_precision": it.get("bias_precision"),
            "clock_period_ns": it.get("clock_period_ns"),
            "part": it.get("part"),
            "weights_in_core": bool(it.get("weights_in_core", False)),
            "weight_file": it.get("weight_file"),
            "n_tiles": int(it.get("n_tiles", 1) or 1),
            # DEBUG: mvau user-directed fold knobs injected via ATLASConfig (bypassing
            # hls4ml's gemm_config). TODO: Ruthwik change this.
            "pe": it.get("pe"),
            "simd": it.get("simd"),
            "k_tiles": it.get("k_tiles"),
            "has_bias": bool(it.get("has_bias", True)),
            # Two-operand (runtime-B) nodes: B beat order. Absent for const_weights items.
            "second_operand_row_major": it.get("second_operand_row_major"),
        }
    if isinstance(cfg, list):
        return [one(it.get("name"), it) for it in cfg]
    if isinstance(cfg, dict):
        keyed = ("gemm_m" in cfg or "m" in cfg) and "name" in cfg
        if keyed:
            return [one(cfg.get("name"), cfg)]
        return [one(name, it) for name, it in cfg.items()]
    raise TypeError(f"Unsupported config format: {type(cfg)}")


def _rtl():
    import rtl
    return rtl


def _golden():
    import golden
    return golden


def _package():
    import package
    return package


class MvauTarget(Target):
    name = "mvau"
    tool = "vitis"

    def geometry(self, shape, **kwargs):
        """Full folding/geometry plan for *shape* ``(m, k, n)``.

        Accepts the same knobs as :func:`geometry.fold_plan`
        (precisions, part, clock_period_ns, reuse_factor, strategy,
        target_cycles, parallelization_factor, n_tiles, weights).
        """
        m, k, n = shape
        return _geom.fold_plan(m, k, n, **kwargs)

    def emit_rtl(self, shape, **kwargs):
        """Filled ``mvu_vvu_axi`` wrapper + the active-high-reset/CE shim."""
        return _rtl().generate_shim(shape, **kwargs)

    def emit_behavioral(self, shape, **kwargs):
        """Same RTL: the FINN cores carry their own FORCE_BEHAVIORAL/VERILATOR
        behavioral path, so there is no separate behavioral emission."""
        return _rtl().generate_shim(shape, force_behavioral=True, **kwargs)

    def golden(self, shape, seed=42, **kwargs):
        return _golden().generate_tb(shape, seed=seed, **kwargs)

    def package(self, shape, cfg):
        cfg = dict(cfg)
        name = cfg.pop("name")
        output_dir = cfg.pop("output_dir")
        # weights_in_core True (or absent) => weight-stationary (baked-B) IP; False =>
        # two-operand (runtime-B) IP fed both operands as streams. The QK^T / A.V
        # attention matmuls are the two-operand case.
        if not cfg.get("weights_in_core", True):
            return _package().generate_two_operand_pkg(shape, name, output_dir, **cfg)
        return _package().generate_mvau_pkg(shape, name, output_dir, **cfg)

    def verify(self, package):
        pkg = Path(package)
        name = pkg.name
        required = [f"{name}_gemm_ip.h", f"{name}_top.cpp", f"{name}_core.cpp",
                    f"{name}.json", f"{name}_tb.cpp", "run_vitis.tcl"]
        missing = [f for f in required
                   if not (pkg / f).is_file() or (pkg / f).stat().st_size == 0]
        # vendored static RTL must be present too
        rtl_missing = [s for s in STATIC_SOURCES if not (pkg / "rtl_static" / s).is_file()]
        if missing or rtl_missing:
            raise RuntimeError(
                f"{name}: package incomplete, missing/empty: {missing + rtl_missing}")
        return True

    def rtl_test(self, cases=None, seeds=None, keep=False):
        return _package().run_vitis_smoke(cases=cases, keep=keep)

    # ── batch orchestration (multi-package; used by the CLI) ─────────────────
    def normalize_config(self, cfg):
        # Self-contained: gemm_ip.config imports tensor_slice's `geometry` by bare
        # name, which collides with mvau's own `geometry` module in sys.modules.
        return _normalize_mvau_items(cfg)

    def combined_header(self, items):
        return _package().gen_combined_header(items)

    def integration_manifest(self, items):
        return _package().gen_integration_manifest(items)

    def sources_tcl(self, items):
        """RTL-blackbox seam: emit ``add_files -blackbox`` for each core so
        hls4ml's Vitis writer brings the IP into the build (unlike the base
        no-op / the header-only ``generic`` target)."""
        return _package().gen_sources_tcl(items)

    def finalize(self, items, output_dir):
        """Post-batch: dedup the identical vendored FINN static RTL across IPs so a
        multi-mvau design doesn't hand Vitis two same-named blackbox .sv files."""
        _package().hoist_shared_static(items, output_dir)


TARGET = MvauTarget()
