"""c-generic target: behavioral-HLS GEMM for Catapult, bound to the Target contract.

The Catapult twin of the (resource-only) Vitis ``v-generic`` target: no RTL
blackbox / hardblock, Catapult synthesizes the emitted C++ directly. Shares
the tool-neutral ReuseFactor-snapping and geometry rules with ``v-generic`` via
``gemm_ip.behavioral`` (see jojo-track/open/catapult-generic-target/plan.md,
step 1) but does not share an emitter: the kernel body (``hls.py``) and
package layout (``package.py``) are Catapult idiom (``ac_int`` / ``ac_fixed``,
``ac_channel``, ``hls_unroll`` / ``hls_pipeline_init_interval``), added in
steps 3-4. Until then every content-producing entry point raises
NotImplementedError.
"""

from ..base import Target
from . import package as _package

from gemm_ip.behavioral import geometry as _behavioral_geometry
from gemm_ip.behavioral import snap_reuse_factor as _snap_reuse_factor


class CGenericTarget(Target):
    name = "generic"
    tool = "catapult"
    # Same ROM layouts v-generic's combined header can read straight from CONFIG_T.
    weight_layouts = ("column_major", "row_major")

    # Resource-only target, same as v-generic: no per-layer knobs.
    knobs = []

    def geometry(self, shape):
        return _behavioral_geometry(shape)

    def emit_rtl(self, shape, **kwargs):
        raise NotImplementedError(
            "generic (catapult): emit_rtl: kernel emitter is step 3")

    def emit_behavioral(self, shape, **kwargs):
        return _package.emit_behavioral(shape, **kwargs)

    def golden(self, shape, **kwargs):
        return _package.golden(shape, **kwargs)

    def package(self, shape, cfg):
        return _package.generate_c_generic_pkg(shape, cfg)

    def verify(self, package):
        return _package.verify(package)

    def rtl_test(self, package=None, **kwargs):
        return _package.rtl_test(package=package, **kwargs)

    # ── batch orchestration (multi-package; used by the CLI) ─────────────────
    def normalize_config(self, cfg):
        # Same manifest item schema as v-generic; ReuseFactor is legalized the
        # same way, before both per-item package() calls and combined_header()
        # see the manifest (mirrors v-generic/flow.py's normalize_config).
        from gemm_ip.config import _normalize_config_items
        items = _normalize_config_items(cfg)
        for item in items:
            _snap_reuse_factor(item)
        return items

    def combined_header(self, items):
        return _package.emit_behavioral(items)

    def integration_manifest(self, items):
        import json
        cores = [{"name": item["name"], "kind": "behavioral", "header": "gemm_ip_combined.h"}
                 for item in items]
        return json.dumps({"tool": "catapult", "flow": "behavioral", "cores": cores}, indent=2)

    def sources_tcl(self, items):
        names = ", ".join(item["name"] for item in items)
        return ("# generic/catapult (behavioral-HLS) GEMM IP: header-only, synthesized from\n"
                "# gemm_ip_combined.h on the firmware include path -- no add_files needed.\n"
                f"# cores: {names}\n"
                'puts "gemm-ip-gen (generic/catapult): behavioral GEMM IP, no blackbox sources to add"\n')


TARGET = CGenericTarget()
