"""generic target: behavioral-HLS GEMM for Vitis, bound to the Target contract.

No RTL blackbox / hardblock — Vitis HLS synthesizes the emitted C++ directly. The
four func types (stream / array x weighted / const_weights) are the hls4ml Vitis seam
(nnet::gemm_{stream,stream_const_weights,array,array_const_weights}); const_weights weights
are baked into <name>_weights.h and fed to the IP by the top.
"""

import subprocess
import sys
from pathlib import Path

_targets_root = str(Path(__file__).resolve().parent.parent)
if _targets_root not in sys.path:
    sys.path.insert(0, _targets_root)
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from base import Target  # noqa: E402
import hls as _hls  # noqa: E402
import golden as _golden  # noqa: E402
import package as _package  # noqa: E402


def _valid_reuse_factors(n_in, n_out):
    """hls4ml's reuse-factor rules (fpga_backend._validate_reuse_factor), verbatim:
    rf must divide n_in*n_out; below n_in the multiplier count must be a multiple of
    n_out; above n_in, rf must be a multiple of n_in."""
    import math
    valid = []
    for rf in range(1, n_in * n_out + 1):
        multfactor = min(n_in, rf)
        multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
        ok = ((multiplier_limit % n_out) == 0) or (rf >= n_in)
        ok = ok and (((rf % n_in) == 0) or (rf < n_in))
        ok = ok and (((n_in * n_out) % rf) == 0)
        if ok:
            valid.append(rf)
    return valid


def _closest_reuse_factor(valid_rf, chosen_rf):
    """hls4ml's get_closest_reuse_factor: nearest valid value, smaller on ties."""
    from bisect import bisect_left
    pos = bisect_left(valid_rf, chosen_rf)
    if pos == 0:
        return valid_rf[0]
    if pos == len(valid_rf):
        return valid_rf[-1]
    before, after = valid_rf[pos - 1], valid_rf[pos]
    return before if (after - chosen_rf) >= (chosen_rf - before) else after


def _snap_reuse_factor(item):
    """Legalize a manifest item's ReuseFactor against hls4ml's validation rules.

    ReuseFactor is a pure pass-through from hls4ml's HLSConfig into the manifest
    (no ATLASConfig knob for it) -- but hls4ml's init_dense skips
    set_closest_reuse_factor for Strategy=GEMM layers, so the raw HLSConfig value
    reaches this manifest unvalidated; the generic resource core then builds the
    remainder regime with a non-dividing block factor (several x the area of the
    snapped point). So the generic target legalizes it itself here: snap, print
    hls4ml's own warning, keep the request as reuse_factor_requested. The combined
    header reads the snapped value from the manifest (gemm_rf<CONFIG_T>) instead of
    hls4ml's CONFIG_T::reuse_factor.
    """
    name = item.get("name", "?")
    n_in = int(item.get("gemm_k", item.get("k", item.get("n_in", 0))) or 0)
    n_out = int(item.get("gemm_n", item.get("n", item.get("n_out", 0))) or 0)
    chosen = int(item.get("reuse_factor", 1) or 1)
    item["reuse_factor_requested"] = chosen
    if n_in <= 0 or n_out <= 0:
        return
    valid = _valid_reuse_factors(n_in, n_out)
    if chosen in valid:
        return
    closest = _closest_reuse_factor(valid, chosen)
    print(f'WARNING: Invalid ReuseFactor={chosen} in layer "{name}".'
          f'Using ReuseFactor={closest} instead. Valid ReuseFactor(s): '
          f'{",".join(map(str, valid))}.')
    item["reuse_factor"] = closest


class GenericTarget(Target):
    name = "generic"
    tool = "vitis"
    # The combined header reads either ROM order straight from CONFIG_T (weights_row_major).
    weight_layouts = ("column_major", "row_major")

    knobs = [
        {"name": "Strategy", "key": "strategy", "type": "enum",
         "choices": ("latency", "resource"), "default": "latency",
         "description": "which generic kernel body to emit"},
    ]

    def geometry(self, shape):
        m, k, n = shape
        return {"gemm_m": m, "gemm_k": k, "gemm_n": n, "n_in": k, "n_out": n}

    def emit_rtl(self, shape, **kwargs):
        raise NotImplementedError(
            "generic is a behavioral-HLS target: Vitis HLS generates the RTL from "
            "C++; there is no hand-written RTL. Use emit_behavioral() / package().")

    def emit_behavioral(self, shape, name="gemm_generic", **kwargs):
        # The synthesizable C++ compute — the four entry points (shape-independent).
        return _hls.gemm_ip_header(name)

    def golden(self, shape, interface="array", weights_in_core=False,
               name="gemm_generic", **kwargs):
        m, k, n = shape
        return _golden.tb_cpp(name, m, k, n, interface, weights_in_core)

    def package(self, shape, cfg):
        cfg = dict(cfg)
        name = cfg.pop("name")
        self.validate_knobs(cfg, name)
        output_dir = cfg.pop("output_dir")
        m, k, n = shape
        # A weight-stationary layer is signalled by a baked weight matrix (the CLI loads
        # it from the layer's .dat). generate_generic_pkg then emits the ROM header and
        # the const_weights standalone top. Drop keys the standalone emitter has no use for
        # (e.g. gemm_k_spatial — a spatial-partition knob for the RTL targets, not the
        # behavioral one) so the config-driven and unit paths share one entry point.
        weight_matrix = cfg.pop("weight_matrix", None)
        cfg.pop("gemm_k_spatial", None)
        # Two-operand routing hints the CLI forwards for the RTL targets: generic derives
        # weights_in_core from the baked weight matrix and has no runtime-B variant, so
        # drop both (weights_in_core would also collide with the explicit kwarg below).
        cfg.pop("weights_in_core", None)
        cfg.pop("second_operand_row_major", None)
        return _package.generate_generic_pkg(
            m, k, n, name, output_dir,
            weights_in_core=weight_matrix is not None,
            weight_matrix=weight_matrix, **cfg)

    def verify(self, package):
        pkg = Path(package)
        name = pkg.name
        required = [f"{name}_gemm_ip.h", f"{name}_config.h", f"{name}_bias.h", f"{name}_top.cpp",
                    f"{name}_tb.cpp", "nnet_types.h", "run_vitis.tcl"]
        missing = [f for f in required
                   if not (pkg / f).is_file() or (pkg / f).stat().st_size == 0]
        if missing:
            raise RuntimeError(f"{name}: package incomplete, missing/empty: {missing}")
        return True

    def rtl_test(self, package=None, **kwargs):
        """Run Vitis csim + csynth on a generated package via ``vitis-run``.

        Requires ``vitis-run`` on PATH (source the Vitis settings first). Returns the
        process-style code: 0 = pass, 1 = fail, 2 = vitis-run not found (skipped).
        """
        import shutil
        if shutil.which("vitis-run") is None:
            print("ERROR: vitis-run not found on PATH (source Vitis settings64.sh)",
                  file=sys.stderr)
            return 2
        if package is None:
            raise ValueError("rtl_test(package=...) needs a generated package dir")
        pkg = Path(package)
        r = subprocess.run(["vitis-run", "--tcl", "run_vitis.tcl", "--mode", "hls"],
                           cwd=str(pkg), text=True, capture_output=True)
        log = r.stdout + r.stderr
        ok = ("GENERIC CSIM PASS" in log) and (r.returncode == 0)
        if not ok:
            print(log[-4000:], file=sys.stderr)
        return 0 if ok else 1

    # ── batch orchestration (multi-package; used by the CLI) ─────────────────
    def normalize_config(self, cfg):
        # Target-independent: same item schema every target consumes.
        from gemm_ip.config import _normalize_config_items
        items = _normalize_config_items(cfg)
        # ReuseFactor is a pure pass-through from hls4ml's HLSConfig (no ATLASConfig
        # knob for it) -- legalize it here, before both the per-item package() calls
        # and combined_header() see the manifest.
        for item in items:
            _snap_reuse_factor(item)
        return items

    def combined_header(self, items):
        # Shape-generic: one templated definition covers every layer. items drives
        # the compile-time gemm_strategy<id> dispatch (latency vs resource per layer).
        return _hls.combined_header(items)

    def integration_manifest(self, items):
        import json
        cores = [{"name": item["name"], "kind": "behavioral", "header": "gemm_ip_combined.h"}
                 for item in items]
        return json.dumps({"tool": "vitis", "flow": "behavioral", "cores": cores}, indent=2)

    def sources_tcl(self, items):
        # Behavioral flow: the combined header IS the implementation and the firmware
        # includes it via the -I<pkg> path, so Vitis needs no add_files. hls4ml sources
        # this file unconditionally, so emit an explanatory no-op rather than nothing.
        names = ", ".join(item["name"] for item in items)
        return ("# generic (behavioral-HLS) GEMM IP: header-only, synthesized from\n"
                "# gemm_ip_combined.h on the firmware include path -- no add_files needed.\n"
                f"# cores: {names}\n"
                'puts "gemm-ip-gen (generic): behavioral GEMM IP, no blackbox sources to add"\n')


TARGET = GenericTarget()
