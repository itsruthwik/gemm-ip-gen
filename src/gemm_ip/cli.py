"""
Unified CLI entry point for gemm-ip-gen.

Usage::

    # tensor_slice (the default target)
    python -m gemm_ip --m 8 --k 8 --n 8 --name gemm_8x8x8

    # From config file
    python -m gemm_ip --config gemm_config.json
"""

import argparse
import json
from pathlib import Path

from gemm_ip.registry import TARGETS


def main():
    parser = argparse.ArgumentParser(
        description="Generate GEMM IP blackbox packages for a hardblock target"
    )
    parser.add_argument("--target", choices=TARGETS, default="tensor_slice",
                        help="Hardblock target (default: tensor_slice)")
    parser.add_argument("--describe", type=str, metavar="TARGET",
                        help="Print JSON capability metadata {name, tool} for TARGET "
                             "and exit (the target->tool source of truth for callers "
                             "that must not hardcode target knowledge)")
    parser.add_argument("--list-targets", action="store_true",
                        help="Print the registered target names as JSON and exit")
    parser.add_argument("--config", type=str, help="Path to gemm_config.json")
    parser.add_argument("--m", type=int, default=8, help="GEMM M (rows per tile)")
    parser.add_argument("--k", type=int, default=8, help="GEMM K (inner dimension)")
    parser.add_argument("--n", type=int, default=8, help="GEMM N (output columns)")
    parser.add_argument("--name", type=str, default="gemm_8x8x8", help="Package name")
    parser.add_argument("--interface", choices=("stream", "array"), default="stream",
                        help="Interface type for generated package metadata/dispatch (default: stream)")
    parser.add_argument("--k-spatial", type=int, default=None,
                        help="Number of spatial K grid partitions (default: full K-chunk unroll)")
    parser.add_argument("--n-frames", type=int, default=1,
                        help="Number of frames the wrapper feeds back-to-back (default: 1). "
                             ">1 builds a multi-frame back-to-back unit package for cosim.")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory (default: ./output)")
    args = parser.parse_args()

    # Capability queries: metadata only, no generation. Each loads at most one target,
    # so the one-target-per-process module-import model (see registry.py) is respected.
    if args.list_targets:
        print(json.dumps({"targets": list(TARGETS)}))
        return
    if args.describe:
        from gemm_ip.registry import load_target
        t = load_target(args.describe)
        print(json.dumps({"name": t.name, "tool": t.tool}))
        return

    _run(args)


def _run(args):
    from gemm_ip.registry import load_target

    target = load_target(args.target)

    if args.config:
        from gemm_ip import weights as _weights
        cfg_path = Path(args.config)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        items = target.normalize_config(cfg)
        for item in items:
            # Weight-stationary (const-weight): bake weights into the core ROM + csim
            # header. The .dat is column-major [n][k] raw ints emitted by hls4ml, path
            # relative to the config file; load_weight_dat returns B as [K, N].
            weight_matrix = None
            if item.get("weights_in_core") and item.get("weight_file"):
                dat = (cfg_path.parent / item["weight_file"]).resolve()
                weight_matrix = _weights.load_weight_dat(str(dat), item["n"], item["k"])
            target.package(
                (item["m"], item["k"], item["n"]),
                {
                    "name": item["name"],
                    "output_dir": args.output_dir,
                    "interface": item.get("interface", "stream"),
                    "output_precision": item.get("output_precision"),
                    "gemm_k_spatial": item.get("gemm_k_spatial"),
                    "input_precision": item.get("input_precision"),
                    "weight_precision": item.get("weight_precision"),
                    "clock_period_ns": item.get("clock_period_ns"),
                    "weight_matrix": weight_matrix,
                    # Forwarded for targets that select core / fold from them (mvau).
                    # tensor_slice tolerates unknowns via **_ignored; generic accepts
                    # part/strategy/reuse_factor as params. (weights_in_core is NOT
                    # forwarded — generic's flow.package sets it explicitly.)
                    "part": item.get("part"),
                    "reuse_factor": item.get("reuse_factor"),
                    "strategy": item.get("strategy"),
                    "parallelization_factor": item.get("parallelization_factor"),
                    "target_cycles": item.get("target_cycles"),
                    "n_tiles": item.get("n_tiles"),
                },
            )
        output_dir = Path(args.output_dir)
        (output_dir / "gemm_ip_combined.h").write_text(target.combined_header(items))
        (output_dir / "integration_manifest.json").write_text(target.integration_manifest(items) + "\n")
        # The tool-agnostic integration seam: the frontend sources this one file, and
        # the target decides what it contains (blackbox add_files vs. header-only no-op).
        (output_dir / "gemm_ip_sources.tcl").write_text(target.sources_tcl(items))
        # Catapult-only artifact: only targets welded to Catapult emit it.
        if hasattr(target, "blackbox_tcl"):
            (output_dir / "catapult_gemm_blackboxes.tcl").write_text(target.blackbox_tcl(items))
        # Optional post-batch step (e.g. mvau dedups shared static RTL across IPs).
        if hasattr(target, "finalize"):
            target.finalize(items, str(output_dir))
    else:
        target.package(
            (args.m, args.k, args.n),
            {
                "name": args.name,
                "output_dir": args.output_dir,
                "interface": args.interface,
                "gemm_k_spatial": args.k_spatial,
                # Standalone unit packages use integer operand codes / int16 result
                # lanes; an integer result type makes the wrapper drain emit
                # value.to_int() so the ac_fixed rescale accumulator converts cleanly
                # to the int16 output lane (the real hls4ml flow passes ac_fixed here).
                "output_precision": "ac_int<16, true>",
                "n_frames": args.n_frames,
            },
        )


if __name__ == "__main__":
    main()
