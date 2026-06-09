"""
Unified CLI entry point for gemm-ip-gen.

Usage::

    # Catapult (default)
    python -m gemm_ip --m 8 --k 8 --n 8 --name gemm_8x8x8

    # Vitis
    python -m gemm_ip --backend vitis --m 8 --k 8 --n 8 --name gemm_8x8x8

    # From config file
    python -m gemm_ip --backend vitis --config gemm_config.json
"""

import argparse
import json
from pathlib import Path

from gemm_ip.metadata import SUPPORTED_BACKENDS, normalize_gemm_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate GEMM IP packages for Catapult or Vitis HLS"
    )
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default="catapult",
                        help="Target HLS backend (default: catapult)")
    parser.add_argument("--config", type=str, help="Path to gemm_config.json")
    parser.add_argument("--m", type=int, default=8, help="GEMM M (rows per tile)")
    parser.add_argument("--k", type=int, default=8, help="GEMM K (inner dimension)")
    parser.add_argument("--n", type=int, default=8, help="GEMM N (output columns)")
    parser.add_argument("--name", type=str, default="gemm_8x8x8", help="Package name")
    parser.add_argument("--interface", choices=("stream", "array"), default="stream",
                        help="Interface type for generated package metadata/dispatch (default: stream)")
    parser.add_argument("--k-spatial", type=int, default=None,
                        help="Number of spatial K grid partitions (default: full K-chunk unroll)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory (default: ./output)")
    args = parser.parse_args()

    if args.backend == "catapult":
        _run_catapult(args)
    elif args.backend == "vitis":
        _run_vitis(args)


def _run_catapult(args):
    from gemm_ip.catapult import generate_catapult_pkg, gen_combined_header, gen_integration_manifest, gen_blackbox_tcl, _normalize_config_items

    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        items = _normalize_config_items(cfg)
        for item in items:
            generate_catapult_pkg(
                item["m"], item["k"], item["n"], item["name"],
                args.output_dir,
                interface=item.get("interface", "stream"),
                output_precision=item.get("output_precision"),
                gemm_k_spatial=item.get("gemm_k_spatial"),
            )
        output_dir = Path(args.output_dir)
        (output_dir / "gemm_ip_combined.h").write_text(gen_combined_header(items))
        (output_dir / "integration_manifest.json").write_text(gen_integration_manifest(items) + "\n")
        (output_dir / "catapult_gemm_blackboxes.tcl").write_text(gen_blackbox_tcl(items))
    else:
        generate_catapult_pkg(
            args.m, args.k, args.n, args.name,
            args.output_dir, interface=args.interface, gemm_k_spatial=args.k_spatial
        )


def _run_vitis(args):
    from gemm_ip.vitis import generate_vitis_pkg, generate_from_config_file

    if args.config:
        generate_from_config_file(args.config, args.output_dir)
    else:
        item = normalize_gemm_config({
            "name": args.name, "m": args.m, "k": args.k, "n": args.n,
            "backend": "vitis", "interface": args.interface, "gemm_k_spatial": args.k_spatial,
        })[0]
        generate_vitis_pkg(item, args.output_dir)


if __name__ == "__main__":
    main()
