#!/usr/bin/env python3
"""Legacy entry point.  Generates Catapult packages from gemm_config.json.

Usage::

    python generate_gemm_ip.py gemm_config.json ./output
"""
import sys
sys.path.insert(0, "src")

import json
from pathlib import Path

from gemm_ip.metadata import normalize_gemm_config
from gemm_ip.catapult import generate_catapult_pkg, gen_combined_header, gen_integration_manifest

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: generate_gemm_ip.py <config.json> <output_dir>", file=sys.stderr)
        sys.exit(1)

    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    layers = normalize_gemm_config(cfg) if "layers" in cfg else cfg.get("layers", [])
    output_dir = sys.argv[2]

    for layer in layers:
        generate_catapult_pkg(
            layer.get("m", 8), layer.get("k", 8), layer.get("n", 8),
            layer.get("name", "layer"),
            output_dir,
            interface=layer.get("interface", "stream"),
        )

    combined_dir = Path(output_dir)
    (combined_dir / "gemm_ip_combined.h").write_text(gen_combined_header(layers))
    (combined_dir / "integration_manifest.json").write_text(gen_integration_manifest(layers) + "\n")
    print(f"Generated {len(layers)} Catapult packages in {output_dir}")
