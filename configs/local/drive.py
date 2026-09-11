"""Drive gemm-ip-gen from a synthetic hls4ml-style gemm_config.json — no hls4ml run.

Emits every mvau node in the config, structurally verifies each package, and prints
the per-node cost-model resources (from the manifest, which now carries them). This
stands in for a full ATLAS flow: it exercises the exact CLI/normalize/package path
hls4ml would drive, on a config shaped like the one hls4ml emits. Repo-local scratch.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # <repo>/configs/local
ROOT = HERE.parents[1]                       # <repo> (gemm-ip-gen)
sys.path.insert(0, str(ROOT / "src"))
from gemm_ip.registry import load_target  # noqa: E402

CFG = HERE / "synthetic_mvau_gemm_config.json"
OUT = ROOT / "temp_space" / "synthetic-config" / "out"   # packages -> gitignored scratch


def main():
    t = load_target("mvau")
    cfg = json.loads(CFG.read_text())
    items = t.normalize_config(cfg)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[config] {CFG.name}: {len(items)} node(s)\n")
    ok = True
    for it in items:
        pkg = t.package(
            (it["m"], it["k"], it["n"]),
            {k: it.get(k) for k in
             ("name", "input_precision", "weight_precision", "output_precision",
              "clock_period_ns", "part", "reuse_factor", "strategy", "target_cycles",
              "parallelization_factor", "n_tiles", "weights_in_core",
              "second_operand_row_major", "interface")} | {"output_dir": str(OUT)},
        )
        try:
            t.verify(pkg)
            v = "OK"
        except Exception as e:  # noqa: BLE001
            v, ok = f"FAIL ({e})", False
        res = it["weights_in_core"] and "baked" or "two-operand"
        print(f"  {it['name']:18} {it['m']:>2}x{it['k']:>3}x{it['n']:>3}  "
              f"rf={it['reuse_factor']:<4} {res:12} verify={v}")

    # Batch artifacts (combined header, manifest, sources tcl) — same as the CLI writes.
    (OUT / "gemm_ip_combined.h").write_text(t.combined_header(items))
    (OUT / "integration_manifest.json").write_text(t.integration_manifest(items) + "\n")
    (OUT / "gemm_ip_sources.tcl").write_text(t.sources_tcl(items))
    t.finalize(items, str(OUT))

    man = json.loads((OUT / "integration_manifest.json").read_text())
    print("\n[manifest] per-node cost-model resources:")
    for c in man["cores"]:
        r = c.get("resources") or {}
        print(f"  {c['name']:18} DSP={r.get('dsp'):>4}  BRAM18={r.get('bram18'):>4}")

    print(f"\n[result] {'ALL PASS' if ok else 'FAILURES'} — packages in {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
