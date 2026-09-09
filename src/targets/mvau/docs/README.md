# mvau target — documentation

The `mvau` target blackboxes FINN's RTL matrix–vector unit (`mvu_vvu_axi` +
DSP-packing cores + `memstream`) into a Vitis-HLS GEMM IP. These docs explain the
knobs and how they compose, in two layers:

1. **[`mvau-rtl-parameters.md`](mvau-rtl-parameters.md)** — the vendored FINN RTL
   itself: every `mvu_vvu_axi` / `memstream` parameter and AXIS port, the derived
   stream widths, the sanity checks the core asserts, and the fixed pipeline
   timing. This is the *contract* the generator must satisfy.

2. **[`config-mapping.md`](config-mapping.md)** — how a gemm-ip-gen config (and the
   hls4ml config it comes from) maps onto those RTL parameters: `ReuseFactor` /
   `fold_axis` resolution to `(PE, SIMD)`, the precision/envelope rules, N-tiling,
   the weight-stationary weight packing, and the requant drain.

Read (1) to understand what the RTL expects; read (2) to understand what a user
knob actually does to the generated hardware.

This target supports Versal parts only, with a single compute core
(`mvu_vvu_8sx9_dsp58`). For the compute core's internal structure (the DSP58
PCOUT cascade, `SEGMENTLEN`), see `finn_space/MVU_space/03_compute_cores.md` at
the repo root.

Source of truth for anything ambiguous, in order: the vendored RTL under
`../rtl_static/`, then `../geometry.py` (folding + derivations), then
`../rtl.py` / `../golden.py` / `../package.py` / `../weightpack.py` (emission).
Where these docs cite a formula, the code is authoritative if they ever diverge.
