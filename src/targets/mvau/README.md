# mvau target — FINN's RTL MVU as a Vitis GEMM IP

`mvau` wraps FINN's matrix–vector unit (`mvu_vvu_axi` + the DSP58 compute core +
`memstream`, vendored under `rtl_static/`) into an RTL blackbox that hls4ml calls
directly from its top-level dataflow region. This file is the operating summary:
what the IP's boundary is, what throughput to expect, when it runs at the
requested ReuseFactor and how to pick the fold knobs so that it does. The knob
semantics and the RTL parameter contract are in `docs/`.

## The boundary

One node = one blackbox = one function, `gemm_stream_<layer>`, which is also the
RTL module name and the JSON `c_function_name`. Its ports are packed bit streams at
hls4ml's own row widths, exact bit counts:

| port | weight-stationary | two-operand (runtime B) |
|---|---|---|
| A in | one `K*act_width`-bit row per beat, M beats per node | same |
| B in | none (weights and bias baked into `memstream`) | one raw B row (`N*w_width`, K beats, row-major) or column (`K*w_width`, N beats, col-major) per beat |
| C out | one `N*out_width`-bit row per beat, M beats per node | same |

Everything between hls4ml's row and the core's beat protocol happens inside the
shim: zero-padding K to `k_pad`, the SF-way SIMD fan-out, the wide-to-narrow
loader gearbox (two-operand), the NF-beat result stitch, bias add and requant.
hls4ml converts its array streams to and from these packed rows with two thin
processes around the call. There is no per-node HLS wrapper and no nested
dataflow region, which is what lets consecutive nodes overlap with no handshake
gap. The manifest's `ports` entry and the per-node `<name>_gemm_ip.h` declaration
carry the contract.

## Throughput model

Terms: `M` rows per node, `SF = k_pad / SIMD`, `NF = n_pad / PE`.

**Within a node (intra-frame).** One row is accepted and one row emitted every
`SF*NF` cycles. This is the II the blackbox JSON reports and it holds on every
fold measured.

**Node to node (inter-frame).** Nodes run back to back through a decoupled
ap_ctrl_chain handshake; up to `MAX_INFLIGHT` nodes may be admitted before the
earlier ones have drained. Weight-stationary nodes have no per-node gap at all:
the interval is exactly `M*SF*NF`. Two-operand nodes additionally have to load
B for each node, and the loader accepts one narrow beat per cycle, so

```
node interval = max( M*SF*NF ,  k_pad*NF  [row-major B]  or  n*SF  [col-major B] )
```

The loader is 2-bank ping-pong, so the load overlaps the previous node's
compute; when the compute term is the larger the node runs at the compute II,
otherwise it is load-bound. The first node of a run always waits for its whole B
load before its first output, which is a one-off cost, not a steady-state one.

**Latency at the ports** (first row in to first row out, B resident):
`fill + 2`, plus `(NF-1)*SF + 2` when `NF > 1`, where `fill = SF +
ceil(ceil(SIMD/3)/SEGMENTLEN) + 2` is the core's pipeline depth. Measured to be
exact across the fold space (`geometry.port_latency_cycles`); the JSON reports it.

## When steady-state II equals ReuseFactor

`ReuseFactor` is `K*N / (PE*SIMD)`, how many times each MAC is used per row. The
II you get is `SF*NF`, so the two coincide only when the conditions below hold.

1. **The fold resolved as requested.** `resolve_fold` honours RF by padding the
   folded dimension, but legalizes out-of-range requests: RF above N for
   `FoldAxis n`, or above `ceil(K/3)` for `FoldAxis k` (SIMD never folds below
   the DSP58's 3 lanes). Check the manifest's `reuse_factor` and
   `effective_reuse` against the request; a warning is printed when they differ.
2. **One fold axis.** With `FoldAxis n` the II is `NF = RF`; with `FoldAxis k` it is
   `SF = RF`. With `FoldAxis kn` both fold by RF and the row II is `RF*RF`; the
   manifest's `effective_reuse` reports that product.
3. **Two-operand nodes are not load-bound.** The load term must not exceed the
   compute term: row-major B needs `k_pad <= M*SF`, col-major B needs
   `n <= M*NF`. Otherwise the interval is the B load count regardless of fold.
4. **The surrounding design sustains it.** The IP never stalls itself, but it is
   paced by whatever feeds it and drains it: hls4ml's pack/unpack processes and
   neighbouring layers must each keep a per-frame interval at or below `M*SF*NF`.
   At small II (RF 1, a few rows per frame) an unpipelined neighbour with its
   ap_ctrl_chain handshake is the usual limiter, not the IP.

## Choosing the fold knobs

Knobs: `ReuseFactor` and `FoldAxis` (`n`, `k`, `kn`), or the direct `PE`/`SIMD`
override; `NTiles`/`KTiles` split a node across parallel cores. `PE` must divide
`n_pad`, `SIMD` must divide `k_pad`, and the DSP58 count is `PE*ceil(SIMD/3)` per
tile.

- **Target II first.** Decide the rows-per-cycle the model needs, `II = SF*NF`,
  then pick the fold that delivers it. For a required II of `R` use `FoldAxis n`
  with `RF = R` (NF folds, SIMD stays `K`, one full DSP cascade per PE) or
  `FoldAxis k` with `RF = R` (SF folds, PE stays `N`). Reach for `kn` only when
  the DSP budget forces a reuse larger than either axis alone allows, and read
  the II as `RF*RF`.
- **Prefer the axis that pads least.** `FoldAxis n` pads N up to a multiple of RF
  and `FoldAxis k` pads K up to a multiple of RF; padding is wasted MACs and, on
  the two-operand loader, wasted load beats (`k_pad`, not `K`, counts).
- **Keep SIMD a multiple of 3 where you can.** A SIMD of 3j fills each DSP58;
  anything else leaves lanes idle in the last DSP of the cascade.
- **For two-operand nodes, fold the axis that hides the load.** Row-major B
  (`k_pad*NF` load beats against `M*SF*NF` compute) is hidden by raising `SF`,
  i.e. `FoldAxis k`, and is exposed by raising `NF`. Col-major B (`n*SF` against
  `M*SF*NF`) is hidden by raising `NF`, i.e. `FoldAxis n`, and exposed by
  raising `SF`. Short-M attention nodes (few query rows) are the ones that go
  load-bound; check condition 3 above before trusting the fold's II.
- **Tiles for capacity, not II.** `NTiles`/`KTiles` add cores in parallel to fit
  a large K or N into per-core limits; the row II is still `SF*NF` of one tile.

## Where things live

- `geometry.py`: fold resolution, padding, DSP estimate, latency and II rules.
- `rtl.py`: the shims (weight-stationary plain/K-tiled/N-tiled, two-operand) and
  the shared decoupled ap_ctrl handshake; `two_operand_ports` is the boundary
  contract for the runtime-B form.
- `golden.py`: the C twins Vitis uses as the blackbox models, and the self-checking
  standalone testbenches.
- `package.py`: package assembly, the blackbox JSON, the declaration headers, the
  manifest, the Vitis smoke runner.
- `run_rtl_tests.py` / `rtl_sv_tb.py`: the xsim regression driving the raw shims
  with random backpressure; its debug mode stamps every beat, which is how the
  numbers above were measured.
- `docs/`: the FINN RTL parameter contract and the full knob mapping.
