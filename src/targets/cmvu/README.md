# cmvu target — choosing KFold / NFold

The cmvu target turns one GEMM layer into a grid of `cmvu_mode1` hard blocks
plus a generated wrapper. It **exposes knobs; it does not pick them.** Every
GEMM layer must be given `KFold` and `NFold`, and the target builds exactly
what they describe (legalizing a non-divisible request, with a warning). The
choice of knobs belongs to whoever writes the design's configuration —
normally the ATLAS config — because it is a design-level trade-off (area vs.
throughput across all layers of a model), not something one layer's
generator can see.

This note is the guidance for making that choice. For the block itself see
`docs/architecture.md` and `docs/mode_1_user_guide.md`; for the wrapper's
design rules see `docs/design.md`.

## Where the knobs go

Per GEMM layer, in the ATLAS config (hls4ml's `ReuseFactor` is ignored by this
target; `FoldAxis`/`MFold` and the tensor_slice fold knobs are rejected):

```python
'ATLASConfig': {
    'Model': {'Backend': 'catapult', 'Target': 'cmvu'},
    'LayerName': {
        'mlp1': {'KFold': 4, 'NFold': 2},
        'mlp2': {'KFold': 4, 'NFold': 2},
        'mlp3': {'KFold': 4, 'NFold': 1},
    },
},
```

## What a fold means

One block is a 4 (K) x 8 (N) int8 tile. A layer computes an (M x K) by
(K x N) product, M rows per frame.

Per axis, with `chunks = ceil(K/4)` (or `ceil(N/8)`) and a fold `F` in
`1..chunks`:

| | K axis | N axis |
|---|---|---|
| spatial blocks | `ks = ceil(chunks_K / KFold)` — a cascade chain | `ns = ceil(chunks_N / NFold)` — rows of the grid |
| temporal passes | `kp = ceil(chunks_K / ks)` | `np = ceil(chunks_N / ns)` |

- **Blocks** = `ks * ns`.
- **Weight tiles per block** = `kp * np`, and a block holds at most **8**.
  This is the one hard limit on folding: a layer needs at least
  `ceil(chunks_K * chunks_N / 8)` blocks.
- `F = 1` is fully spatial (one pass, most blocks); larger `F` trades blocks
  for passes. A request that does not divide evenly legalizes to the
  nearest achievable pass count (e.g. 5 chunks at `F = 4` gives 2 spatial x 3
  passes) and pads; prefer folds that divide `chunks` so no multipliers
  idle on padding.

## What it costs in time

Each A row is replayed once per (N group, K pass): **one row every
`kp * np` cycles**, back to back. Call this the layer's reuse factor
per row; per frame the layer needs `M * kp * np` cycles of work.

- **First-result latency** is about `kp*np + 6 + (ks - 1) + (ns - 1) + 1`
  cycles (block pipeline 6, one cycle per cascade and per grid-row hop).
- **Layers with compile-time weights overlap consecutive frames**, so their
  frame interval is the work alone: **`II = M * kp * np`**. Measured at 3 ns:
  fc 1x32x8 with `KFold=8` (1 block) runs at II 8; fc 1x64x16 with `KFold=8`
  (4 blocks) at II 8.
- **Runtime-B layers** (both operands at run time, e.g. attention `QK^T`,
  `attention x V`) load a new B every frame. When two copies of the layer's
  `kp * np` tiles fit in a block's 8 slots (`kp * np <= 4`), the next
  frame's B loads while the current frame computes, so
  **`II = max(M * kp * np, B load + 2)`**. Otherwise the load waits for the
  current frame's last row, and `II` is about
  `B load + M * kp * np + (block latency and skew) + a few cycles`.
  The B load is one cycle per N column (rounded up to whole 8-column
  groups); with column-major B and `kp > 2`, each 8-column group is followed
  by `4` stall cycles per staged tile (`kp - 2` tiles, or `kp - 1` for an
  N group whose first slot is odd). Row-major B adds `4 * (np - 1)` stall
  cycles per 4 K rows instead.

## Choosing folds for a design

A design's frame interval is set by its slowest stage (every hls4ml layer
is its own pipelined block), so pick folds per layer against one target:

1. **Pick a target frame interval `II`** for the whole design.
2. **For each GEMM layer, choose the fewest blocks with `M * kp * np <= II`**
   and `kp * np <= 8`. Folding further than the target buys nothing; folding
   less leaves a stage slower than the rest.
3. **Prefer folds that divide the chunk counts evenly** (less padding). Among
   equal block counts the choice between folding K or N only moves latency
   slightly (one cycle per cascade block or grid row).
4. **Check the non-GEMM stages** (activations, quantizers, reshapes) meet the
   same `II`; Catapult's schedule report lists each stage's throughput.
5. **For runtime-B layers, keep `kp * np <= 4`** so B is double-buffered, and
   check the B load (see above) fits under `M * kp * np`; keep `ns` small, as
   every block row adds 8 load cycles. Deep K folding is fine but each staged
   tile adds load stalls.

Consequences worth knowing:

- With compile-time weights, `blocks x II` is roughly the layer's tile count
  `chunks_K * chunks_N` (times `M`), so halving `II` roughly doubles blocks.
- Because a block holds 8 tiles, `kp * np` never exceeds 8: at its fewest
  blocks an `M = 1` layer runs at `II = min(8, chunks_K * chunks_N)`, and no
  folding makes it slower.
- Different layers must match in **cycles per frame** (`M * kp * np`), not
  in folds: a conv with `M = 100` at `kp * np = 1` is already `II = 100`.

Worked example (MLP, 3 layers with M = 1, measured at 3 ns):

| Layer | Shape | Folds | Blocks | `kp * np` |
|---|---|---|---|---|
| mlp1 | 1x64x32 | `KFold=4, NFold=2` | 8 | 8 |
| mlp2 | 1x32x16 | `KFold=4, NFold=2` | 2 | 8 |
| mlp3 | 1x16x8 | `KFold=4, NFold=1` | 1 | 4 |

11 blocks, design II 8, latency 55 cycles. Targeting II 4 instead needs about
16 + 4 + 1 = 21 blocks.

Worked example with runtime B (single-head attention, M = 8, measured at
3 ns): projections 8x16x16 at `KFold=4, NFold=1` (2 blocks each), `QK^T`
8x16x8 at `KFold=4, NFold=1` (1 block, 2 staged tiles), `attention x V`
8x8x16 at `KFold=2, NFold=2` (1 block). Every GEMM is `M * kp * np = 32`
with its B double-buffered; the design runs at II 32, the non-GEMM stages
(softmax, transposes, head split/merge) scheduling at 10-27 cycles.

## Limits the knobs cannot lift

Set by the hard block, rejected at package time with the layer named:
int8 operand codes; output width up to 16 bits; output rounding `RND` or
`TRN` and overflow `WRAP` only; bias exactly representable at the product's
fractional precision; stream interface only. See `docs/design.md`.
