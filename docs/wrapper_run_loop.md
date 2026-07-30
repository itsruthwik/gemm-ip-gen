# The Wrapper RUN Loop

This note documents the frame schedule of the generated Catapult C++ wrapper —
the `RUN` / `RUN_ARRAY` loop emitted into every package's public header by
`gen_public_header` (`src/gemm_ip/catapult.py`). For the timing formulas
(`latency_cycles`, frame slots, per-mode `first_out`) see
`wrapper_timing_model.md`.

## Execution model

The core (`{name}_ccore`, bound to `{name}_core.v` via `ac_blackbox`) is a
stateful blackbox that advances **one clock cycle per `gemm.run()` call**. The
wrapper's loop structure therefore *is* the cycle schedule: one loop iteration
= one `run()` call = one core cycle.

One call of a wrapper entry point processes **one GEMM frame**: M rows of A
against the N weight columns, producing M result rows. Entry points:

- `{name}_gemm_ip_stream_const_weights(a_stream, weight_cols, biases, res_stream)`
  — A rows from an `ac_channel`, weights resident in an array.
- `{name}_gemm_ip_stream(...)` — thin variant that first reads the N weight
  columns from a channel, then calls the const-weights version.
- `{name}_gemm_ip_array(a_rows[], weight_cols[], biases[], results[])` — array
  in/out (the einsum / attention interface).

## Function anatomy

```cpp
static {name}_ccore gemm;        // the stateful core, shared across calls
int captured = 0;                // result rows captured so far (this frame)
ac_int<bias_bits, false> bias_packed = 0;

BIAS_PACK: ...                   // unrolled; packs ZERO bias for the core —
                                 // the real bias is added at capture time in
                                 // full precision, after the rescale

RUN: for (step = 0; step < total_steps; step++) { ... }  // the frame schedule

DRAIN_PADDED_ROWS: ...           // MR - M idle calls (8-row burst headroom)
```

```text
period      = total_beats + 1
total_steps = (n_frames - 1) * period + first_out + MR + 6
```

`first_out` is the mode-aware first output offset (full-K-spatial packages get
the shorter single-pass value); `MR = grid_rows * 8 >= M`. The `+6` is the
port-lag tail (3 calls worst-case RTL lag + 2 spare + 1 for the preload beat).
Generation fails hard if the single-frame budget `first_out + M + 6` cannot
cover `total_beats + 2` — i.e. if the loop would be shorter than the feed.

## What one RUN iteration contains

Every iteration executes the same body; step-index guards select what is
active:

**1. Step decode + operand packing** — active only on a frame's data beats:

```cpp
bool in_feed     = step < feed_total;              // feed_total = n_frames * period
int  p           = in_feed ? step % period : period;
bool feeding_now = in_feed && p >= 1 && p <= total_beats;
int  pf = p - 1;
int  kc = pf / input_beats;        // K-chunk index   (chunked packages)
int  t  = pf % input_beats;        // row/col beat within the pass
```

`p == 0` is the frame's leading preload/idle beat; `p == 1..total_beats` are
its data beats. With `n_frames == 1` the whole feed is one frame, so
`feed_total = period` and every step past it is an idle drain call.

- *Chunked packages* (`gemm_k_spatial == 1`): the feed makes `k_chunks` passes
  of `max(M,N)` beats. During chunk 0 each logical A row is read from the
  source once (`a_stream.read()` / `a_rows[t]`) and its 8 K-bytes are packed
  into the per-tile 64-bit lane of the blackbox word (`ROW_PACK_DIRECT`);
  the remaining K-chunks are pre-packed into `a_replay[k_chunks][input_beats]`
  and replayed on the later passes. B columns are packed from `weight_cols`
  every pass (`COL_PACK`).
- *Full-K-spatial packages* (`gemm_k_spatial == k_chunks > 1`): a single
  `max(M,N)`-beat pass; each beat carries one logical A row / B column with
  **all** K chunks packed into a widened `64·k_chunks`-bit word
  (`ROW_PACK_FULL_KC` / `COL_PACK_FULL_KC`). No replay storage.

Outside the feed window the words are zero and this section is inert.

**2. The core call** — exactly one per iteration:

```cpp
ac_int<1,false> feed_valid    = feeding_now ? 1 : 0;
ac_int<1,false> frame_preload = (in_feed && p == 0) ? 1 : 0;
gemm.run(a_rows, b_cols, bias_packed, frame_preload, feed_valid, c_row, v, l);
```

The `p == 0` beat is the frame delimiter the core keys on: it is the
non-`in_valid` call that drops the core's `feeding` flag so the next frame
allocates a fresh slot. Beats `p == 1..total_beats` are the `in_valid` data
beats (the core's frame clock is 0 at the first of them). Every step past the
feed region is an idle call that advances the core's clock.

`frame_preload` must stay a *live* signal. The behavioral core ignores it
(bias is zero and added in the drain), but if it folds to a compile-time
constant, VTR proves the structural core's `S_IDLE -> S_PRELOAD -> S_RUN` arm
unreachable, concludes the tensor_slice result path is dead, and prunes every
slice. Pulsing it on the frame's mandatory idle beat costs nothing.

**3. Output capture** — polled on **every** iteration:

```cpp
if (v) {
    if (captured < M) {
        res_T out_pack;
        for (col = 0; col < N; col++) {          // unrolled
            ac_int<16,true> raw_val = c_row.slc<16>(col_tile*128 + col_local*16);
            accum_t value = (rescale raw_val by 2^-(frac_a + frac_b))
                          + biases[col];          // full-precision bias here
            out_pack[col] = <cast to result type>; // quantize (round/saturate)
        }
        res_stream.write(out_pack);   // array interface: results[captured] = out_pack;
    }
    captured++;
}
```

Because the poll runs from step 0, rows are captured whenever they emerge —
including while later feed beats of the same frame are still being issued
(deep-K full-K shapes finish their wave shortly after the short feed). The
core emits raw integer dot products in 16-bit lanes; rescale, bias, and
result-type quantization all live here in the wrapper.

## Cycle budget

Frame timeline in run-call indices, for the frame starting at call
`f = (frame_index) * period`: call `f` is the preload beat, call `f+1` the
first data beat (core clock 0), so result row `r` appears at call
`f + first_out + 2 + r` from the core model, and the RTL's registered output
stages may add up to 3 calls of lag. The bound

```
total_steps = (n_frames - 1) * period   start of the last frame
            + (first_out + MR + 1)      its last row, core schedule
            + 3                         worst RTL port lag
            + 2                         spare
```

guarantees every row lands inside the loop's capture window; generation fails
hard if the budget cannot cover the feed itself. Note the tail uses `MR`, not
`M`, so shapes with a non-8-aligned M get the padded-row headroom too.
`DRAIN_PADDED_ROWS` then
issues `MR − M` idle calls (zero for 8-aligned M) as flush headroom for the
hard block's 8-row burst granularity.

## Timing diagram — 8×8×8

`total_beats = 8`, `first_out = 16`, `M = MR = 8`, `n_frames = 1`, so
`total_steps = 30`. One column =
one RUN iteration = one core cycle:

```
RUN step      :  0 | 1  2  3  4  5  6  7  8 | 9 ......... 17 |18 19 20 21 22 23 24 25 |26 27 28 29
phase         :  P |◄———————— FEED ————————►|◄—— wave wait ——►|◄———— rows emerge ————►|◄— spare —►
              :    |                        |                |                        |
preload_valid :  1 | 0  0  0  0  0  0  0  0 | 0           0  | 0  0  0  0  0  0  0  0 | 0  0  0  0
in_valid      :  0 | 1  1  1  1  1  1  1  1 | 0           0  | 0  0  0  0  0  0  0  0 | 0  0  0  0
a_rows        :  - |A0 A1 A2 A3 A4 A5 A6 A7 | -           -  | -  -  -  -  -  -  -  - | -  -  -  -
b_cols (wgts) :  - |W0 W1 W2 W3 W4 W5 W6 W7 | -           -  | -  -  -  -  -  -  -  - | -  -  -  -
              :    |                        |                |                        |
core clock cc :  - | 0  1  2  3  4  5  6  7 | 8 ......... 16 |17 18 19 20 21 22 23 24 |25 26 27 28
v (out_valid) :  0 | 0  0  0  0  0  0  0  0 | 0           0  | 1  1  1  1  1  1  1  1 | 0  0  0  0
c_row         :  - | -  -  -  -  -  -  -  - | -           -  |R0 R1 R2 R3 R4 R5 R6 R7 | -  -  -  -
res_stream    :    |                        |                |r0 r1 r2 r3 r4 r5 r6 r7 |
captured      :  0 | 0  0  0  0  0  0  0  0 | 0           0  | 1  2  3  4  5  6  7  8 | 8  8  8  8
```

In RTL cosimulation the `v`/`c_row` group may slip up to 3 columns right; the
spare columns absorb it.

## Timing diagram — conv2d full-K (16×72×8)

`total_beats = 16` (single full-K pass), `first_out = 80`, `M = MR = 16`, so
`total_steps = 102` — phase view:

```
step:   0 |1 ............... 16 |17 ......................... 81 |82 ........ 97 |98 ... 101
        P |◄————— FEED ———————►|◄—— wave (72+8−16 = 64 idle) ———►|◄— 16 rows ——►|◄— spare —►
                 in_valid=1                 in_valid=0                 v=1
```

The chunked package of the same shape feeds `9 × 16 = 144` beats, with rows at
steps 146..161 and `total_steps = 166`.

## Frame protocol invariants

The core relies on exactly the protocol this loop produces:

- every frame begins with one preload/idle call (`preload_valid=1`,
  `in_valid=0`) at `p == 0`;
- the frame's data beats are contiguous `in_valid=1` calls — any
  non-`in_valid` call ends the frame's feed;
- `first_out >= total_beats` by construction, so rows never emerge before the
  feed completes;
- the M-row emission window of a frame is shorter than the minimum frame
  period (`M <= total_beats < total_beats + 1`), so windows of consecutive
  frames never overlap.

The core itself pipelines frames (frame-slot scheduler, minimum frame period
`total_beats + 1` calls — see *Frame Pipelining* in `wrapper_timing_model.md`).
This loop always overlaps a frame's feed with its own compute/drain window.
Overlap *across* frames happens two ways:

- `n_frames > 1` packages drive it directly from this loop — the frames are
  contiguous at `period` spacing, so frame `t+1` feeds while frame `t` drains.
  These are **simulation** packages for measuring back-to-back throughput.
- `n_frames == 1` (the real hls4ml flow) is one frame per wrapper call, so
  cross-frame overlap depends on a caller issuing multiple frames through one
  core (multi-frame conv tiling, einsum head loops).
