# The Wrapper RUN Loop

This note documents the frame schedule of the generated Catapult C++ wrapper —
the `RUN` / `RUN_ARRAY` loop emitted into every package's public header by
`gen_public_header` (`src/targets/tensor_slice/package.py`). For the timing formulas
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

The two stream entries have a second form. Under `__SYNTHESIS__ &&
BLACKBOX_FLOW` they are free-running single-cycle design blocks (see
*Free-running stream entries* below); the loop described in the rest of this
note is the csim (`#else`) body of those entries and the only body of the
array entry, both of which process one frame per call.

## Function anatomy

```cpp
static {name}_ccore gemm;        // the stateful core, shared across calls
int captured = 0;                // result rows captured so far (this frame)
                                 // (no bias packing: the bias is baked into the core)

RUN: for (step = 0; step < total_steps; step++) { ... }  // the frame schedule

// The structural core's padded tail rows (physical row count above logical M)
// are terminated by op1, not by pe_reset: pe_reset is tied low on every slice
// (each slice masks it on any start edge and self-clears at capture, so
// `rst` is the only recovery path). Fold-M retains a conservative legacy
// flush hook (`DRAIN_ARRAY_WL_PADDED_ROWS`/`DRAIN_ARRAY_PADDED_ROWS`, `physical_rows - m`
// idle calls); it's normally zero-trip because fold-M's generated core M is
// tile-aligned today.
```

```text
period      = total_beats
total_steps = (n_frames - 1) * period + first_out + M + 4
```

`first_out` is the mode-aware first output offset (full-K-spatial packages get
the shorter single-pass value). The `+4` is the port-lag tail (3 calls
worst-case RTL lag + 1 spare). There is no preload beat: a frame is exactly
`total_beats` `in_valid` calls and the next frame's first beat follows its
last directly. The tail is sized on
the logical row count `M`, not the physical tile burst (`grid_rows * 8`): the
structural core's op1 signal stops the drain at the logical M'th row, so the
remaining physical rows never need to be walked by the loop. A combined
K/N-fold package adds one extra `period` of slack (`_struct_tail`) on top of
this budget, to guarantee the loop outlives the structural core's registered
handshake latency on the last frame.
Generation fails hard if the single-frame budget `first_out + M + 4` cannot
cover `total_beats + 2` — i.e. if the loop would be shorter than the feed.

## What one RUN iteration contains

Every iteration executes the same body; step-index guards select what is
active:

**1. Step decode + operand packing** — active only on a frame's data beats:

```cpp
bool in_feed     = step < feed_total;              // feed_total = n_frames * period
int  p           = in_feed ? step % period : period;
bool feeding_now = in_feed && p < total_beats;
int  pf = p;
int  kc = pf / input_beats;        // K pass index (0..passes-1)
int  t  = pf % input_beats;        // row/col beat within the pass
```

`p == 0..total_beats-1` are the frame's data beats (there is no leading
preload/idle beat). With `n_frames == 1` the whole feed is one frame, so
`feed_total = period` and every step past it is an idle drain call.

- *Chunked packages* (`k_spatial == 1`, `passes == k_chunks`): the feed makes
  `k_chunks` passes of `max(M,N)` beats. During pass 0 each logical A row is
  read from the source once (`a_stream.read()` / `a_rows[t]`) and its 8
  K-bytes are packed into the per-tile 64-bit lane of the blackbox word
  (`ROW_PACK_DIRECT`); the remaining passes are pre-packed into
  `a_replay[passes-1][M]` and replayed later. B columns are packed
  from `weight_cols` every pass (`COL_PACK`).
- *Full-K packages* (`k_spatial == k_chunks`, `passes == 1`): a single
  `max(M,N)`-beat pass; each beat carries one logical A row / B column with
  **all** K chunks packed into a widened `64*k_spatial`-bit word. No replay
  storage is used (no `a_replay` array is declared at all).
- *General (`1 < k_spatial < k_chunks`, multi-pass narrow word)*: the same
  `a_replay[passes-1][M]` replay mechanism as the chunked case, but
  each pass's word carries `k_spatial` K chunks (`64*k_spatial` bits) instead
  of one. B is never replayed at any `k_spatial` -- `weight_cols` is a plain
  array, so every pass simply re-slices the columns it already has in hand.

Outside the feed window the words are zero and this section is inert.

**2. The core call** — exactly one per iteration:

```cpp
ac_int<1,false> feed_valid    = feeding_now ? 1 : 0;
ac_int<1,false> frame_preload = 0;
gemm.run(a_rows, b_cols, bias_packed, frame_preload, feed_valid, c_row, v, l);
```

The cores have no preload stage: a frame starts on its first `in_valid` beat
(chunk/pass 0, beat 0) and the core's frame clock is 0 there. Consecutive
frames may follow each other with no idle call, but a frame's own beats must be
contiguous (no idle cycle inside a frame; a gap between frames is fine, and an
`en` freeze at any point is safe). `preload_valid` (`frame_preload`) is an
unused port kept only so the wrappers still connect; it is tied to 0. Every
step past the feed region is an idle call that advances the core's clock.

**3. Output capture** — polled on **every** iteration:

```cpp
if (v) {
    if (captured < M) {
        res_T out_pack;
        for (col = 0; col < N; col++) {          // unrolled
            ac_int<W,true> raw_val = c_row.slc<W>(col_tile*8*W + col_local*W);  // W = out_width
            out_pack[col] = <reinterpret raw_val as the result type's mantissa>;
        }
        res_stream.write(out_pack);   // array interface: results[captured] = out_pack;
    }
    captured++;
}
```

Because the poll runs from step 0, rows are captured whenever they emerge —
including while later feed beats of the same frame are still being issued
(deep-K full-K shapes finish their wave shortly after the short feed). The
core emits finished result codes in `out_width`-bit lanes: requantization
and the baked bias both live in the core (see `rtl_contract.md`,
"Requantization and Bias"), so the capture is a pure unpack.

## Cycle budget

Frame timeline in run-call indices, for the frame starting at call
`f = (frame_index) * period`: call `f` is the first data beat (core clock 0),
so result row `r` appears at about call `f + first_out + r` from the core
model, and the RTL's registered output stages may add up to 3 calls of lag.
The bound

```
total_steps = (n_frames - 1) * period   start of the last frame
            + (first_out + M)           its last logical row, core schedule
            + 3                         worst RTL port lag
            + 1                         spare
```

guarantees every logical row lands inside the loop's capture window; generation
fails hard if the budget cannot cover the feed itself. K- and N-fold packages
size the tail on the logical `M`, not the physical `MR = ceil(M/8)*8`: op1
stops the structural core's drain at the logical M'th row, so the masked
physical tail rows never need loop headroom (pe_reset plays no part in this —
it stays tied low throughout). Fold-M retains its conservative legacy flush
hook (`DRAIN_ARRAY_WL_PADDED_ROWS`/`DRAIN_ARRAY_PADDED_ROWS`, `physical_rows - m`
idle calls); its generated core M is tile-aligned today, so that hook is
zero-trip.

## Timing diagram — 8×8×8

`total_beats = 8`, `first_out = 16`, `M = 8`, `n_frames = 1`, so
`total_steps = 28`. One column = one RUN iteration = one core cycle (row
positions are the behavioral core's):

```
RUN step      : 0  1  2  3  4  5  6  7 | 8 ......... 15 |16 17 18 19 20 21 22 23 |24 25 26 27
phase         :◄———————— FEED ————————►|◄—— wave wait ——►|◄———— rows emerge ————►|◄— spare —►
              :                        |                 |                        |
preload_valid : 0  0  0  0  0  0  0  0 | 0           0   | 0  0  0  0  0  0  0  0 | 0  0  0  0
in_valid      : 1  1  1  1  1  1  1  1 | 0           0   | 0  0  0  0  0  0  0  0 | 0  0  0  0
a_rows        :A0 A1 A2 A3 A4 A5 A6 A7 | -           -   | -  -  -  -  -  -  -  - | -  -  -  -
b_cols (wgts) :W0 W1 W2 W3 W4 W5 W6 W7 | -           -   | -  -  -  -  -  -  -  - | -  -  -  -
              :                        |                 |                        |
core clock cc : 0  1  2  3  4  5  6  7 | 8 ......... 15  |16 17 18 19 20 21 22 23 |24 25 26 27
v (out_valid) : 0  0  0  0  0  0  0  0 | 0           0   | 1  1  1  1  1  1  1  1 | 0  0  0  0
c_row         : -  -  -  -  -  -  -  - | -           -   |R0 R1 R2 R3 R4 R5 R6 R7 | -  -  -  -
res_stream    :                        |                 |r0 r1 r2 r3 r4 r5 r6 r7 |
captured      : 0  0  0  0  0  0  0  0 | 0           0   | 1  2  3  4  5  6  7  8 | 8  8  8  8
```

In RTL cosimulation the `v`/`c_row` group may slip up to 3 columns right; the
spare columns absorb it.

## Timing diagram — conv2d full-K (16×72×8)

`total_beats = 16` (single full-K pass), `first_out = 80`, `M = 16`, so
`total_steps = 100` — phase view:

```
step:   0 ............... 15 |16 ......................... 79 |80 ........ 95 |96 ... 99
        ◄————— FEED ———————►|◄—— wave (72+8−16 = 64 idle) ———►|◄— 16 rows ——►|◄— spare —►
             in_valid=1                 in_valid=0                 v=1
```

The chunked package of the same shape feeds `9 × 16 = 144` beats, with rows at
steps 144..159 and `total_steps = 164`.

## Frame protocol invariants

The core relies on exactly the protocol this loop produces:

- a frame begins on its first `in_valid` beat; there is no preload/idle call
  (`preload_valid` is tied to 0 and unused);
- the frame's `total_beats` data beats are contiguous `in_valid=1` calls —
  no idle cycle inside a frame. Idle cycles between frames are allowed, and
  the next frame may start on the very next call;
- `first_out >= total_beats` by construction, so rows never emerge before the
  feed completes;
- the M-row emission window of a frame is no longer than the frame period
  (`M <= total_beats`), so windows of consecutive frames never overlap.

The core itself pipelines frames (frame-slot scheduler, minimum frame period
`total_beats` calls — see *Frame Pipelining* in `wrapper_timing_model.md`).
This loop always overlaps a frame's feed with its own compute/drain window.
Overlap *across* frames happens three ways:

- `n_frames > 1` packages drive it directly from this loop — the frames are
  contiguous at `period` spacing, so frame `t+1` feeds while frame `t` drains.
  These are **simulation** packages for measuring back-to-back throughput.
- `n_frames == 1` csim / array entries are one frame per wrapper call, so
  cross-frame overlap there depends on a caller issuing multiple frames
  through one core (multi-frame conv tiling, einsum head loops).
- The synthesized stream entries are free-running blocks (next section), so
  consecutive hls4ml frames overlap at the feed period without any caller
  cooperation.

## Free-running stream entries

Under `__SYNTHESIS__ && BLACKBOX_FLOW`, the stream and const-weight stream
entries are single-cycle `hls_design` blocks: one wrapper clock per call, so
successive frames overlap and the frame interval is the feed length
(`m_passes * n_passes * total_beats`) rather than feed + drain + call
overhead.

- **A-row queue.** The next group's A rows wait in a fixed-slot shift queue fed
  from one non-blocking `nb_read` site. The oldest row is always the top slot
  (a constant head), a pass-0 beat pops it and shifts the slots up, and
  otherwise the slots below the highest empty one close the gap. A
  run-time-indexed latch instead would build wide write decoders and read
  muxes that grew wide conv layers by about half and cost Fmax.
- **Group start.** The core has no backpressure and needs every group's beats
  contiguous, so a group starts only once its core rows are all held. Groups
  (`m_passes * n_passes` per frame, mg-major) then run back to back with no
  bubble whenever the stream keeps up.
- **No blocking reads.** A blocking read (or `available()` wait) stalls the
  whole block, core drain included, whenever the stream is empty, which would
  hold the previous frame's outputs until the next frame's first row arrives.
- **Runtime B.** The runtime-B latch holds all `logical_n` columns for the whole
  frame (every M-group re-reads them) and is refilled during the last M-group.
- **Outputs.** Rows are written one per `out_valid` pulse. Under fold-N the
  slices of earlier N-groups wait in a small `c_buf` and the last N-group's
  pulse completes and writes each full row. Fold-N also registers
  `out_valid`/`c_row` before `c_buf` to take the core's row-done logic off the
  capture path; other configs keep the direct capture, because Catapult turned
  the registered row counter into a free-running one and lost phase with padded
  M groups.

Catapult's `cycle.rpt` reports these blocks per call (latency 2, II 1), so it
does not give the frame interval. Measure it from the cosim end time at two
frame counts and divide the difference by the frame-count difference.

## Fold-M multi-frame schedule (FoldAxis="m")

A fold-M package reuses this exact loop with `n_frames = m_passes`; the only
differences are what `M` means and what the loop reads/captures:

- The core is generated for `M_g = 8*mg` rows (`k_spatial = K_CHUNKS`, one K
  pass — see `rtl_contract.md`'s FoldAxis section), so `m` in the formulas
  above (`total_beats`, `period`, `first_out`, ...) is `M_g`, not the logical
  M. `logical_m` (the true M) is a second, separate quantity threaded through
  only the two places below.
- **Feed gating**: frame `g`'s beat `t` is global row `g*M_g + t`. The loop
  reads `a_stream`/`a_rows[]` only when that global row is `< logical_m` (an
  extra `g * M_g + t < logical_m` term on the feed condition); beats past
  `logical_m` feed zero A instead of reading — there are exactly `logical_m`
  real rows available, not `m_passes * M_g`. Only the last frame ever has
  padding rows.
- **Capture**: fold-M drops the b2b capture body's `n_frames * m`-row bound
  (which would keep every frame's rows, all real) for the single-frame-style
  body's bound (`captured < logical_m`, the same guard phase 1 uses for one
  frame). `captured` is one counter declared outside the loop, incremented on
  every `v` pulse across every frame, so this is exactly "keep the first
  `logical_m` pulses in emission order" — since frames retire in order and
  only the last frame's tail is padding, that is exactly the real rows.
- No replay buffer, no C buffer, no ROM reshaping: K is a single pass (no
  replay), each frame's rows are complete and in order at the end of its own
  pass (no cross-frame buffering), and the ROM address and A-replay counters
  wrap at the frame's last beat, so the ROM re-reads the same `N` (or
  `K_CHUNKS * N`) entries for every frame — no group-scoped addressing needed.
- RF=1 (`m_passes == 1`) is a single frame: the loop's fold-M-only feed/
  capture text collapses to the ordinary single-frame path with `logical_m ==
  M_g`, matching today's (`FoldAxis="k"`) single-frame hardware.

## Fold-N multi-frame schedule (FoldAxis="n")

A fold-N package reuses the same loop with `n_frames = n_passes`, but unlike
fold-M (which pads spare ROWS in the last frame) every frame emits M REAL
rows; only the tail COLUMNS of the last group may be padding. The core is
generated for `N_g = 8*cg` columns (`k_spatial = K_CHUNKS`, one K pass, M
fully spatial); `n` in the RUN-loop formulas above is `N_g`, not the logical
N. `logical_n` (the true N) is a second quantity used only where noted below.

- **A replay**: A is read once per column-group's frame but must be
  IDENTICAL across every frame (same M rows, only the B columns and output
  group change), and the stream is single-read. Frame 0 (`g == 0`) reads
  `a_stream` and stores the packed row in a NEW replay buffer,
  `a_replay_n[M]` (one shared slot -- the replayed value is the same for
  every later frame, unlike the K-multipass replay buffer, which needs a
  distinct slot per pass because each pass slices different K bytes from the
  same beat). Frames `g >= 1` read `a_rows = a_replay_n[t]` instead of the
  channel. The array entry needs no replay at all: `a_rows[]` is already
  randomly addressable, so every frame just re-reads it.
- **B restriction**: frame `g`'s B beats are restricted to group `g`'s
  columns. Two-stream: `weight_cols[g * N_g + t]` instead of `weight_cols[t]`
  (both the stream and array RUN loops). Weight-stationary: the RTL ROM holds
  every group back to back and self-addresses by frame boundary (see
  `rtl_contract.md`'s FoldAxis section) -- the wrapper feeds no B beat at all.
- **C assembly (capture + a separate emission loop)**: unlike fold-M (which
  can write each row's res_T the moment it emerges, since every row already
  carries the full logical N), fold-N's per-frame `out_valid` rows only carry
  `N_g` columns of ONE group -- a full logical row does not exist until every
  group has landed. The RUN loop's capture is reduced to a raw store: a
  `c_buf[M][n_passes * N_g]` buffer (`out_width`-bit lanes, already requantised and biased by the core) is
  written at `c_buf[row][g * N_g + col]` as each frame's rows emerge (`gOut =
  captured / M`, `rowOut = captured % M`, from the same monotonic `captured`
  counter phase 1 uses -- frames retire in order and every row is real, so
  this is exactly "group g's rows land at group g's column offset"). AFTER
  the RUN loop, a separate `EMIT_FOLD_N` loop of M iterations assembles the
  full logical-N row once per row, reading `c_buf[row][0..logical_n)` and
  dropping columns `>= logical_n` (the last group's padding tail); the lanes
  are already final codes (the core indexes its baked bias by the emitting
  frame's group), so this is a pure unpack. The stream entry writes each
  assembled row to `res_stream`; the array entry writes `results[row][col]`
  directly (element-wise, like phase 1's array capture, to keep the loop
  HLS-unrollable).
- No row padding, no cross-frame row buffering: M is never folded under
  `FoldAxis="n"`, so every frame's M rows are real and complete at the end of
  its own pass -- only the COLUMN assembly is deferred to the emission loop.
- RF=1 (`n_passes == 1`) is a single frame: `logical_n == N_g`, the A replay
  buffer and B restriction are unused (`g` is always 0), and the emission
  loop's `c_buf` round-trip reproduces today's inline per-call capture
  exactly (same rescale/bias/cast text, just deferred by one loop) --
  byte-identical text to `FoldAxis="k"`'s single-frame path (n_passes==1
  short-circuits the c_buf/emission text entirely; see
  `test_generate_catapult_pkg_fold_axis_n_rf1_byte_identical_to_k`).
