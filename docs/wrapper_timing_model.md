# Wrapper Timing Model

This note captures the active timing model for the hls4ml-native Catapult GEMM IP
wrapper.

## Hardblock Geometry

The tensor-slice GEMM hardblock is built from `8x8` tiles.

For GEMM shape `M x K x N`:

```text
grid_rows = ceil(M / 8)
grid_cols = ceil(N / 8)
PHYSICAL_ROWS = grid_rows * 8
LOGICAL_ROWS  = M
```

`PHYSICAL_ROWS` is the tensor-slice burst size, including masked padded rows.
The generated structural wrapper exposes only `LOGICAL_ROWS`: after its final
logical row, it pulses `op[1]` (`drain_stop`). The tensor-slice hardware
contract defines that pulse as aborting the remaining masked burst with no
effect on accumulators or shadow contents, so the next operation may start
without a visible padded drain. `pe_reset` is tied off (`1'b0`) in the
wrapper instantiation -- the slice never acts on it, and `rst` is the only
recovery path.

## Input Feed Contract

The feed path is row/col streaming with K-chunks, organised as `n_frames`
back-to-back frames of period `total_beats`. Each frame is `total_beats`
contiguous data beats; there is no preload/idle beat, and the next frame's
first beat may follow immediately:

```text
k_chunks    = ceil(K / 8)
input_beats = max(M, N)
total_beats = input_beats                 (full-K-spatial)
            | k_chunks * input_beats      (chunked)
period      = total_beats
feed_total  = n_frames * period

for step in 0..total_steps:
    in_feed     = step < feed_total
    p           = in_feed ? step % period : period
    feeding_now = in_feed and p < total_beats
    beat t = p % input_beats, chunk kc = p / input_beats

    frame_preload = 0                      # unused port, tied to 0
    feed_valid    = feeding_now

    pack a_beat[t] into a_rows (K lanes kc*8..kc*8+7; all chunks if full-K)
    pack b_beat[t] into b_cols (K lanes kc*8..kc*8+7; all chunks if full-K)
    gemm.run(a_rows, b_cols, bias_packed, frame_preload, feed_valid, ...)
```

`bias_packed` is always **zero** — the core is a pure integer matmul and the
real bias is added in the wrapper's capture path (see *Output capture*).
`preload_valid` is an unused port kept so the wrappers still connect.

## Dead-Cycle Formula

Matches the behavioral grid timing — feed beats plus the systolic K+N wave
remainder. A ReuseFactor legalizes to `passes` sweeps of K over `k_spatial`
parallel chunks (`passes == 1` is full-K: every K chunk fed spatially in one
`max(M,N)`-beat pass, so its first output arrives earliest; `k_spatial == 1`
is chunked: `k_chunks` serialized passes). `latency_cycles` delegates to
`geometry.latency_first_out(m, k, n, k_spatial)`:

```python
def latency_cycles(m, k, n, grid_rows, grid_cols, k_spatial=1):
    return latency_first_out(m, k, n, k_spatial)   # total_beats = passes * max(m, n) + wave remainder

def dead_cycles_raw(m, k, n, grid_cols, k_spatial=1):
    return latency_cycles(m, k, n, grid_rows=1, grid_cols=grid_cols,
                          k_spatial=k_spatial) + 1

def dead_cycles(m, k, n, grid_cols, k_spatial=1):
    return dead_cycles_raw(m, k, n, grid_cols, k_spatial) + 1
```

By construction `first_out >= total feed beats`, so the first output row
always lands inside the RUN loop's capture window. The same formula is emitted
into three coordinated places — the C++ clk_cnt sim core, the behavioral
Verilog grid's `FIRST_OUT` localparam, and the wrapper's RUN call budget — and
package generation cross-asserts the behavioral localparam against
`latency_cycles` (`_assert_core_first_out`). `k_chunks == 1` designs are unaffected (the two
branches coincide).

This is the behavioral model's systolic abstraction; the structural
(`SYNTHESIS`) branch is synthesis-only and is not a cycle-accurate reference.

## Frame Pipelining (sim core)

The sim core (C++ ccore `#else` branch and the behavioral Verilog grid) is a
frame-slot scheduler: each frame gets a private operand buffer and cycle
counter, so the feed of frame t+1 may overlap the compute/drain of frame t.
A frame starts at the first `in_valid` call after a non-`in_valid` call, or
right after the previous frame's last beat (`total_beats` beats later) when
frames are gapless, and emits its M result rows at `[first_out, first_out+M)`
of its own clock.

- Minimum frame period: `total_beats` calls (data beats only) —
  back-to-back frames sustain ~one result row per cycle for square 8-row
  frames. Emission windows of consecutive frames cannot overlap because
  `M <= total_beats`.
- Slot count: `ceil((first_out + M) / total_beats) + 1` frames in
  flight (one spare so an allocating frame never lands on a draining slot).
- The generated wrapper's merged RUN loop overlaps a frame's feed with its
  own compute/drain window (see Active Wrapper Schedule); cross-frame overlap
  is exploitable by callers that issue back-to-back frames through one core
  (multi-frame conv tiling, einsum head loops).

Grid-level latency:
```
beats = max(M,N)               (full-K)   |   k_chunks × max(M,N)   (chunked)
beh   = beats + max(0, K+N − beats) + M  (+1 sync)
wrap  = beh + 3   (Catapult wrapper: bias + transition + register)
II    = beats   (back-to-back with shadow FIFO)
```

Examples:

| Shape | k_chunks | mode | first_out | blind | beh | wrap | b2b II |
|---|---:|---|---:|---:|---:|---:|---:|
| 8×8×8 | 1 | chunked | 16 | 18 | 25 | 28 | 8 |
| 16×8×8 | 1 | chunked | 16 | 18 | 33 | 36 | 16 |
| 16×16×16 | 2 | chunked | 32 | 34 | 49 | 52 | 32 |
| 16×16×16 | 2 | full-K | 32 | 34 | 49 | 52 | 16 |
| 9×17×10 | 3 | chunked | 30 | 32 | 40 | 43 | 48 |
| 16×72×8 | 9 | chunked | 144 | 146 | 161 | 164 | 144 |
| 16×72×8 | 9 | full-K | 80 | 82 | 97 | 100 | 16 |
| 25×81×10 | 11 | chunked | 275 | 277 | 301 | 304 | 275 |
| 25×81×10 | 11 | full-K | 91 | 93 | 117 | 120 | 25 |

`b2b II` is the feed period `passes * feed_beats` (no preload beat). The 9×17×10
row pads each pass to 16 beats (`feed_beats`), so its true `first_out` is 48; the
`blind`/`beh`/`wrap` columns predate both that padding and the one-cycle output
latency reduction from removing the preload stage.

## Active Wrapper Schedule

See `wrapper_run_loop.md` for the full anatomy and cycle-by-cycle timing
diagrams of this loop.

The generated wrapper uses a single merged RUN loop covering every frame:
within a frame `p == 0..total_beats-1` are the data beats (no preload beat),
and EVERY step polls `out_valid`, so rows are captured as they
emerge instead of in a separate drain loop after the feed (the feed of a frame
overlaps its own compute/drain window, and with `n_frames > 1` it also overlaps
the previous frame's):

- `BIAS_PACK`: N iterations (unrolled), packing **zero** bias
- `RUN` / `RUN_ARRAY`: `total_steps` iterations, II=1 pipelined, where

  ```text
  total_steps = (n_frames - 1) * period + first_out + M + 4
  ```

  The `+4` is the port-lag tail: worst-case RTL port lag is 3 calls, plus 1
  spare. A combined K/N-fold package adds one more `period` of slack.
  The tail is sized on logical `M`, not the tile-padded physical row count.
- `DRAIN_PADDED_ROWS` is removed for K- and N-fold packages. Fold-M retains
  its conservative legacy flush hook (currently zero-trip because its core M
  is tile-aligned).

Generation fails hard (`RuntimeError`) if the single-frame budget
`run_calls = first_out + M + 4` cannot even cover `total_beats + 2`.

For `8x8x8` (`total_beats = 8`, `first_out = 16`, `M = 8`, `n_frames = 1`):

```text
BIAS_PACK          8 iterations (unrolled)
RUN               28 iterations (II=1; 8 feed steps + capture window)
DRAIN_PADDED       0 iterations
```

versus the previous serialized schedule (`FEED` 9 + `DRAIN` 26 = 35 calls):
the merged loop saves ~`total_beats` calls of per-frame function latency.
Function latency / call II is what shrinks; the first output row
comes one call earlier than with the old preload beat.

## Blackbox Binding

The wrapper must keep Catapult scheduling decoupled from the internal tensor-slice
latency:

```cpp
ac_blackbox()
    .entity("<name>_core")
    .verilog_files("<name>_core.v")
    .outputs("c_row out_valid out_last")
    .latency(1)
    .init_delay(1)
    .clock_name("clk")
    .posedge_clock(true)
    .sync_reset_name("rst")
    .active_high_sync_reset(true)
    .start_name("en")
    .has_state(true);
```

Do not replace this with a full hardblock latency declaration. Declaring the full
RTL latency makes Catapult schedule excessive pipeline depth around the blackbox.

## Latency Regression Signals

- `RUN` has more than `(n_frames - 1) * period + first_out + M + 4` iterations.
- Catapult reports large local array load loops before `RUN`.
- `READ_B_COLS` uses `hls_unroll` instead of `hls_pipeline_init_interval 1` —
  causes RAM port scheduling failures.
- Separate `PRELOAD_BIAS` or `DRAIN` loop exists instead of being folded into the RUN loop.
- SCVerify reports non-zero comparison errors when C++ sim model uses old per-KK data indexing.

## Back-to-Back Multi-Frame Feed

The wrapper feeds `n_frames` frames back-to-back to exploit the behavioral core's
frame-slot scheduler (`FRAME_SLOTS`), so the feed of frame *t+1* overlaps the
compute/drain of frame *t*. This is controlled by the generated `n_frames` constant
(`generate_catapult_pkg(..., n_frames=N)` / `python -m gemm_ip --n-frames N`):

- `n_frames == 1` (default, the real hls4ml flow): one frame per wrapper call.
- `n_frames > 1`: a standalone **simulation** package that feeds N frames in one
  call, used to demonstrate/measure back-to-back throughput via Catapult scverify.

### No bias step, no preload

The core never receives bias: `bias_packed` is hardwired to zero and the real
bias is added in the wrapper's capture path, post-rescale, in full precision.
No cycle is spent on a bias preload, and the cores have no preload stage at
all: a frame starts on its first `in_valid` beat (chunk/pass 0). `preload_valid`
is an unused port kept for compatibility and tied to 0.

### Frame schedule

```text
period      = total_beats              # total_beats in_valid beats, no idle beat
feed_total  = n_frames * period
total_steps = (n_frames - 1) * period + first_out + M + 4
```

Each frame is `total_beats` contiguous `in_valid=1` beats; the next frame may
start on the very next call. There must be no idle cycle inside a frame
(between frames is fine; an `en` freeze anywhere is safe). The weight-ROM
address and A-replay counters wrap at the frame's last beat, and row release
uses `done_count` plus `done_mat_mul`, which makes the output latency one
cycle shorter than with the old preload beat. Steady-state **frame II =
total_beats**, while each frame's first-in→last-out latency stays about
`first_out + M` (II < latency = pipelined).

`total_steps` is sized from the *last* frame's start (`(n_frames-1) * period`)
plus one full drain tail, not from `feed_total` — sizing on `feed_total` would
over-run by a whole period per frame and reintroduce serialized latency at
`n_frames == 1`.

### Free-running stream entries

Under `__SYNTHESIS__ && BLACKBOX_FLOW` the stream and const-weight stream
entries are free-running single-cycle design blocks, so the frame interval of
the synthesized IP is the feed period (`m_passes * n_passes * total_beats`),
not feed + drain + call overhead. See *Free-running stream entries* in
`wrapper_run_loop.md` for the A-row queue and output rules. The csim
(`#else`) path and the array entry keep the one-frame-per-call loop above.
Catapult's `cycle.rpt` reports these blocks per call (latency 2, II 1), so
measure the frame interval from the cosim end time at two frame counts.

### Output capture

Both branches emit exactly **M** `out_valid` pulses per frame. The structural
branch uses `op[1]` (`drain_stop`) after the final logical row to suppress the
remaining masked rows in its physical tile burst; `pe_reset` is tied off, not
used for this. The behavioral branch has no physical tail. The wrapper writes
the first `n_frames * M` pulses to `res_stream`.

### Measured (Catapult synth + scverify RTL cosim, msim, 5 frames)

Run through the real Catapult flow (`run_catapult.tcl` now ends with
`flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim sim`,
exercising the `ifndef SYNTHESIS` core — no `-DSYNTHESIS`). Frame II is read from
the core's `BEH_II` `$display` lines; all cells bit-exact (`error count = 0`).

| shape (MxKxN) | path | total_beats | per-frame latency (first_out+M) | steady frame II |
|---|---|---:|---:|---:|
| 8x8x16   | chunked (kc=1)       | 16 | 32 | **17** |
| 16x16x16 | full-K-spatial (kc=2) | 16 | 48 | **17** |
| 15x8x16  | chunked, non-square M | 16 | 39 | **17** |

II was `total_beats + 1` in every case when these were measured (with the
old preload beat; it is now `total_beats`), versus the serialized wrapper whose II equals
the full per-frame latency. This isolates the large-bench GEMM-vs-baseline latency
gap as a wrapper-feed (serialization) artifact, not a core limitation — demonstrated
on the actual `ifndef SYNTHESIS` RTL path.

### Reproduce

```bash
PYTHONPATH=src python3 -m gemm_ip --m 8 --k 8 --n 16 --name gemm_8x8x16 \
    --output_dir b2b_work --n-frames 5
cd b2b_work/gemm_8x8x16 && catapult -product ultra -shell -f run_catapult.tcl
# grep the transcript for: "error count", "BEH_II", "Simulation PASSED"
```

Note: the `n_frames > 1` package raises `MEM_MAP_THRESHOLD`/`REGISTER_THRESHOLD`
so the small operand/replay arrays map to registers (the back-to-back modulo feed
indexing otherwise contends for limited RAM read ports). This is a sim-only
relaxation, not an area-optimised build.
