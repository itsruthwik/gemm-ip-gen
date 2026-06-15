# Wrapper Timing Model

This note captures the active timing model for the hls4ml-native Catapult GEMM IP
wrapper.

## Hardblock Geometry

The tensor-slice GEMM hardblock is built from `8x8` tiles.

For GEMM shape `M x K x N`:

```text
grid_rows = ceil(M / 8)
grid_cols = ceil(N / 8)
MR        = grid_rows * 8
```

`MR` is the number of physical output rows emitted by the hardblock, including
padded rows. The hls4ml wrapper writes only the first `M` rows to `res_stream` and
drains the remaining padded rows.

## Input Feed Contract

The feed path is row/col streaming with K-chunks:

```text
k_chunks = ceil(K / 8)
input_beats = max(M, N)

for step in 0..(k_chunks * input_beats):
    if step == 0:  preload_valid = 1, in_valid = 0  (bias preload)
    if step > 0:   preload_valid = 0, in_valid = 1  (data feed)
    beat t = (step-1) % input_beats, chunk kc = (step-1) / input_beats
    pack a_beat[t] into a_rows (K lanes kc*8..kc*8+7)
    pack b_beat[t] into b_cols (K lanes kc*8..kc*8+7)
    gemm.run(a_rows, b_cols, bias_cols, preload_valid, in_valid, ...)
```

## Dead-Cycle Formula

Matches the behavioral grid timing — feed beats plus the systolic K+N wave
remainder. Full-K-spatial mode (`gemm_k_spatial == k_chunks > 1`) feeds every
K chunk spatially in one `max(M,N)`-beat pass, so its first output arrives
correspondingly earlier; chunked mode serializes the chunks:

```python
def latency_cycles(m, k, n, grid_rows, grid_cols, full_k_spatial=False):
    k_chunks = ceil(k / 8)
    input_beats = max(m, n)
    total_beats = input_beats if full_k_spatial else k_chunks * input_beats
    return total_beats + max(0, k + n - total_beats)

def dead_cycles_raw(m, k, n, grid_cols, full_k_spatial=False):
    return latency_cycles(m, k, n, grid_rows=1, grid_cols=grid_cols,
                          full_k_spatial=full_k_spatial) + 1

def dead_cycles(m, k, n, grid_cols, full_k_spatial=False):
    return dead_cycles_raw(m, k, n, grid_cols, full_k_spatial) + 1
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
A frame starts at the first `in_valid` call after a non-`in_valid` call (the
FEED protocol always inserts the preload step between frames), and emits its
M result rows at `[first_out+1, first_out+1+M)` of its own clock.

- Minimum frame period: `total_beats + 1` calls (preload + data beats) —
  back-to-back frames sustain ~one result row per cycle for square 8-row
  frames. Emission windows of consecutive frames cannot overlap because
  `M <= total_beats < total_beats + 1`.
- Slot count: `ceil((first_out + 1 + M) / (total_beats + 1)) + 1` frames in
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
II    = beats + 1   (back-to-back with shadow FIFO)
```

Examples:

| Shape | k_chunks | mode | first_out | blind | beh | wrap | b2b II |
|---|---:|---|---:|---:|---:|---:|---:|
| 8×8×8 | 1 | chunked | 16 | 18 | 25 | 28 | 9 |
| 16×8×8 | 1 | chunked | 16 | 18 | 33 | 36 | 17 |
| 16×16×16 | 2 | chunked | 32 | 34 | 49 | 52 | 33 |
| 16×16×16 | 2 | full-K | 32 | 34 | 49 | 52 | 17 |
| 9×17×10 | 3 | chunked | 30 | 32 | 40 | 43 | 31 |
| 16×72×8 | 9 | chunked | 144 | 146 | 161 | 164 | 145 |
| 16×72×8 | 9 | full-K | 80 | 82 | 97 | 100 | 17 |
| 25×81×10 | 11 | chunked | 275 | 277 | 301 | 304 | 276 |
| 25×81×10 | 11 | full-K | 91 | 93 | 117 | 120 | 26 |

## Active Wrapper Schedule

See `wrapper_run_loop.md` for the full anatomy and cycle-by-cycle timing
diagrams of this loop.

The generated wrapper uses a single merged RUN loop per frame: steps
`0..total_beats` preload + feed the frame's beats, and EVERY step polls
`out_valid`, so rows are captured as they emerge instead of in a separate
drain loop after the feed (the feed of the frame overlaps its own
compute/drain window):

- `BIAS_PACK`: N iterations (unrolled)
- `RUN` / `RUN_ARRAY`: `first_out + M + 6` iterations, II=1 pipelined
  (step 0 = preload, steps 1..total_beats = data beats, all steps poll)
- `DRAIN_PADDED_ROWS`: `MR - M` iterations, II=1 pipelined

For `8x8x8`:

```text
BIAS_PACK          8 iterations (unrolled)
RUN               30 iterations (II=1; 9 feed steps + capture window)
DRAIN_PADDED       0 iterations
```

versus the previous serialized schedule (`FEED` 9 + `DRAIN` 26 = 35 calls):
the merged loop saves ~`total_beats` calls of per-frame function latency.
The first output row is unchanged (still run-call index `first_out + 2`), so
measured first-output latency is identical; function latency / call II is
what shrinks.

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

- `RUN` has more than `first_out + M + 6` iterations.
- Catapult reports large local array load loops before `RUN`.
- `preload_valid` is hardwired to `1'b0` in the Catapult-generated RTL (means `feed_preload_valid` is compile-time constant).
- `READ_A_ROWS`/`READ_B_COLS` use `hls_unroll` instead of `hls_pipeline_init_interval 1` — causes RAM port scheduling failures.
- Separate `PRELOAD_BIAS` or `DRAIN` loop exists instead of being folded into the RUN loop.
- SCVerify reports non-zero comparison errors when C++ sim model uses old per-KK data indexing.

## Back-to-Back Multi-Frame Feed (preload-free)

The wrapper feeds `n_frames` frames back-to-back to exploit the behavioral core's
frame-slot scheduler (`FRAME_SLOTS`), so the feed of frame *t+1* overlaps the
compute/drain of frame *t*. This is controlled by the generated `n_frames` constant
(`generate_catapult_pkg(..., n_frames=N)` / `python -m gemm_ip --n-frames N`):

- `n_frames == 1` (default, the real hls4ml flow): one frame per wrapper call.
- `n_frames > 1`: a standalone **simulation** package that feeds N frames in one
  call, used to demonstrate/measure back-to-back throughput via Catapult scverify.

### No preload step

`preload_valid` is tied to `0` for every `gemm.run()` call. The behavioral
(`ifndef SYNTHESIS`) core never reads `preload_valid` — it allocates a slot on the
first `in_valid` after idle and captures (zero) bias there — and bias is added in
the wrapper's drain capture (`bias_packed = 0`, real bias applied post-rescale).
So the RTL core never sees bias and the loop spends no cycle on preload.

### Frame schedule

```text
period      = total_beats + 1          # total_beats in_valid beats + ONE in_valid=0 separator
feed_total  = n_frames * period
total_steps = feed_total + first_out + MR + 5   # tail drains the last frame
```

Each frame is `total_beats` of `in_valid=1` followed by exactly ONE `in_valid=0`
separator (no separator before the first frame). The separator drops the core's
`feeding` flag so the next frame allocates a fresh slot — continuous `in_valid`
would merge two frames. Steady-state **frame II = total_beats + 1**, while each
frame's first-in→last-out latency stays `first_out + M` (II < latency = pipelined).

### Output capture

The behavioral core emits exactly **M** `out_valid` pulses per frame
(`TOTAL_ROWS = M` in the `ifndef SYNTHESIS` branch — note this differs from the
structural branch's `MR`), retiring in frame order. The wrapper writes the first
`n_frames * M` pulses to `res_stream`.

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

II = `total_beats + 1` in every case, versus the serialized wrapper whose II equals
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
