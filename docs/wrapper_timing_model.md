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
always lands inside the DRAIN window. The same formula is emitted into three
coordinated places — the C++ clk_cnt sim core, the behavioral Verilog grid's
`FIRST_OUT` localparam, and the wrapper's DRAIN trip count — and package
generation cross-asserts the behavioral localparam against `latency_cycles`
(`_assert_core_first_out`). `k_chunks == 1` designs are unaffected (the two
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
- The generated wrapper still issues frames sequentially (FEED then DRAIN
  within one call), so per-frame latencies are unchanged; the pipelining is
  exploitable by callers that issue back-to-back frames (multi-frame conv
  tiling, einsum head loops, back-to-back samples).

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

The generated wrapper uses a single merged FEED loop (bias preload folded into step 0):

- `BIAS_PACK`: N iterations (unrolled)
- `READ_A_ROWS`: M iterations, II=1 pipelined
- `READ_B_COLS`: N iterations, II=1 pipelined
- `FEED`: `k_chunks × max(M,N) + 1` iterations (step 0 = preload, steps 1+ = data), II=1 pipelined
- `DRAIN`: `dead_cycles + M` iterations, II=1 pipelined
- `DRAIN_PADDED_ROWS`: `MR - M` iterations, II=1 pipelined

For `8x8x8`:

```text
BIAS_PACK          8 iterations (unrolled)
READ_A_ROWS        8 iterations (II=1)
READ_B_COLS        8 iterations (II=1)
FEED               9 iterations (II=1, step 0 = bias preload)
DRAIN             26 iterations (II=1)
DRAIN_PADDED       0 iterations
```

Catapult 2026.1 synthesis (nangate-45nm):

- `/core` latency: `38` cycles
- `/core` throughput: `40` cycles
- `FEED + READ_A_ROWS + READ_B_COLS` merged: 17 iterations, II=1
- `DRAIN`: 28 iterations, II=1

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

- `FEED` has more than `k_chunks × max(M,N) + 1` iterations.
- Catapult reports large local array load loops before `FEED`.
- `preload_valid` is hardwired to `1'b0` in the Catapult-generated RTL (means `feed_preload_valid` is compile-time constant).
- `READ_A_ROWS`/`READ_B_COLS` use `hls_unroll` instead of `hls_pipeline_init_interval 1` — causes RAM port scheduling failures.
- Separate `PRELOAD_BIAS` loop exists instead of being folded into FEED step 0.
- SCVerify reports non-zero comparison errors when C++ sim model uses old per-KK data indexing.
