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

The low-latency feed path is one beat per `kk`:

```text
for kk in 0..K-1:
    a_beat = a_beat_stream.read()
    b_beat = b_beat_stream.read()
    pack a_beat directly into a_rows
    pack b_beat directly into b_cols
    gemm.run(a_rows, b_cols, in_valid=1, ...)
```

Do not add:

- local activation matrices
- local weight matrices
- prepacked beat arrays
- row-major weight reshaping
- shift-register or random-access repacking schemes

Those approaches move work into the Catapult wrapper and have previously caused
large latency regressions or memory-port scheduling failures.

## Dead-Cycle Formula

The first valid output row depends on column-wise systolic propagation.

```python
def latency_cycles(k, grid_cols):
    return (grid_cols - 1) * 8 + k + 10

def dead_cycles_raw(grid_cols):
    return (grid_cols - 1) * 8 + 10

def dead_cycles(grid_cols):
    return dead_cycles_raw(grid_cols) + 1
```

`dead_cycles_raw` is the number of cycles between the end of input feed and the
first physical valid output. `dead_cycles` adds the registered boundary implied by
the Catapult blackbox declaration:

```cpp
.latency(1)
.init_delay(1)
```

The key property is that `dead_cycles_raw` depends on `grid_cols`, not on
`grid_rows`. Additional tile rows increase the number of output rows to drain;
they do not delay the first row.

Examples:

| Shape | `grid_rows` | `grid_cols` | `dead_cycles_raw` | `dead_cycles` |
|---|---:|---:|---:|---:|
| `8x8x8` | 1 | 1 | 10 | 11 |
| `16x8x8` | 2 | 1 | 10 | 11 |
| `16x16x16` | 2 | 2 | 18 | 19 |
| `32x32x32` | 4 | 4 | 34 | 35 |

## Active Wrapper Schedule

The current hls4ml-native wrapper uses a single shape-specific function with:

- `FEED`: `K` iterations
- `DRAIN_WRITE`: `dead_cycles(grid_cols) + M` iterations
- `DRAIN_PADDED_ROWS`: `MR - M` iterations

`DRAIN_WRITE` checks `out_valid` and only writes real rows while `captured < M`.
`DRAIN_PADDED_ROWS` keeps stepping the hardblock long enough to consume any padded
tile rows.

For `8x8x8`, the desired structure is therefore:

```text
FEED              8 iterations
DRAIN_WRITE      19 iterations
DRAIN_PADDED     0 iterations
```

The earlier standalone passing package reported approximately:

- `/core` latency: `28` cycles
- `/core` throughput: `31` cycles
- `FEED`: 8 iterations, II=1
- `DRAIN_WRITE`: 19 iterations, II=1

Those numbers are the practical latency target for the hls4ml-native wrapper when
the feed path remains direct.

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

The following are signs that the wrapper has drifted away from the intended design:

- `FEED` takes much more than `K` useful cycles.
- Catapult reports large local array load loops before `FEED`.
- Weight-side packing dominates the schedule.
- Generated code contains `PREPACK_BEATS`, `a_packed`, `b_packed`, or local
  matrix buffers in the hardblock wrapper path.
- `8x8x8` wrapper latency is far above the old standalone result.

When this happens, restore the direct per-`kk` beat feed before debugging the
hardblock itself.
