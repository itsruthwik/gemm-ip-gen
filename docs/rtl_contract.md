# GEMM-IP RTL Contract

## Architecture

The public hls4ml GEMM-IP contract is row/column based:

```text
A = activations or im2col rows  (M x K)
B = transposed weight columns   (K x N)
C = A @ B + bias                (M x N)
```

Simulation and synthesis use a single combined RTL file (`{name}_core.v`)
with an `ifndef SYNTHESIS` guard:

- `ifndef SYNTHESIS` — behavioral GEMM model (double-buffer, row/col streaming,
  shadow FIFO for pipelined back-to-back). Used for iverilog regression and
  Catapult SCVerify co-simulation.
- `else` — structural grid of black-box `tensor_slice_int8` tiles. Used for
  Catapult HLS synthesis and downstream implementation.

The package adapters preserve the public row/column API. For synthesis, they
pack public rows/columns into 8-lane K chunks before driving the RTL wrapper.

## RTL Blackbox Interface

Catapult-style core:

```verilog
module {name}_core(
    clk, rst, en,
    a_rows, b_cols, bias_cols,
    preload_valid, in_valid,
    c_row, out_valid, out_last
);
```

## Public Data Layout

For dimensions `M x K x N`:

- public A stream has `M` beats, one K-wide activation/im2col row per beat
- public B stream has `N` beats, one K-wide transposed weight column per beat
- public C stream has `M` beats, one N-wide result row per beat
- `GRID_ROWS = ceil(M / 8)`
- `GRID_COLS = ceil(N / 8)`
- `K_CHUNKS = ceil(K / 8)`

Dense and Conv/im2col both use the same math: `C = A @ B + bias`.

## Synth RTL Chunk Layout

The generated synth RTL wrapper consumes chunk-local 64-bit tile lanes:

| Stream | Width | Beats into RTL | Layout |
|--------|-------|----------------|--------|
| bias | `GRID_COLS * 64` | 1 | byte `tile*64 + lane*8` = `bias[tile*8 + lane]` |
| A | `GRID_ROWS * 64` | `K_CHUNKS * max(M,N)` | for chunk `kc`, beat `t` carries row `t`, K lanes `kc*8 + 0..7` |
| B | `GRID_COLS * 64` | `K_CHUNKS * max(M,N)` | for chunk `kc`, beat `t` carries column `t`, K lanes `kc*8 + 0..7` |
| C | `GRID_COLS * 64` | `GRID_ROWS * 8` | one packed output row per beat; tile `c` carries 8 columns |

Invalid tail M/N/K lanes are masked to zero.

## ReuseFactor and K passes

`ReuseFactor` (RF) is the number of sequential passes each input vector's K
reduction takes over the array -- never a cycle count or an initiation
interval. `k_spatial` parallel K partitions cover `K_CHUNKS = ceil(K/8)`
chunks in `passes = ceil(K_CHUNKS / k_spatial)` passes; the legalized RF is
`passes`. Requests above `K_CHUNKS` legalize down to `K_CHUNKS` (chunked) with
a warning; RF=1 legalizes to `k_spatial = K_CHUNKS` (full-K, one pass). These
are the two endpoints of one general layout:

| | chunked (`k_spatial == 1`) | full-K (`k_spatial == K_CHUNKS`, `passes == 1`) |
|---|---|---|
| grid | one tensor-slice grid | `K_CHUNKS` grid partitions |
| feed beats | `passes * max(M,N)` = `K_CHUNKS * max(M,N)` | `max(M,N)` (single pass) |
| A word width | `GRID_ROWS * 64` | `64 * k_spatial` |
| B word width | `GRID_COLS * 64` | `64 * k_spatial` |
| per-beat content | one K chunk of row/col `t` | **all** K chunks of row/col `t` |
| wrapper storage | `a_replay[passes-1][M]` | none (no replay array declared) |

In general, a narrow-word package (`k_spatial > 1`) uses a `64 * k_spatial`-bit
word per beat: pass `q`'s beat `t` carries K chunks `q*k_spatial .. q*k_spatial
+ k_spatial - 1` of row/col `t`, chunk `p` within the word at bit offset
`p*64`. K is padded to `passes * k_spatial` chunks (`K_CHUNKS_PAD`), with the
padding chunks masked to zero. The RTL re-inserts the grid row/column tile
offset from the beat index, so the deep input FIFO never stores the
always-zero grid padding. `k_spatial == 1` keeps the single-chunk grid-padded
width instead (today's chunked layout).

Intermediate K-spatial partial sums are INT16, so partition-level overflow is
possible; correctness depends on quantized operand ranges and partition size.
The generator prints a warning for every package with `k_spatial > 1`.

## FoldAxis and fold-M

`FoldAxis` selects which dimension `ReuseFactor` folds: `k` (default, the
model above) or `m`. Under `m`, K and N stay fully spatial (`k_spatial =
K_CHUNKS`, one K pass); RF folds row tiles instead of K chunks. `GRID_ROWS =
ceil(M/8)` row tiles are covered by `mg = ceil(GRID_ROWS / RF)` row tiles per
group, in `m_passes = ceil(GRID_ROWS / mg)` back-to-back frames; the
legalized RF is `m_passes`. Requests above `GRID_ROWS` legalize down to
`GRID_ROWS` (one row tile per frame) with a warning; RF=1 legalizes to
`mg = GRID_ROWS`, `m_passes = 1` -- one frame, functionally today's
single-frame hardware. The legal range is `1..GRID_ROWS`.

No new RTL: the core generated is the plain rf=1 (full-K, full-N) core sized
for `M_g = 8*mg` rows. The wrapper issues `m_passes` frames of that core
back-to-back (the same frame-slot pipelining the core already has); frame `g`
covers logical rows `[g*M_g, (g+1)*M_g)`. The last frame's rows past the
logical M are padding (zero-fed A), dropped at capture in emission order.
Multipliers = `64 * mg * GRID_COLS * K_CHUNKS`; per-vector `effective_reuse`
stays `1` (each row's MACs happen once) -- the manifest reports
`reuse_factor` (the fold-M pass count) and `effective_reuse` separately so
the two are never confused. See `wrapper_run_loop.md` for the multi-frame
feed/capture schedule.

## Synth Protocol

1. The wrapper pulses `preload_valid` for one cycle at the head of each frame.
   The bias word is zero — see *Bias* below — so this pulse exists to keep the
   `S_IDLE -> S_PRELOAD -> S_RUN` arm live, not to load coefficients.
2. For each K chunk:
   - pulse `start_mat_mul` on the first A/B beat of that chunk
   - assert `pe_reset` only on chunk 0
   - drive `validity_mask_a_cols_b_rows` and `final_mat_mul_size` for that chunk
   - feed `max(M,N)` row/column beats
3. Intermediate outputs from non-final K chunks are ignored.
4. After the final K chunk, the output collector releases one tile-row at a
   time and concatenates its column tiles into full output rows.

## Bias

The core is a **pure integer matmul**. `bias_cols` is driven with zero by every
generated wrapper; the real bias is added in the C++ wrapper's capture path,
after the `2^-(frac_a + frac_b)` rescale, in full `accum_t` precision, and
before the result-type quantization (Keras order: matmul + bias, then quantize).
The `bias_cols` port and the preload phase are retained in the RTL contract, but
no generated flow uses them to carry coefficients.

## Output Collector

Output readout is gated by the tensor-slice `op[0]` (`out_ctrl`) input, with
**zero parking storage**:

- `op[0] = 1` — the tile HOLDS its completed result internally (including after
  intermediate K chunks) and emits nothing.
- `op[0] = 0` — the tile shifts one result row per cycle onto `c_data_out`,
  qualified by `c_data_available`.

All tiles in the grid finish together, so the column tiles of one tile-row
concatenate as pure wiring. The wrapper holds every tile-row (`op[0] = 1`) and
releases them one at a time in row-major order for their 8-row bursts.

This replaced an earlier per-tile delay-line alignment pyramid, whose shift
registers cost sum-of-delays x 129 FFs (~6.2k on a 2x2 grid, ~29k on 4x2). That
pyramid — and its unused buffered-synth generator — has since been removed.

## Tensor-Slice Assumption

The synth wrapper does not define the tensor-slice module. It instantiates
each slice as a black box:

```verilog
(* black_box = "true" *) (* keep = "true" *) tensor_slice_int8 slice_rX_cY (...);
```

'tensor_slice_int8' module directly maps to a hardblock in the VTR architecture, with the following properties:

- accept one A row tile and one B column tile per cycle
- internally skew/diagonalize row/column inputs for its systolic array
- preserve PE accumulators across repeated `start_mat_mul` operations when
  `pe_reset` is not asserted
- emit final accumulated `c_data` after the last K chunk

The generated wrapper does not change the tensor-slice port list and does not
inline or concatenate tensor-slice RTL.

## Wrapper Responsibilities

The synth wrapper:

- instantiates `GRID_ROWS x GRID_COLS` tensor slices
- sets each slice `a_loc` and `b_loc`
- wires A chains left-to-right and B chains top-to-bottom
- drives only the top/left boundaries from current A/B input beats
- drives non-boundary primary inputs to zero
- applies row, column, and per-chunk K masks
- preloads bias into each column tile
- uses every slice output to assemble full C rows

It must not contain a second GEMM datapath.
