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
    a_rows, b_cols,
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

## FoldAxis and fold-N

`m` above folds M behind FoldAxis; `n` folds N the same way. Under `n`, K and M
stay fully spatial (`k_spatial = K_CHUNKS`, one K pass, all row tiles in every
frame); RF folds column tiles instead of K chunks or row tiles. `GRID_COLS =
ceil(N/8)` column tiles are covered by `cg = ceil(GRID_COLS / RF)` column
tiles per group, in `n_passes = ceil(GRID_COLS / cg)` back-to-back frames; the
legalized RF is `n_passes`. Requests above `GRID_COLS` legalize down to
`GRID_COLS` (one column tile per frame) with a warning; RF=1 legalizes to
`cg = GRID_COLS`, `n_passes = 1` -- one frame, functionally today's
single-frame hardware. The legal range is `1..GRID_COLS`.

No new RTL beyond the weight ROM's group counter (see below): the core
generated is the plain rf=1 core sized for `N_g = 8*cg` columns. The wrapper
issues `n_passes` frames of that core back-to-back; frame `g` covers logical
columns `[g*N_g, (g+1)*N_g)`. Unlike fold-M (which pads spare ROWS in the last
frame), every frame here emits M REAL rows -- only the tail column tiles of
the LAST group may be padding (silently zero, dropped past the logical N
bound when the wrapper assembles the full row). Multipliers = `64 * GRID_ROWS
* cg * K_CHUNKS`; `effective_reuse` tracks `n_passes` (each frame reuses the
array once per group, unlike fold-M's per-vector reuse of 1) -- the manifest
reports `reuse_factor` (the fold-N pass count) and `effective_reuse`
separately.

**Weight ROM group addressing** (weight-stationary only): the ROM holds
`n_passes * N_g` entries, group `g`'s columns at base `g * N_g` (the packer
zero-pads any tile columns past the real N). The existing `rom_addr`
register already steps to the next block's base on every frame's wrap beat
(`rom_addr + 1 == base + N_g`, same mechanism the K-multipass ROM uses
between K passes); fold-N adds a `grp_ctr` register, only emitted when
`n_passes > 1`, that gates the idle-beat rewind: `rom_addr` only rewinds to 0
once every group has been visited (`grp_ctr` reaches `n_passes - 1`),
otherwise it holds the wrap's next-base value and `grp_ctr` advances. A
`grp_was_feeding` register distinguishes a REAL frame boundary (the idle beat
right after a frame fed real beats) from any OTHER idle cycle (reset settle
beats, or a testbench's trailing wait) -- only a real boundary may advance or
wrap the group counter, since consecutive idle cycles are common outside a
frame's feed window and must not double-advance it. With `n_passes == 1` none
of this is emitted, so the ROM block is byte-for-byte today's.

The two-operand (external `b_cols`) path needs no RTL change: the wrapper
already re-inserts the tile offset by beat index, so fold-N just restricts
each frame's B beats to `weight_cols[g * N_g .. g * N_g + N_g)`.

See `wrapper_run_loop.md` for the multi-frame feed/capture/emission schedule
and this file's fold-M section above for the row-fold analogue.

## Synth Protocol

1. The wrapper pulses `preload_valid` for one cycle at the head of each frame.
   The bias word is zero — see *Bias* below — so this pulse exists to keep the
   `S_IDLE -> S_PRELOAD -> S_RUN` arm live, not to load coefficients.
2. For each K chunk:
   - pulse `start_mat_mul` on the first A/B beat of that chunk
   - assert `pe_reset` on chunk 0 to clear accumulators (cold start only --
     `pe_reset` no longer has any role at end-of-output; see *Output
     Collector* below)
   - drive `validity_mask_a_cols_b_rows` and `final_mat_mul_size` for that chunk
   - feed `max(M,N)` row/column beats
3. Intermediate outputs from non-final K chunks are ignored.
4. After the final K chunk, the output collector releases one tile-row at a
   time and concatenates its column tiles into full output rows.

## Requantization and Bias

The slices are a **pure integer matmul**; the core owns everything after it, in
two stages, and `c_row` carries finished result codes of `out_width` bits per
column (`c_bits = GRID_COLS * 8 * out_width`).

- **Stage 1, in the slice.** The full K contraction accumulates in 32 bits. The
  slice's 16-bit output is that sum after a round-half-up shift by `S1` and a
  wrap to 16 bits. `S1` is an IP parameter set out of band; the generator
  records it as a comment above each instantiation (VTR's hard-block model has
  no parameters, so it is not a Verilog override). `S1 = 0` is a pass-through
  and is the normal case: the generator derives `S1` from the layer's
  `accum_t` as the smallest shift that makes the gemm-scale accumulator fit
  16 bits, and warns when it is nonzero (that layer double-rounds).
- **Stage 2, in the wrapper.** The 16-bit partials of the K partitions are
  summed in 16-bit wrapping arithmetic (one term in the chunked path), the
  bias is added at that intermediate scale (`frac_a + frac_b - S1`), then a
  round-half-up shift by `S2` and a wrap to `out_width`. `S1 + S2 =
  frac_a + frac_b - frac_out`. No saturation anywhere.

The bias is a **compile-time constant**: one 16-bit signed lane per column,
baked into the core as a flat `wire` from the same codes list the C behavioral
core and mvau use (`gemm_ip/biasrom.py`). It is a flat wire, not a `reg`
array, because parmys would otherwise infer an unclocked memory and vpr would
abort. Under fold-N the wire holds `n_passes * core_n` lanes and is indexed by
the emitting frame's column group, latched when the frame is allocated (the
live feed-side group counter has already advanced by emit time). There is no
bias port. The preload phase remains in the protocol only to keep the
structural core's `S_PRELOAD` arm live.

The hls4ml-facing drain is a pure unpack: it slices `out_width` bits per
column and reinterprets them as the result type's mantissa.

The behavioral sim branch folds the whole contraction into one exact
accumulator before stage 1, so it does not mirror the per-partition stage 1
of the structural branch. The two agree exactly when `S1 = 0`; aligning them
structurally is a separate item.

## Output Collector

Output readout is gated by a three-bit tensor-slice `op` input, with
**zero parking storage**:

- `op[0]` (`out_ctrl`, level) — `1` HOLDS the tile's completed result
  internally (including after intermediate K chunks) and emits nothing; `0`
  shifts one result row per cycle onto `c_data_out`, qualified by
  `c_data_available`. Readout sources a **shadow bank**, not the live PE
  accumulators, so a drain in progress never observes a result the array is
  still computing.
- `op[1]` (`drain_stop`, 1-cycle pulse) — terminates the remaining
  masked/padded tail of the current drain burst early. It has no effect on
  the accumulators or on shadow contents.
- `op[2]` (`shadow_swap`, 1-cycle pulse) — snapshots the final PE
  accumulators into the shadow bank and clears the accumulators in the same
  cycle, freeing the array to start the next group's `start_mat_mul` before
  the shadow bank has finished draining.

Splitting drain control (`op[1]`/`op[2]`) out of `pe_reset` decouples drain
from the accumulator lifecycle: one group can drain from its shadow bank
while the next group is already computing into the (now-cleared) live
accumulators. Legacy free-run (no shadow bank in use, no drain-stop) is
`op = 3'b000`.

All tiles in the grid finish together, so the column tiles of one tile-row
concatenate as pure wiring. The wrapper holds every tile-row (`op[0] = 1`) and
releases them one at a time in row-major order for their 8-row bursts,
pulsing `op[1]` to cut a burst short and `op[2]` once the group's final K
chunk lands.

This replaced an earlier per-tile delay-line alignment pyramid, whose shift
registers cost sum-of-delays x 129 FFs (~6.2k on a 2x2 grid, ~29k on 4x2). That
pyramid — and its unused buffered-synth generator — has since been removed.

The `op`/`pe_reset` pins live only on the synth branch's `tensor_slice_int8`
black-box instantiation (see *Tensor-Slice Assumption* below); the
behavioral sim model and the C behavioral core are grid-level
compute-at-emit models that realize the same compute/drain overlap through
their own frame/slot schedulers and carry no `op` state. Local regression is
therefore compile-check only for the `op` contract itself; functional
validation of the pins is deferred to the separate hardblock project's
cosim. See `wrapper_run_loop.md` for the multi-frame feed/capture schedule
this overlap builds on.

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
- on `op[2]` (`shadow_swap`), snapshot the final accumulated result into an
  internal shadow bank and clear the accumulators, so the next group's
  `start_mat_mul` can begin immediately
- gate readout by `op[0]` (`out_ctrl`) from the shadow bank, independent of
  the live accumulators' state (see *Output Collector* above)
- on `op[1]` (`drain_stop`), cut the remaining masked/padded drain burst
  short, with no effect on accumulators or shadow contents
- `pe_reset` clears the PE accumulators only (cold start / error recovery);
  it does not touch drain or the shadow bank and is not asserted at
  end-of-output

This is a **model-level contract, not an in-repo implementation**: the
`tensor_slice_int8` hardblock is a separate project, and neither the
behavioral sim model (`behav_grid`) nor the C behavioral core in
`package.py` instantiates it or drives these pins -- both are grid-level
compute-at-emit models whose own frame/slot schedulers realize the same
compute/drain overlap without an `op` port. Validating the pins themselves
against real hardblock RTL is out of scope for this repo's regression;
that happens in the hardblock project's cosim.

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
