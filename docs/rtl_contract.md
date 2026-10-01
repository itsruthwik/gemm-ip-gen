# GEMM-IP RTL Contract

## Architecture

The public hls4ml GEMM-IP contract is row/column based:

```text
A = activations or im2col rows  (M x K)
B = transposed weight columns   (K x N)
C = A @ B + bias                (M x N)
```

Simulation and synthesis share a single structural RTL file (`{name}_core.v`):
a grid of black-box `tensor_slice_int8_atlas` tiles, one source of truth for
iverilog regression, Catapult SCVerify co-simulation, Catapult HLS synthesis,
and downstream implementation. There is no separate behavioral GEMM branch —
the earlier `ifndef SYNTHESIS` behavioral/synth split is gone.

The package adapters preserve the public row/column API. For synthesis, they
pack public rows/columns into 8-lane K chunks before driving the RTL wrapper.
Quantization is symmetric only: nonzero zero-points raise at generation time,
there is no zero-point datapath in the RTL.

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

Intermediate K-spatial partial sums are INT32, so partition-level overflow is
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
between K passes). Frames may be gapless (no idle beat between them), so the ROM
entries are laid out in feed order and `rom_addr` simply wraps to 0 past the
last entry, on the last beat of the last group's frame; the wrap itself is the
frame boundary, so no group counter stepped by idle beats is needed (it would
drift from the frame sequence). Idle cycles hold `beat_ctr` and `rom_addr`
(they advance on accepted beats only), so reset settle beats, a paused frame
and a testbench's trailing wait cannot disturb the address. The A-row replay
counters wrap the same way at the frame's last beat.

The two-operand (external `b_cols`) path needs no RTL change: the wrapper
already re-inserts the tile offset by beat index, so fold-N just restricts
each frame's B beats to `weight_cols[g * N_g .. g * N_g + N_g)`.

See `wrapper_run_loop.md` for the multi-frame feed/capture/emission schedule
and this file's fold-M section above for the row-fold analogue.

## Synth Protocol

1. There is no preload stage: a frame starts on its first `in_valid` beat, and
   frames may follow each other with no idle cycle. An idle beat inside a frame
   pauses the core (see *Idle-beat pause* below). `preload_valid` is an unused port kept so wrappers still connect.
   The bias is compile-time — see *Bias* below.
2. For each K chunk:
   - pulse `start_mat_mul` on the first A/B beat of that chunk
   - drive `validity_mask_a_cols_b_rows` for that chunk
   - feed `max(M,N)` row/column beats

   `pe_reset` and `final_mat_mul_size` are tied off (`1'b0` / `8'd0`) in the
   wrapper instantiation: the slice never acts on them, and `rst` is the only
   recovery path (see *Tensor-Slice Assumption* below).
3. Intermediate outputs from non-final K chunks are ignored.
4. After the final K chunk, the output collector releases one tile-row at a
   time and concatenates its column tiles into full output rows.

### Idle-beat pause

The core tolerates idle beats inside a frame. `frame_open` is set on a frame's
first accepted beat and cleared on its last; `pause = frame_open && !in_valid_q`.
The slices and all core sequential state run on `en && !pause`, so a pause
freezes the core as a unit, exactly like a core-level `en` freeze. `out_valid`
and `out_last` are held low while paused so no output row repeats. Idle beats
between frames do not pause, so the previous frame's drain continues. The
weight-ROM and A-replay counters advance on accepted beats only. With no gaps
the feed period (input beats per frame) and the latency are unchanged.

Trade-off: a pause also freezes the previous frame's drain (there is one `en`
per slice), so a bursty upstream such as im2col can cost a few cycles of
frame interval. A small read-ahead was measured not to help.

## Requantization and Bias

The slices are a **pure integer matmul**; the core owns everything after it, in
two stages, and `c_row` carries finished result codes of `out_width` bits per
column (`c_bits = GRID_COLS * 8 * out_width`).

- **Stage 1, in the slice.** The full K contraction accumulates in 32 bits. The
  slice's 32-bit output lane is that sum after a round-half-up shift by `S1`
  and a wrap to 32 bits. `S1` is an IP parameter set out of band; the generator
  records it as a comment above each instantiation (VTR's hard-block model has
  no parameters, so it is not a Verilog override). `S1 = 0` is a pass-through
  (no rounding) and is the normal case: the generator derives `S1` from the
  layer's `accum_t` as the smallest shift that makes the gemm-scale accumulator
  fit 32 bits.
- **Stage 2, in the wrapper.** The 32-bit partials of the K partitions are
  summed in 32-bit wrapping arithmetic (one term in the chunked path), the
  bias is added at that intermediate scale (`frac_a + frac_b - S1`), then a
  round-half-up shift by `S2` and a wrap to `out_width`. `S1 + S2 =
  frac_a + frac_b - frac_out`. No saturation anywhere.

The requant is meant to be **one exact step in the result type's own mode**
(`RND` = round half up, `TRN` = floor), and the generator picks `S1`/`S2` to
get there:

- `S1 = 0`: stage 2 does the whole shift. For a `TRN` result the floor is
  `floor(x / 2^S2) = round_half_up(x - 2^(S2-1), S2)`, so the generator folds
  `-2^(S2-1)` into the baked bias codes (creating them for a bias-free layer);
  the emitters keep a single stage-2 form.
- `S1 > 0` on an `RND` result: the whole shift moves into the slice (`S1 =
  frac_a + frac_b - frac_out`, `S2 = 0`) so it rounds once. The bias then lands
  at the output scale, so this needs a bias-free layer or a bias no finer than
  the result.
- Otherwise (`S1 > 0` with a `TRN` result, or with a bias finer than the
  result) the two-stage path stays and can differ from the reference by one
  output LSB; the generator warns. The same holds for `S1 > 0` with
  `k_spatial > 1`, where every K partition is rounded before the partials are
  summed; folding K fully in time avoids it.

The bias is a **compile-time constant**: one 32-bit signed lane per column,
baked into the core as a flat `wire` from the same codes list the C behavioral
core and mvau use (`gemm_ip/biasrom.py`). It is a flat wire, not a `reg`
array, because parmys would otherwise infer an unclocked memory and vpr would
abort. Under fold-N the wire holds `n_passes * core_n` lanes and is indexed by
the emitting frame's column group, latched when the frame is allocated (the
live feed-side group counter has already advanced by emit time). There is no
bias port, and there is no preload phase.

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
- `op[2]` (commit tag, level asserted at the committed pass's `start_mat_mul`
  edge) — tags the wave slot allocated on that edge as committed; only a
  committed wave's per-cell captures land in the shadow bank. Non-final K
  passes of a multi-pass frame start with `op[2] = 0` so their partial sums
  keep accumulating without being captured.

`op[1]`/`op[2]` replace the accumulator lifecycle that `pe_reset` used to
own: one wave can drain from its shadow bank while the next wave is already
computing into the (self-clearing, per-cell) live accumulators.

All tiles in the grid finish together, so the column tiles of one tile-row
concatenate as pure wiring. The wrapper holds every tile-row (`op[0] = 1`) and
releases them one at a time in row-major order for their 8-row bursts,
pulsing `op[1]` to cut a burst short; `op[2]` rides the final K pass's
`start_mat_mul` rather than pulsing at drain time.

This replaced an earlier per-tile delay-line alignment pyramid, whose shift
registers cost sum-of-delays x 129 FFs (~6.2k on a 2x2 grid, ~29k on 4x2). That
pyramid — and its unused buffered-synth generator — has since been removed.

The `op`/`pe_reset` pins live only on the `tensor_slice_int8_atlas` black-box
instantiation (see *Tensor-Slice Assumption* below); the RTL is structural
only now, so there is no separate behavioral grid model driving these pins.
Local regression is therefore compile-check only for the `op` contract
itself; functional validation of the pins is deferred to the separate
hardblock project's cosim. See `wrapper_run_loop.md` for the multi-frame
feed/capture schedule this overlap builds on.

## Tensor-Slice Assumption

The synth wrapper does not define the tensor-slice module. It instantiates
each slice as a black box:

```verilog
(* blackbox *) tensor_slice_int8_atlas slice_rX_cY (...);
```

`tensor_slice_int8_atlas` (22 ports) directly maps to a hardblock in the VTR
architecture, with the following properties:

- accept one A row tile and one B column tile per cycle
- internally skew/diagonalize row/column inputs for its systolic array
- per-cell accumulators self-clear at capture; a wave's cells are captured
  into the shadow bank only when that wave's `op[2]` commit tag was asserted
  at its `start_mat_mul` edge, freeing the array for the next wave without
  waiting for the shadow bank to drain
- gate readout by `op[0]` (`out_ctrl`) from the shadow bank, independent of
  the live accumulators' state (see *Output Collector* above)
- on `op[1]` (`drain_stop`), cut the remaining masked/padded drain burst
  short, with no effect on accumulators or shadow contents
- `pe_reset` and `final_mat_mul_size` are tied off in the wrapper
  instantiation (`1'b0` / `8'd0`); the slice never acts on either, so `rst`
  is the only recovery path
- one stage-1 round-half-up shift input, `shift_amount[3:0]`, latched at
  `start_mat_mul`
- symmetric-only quantization: no zero-point pins or datapath
- the module definition carries `(* blackbox *)` so yosys/parmys discards the
  body and maps instances to the VTR hard-block model (simulators ignore the
  attribute)

This is a **model-level contract, not an in-repo implementation**: the
`tensor_slice_int8_atlas` hardblock ships with the target
(`src/targets/tensor_slice/tensor_slice_int8_atlas.v`) but its internals are a
separate project's concern. The RTL wrapper is structural only -- there is no
behavioral grid model instantiating or driving these pins in this repo.
Validating the pins themselves against real hardblock RTL is out of scope for
this repo's regression; that happens in the hardblock project's cosim.

The generated wrapper does not change the tensor-slice port list and does not
inline or concatenate tensor-slice RTL. The model ships once per package
root (per target), not once per layer: `finalize()` writes it, and
`sources_tcl()` adds it once via `gemm_ip_sources.tcl`; a standalone
`run_catapult.tcl` adds `../tensor_slice_int8_atlas.v` and uses
`-RESET_KIND sync` so the exported RTL goes straight into VTR.

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
