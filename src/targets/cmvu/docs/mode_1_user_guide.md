# Mode 1 user guide

This guide explains how to integrate the V1 Mode-1 CMVU (`cmvu_mode1`): load weight tiles, compute with a
resident tile, preload another tile concurrently, switch tiles safely, and interpret the result interface.
The normative micro-architecture specification remains [architecture.md](architecture.md).

> **Hardened block, not parameterizable soft IP.** `cmvu_mode1` is a fixed-function block
> specified, validated, and taken to GDS at the V1 defaults (`k=4, n=8`, int8 inputs/weights,
> int32 output lanes, `L=6`,
> 8 resident weight tiles). Every runtime-configurable aspect — operand signedness, requant
> shift, effective output width, tile selection, framing, per-lane bias — is exposed as a **port**. Do not
> re-parameterize the module to alter its behavior: non-default parameter values are outside
> the validated configuration and are not covered by this guide.

## 1. Default configuration

The V1 defaults implement a `k=4` by `n=8` multiplier grid:

```text
y[j] = requant(bias[j] + cascade_in[j] + sum(i=0..3, x[i] * W[i][j]))
```

- Inputs and weights are int8 lanes; `a_signed` and `b_signed` select signed or unsigned interpretation.
- There are eight output lanes.
- Accumulation and cascade values are int32.
- `y_out` contains eight requantized int32 lanes (256 bits total). Each lane is an effective
  width `W = out_w + 1` (`1…32`) value, with the high bits sign-extended; `out_w=31` gives the
  full 32-bit result, `out_w=7` puts the V1 int8 result in the low byte.
- Default input-to-output latency is six cycles; `y_valid` identifies valid output cycles.
- The weight memory contains 8 slots by default. Each slot holds one `4×8` tile (256 bits).

## 2. Weight-tile packing

The internal representation is always row-major. Weight `W[i][j]` occupies:

```text
tile[((i * 8 + j) * 8) +: 8]
```

The loader accepts row-major and column-major wire formats and writes both into this same representation.

### Row-major load

A tile takes four accepted beats. Each beat supplies one complete row:

| Beat | `b_in[63:0]` |
|---:|---|
| 0 | `W[0][0]` in bits `[7:0]` through `W[0][7]` in `[63:56]` |
| 1 | `W[1][0]` through `W[1][7]` |
| 2 | `W[2][0]` through `W[2][7]` |
| 3 | `W[3][0]` through `W[3][7]` |

Drive `w_col_major=0`. Assert `w_we` on each accepted beat.

### Column-major load

A tile takes eight accepted beats. Each beat supplies one complete column in the low half of `b_in`:

| Beat | `b_in[31:0]` | `b_in[63:32]` |
|---:|---|---|
| `c` | `W[0][c]` in `[7:0]`, `W[1][c]` in `[15:8]`, `W[2][c]` in `[23:16]`, `W[3][c]` in `[31:24]` | ignored |

Drive `w_col_major=1`; beats `c=0..7` load columns in ascending order.

### Paired column-major load

Set `w_col_major=1` and `w_dual_tile=1`. `w_tile_sel` is one **even** base address `2p`; the only
destination pair is `(2p,2p+1)`. For each of eight beats `c=0..7`, `b_in[31:0]` supplies column `c`
of tile `2p` and `b_in[63:32]` supplies column `c` of tile `2p+1`, each packed rows 0–3 LSB first.
Both tiles are complete after beat 7. `w_dual_tile=1,w_col_major=0` is invalid.

### Transaction rules

- Assert `w_load_start` together with `w_we` on the first beat. This is recommended for every load.
- The first accepted beat captures `w_tile_sel`, `w_col_major`, and `w_dual_tile`; changes to those pins during the remaining
  beats do not alter the transaction.
- `w_we=0` inserts a bubble. The beat counter advances only when `w_we=1`.
- Asserting `w_load_start` with a later accepted beat restarts at beat zero and captures a new address and
  format. A partially written slot must not be used for computation.
- After reset, memory contents are undefined. Load a slot completely before selecting it for computation.

## 3. Compute while loading another tile

The memory has an independent synchronous read port and write port. Therefore the CMVU can read one active
tile every compute cycle while loading a different slot:

```text
compute read address: tile_sel   = active slot A
load write address:   w_tile_sel = free slot B
```

This is the normal way to hide weight-load time. Row-major loading takes four accepted cycles; either
column-major form takes eight. During a concurrent valid computation, a paired load must keep `tile_sel`
outside both destinations. Neither operation changes compute latency or throughput.

The mandatory safety rule during concurrent operation is:

> When `valid=1`, never write the slot selected by `tile_sel`.

Loading the selected slot while `valid=0` is safe and is the normal way to initialize memory before the
first computation. A same-address read/write returns the old data for that collision cycle, but later reads
see the updated bytes; therefore a valid computation must not overlap writes to its selected tile.

### Safe row-major preload and switch

The following schedule computes continuously with tile A while loading tile B:

| Cycle presented before edge | `tile_sel` | Compute | `w_tile_sel` | `w_we` | Load action |
|---:|---|---|---|---:|---|
| 0 | A | input using A | B | 1 | B row 0, `w_load_start=1` |
| 1 | A | input using A | B | 1 | B row 1 |
| 2 | A | input using A | B | 1 | B row 2 |
| 3 | A | input using A | B | 1 | B row 3; load completes at edge |
| 4 | B | first input using B | don't care | 0 | no write |

Do not select B for a valid computation on the same edge that accepts B's final write beat. Present B's
`tile_sel` and its first matching activation in the following cycle. The memory registers the read address
while the mandatory input register captures that activation, so the tile and activation remain aligned.
This transition does not require an idle compute cycle: cycle 3 can accept an A input and cycle 4 a B input.

## 4. Basic single-pass computation

For a standalone `4×8` tile with no temporal-K folding:

1. Select a completely loaded slot with `tile_sel`.
2. Pack `x[0..3]` into `a_in` (`K*IN_WIDTH` = 32 bits), element 0 least-significant.
3. Present `a_signed`, `b_signed`, `shift_amt`, `out_w`, and `bias_in` with the corresponding valid input.
4. When the block is not part of a sum-cascade chain, set `CASCADE_EN=0` and leave `cascade_in`
   **dangling** — do not tie it to `0`/`1`. A chain tail's `cascade_out` is likewise left dangling.
5. Assert `valid=1`, `acc_first=1`, and `acc_last=1` for the input.
   (`mode`/`start` ports removed 2026-09-17.)
6. Sample `y_out` only when `y_valid=1`. With all default register banks present, this is six cycles after
   the corresponding valid input.

Lane `j` is returned in `y_out[j*32 +: 32]` (int32 lane; effective `W = out_w+1` bits sign-extended). `cascade_out[j*32 +: 32]` carries the full-precision int32
partial/final sum for spatial chaining and appears one cycle earlier than `y_out` at the default configuration.
`cascade_out` is valid at that fixed latency after its input and **holds its last accumulated partial on
later (invalid) cycles**; sample it only in the valid window. When the block is used standalone (not in a
chain), set `CASCADE_EN=0` and leave `cascade_in` dangling rather than tying it to `0` — a nonzero
`cascade_in` is live at the reduce node and would show through.

## 5. Bias and requantization

`bias_in` contains one signed int32 value per output lane (`BIAS_WIDTH=32`, accumulator scale,
sign-extended to `ACC_WIDTH` — a pass-through at the default since `BIAS_WIDTH == ACC_WIDTH`). On
`acc_first`, bias replaces the initial zero accumulator value, so it is added exactly once even when
temporal-K folding uses multiple passes.

Final outputs apply:

```text
arithmetic right shift by shift_amt (floor/TRN) -> wrap to W = out_w+1 bits -> sign-extend to int32
```

The block has no rounding-mode input; it always truncates. Round-half-up (`RND`) output is produced by
folding the rounding constant `1 << (shift_amt-1)` into `bias_in` before it reaches the block (accumulator
scale), so both hls4ml `RND` and `TRN` output modes are served by this one requant path.

`out_w` encodes `W-1` (0..31), uniformly across all lanes: `out_w=31` gives the full int32 result,
`out_w=7` gives the old int8 result in the low byte (high bits sign-extended). The effective result is not
saturated. Because the high `16-W` bits are just the sign fill, an integrator that needs only `W` bits may
leave those high `y_out` bits unconnected. `cascade_out` remains full int32 and is not requantized.

## 6. Folding and composition

Matrices larger than one `4×8` tile use temporal folding, spatial composition, or both. These cases require
careful `acc_first`, `acc_last`, cascade, and tile schedules. For standalone Mode 1, issue each
output group's K slices consecutively (N-outer/K-inner); replay the K-slice activation set for every group.
`y_valid` marks every pass, while `done` marks each completed group.

The group cadence is `num_K` issued cycles; a full `num_N`-group vector takes
`num_K × num_N` cycles. `done` must therefore be qualified as a group-completion marker, not
a full-vector marker. There is one accumulator per output lane: no slot counter, depth parameter,
or N-inner scheduling contract applies to standalone `cmvu_mode1`. See
[architecture.md](architecture.md):

- §1 for K/N folding and initiation intervals.
- §6 for temporal accumulation and bias-once behavior.
- §7 for spatial cascade timing.
- §8 for framing and interfaces.
- §11.3 for accumulator contracts.

## 7. Integration checklist

- Load every selected slot completely after reset.
- Assert `w_load_start` on the first accepted load beat.
- On every `valid=1` compute cycle, keep the active read slot and every load destination different.
- Do not use a partially loaded or abandoned slot.
- Switch `tile_sel` only with the activation that should use the new tile.
- Hold or present signedness, shift, `out_w`, bias, and framing with their corresponding valid input.
- Drive `out_w = RESULT_WIDTH-1` (31) for full-width int32 output; smaller values sign-extend a
  narrower effective result into each lane.
- Observe `y_valid`; do not infer output validity from a fixed software delay alone.
- Leave unused cascade pins dangling (chain head `cascade_in`, chain tail `cascade_out`); never tie them
  to `0`/`1`, and set `CASCADE_EN=0` on a block whose `cascade_in` is unused.
