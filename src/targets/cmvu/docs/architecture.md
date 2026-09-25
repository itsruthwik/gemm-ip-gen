# CMVU Architecture Specification

The **Constant Matrix-Vector Unit (CMVU)** computes matrix–vector and matrix–matrix products in integer
arithmetic — the core arithmetic of machine-learning inference. It is a coarse-grained, parameterized,
pipelined, **inner-product (adder-tree)** block on a multiplier array:

Computes `y = W · x` for a weight matrix `W` held in block-local memory and an input vector `x` that
streams through. The contraction is reduced **in space** by adder trees; one output vector leaves per
cycle. This is the weight-stationary / GEMM path (and the founding constant-matrix workload).

This page is the specification the RTL implements; the testbenches prove the RTL matches it.
For integration procedures and cycle-level load/switch examples, see the
[Mode 1 user guide](mode_1_user_guide.md).

> **Version scope (v1.1):** this specification covers the single-mode Mode-1 block
> (`cmvu_mode1`, inner-product / adder-tree) only. Mode 2 (outer-product) and the unified
> dual-mode `cmvu` were removed in v1.1 and live on only in the frozen `v1.0` branch.

## 1. Function and dimensions

The physical multiplier grid is **`k` rows × `n` cols**. Rows index the (spatial) contraction and cols
index the output lanes:

```
y[j] = Σ_i  x[i] · W[i][j]        i = 0 … k-1,  j = 0 … n-1
```

| Name | Meaning | Sets |
|---|---|---|
| **k** = grid **rows** | spatial contraction length per pass | adder-tree depth, activation slice width |
| **n** = grid **cols** | output lanes | number of trees, `y`/cascade width |

**V1 reference grid: `k = 4`, `n = 8`** (4 rows × 8 cols). Dimensions are `parameter`s; these are the
validated defaults.

| Quantity | Formula | V1 (k=4, n=8) |
|---|---|---|
| Multipliers | `k × n` | 32 |
| Adder trees | `n` | 8 |
| Adder-tree depth | `⌈log₂ k⌉` | 2 (sums 4 terms) |
| Activation slice | `k × IN_WIDTH` | 32 b |
| Output `y_out` width | `n × RESULT_WIDTH` | 128 b (int16 lanes, effective `W`-bit sign-extended) |
| Cascade width | `n × ACC_WIDTH` | 256 b (int32 partials) |
| Weight tile size | `k × n × COEF_WIDTH` | 256 b |
| Pipeline latency `L` | see §5 | 6 |

### Folding — arbitrary matrix sizes on a fixed grid

A logical weight matrix `K × N` larger than the physical `k × n` grid is **tiled onto the grid and
folded in time**. With `num_K = K/k` contraction tiles and `num_N = N/n` output-lane tiles:

```
issued-pass II       = 1
completed-group II   = num_K
full-vector II       = num_K · num_N
```

- **Standalone Mode-1 port utilization is one issued pass per cycle.** The complete K-slice activation
  set is replayed `num_N` times, once per output group (N-outer/K-inner K-first schedule).
- **Multiplier utilization stays 100%** — the grid does `k·n` MACs every cycle, and one output vector
  emerges every `num_K·num_N` cycles (`= 2k·2n / (k·n)` etc.), which is the optimal rate for a `K×N`
  contraction on a `k×n` grid.
- A different `k×n` **weight sub-tile feeds the grid every cycle** while folding, so the operand memory
  read port delivers a **full tile per cycle** (§4). (The "read at tile-switch rate" applies only to the
  degenerate single-tile stationary case, `num_K = num_N = 1`.)

Adjacent K passes use the same single lane accumulator register; a one-cycle launch gap is sufficient.

## 2. Number format

Integer only in V1.

| Parameter | Value | Meaning |
|---|---|---|
| `IN_WIDTH` | 8 | input / activation element width (int8) |
| `COEF_WIDTH` | 8 | weight / second-operand element width (int8) |
| `ACC_WIDTH` | 32 | internal accumulate / **cascade** width (int32) |
| `RESULT_WIDTH` | 16 | physical **result** lane width (int16) — the `y_out` lane width |
| `SHIFT_WIDTH` | 5 | width of the runtime `shift_amt` (covers shifts 0…`ACC_WIDTH`−1) |
| `out_w` | runtime | effective result width minus 1: `W = out_w + 1`, `1 ≤ W ≤ RESULT_WIDTH` (uniform across lanes) |

- **`y_out` is `n × RESULT_WIDTH` = 8 × 16 = 128 b** — the final requantized result bus (int16 lanes,
  each a sign-extended `W`-bit value). The
  full-precision int32 path is the separate **`cascade_out` (`n × ACC_WIDTH` = 256 b)**, used for
  chaining; the two are distinct ports (§7, §8).

- **Reduction is full-precision int32 end to end.** Products, adder trees, cascade, and the accumulator
  registers all carry int32.
- **Output requantization = shift → wrap-to-`W` → sign-extend, applied ONCE at the final emit.**
  When a full result is produced (the final contraction pass) the value is requantized:

  ```
  q = sum >>> shift_amt                                 // arithmetic (sign-aware) shift right = floor
  y_eff = q[W-1 : 0]                                    // WRAP to W = out_w+1 bits
  y = sign_extend(y_eff, RESULT_WIDTH)                  // high bits = copies of bit W-1
  ```

  Overflow policy is **WRAP** (two's-complement truncation to `W`), not saturate — a free
  bit-slice rather than a compare-and-clamp; the high `RESULT_WIDTH−W` bits are the sign fill, not
  accumulator data. `W = RESULT_WIDTH` (`out_w = RESULT_WIDTH−1`) reproduces a plain full-width wrap;
  `W = 8` (`out_w = 7`) puts the V1 int8 result in the low byte, sign-extended. Requant is combinational
  (sits between `REG_RED` and
  `REG_OUT`), so it adds no latency (`L` unchanged); `shift_amt = 0` is a plain wrap (no scaling).
- **Requantization is truncating (TRN/floor) only — the block has no rounding-mode input.** Rounding,
  when wanted, is supplied by the integrator through the 32-bit `bias_in` (§8): the rounding constant
  `1 << (shift_amt-1)` is folded into the bias at accumulator scale before it reaches the block, so a
  floor here plus a pre-rounded bias reproduces round-half-up (`RND`) at the output, and a plain bias
  (or none) reproduces `TRN`. One requant path therefore serves both hls4ml output modes exactly.
- **`y_out` is the final `RESULT_WIDTH`-bit result (effective `W`-bit, sign-extended); full precision goes
  out `cascade_out`.** Since `y_out` lanes are `RESULT_WIDTH`-wide, a full-int32 partial cannot ride it —
  the un-requantized int32 `sum`
  is emitted on **`cascade_out`** (256 b, tapped at `REG_RED`) for chaining. A cascade tail therefore
  reads its neighbour's `cascade_out` and emits the final result on its own `y_out`.
- **`shift_amt` and `out_w` are runtime per-frame inputs** (not compile-time parameters): the block is
  time-multiplexed across heterogeneous layers, each with its own requantization scale and effective
  output width, so baking either would force reconfiguration. They change only at frame boundaries (like
  `tile_sel`) and travel with the side-band, delayed by `L`.
- **Per-operand signedness is a runtime mode** (`a_signed`, `b_signed`), changeable per frame. Default
  signed/signed.

## 3. Dataflow

The block is a single-mode inner-product (adder-tree) datapath over 32 multipliers.

### Mode 1 — inner-product (adder-tree)

```
x (k-wide slice)  →  Multiply Array (k×n)  →  Adder Trees (×n, depth ⌈log₂k⌉)  →  Reduce  →  y
                     W  ↑ (held tile, read from operand memory)
```

- The `k`-wide activation slice is **broadcast across the n columns**; each of the **n adder trees**
  sums its `k` products (one output lane). The trees are independent — no cross-lane summation.
- Contraction beyond `k` is handled by **folding** (§1): a feed-forward tap-accumulate over `num_K`
  slices, and/or spatial cascade across blocks (§7).
- Feed-forward → one output vector per (folded) output cadence; latency `L` (§5).

![Mode-1 grid dataflow: x broadcast across columns, per-lane adder-tree reduction](diagrams/cmvu-grid-dataflow.svg)

![Mode-1 datapath: SRAM→grid→trees→reduce(+cascade,+accum/bias)→requant→REG_OUT](diagrams/cmvu-datapath.svg)

## 4. Operand memory

A **block-local operand memory** feeds the multiplier array. It has a **64 b write / load port** and a
**wide 256 b tile read port** (one `k×n` tile per cycle — required by folding, §1). It is organized as
**two 1 KB banks (2 KB total)** in the generic primitive default; the V1 Mode-1
instantiation uses 8 resident tiles (256 B across both banks).

The read port is a **synchronous, single-cycle registered read**: the read address is registered
inside the memory, and the addressed word is valid the cycle *after* the address is presented — there
is no asynchronous/combinational read path. The write port is independent of the read port (a 1R1W
memory), and same-address read/write collisions resolve to **read-returns-prior-contents** (the read
samples the word as it stood before that cycle's write lands). On a cycle with `valid=1`, the caller must
not write the slot selected by `tile_sel`; a valid computation must never consume a tile while it is being
modified. With `valid=0`, loading the selected slot is safe because the read value is not consumed.

| Parameter | Value | Meaning |
|---|---|---|
| `M_MEM_BANKS` | 2 | interleaved banks (address bit 0 selects the bank) |
| `M_MEM_TILES` | 32 / bank | resident `k×n` tiles per bank in the generic primitive default (32 × 256 b = 1 KB/bank, **2 KB total**). The V1 Mode-1 instantiation uses `M_MEM_TILES=8` (8 tiles, 256 B across both banks) — see "Mode 1 (weights held)" below |

- **Weights held.** The 64-bit `B`/operand-in port accepts either of two load formats while
  keeping the stored tile in the same canonical row-major representation:
  - **Row-major:** four accepted beats, one eight-int8 row per beat (`W[r][0]` in the least-significant
    byte through `W[r][7]` in the most-significant byte), rows 0 through 3.
  - **Column-major:** eight accepted beats, one four-int8 column per beat in `b_in[31:0]`
    (`W[0][c]` in the least-significant byte through `W[3][c]` in bits `[31:24]`), columns 0 through 7;
    `b_in[63:32]` is ignored when `w_dual_tile=0`.
  - **Paired column-major:** with `w_col_major=1,w_dual_tile=1`, one even `w_tile_sel=2p` selects
    exactly `(2p,2p+1)`. On each beat `c`, the lower and upper 32-bit halves carry that column for
    the even and odd tile. Both canonical row-major tiles complete after eight beats. `(0,1)` format
    (`w_col_major=0,w_dual_tile=1`) and an odd base are invalid.
  The physical implementation interleaves logical tiles across two WORDS/2-entry 1R1W banks: address bit 0
  selects bank and upper bits select pair index. This permits one write per bank in paired mode without
  a true 2W bank. `w_col_major`, `w_dual_tile`, and `w_tile_sel` are captured on the first accepted beat. `w_load_start`, asserted with
  `w_we`, explicitly starts or restarts a transaction at beat zero; a restart abandons the incomplete
  contents of the previous transaction. Pin changes after the first beat do not redirect or reformat the
  transaction. Completed tiles are **held and reused** across input rows and read a full tile per cycle in
  the folding cycle-through pattern. The whole `K×N` weight set is kept resident.
  - Mode 1 uses **both banks — 8 tiles, 256 B at the V1 default** — for weight residency (`tile_sel` is
    `⌈log₂8⌉ = 3` b).
  - **Single memory, no replicated buffer.** Bubble-free tile switching comes from the *many slots being
    the buffer*: preload the next tile into a **free slot `j`** while computing the active tile `i`, then
    point `tile_sel` at `j`. **Caller contract:** whenever `valid=1`, no write destination may equal
    `tile_sel`. Loading the selected slot while `valid=0` is permitted.

## 5. Register and latency model

Every pipeline stage is a register bank; on all stages but one, whether the bank is a real register or
a combinational pass-through is a **static, compile-time choice** — **latency = the number of present
banks** on the input→output path. Default is all-present.

| Stage | Bank | Cycles (k=4) |
|---|---|---|
| operand mux | — (combinational) | 0 |
| input | `REG_IN` (always present, non-bypassable) | 1 |
| multiply | `REG_MULT` | 1 |
| adder tree | `REG_TREE[0 … ⌈log₂k⌉−1]` | 2 |
| reduce | `REG_RED` (the 3-input reduce node) | 1 |
| output | `REG_OUT` | 1 |

**`L = 1 + 1 + 2 + 1 + 1 = 6`** at k=4 (Mode 1). The shallower depth-2 tree gives one cycle less latency
than a depth-3 (k=8) tree.

- **`REG_IN` is mandatory (not bypassable), unlike the other four banks.** The operand memory's read
  port is a synchronous, single-cycle registered read (§4): the address is registered inside the
  memory, so the addressed tile is valid one cycle after `tile_sel` is presented. `REG_IN` registering
  the incoming activation by that same one cycle is exactly what keeps the activation aligned with its
  paired weight tile at the multiply stage — removing `REG_IN` would desynchronize them by a cycle. The
  other four banks (`REG_MULT`, `REG_TREE[*]`, `REG_RED`, `REG_OUT`) remain independently bypassable,
  and `L` stays 6 at the all-present default (`REG_IN` simply moved from "present by default parameter
  choice" to "always present").

- **`cascade_out` taps `REG_RED` (latency `L−1` = 5); `y_out` taps `REG_OUT` (latency `L` = 6).** Tapping
  cascade one stage earlier makes the **cascade hop cost exactly 1 cycle**.
- **`cascade_out` holds its last accumulated partial across invalid cycles.** `REG_RED` is free-running:
  on a cycle with no valid data at the reduce node, `sum = local_partial(0) + cascade_term + acc_reg`,
  which is the held accumulator value once the pipeline has drained. The hold is therefore a consequence of
  the running accumulator, **not a separate enable**: `cascade_in` is live at the reduce node (§7), so a
  nonzero idle `cascade_in` (or idle `a_in`) shows through. Callers sample `cascade_out` only at the
  fixed-latency window after a valid input and must drive `cascade_in=0` when no partial is intended.
- The **broadcast forward register** (§7) is off the local critical path — forwarding does not add to a
  block's own `L`.
- The **requantization stage** (§2) is combinational, between `REG_RED` and `REG_OUT` on the `y_out`
  path only; it adds no bank, and `cascade_out` (tapped at `REG_RED`) stays full int32.

## 6. Accumulation

For shipped standalone `cmvu_mode1`, accumulation is **single-slot**: one `n × ACC_WIDTH`
register bank, one running accumulator per output lane. There is no `ACC_BUF_DEPTH`, slot
counter, or N-inner interleave. For an output group, `acc_first` on K pass 0 selects its
lane bias; following passes read the running accumulator; `acc_last` on the final K pass marks
that group's `done`. The reference RTL resets the registers to zero, but `acc_first` remains
mandatory on every group so prior-group state is discarded. Bias, shift, `acc_first`, and
`acc_last` travel with their data; the sideband is `{acc_last, acc_first}`, not an address.

**Mode 1** accumulation is realized at the **reduce node** together with one lane accumulator bank. The
reduce node is a **single-cycle combinational 3-input int32 adder**:

```
sum = local_partial + cascade_term + acc_term

  cascade_term ∈ { neighbor cascade_in (spatial-K), 0 }
  acc_term     ∈ { acc_reg lane, bias_in lane on acc_first }

  → cascade_out ← sum            (always, for further chaining; full int32, un-requantized; holds
                                  across invalid cycles — see §5)
  → acc_reg     ← sum
  → y_out       ← requant(sum)   (every valid pass; round→shift→wrap-to-W→sign-extend per §2, bypassable)
```

Three inputs let a single block reduce contraction terms across **space and time simultaneously**. The
adder is single-cycle combinational (fixed by §5's `L`; a pipelined variant would raise `L` and shift the
cascade tap).

**First-class bias inject.** A per-output-lane signed int32 bias (`bias_in`, §8, `BIAS_WIDTH=32`),
sign-extended to `ACC_WIDTH` (a pass-through at the BIAS_WIDTH==ACC_WIDTH default), is added **exactly once
per output**: `acc_first` initializes `acc_term` to that lane's `bias_in`. Every subsequent K pass reads
the running `acc_reg` partial; bias is folded exactly once and is never reintroduced, while `acc_last`
qualifies the group's delayed `done` without suppressing intermediate `y_out` values. In single-pass
operation (`acc_first == acc_last` on the same cycle) this degenerates to
`y_out = requant(bias + local_partial + cascade_term)`. `bias_in` is a **feed-forward side
input**: it is delayed through the same matched register chain that carries `{acc_last, acc_first}`
to the reduce node (§8), so it arrives aligned with `acc_first` without adding a pipeline stage — `L` and
`II` are unaffected. **Composition caveat:** in a cascade or spatial array of blocks, bias must be applied
at exactly one block in the chain (double-adding it at every block would multiply it by the tile count) —
the reference array wrapper (§7) ties every block's `bias_in` to `0`; per-block bias composition across a
spatial array is deferred beyond this specification.

**Accumulator** (single running register per output lane):

| Parameter | Value | Meaning |
|---|---|---|
| accumulator | `n × 32` b | one running accumulator register per output lane |

- One `n × 32` accumulator register bank; **one read + one write every issued cycle** (read-modify-write).
- Holds one running accumulator per lane for K-first folding.

## 7. Composition — cascade and broadcast

Two **distinct** neighbor networks compose blocks in space (primarily a Mode-1 mechanism).

- **Sum-cascade (contraction direction).** Full int32 partial-`y` flows block→block; each block's reduce
  node adds `cascade_in`, so the chain **tail emits the fully-reduced result**. Width `n × 32` = 256 b.
  No external reduction is ever required.
  - **`cascade_in` is consumed *directly* (combinationally) at the reduce node — it is NOT re-registered
    through the receiving block's `REG_IN`/`REG_MULT`/`REG_TREE` stages.** This is what makes the hop cost
    exactly 1 cycle: an upstream `cascade_out` is available at `L−1` (tapped at `REG_RED`), and the
    downstream reduce node combines it in the same cycle its own local partial arrives. **Timing contract:**
    a block's `cascade_in` must be **presented aligned with that block's local partial at the reduce node**,
    i.e. `D = present(REG_IN) + present(REG_MULT) + Σ present(REG_TREE)` cycles (= 4 at the all-present
    default) after the block's corresponding `a_in`. In a chain this is met by skewing each successive
    block's activation input by `+1` cycle (the cascade hop skew), so `cascade_out(L−1) → cascade_in`
    lines up with just one cycle of skew. Re-delaying `cascade_in` inside the block would defeat the
    REG_RED tap and inflate the hop to `L−1`; do not.
  - **Cascade port rule (inter-block wiring).** `cascade_out` may connect *only* to another block's
    `cascade_in`, and `cascade_in` *only* from a `cascade_out` — never driven or read by wrapper/fabric
    logic. Chain-endpoint pins that are unused (a chain head's `cascade_in`, a chain tail's `cascade_out`,
    or both on a standalone single block) are left **dangling**, not tied to `0`/`1`. A block whose
    `cascade_in` is unused sets `CASCADE_EN = 0` so the dangling input is ignored; `CASCADE_EN` only
    zeroes the `cascade_in` term — `cascade_out` is unaffected and still carries the local partial.
- **Broadcast (output-lane direction).** The same input vector slice is delivered to every block in a
  row of output-lane tiles as a **registered forward chain**, which keeps interconnect local (helping timing
  closure). Each
  block produces a disjoint slice of `y`; slices concatenate.
  - `BCAST_REG_HOPS` = 1 (default): one register per hop → broadcast hop skew = 1 cycle. The forward
    register is **always present**, decoupled from `REG_IN`.

**Grid skew and latency.** For a block reached by `h_b` broadcast hops and `h_c` cascade hops, its input
is skewed by `h_b + h_c` cycles and its output appears at `L + h_b + h_c`.

**Composition-layer signals — no block-level broadcast port.** The block itself (`cmvu`/`cmvu_mode1`) has
**no `x_bcast_in`/`x_bcast_out` pins** (§8) — spatial-N delivery is realized entirely by the **wrapper**
that instantiates multiple blocks, not by a port the block exposes. In the reference wrapper
(`cmvu_array`), this is a fully internal register chain: one shared external activation bus enters the
wrapper, is skewed per cascade-column by the alignment chain above, and is then forwarded down each
column's row of blocks by one `cmvu_regbank` hop per row (the same registered-forward-chain mechanism
described above, just internal to the wrapper rather than crossing an inter-block pin). A different
composition strategy is equally valid and stays within this section's contract as long as it is a
registered hop with the stated 1-cycle skew per hop; e.g. **direct `a_in` fan-out** (each block reads the
same wire, no forward register, zero skew) is a legitimate degenerate composition for wrappers where
Fmax/fan-out is not a concern, at the cost of losing the "local wires only" property. Either way this is a
**wrapper/fabric-level design choice**, not a block-interface requirement.

*(Note: with the wider `n = 8`, the int32 **cascade** bus is 256 b — the deliberate cost of the 4×8 grid;
larger local contraction was traded away because folding (§1) makes low local-contraction cheap. The
requantized `y_out` bus is 128 b, int16 per lane.)*

## 8. Control and interface

### Standalone Mode-1 K-first contract

Schedule output groups outermost and K slices innermost. Issue every pass on consecutive cycles:
the K-slice activation set is replayed `num_N` times, once per output group. The port is occupied
for every issued pass; a group completes every `num_K` cycles and a full vector every
`num_K × num_N` cycles. Same-slot launch gap is at least one cycle, and back-to-back K passes
and group boundaries are permitted. `y_valid` accompanies every valid pass, including intermediate
partials. `done` is delayed `acc_last`, so it marks each completed group, not a whole vector.
No ready/backpressure behavior is provided, and no `start`/frame marker exists (`start` and `mode`
were removed from `cmvu_mode1` on 2026-09-17; the removed unified block is gone in v1.1).

- **Contract:** `valid` + **fixed latency**, no `ready`/backpressure. `y_valid` is `valid` delayed by `L`.
- **Framing:** `acc_first` initializes a group and `acc_last` marks its final K pass;
  the sideband is `{acc_last, acc_first, shift_amt, out_w}`.
- **Schedule:** N-outer/K-inner K-first, one issued pass per cycle; group/full-vector cadence
  is `num_K`/`num_K·num_N`; same accumulator launch gap is at least one.
- **`done`** is delayed `acc_last` for each group.

### Interface

This is the interface of **`cmvu_mode1`** (`cmvu_mode1.sv` in the RTL — the shipped block).
Widths shown are the V1 default
(`k=4, n=8, M_MEM_TILES=8, SHIFT_WIDTH=5, ACC_WIDTH=32, RESULT_WIDTH=16`); `tile_sel`/`w_tile_sel` are
`⌈log₂ M_MEM_TILES⌉` bits, generically (3 b at the V1 default). The underlying `cmvu_w_mem`
primitive defaults to 64 slots (32/bank, §4); the V1 Mode-1 instantiation uses 8.

| Signal | Dir | Width (k=4,n=8) | Description |
|---|---|---|---|
| `clk`, `rst` | in | 1 each | clock / reset (async reset, §5) |
| `valid` | in | 1 | input-stream valid; gates each compute cycle |
| `acc_first` | in | 1 | first K pass of the current output group (selects `bias_in` instead of prior accumulator state) |
| `acc_last` | in | 1 | final K pass of the current output group (drives delayed per-group `done`) |
| `a_in` (activation) | in | 32 b (`K*IN_WIDTH`) | 4-elem input slice |
| `b_in` (operand-in) | in | 64 b | row-major: eight int8 values/beat, four beats. Column-major: `b_in[31:0]` carries rows 0–3 of one column, eight beats; upper 32 bits ignored |
| `w_we` | in | 1 | accepts one weight-load beat when asserted |
| `w_load_start` | in | 1 | asserted with `w_we` to start or restart a load transaction at beat zero and capture its format/address; optional for uninterrupted legacy row-major bursts after reset/completion |
| `w_col_major` | in | 1 | load format captured at transaction start: `0` = four row-major beats, `1` = eight single-tile column-major beats |
| `w_dual_tile` | in | 1 | with `w_col_major=1`, select the eight-beat strict even/odd paired-column load; invalid with row-major |
| `tile_sel` | in | ⌈log₂ M_MEM_TILES⌉ = 3 at V1 default | active **read** tile slot (8 tiles across both banks at the default) |
| `w_tile_sel` | in | ⌈log₂ M_MEM_TILES⌉ = 3 at V1 default | **write**-target tile slot for the `b_in`/`w_we` weight-load protocol (independent of `tile_sel`, so a free slot can be preloaded while another is read — §4's caller contract) |
| `a_signed`, `b_signed` | in | 1 each | runtime per-operand signedness |
| `shift_amt` | in | `SHIFT_WIDTH` = 5 | runtime per-frame requant right-shift |
| `out_w` | in | `⌈log₂ RESULT_WIDTH⌉` = 4 | runtime effective result width **minus one** (`W = out_w+1`, 1…16), uniform across all lanes; high `RESULT_WIDTH−W` bits are the sign fill (§2) |
| `cascade_in` | in | `n × ACC_WIDTH` = 256 b | partial-`y` from the previous block in a sum-cascade chain (§7), consumed directly, combinationally, at the reduce node |
| `y_out` | out | `n × RESULT_WIDTH` = 128 b | requantized result, all 8 lanes/cycle (one beat), int16 lanes |
| `cascade_out` | out | `n × ACC_WIDTH` = 256 b | partial-`y` to the next block (tapped at `REG_RED`, latency `L−1`; full int32) |
| `bias_in` (see note below) | in | `n × BIAS_WIDTH` = 256 b | per-lane signed int32 bias (accumulator scale), sign-extended, added exactly once per output at `acc_first` (§6) |
| `y_valid`, `done` | out | 1 each | output-valid / per-group-complete |

**No `x_bcast_in`/`x_bcast_out` block ports.** The block itself has no broadcast pins — broadcast
(output-lane composition, §7) is realized **entirely in the composition layer** (the wrapper/fabric
around one or more `cmvu_mode1` instances), not as an interface the block exposes. See the composition-layer
note at the end of §7.

**Per-block primitive.** `cmvu_mode1.sv` is the standalone RTL model of the table above and serves as
`cmvu_array`'s per-block primitive (§7).

**`bias_in`.** In: `n × BIAS_WIDTH` =
256 b (`BIAS_WIDTH = 32`) — one signed int32 bias per output lane, at accumulator scale
(`2^(in_frac + w_frac)`, lane `j` at
`bias_in[j*BIAS_WIDTH +: BIAS_WIDTH]`), sign-extended to `ACC_WIDTH` at the reduce node and consumed
at `acc_first` per §6's first-class bias inject. It is a feed-forward side input carried to the reduce
node through a matched delay chain (mirroring the `{acc_last, acc_first}` side-band above), so it
needs no `L`/`II` change. `cmvu_array` (§7) ties every block's `bias_in` to `0` — bias composition across
a spatial array is deferred beyond this specification (§6).

## 9. Datapath diagram

**Datapath/dataflow figures (grid dataflow, Mode-1 datapath) live in §3,
next to the dataflow they illustrate — see there.** This section holds only the two views that cut across
the block and its composition: the top-level block overview and multi-block spatial composition.

![CMVU top-level overview](diagrams/cmvu-overview.svg)

Top-level block: `a_in` feeds the multiplier array, `b_in` feeds the operand SRAM (load)
on top and the SRAM feeds the array below; the adder-trees box attached below the array
reduces per-lane into the separate 3-input reduce node (local + cascade + accumulator/bias)
with `cascade_in` entering from the top (outside the block), the single-slot accumulator
register feeding back into reduce, then requant and `y_out`/`cascade_out`.

![CMVU spatial tiling](diagrams/cmvu-tiling.svg)

Spatial tiling (§7): a 2×2 array of blocks — cascade (space, reduces contraction K) flows horizontally,
the broadcast forward chain (output lanes N) flows vertically; skew = broadcast hops + cascade hops.

## 10. Tiling summary

Standalone temporal folding is K-first only: temporal K uses the one lane accumulator bank;
temporal N is represented by repeating the complete K-slice sequence for each output group.
There is no standalone temporal-N slot mechanism.

| Dimension | Decomposition | Mechanism |
|---|---|---|
| **Contraction (K)** | space, then time | sum-cascade (space) → single-slot accumulator (time) |
| **Output lanes (N)** | space | broadcast forward chain (input reuse; disjoint output slices) |
| **stream length** | time | successive input vectors |

Folding law (grid `k×n`, logical `K×N`): issued/group/full-vector cadence is
**1/`num_K`/`num_K·num_N`**. The K-slice activation set is replayed once per output group; weight read =
one `k×n` tile/cycle, multiplier utilization 100%.

## 11. Data conventions and contracts

These fix the ambiguities an RTL author must resolve; they are conventions (choose once, apply everywhere),
not architectural choices.

### 11.1 Bit-ordering / packing (element 0 in the LSBs)

- **Vector ports** pack **element 0 in the least-significant bits**. On `a_in` (and on a
  wrapper-level broadcast wire, where a composition wrapper provides one — there is no `x_bcast`
  block port, §7), activation
  element `i` occupies bits `[i·IN_WIDTH +: IN_WIDTH]`. On **`y_out`**, result lane `j` occupies
  `[j·RESULT_WIDTH +: RESULT_WIDTH]` (int16, effective `W`-bit sign-extended, §2). On **`cascade_in`/`cascade_out`**, int32 lane `j` occupies
  `[j·ACC_WIDTH +: ACC_WIDTH]`.
- **Weight tile** is packed **row-major (contraction-row `i` major, output-col `j` minor)**: element
  `W[i][j]` is at linear index `i·n + j`, i.e. bits `[(i·n + j)·COEF_WIDTH +: COEF_WIDTH]`.
- Neighbor connections (cascade, broadcast) use the same packing so tiled blocks wire up consistently.

### 11.2 Multiplier internal precision and signedness

- Each operand is extended to `IN_WIDTH+1` (= 9) bits according to its runtime flag: **`*_signed=1` →
  sign-extend; `*_signed=0` → zero-extend** (prepend 0). Multiplying two 9-bit signed values yields an
  18-bit signed product — correct for all four signed/unsigned combinations.
- **Each product is then sign-extended to `ACC_WIDTH` (32 b), and the adder trees, cascade, reduce node,
  and accumulator registers all carry full int32.** This progressive-to-32 rule is correct for any
  `k`/`n` (an int8·int8 grid sum cannot overflow int32 at any realistic tile size); no per-tree-level width
  bookkeeping is needed.
- **This headroom claim covers the spatial grid sum only** (one reduce-node evaluation: adder-tree partial
  + a bounded cascade depth) — it does **not** bound unbounded *temporal* accumulation through the
  lane-accumulator RMW loop, whose 32-bit headroom is a function of accumulation length and is a
  **caller responsibility** to size, not a hardware guarantee (caller obligation, §12).

### 11.3 Single-slot accumulation

`acc_first` selects bias initialization; each following K pass reads the lane register and writes the
new sum. `acc_last` marks that group's delayed `done`; `y_valid` remains asserted for every issued pass.
The reference RTL resets the register bank, but `acc_first` is mandatory per group.

### 11.4 Composition boundary tie-offs

- At the **head of a cascade chain**, `cascade_in` **must be tied to 0** (or `cascade_term` gated off by
  config) so no partial sum is injected. Unused neighbor inputs are **tied to 0, never left floating** — a
  config bit disables `cascade_term`/broadcast so an unwired input cannot inject garbage.
- `x_bcast_in` at a broadcast-chain head is externally driven (or the block is the head, fed from `a_in`).
- `done` and completion are **per-block**; system-level "all-tiles-done" aggregation across a tiled/cascaded
  array is the fabric's responsibility, out of scope for a single block.

## 12. Support matrix, constraints & validation status

### Current standalone status (authoritative)

| K folding | N folding | Standalone mechanism | Current evidence |
|---|---|---|---|
| spatial | spatial | cascade + broadcast wrapper | `cmvu_array` target in the full regression |
| temporal | temporal | one lane accumulator, N-outer/K-inner replay | `cmvu_mode1_kfirst` target + perf Phase C (both in the full regression) |
| spatial | temporal | cascade feeding K-first lane accumulation | exercised by `cmvu_gemv`'s combined tiled shapes (full regression) |
| temporal | spatial | K-first lane accumulation + wrapper broadcast | exercised by `cmvu_gemv`'s combined tiled shapes (full regression) |

`cmvu_mode1` is the standalone block; `cmvu_array` supplies spatial composition.

**Regression status (2026-09-22).** `./run_tests_vcs.sh` passes all nine targets — `cmvu_mode1`,
`cmvu_mode1_kfirst`, `cmvu_array`, `cmvu_sram`, `cmvu_load_formats`,
`cmvu_sweep` (10 configs), `cmvu_perf`, `cmvu_edge`, `cmvu_gemv` — with exact arithmetic, exact
per-bank latency, II laws, and the K-first group cadence checked cycle-by-cycle. `cmvu_edge` also sweeps
the runtime effective width `out_w` (W = 1/7/8/15/16), checking the wrap-to-`W` + sign-extension
invariants and the W=8 ↔ V1 int8 result. A Verilator 5.x build is
also used as a second-opinion debug simulator (`run_tests.sh`); the `sweep` and `edge` targets are
additionally cross-checked green under Verilator.

**Synthesis status (2026-09-20).** The V1 Mode-1 RTL was taken through the full ASIC flow
(DC synthesis → Innovus place-and-route → PT signoff → genlibdb, FreePDK45, 4.0 ns target) to
signoff-clean GDS: setup WNS +0.315 ns, hold WNS +0.065 ns, no violations. See
`asic-work/cmvu-mode1/README.md` for the full result table and layout renders. (That physical result
predates the int16 `y_out`/`out_w` change; the flow has not been re-run for it.)

Caller obligations: assert `acc_first` on each group and `acc_last` only on its final valid K pass;
drive `out_w` with each valid frame (`RESULT_WIDTH−1` for full width);
`y_valid` marks every pass and delayed `done` marks each group; keep launch gap at least one cycle;
replay the K-slice activation set per group; avoid active tile read/write collisions; align cascade at
the reduce node; and size int32 temporal headroom for the workload.
No dynamic ready/backpressure or broader wrapper changes are in scope. The 8-tile DC baseline is preserved
in `asic-work/cmvu-mode1/snap-8tiles/` for reference.
