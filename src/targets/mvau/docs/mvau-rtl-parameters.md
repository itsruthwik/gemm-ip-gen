# FINN MVAU RTL — parameter reference

The compute is FINN's `mvu_vvu_axi` (vendored in `../rtl_static/`), an AXI-Stream
matrix–vector (MVU) / vector–vector (VVU) unit that delegates the MACs to one of
three DSP-packing cores. In the weight-stationary path a FINN `memstream` sits in
front of it as the weight ROM. This file documents both, exactly as the vendored
RTL declares them — it is the contract the generator (`../rtl.py`) fills.

We use the MVU mode only (`IS_MVU=1`): a GEMM `C[M,N] = A[M,K] · B[K,N]` maps to
one MVU as **`W := Bᵀ`, so `MH = N` and `MW = K`**; rows of `A` stream as
activations (`numInputVectors = M`), rows of `C` are collected.

---

## 1. `mvu_vvu_axi` parameters

### Core identity

| Parameter | Type / default | Meaning |
|---|---|---|
| `IS_MVU` | `bit` | 1 = matrix–vector (shared activation vector across output channels). We always set 1. 0 = vector–vector (element-wise), DSP58/LUT only. |
| `COMPUTE_CORE` | string | Which DSP-packing core to instantiate (see §4). |

### Shape (per instance)

| Parameter | Meaning | Constraint |
|---|---|---|
| `MW` | matrix width = **K** (the reduction/inner dim) | `MW % SIMD == 0` (asserted) |
| `MH` | matrix height = **N** (output channels); for an N-tile it is `n_tile` | `MH % PE == 0` (asserted) |
| `PE`  | processing elements — **spatial output-channel parallelism** | divides `MH` |
| `SIMD`| SIMD lanes — **spatial reduction (K) parallelism** | divides `MW` |

Two sequential fold counts follow directly (localparams in the RTL):

```
SF = MW / SIMD     // synapse folds  — K streamed in SF beats of SIMD
NF = MH / PE       // neuron folds   — N produced in NF beats of PE
```

One input vector costs `SF · NF` cycles at II=1. **Full unfold along MH (`PE=MH`,
so `NF=1`) instantiates no replay buffer** (FINN folding hint, line 43 of the RTL).

### Operand precision

| Parameter | Type / default | Meaning |
|---|---|---|
| `ACTIVATION_WIDTH` | `int` | activation bit width. `≤ 8`, or `= 9` only if signed **and** DSP58 (asserted). |
| `WEIGHT_WIDTH` | `int` | weight bit width. `≤ 8` (asserted). |
| `ACCU_WIDTH` | `int` | accumulator width. Must be sized by the caller to avoid overflow — the cores assert on overflow in sim. |
| `NARROW_WEIGHTS` | `bit`, `0` | 1 ⇒ weights never use the most-negative code (`-2^(w-1)`). Mandatory `1` for the 4-bit core on DSP48**E1**. |
| `SIGNED_ACTIVATIONS` | `bit`, `0` | 1 ⇒ activations are signed two's-complement. |

### Implementation / timing

| Parameter | Type / default | Meaning |
|---|---|---|
| `SEGMENTLEN` | `int`, `0` | DSP58 cascade segment length. `0` warns and defaults to chain length `⌈(SIMD+2)/3⌉`; must be `≤ ⌈(SIMD+2)/3⌉` (asserted). Ignored by DSP48 cores. |
| `PUMPED_COMPUTE` | `bit`, `0` | 1 ⇒ double-pump the DSPs on `ap_clk2x` (2× MAC throughput per DSP). We set 0 and tie `ap_clk2x` low. |
| `FORCE_BEHAVIORAL` | `bit`, `0` | 1 ⇒ use the behavioral (Verilator/unisim-free) MAC model instead of DSP primitives. 1 for fast behavioral cosim, 0 for impl-ready DSP48/DSP58. |
| `M_REG_LUT` | `bit`, `1` | output pipeline register placed in LUTs. |

### Derived stream widths (localparams — do **not** pass these)

```
WEIGHT_STREAM_WIDTH    = PE * SIMD * WEIGHT_WIDTH
INPUT_STREAM_WIDTH     = (IS_MVU ? 1 : PE) * SIMD * ACTIVATION_WIDTH   // MVU: SIMD*AW
OUTPUT_STREAM_WIDTH    = PE * ACCU_WIDTH
*_STREAM_WIDTH_BA      = ceil(*_STREAM_WIDTH / 8) * 8                  // byte-aligned AXIS
```

`gemm-ip-gen` recomputes these identically in `geometry.stream_widths` /
`geometry._ba`; the shim and C twin must agree with them bit-for-bit.

### AXIS ports

| Port group | Signals | Width |
|---|---|---|
| weights (in) | `s_axis_weights_{tdata,tvalid,tready}` | `WEIGHT_STREAM_WIDTH_BA` |
| input / activations (in) | `s_axis_input_{tdata,tvalid,tready}` | `INPUT_STREAM_WIDTH_BA` |
| output / results (out) | `m_axis_output_{tdata,tvalid,tready}` | `OUTPUT_STREAM_WIDTH_BA` |
| global | `ap_clk`, `ap_clk2x`, `ap_rst_n` (active-**low**) | — |

### Sanity checks the core asserts (`initial` block)

- `MW % SIMD == 0` and `MH % PE == 0` — folding must be divisor-exact.
- `WEIGHT_WIDTH ≤ 8`; `ACTIVATION_WIDTH ≤ 8` (or `= 9` signed on DSP58).
- DSP58: `SEGMENTLEN ≤ ⌈(SIMD+2)/3⌉`.
- VVU (`IS_MVU=0`): only on `mvu_vvu_8sx9_dsp58` or `mvu_vvu_lut`.

---

## 2. Beat protocol (what the generator's C twin/TB must reproduce)

Per input vector, in `(nf outer, sf inner)` order for weights:

- **activations** — `SF` beats; beat `sf` carries `x[sf·SIMD + s]` at bit `s·AW`.
  Internally the `replay_buffer` (`LEN=SF`, `REP=NF`) re-streams them `NF` times.
- **weights** — `NF·SF` beats; beat `(nf, sf)` holds `W[nf·PE+pe][sf·SIMD+s]`
  packed `[pe][simd]` LSB-first at bit `(pe·SIMD + s)·WEIGHT_WIDTH`.
- **output** — `NF` beats; beat `nf` holds `PE` accumulators for output rows
  `nf·PE .. nf·PE+PE-1`, each at bit `pe·ACCU_WIDTH`.

This is encoded in `../golden.py` (C twin) and the weight-stationary packing in
`../weightpack.py`.

---

## 3. Deterministic timing

The RTL is a fixed pipeline, so latency and II are exact (calibrated against XSIM,
see `geometry.latency_cycles` / `geometry.output_ii`):

```
II (steady state)  = SF                       // one output every SF cycles
fill latency       = SF + 5                    // DSP48 (8sx8u / 4sx4u): 5 fixed stages
                   = SF + ⌈CHAINLEN/SEGMENTLEN⌉ + 2   // DSP58, CHAINLEN = ⌈SIMD/3⌉
```

---

## 4. Compute cores (`COMPUTE_CORE`)

| Core | Device / operands | DSP packing (MACs/DSP) | DSP estimate (per instance) |
|---|---|---|---|
| `mvu_4sx4u_dsp48e1` / `…e2` | DSP48, ≤4b weight & act | 4 (packed along **PE**) | `⌈PE/4⌉ · SIMD` |
| `mvu_8sx8u_dsp48` | DSP48, (4,8]b | 2 (packed along **PE**) | `⌈PE/2⌉ · SIMD` |
| `mvu_vvu_8sx9_dsp58` | DSP58 (Versal), [4,9]b | 3 (packed along **SIMD**) | `PE · ⌈SIMD/3⌉` |

Which one is selected from the FPGA part + operand widths is `geometry.select_core`
(see the mapping doc). **The packing dimension matters for tiling:** DSP48 packs
along PE, so splitting PE (N-tiling) can break packing unless each tile keeps
`PE ≥ packing factor`; DSP58 packs along SIMD, so it is tiling-neutral.

---

## 5. `memstream` (weight-stationary weight ROM)

In the weight-stationary path the shim instantiates one `memstream` per MVU
(vendored `../rtl_static/memstream.sv`) as the baked weight ROM feeding
`s_axis_weights` — there is no external weight port.

| Parameter | Meaning |
|---|---|
| `DEPTH` | number of words = `WMEM = NF · SF` (weight beats per input vector). |
| `WIDTH` | word width = the byte-aligned weight-stream width (`WEIGHT_STREAM_WIDTH_BA`); packed weights sit in the low bits, high bits zero. |
| `INIT_FILE` | `$readmemh` path to the packed `.dat` (see `../weightpack.py`). **Absolute** — a relative path is unresolvable in Vitis cosim's XSIM working dir. |
| `RAM_STYLE` | `"auto"` (Vitis picks BRAM/LUTRAM by size). |

The ROM streams its `DEPTH` words cyclically (`ovld`/`ordy` handshake, data-independent
timing) so every input vector re-reads the same weights. The config/readback port
(`config_*`) is tied off. `INIT_FILE=""` would leave it uninitialized — the
generator always provides a path.

---

## 6. Activation buffering (`replay_buffer`)

The activation-side counterpart to the weight ROM. FINN's `replay_buffer`
(`../rtl_static/replay_buffer.sv`) is instantiated **inside `mvu_vvu_axi`** as
`activation_replay` — one per MVU core; the generator's shim and HLS wrapper hold
no activation vector buffer of their own.

```
replay_buffer #(.LEN(SF), .REP(IS_MVU ? NF : 1), .W(SIMD·ACTIVATION_WIDTH))
```

Its job is to **reuse one input vector across the `NF` neuron-folds**: each of the
`NF = N/PE` output passes needs the whole K-vector again, so the vector is stored
once and replayed `NF` times. (Weights are *not* replayed — all `NF·SF` weight
beats are distinct.)

| Aspect | Value | Set by |
|---|---|---|
| `LEN` | `SF = K/SIMD` items (the full K-vector, one word per beat) | **temporal K** — buffer *depth* |
| `REP` | `NF = N/PE` replays | **N-fold** — presence + replay count |
| `W` (word) | `SIMD · ACTIVATION_WIDTH` bits | one activation beat |
| storage | `Mem[2^⌈log₂ SF⌉]` words ≈ one input vector (`≈ K·ACT_WIDTH` bits) | — |
| RAM style | tool-inferred (no `RAM_STYLE`): LUTRAM/SRL when shallow, BRAM when `SF` is deep | — |

**It vanishes at full N-unroll.** When `PE = MH` (`NF = 1` ⇒ `REP = 1`) the module
takes its passthrough branch — no `Mem` array is instantiated and activations
stream straight through, consumed once. So the vector buffer exists **only when N
is folded** (`PE < N`); its depth tracks the temporal-K span `SF`, its
existence/replay-count the N-fold `NF`.

**N-tiling cost.** Each tile is its own `mvu_vvu_axi`, hence its own
`replay_buffer`, so the activation vector is buffered `n_tiles` times (one copy per
tile), all fed by the shim's activation broadcast. This is the concrete reason a
single large-PE core is more activation-efficient than `n_tiles` tiles (see the
larger-PE-vs-N-tiling table in `config-mapping.md` §6).

> The only other activation storage is the HLS wrapper's dataflow FIFOs (`a_s`,
> depth 4) — streaming plumbing between `feed_a` and the core, not a vector buffer.
