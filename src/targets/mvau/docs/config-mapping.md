# Config mapping — hls4ml / gemm-ip-gen → MVAU RTL

This file explains what each user knob actually does to the generated hardware:
how a gemm config maps onto the `mvu_vvu_axi` / `memstream` parameters documented
in [`mvau-rtl-parameters.md`](mvau-rtl-parameters.md). The authoritative code is
`../geometry.py` (`fold_plan` and helpers); this is the narrative.

## Config path

```
hls4ml gemm_config.json (per-GEMM item)
        │  flow._normalize_mvau_items          ← names/defaults, self-contained
        ▼
mvau item  ──►  cli builds package cfg  ──►  package.generate_mvau_pkg
                                                   │  geometry.fold_plan(m,k,n, **knobs)
                                                   ▼
                                            plan  ──►  rtl / golden / package emission
```

Each GEMM in a model is normalized and folded **independently** by its own knobs,
producing its own single blackbox IP; the combined header + manifest stitch them
into the dataflow. Nothing is shared or auto-balanced across layers.

---

## 1. Knob reference

`M,K,N` come from the GEMM shape (`gemm_m|m`, `gemm_k|k|n_in`, `gemm_n|n|n_out`).
Recall the mapping: **`MW = K`, `MH = N`, `numInputVectors = M`**.

| hls4ml / config field | mvau effect | notes |
|---|---|---|
| `input_precision`  | `ACTIVATION_WIDTH`, `SIGNED_ACTIVATIONS`, activation frac | `u?fixed<W,I>`; frac = `W−I` |
| `weight_precision` | `WEIGHT_WIDTH`, weight frac | selects 4-bit vs 8-bit core |
| `output_precision` | drain output type `ap_fixed<W,I,AP_RND,AP_SAT>` | frac = `W−I` |
| `part` | DSP generation → `COMPUTE_CORE`, `SEGMENTLEN` | see §3 |
| `clock_period_ns` | DSP58 `SEGMENTLEN` (cascade split) | ignored on DSP48 |
| `strategy` | fold target: `latency` ⇒ full unroll | see §4 |
| `reuse_factor` | fold target = RF (per-vector II) | the throughput/area dial |
| `target_cycles` | fold target = `⌈target_cycles / M⌉` | whole-frame budget |
| `n_tiles` | split **N** into `n_tiles` stitched MVU cores | see §6 |
| `weights_in_core` + `weight_file` | bake weights into the `memstream` ROM | see §7 |
| `bias` (+ `has_bias`) | per-column bias added in the accumulator domain | see §8 |
| `parallelization_factor` | **currently inert** — carried but not used by the fold | use `reuse_factor` instead |
| `accum_precision` | **currently inert** — `ACCU_WIDTH` is always auto-sized (§5) | — |
| `interface` | `stream` only; `array` (io_parallel) is rejected | MVU is a streaming core |

---

## 2. Precision parsing & the envelope

Precisions are `ac_fixed`/`ap_fixed`-style strings, `u?fixed<W,I>`:

- width `W` → `ACTIVATION_WIDTH` / `WEIGHT_WIDTH` / output width
- `ufixed` ⇒ unsigned (`SIGNED_ACTIVATIONS=0`); `fixed` ⇒ signed
- fractional bits `frac = W − I`; operands are fed as integer **codes** (`value·2^frac`)

**Envelope (`geometry.check_envelope`) — rejected outright, no fallback:**
- `WEIGHT_WIDTH > 8` → error.
- `ACTIVATION_WIDTH > 8` → error, unless `= 9` **and** signed **and** DSP58.

There is no bit-plane decomposition and no float path. "Larger GEMM" only ever
means larger `M/K/N`, never wider precision.

---

## 3. Part → DSP generation → core

`geometry.dsp_block_for_part` then `select_core`:

| Part prefix | DSP block | 8-bit core | 4-bit core |
|---|---|---|---|
| `xcv[cpmeh]…` (Versal) | DSP58 | `mvu_vvu_8sx9_dsp58` | `mvu_vvu_8sx9_dsp58` |
| `xc{vu,ku,zu,au,u}…` (US/US+) | DSP48E2 | `mvu_8sx8u_dsp48` | `mvu_4sx4u_dsp48e2` |
| `xc7…` (7-series) | DSP48E1 | `mvu_8sx8u_dsp48` | `mvu_4sx4u_dsp48e1` |
| otherwise | DSP48E2 (default) | — | — |

`NARROW_WEIGHTS` is derived from the actual weights when known; the 4-bit core on
DSP48**E1** *requires* `NARROW_WEIGHTS=1` (fold_plan errors otherwise).

---

## 4. The folding search (PE, SIMD)

`fold(k, n_tile, weight_width, target)` mirrors FINN's `SetFolding`:

1. **Fold target** (`target_from_knobs`):
   - `strategy=latency` / `reuse_factor=1` → `target=1` → full unroll.
   - `reuse_factor=R` → `target=R` (per-vector II ≈ R).
   - `target_cycles=T` → `target=⌈T/M⌉`.
2. **Ramp SIMD** over divisors of `K`, stopping when the cycle target is met **or**
   the weight-stream-width cap is hit: `WEIGHT_WIDTH · SIMD ≤ WWIDTH_MAX (36)`.
3. **Ramp PE** over divisors of `n_tile` (= N when untiled) until the cycle target
   is met.

Folding is **divisor-exact** (`K % SIMD == 0`, `N % PE == 0`) by construction, so
the RTL sanity checks always pass.

### SIMD is hard-capped (spatial K is bounded)

`WEIGHT_WIDTH · SIMD ≤ 36` caps the spatial reduction:

| weight width | max SIMD |
|---|---|
| 8-bit | **4** |
| 4-bit | 9 |
| 2-bit | 18 |

`SIMD` is then the largest **divisor of K** at or below that cap. This is why
K is largely a *temporal* dimension — see §5.

### PE is the throughput/area dial

With `target=1` (latency) PE ramps to the largest divisor of `n_tile` → full
unroll (`PE = n_tile`, `NF = 1`). Higher `reuse_factor` stops PE earlier → smaller
PE, larger `NF`, fewer DSPs, more cycles.

### Divisor snapping (a real sharp edge)

Because PE must divide N and SIMD must divide K, awkward dims collapse. `N` prime
⇒ `PE ∈ {1, N}` only: either full unroll (huge DSP) or fully serial. `plan
["reuse_factor_snapped"]` flags when the achieved II ≠ the requested RF.

---

## 5. Derived quantities

```
SF   = K / SIMD                      # synapse (K) folds — temporal K
NF   = n_tile / PE                   # neuron  (N) folds
WMEM = SF · NF                       # weight beats / vector = memstream DEPTH (per tile)
ACCU = ⌈log2 K⌉ + WEIGHT_WIDTH + ACTIVATION_WIDTH + 1     # auto-sized (guard=1)

weight_stream_width_ba = ceil(PE·SIMD·WEIGHT_WIDTH / 8)·8
input_stream_width_ba  = ceil(SIMD·ACTIVATION_WIDTH / 8)·8
output_stream_width_ba = ceil(PE·ACCU / 8)·8              # per tile (PB)

II   = SF                            # one output every SF cycles
lat  = SF + 5                        # DSP48;  SF + ⌈⌈SIMD/3⌉/SEGMENTLEN⌉ + 2 on DSP58
```

`ACCU_WIDTH` is always computed here (any `accum_precision` knob is ignored), so
the accumulator never overflows for a length-K dot product.

---

## 6. K vs N — spatial and temporal

The two GEMM dimensions are treated asymmetrically, because K is a **reduction**
and N is a **map**:

| | spatial (parallel) | temporal (sequential) | can it be fully spatial? |
|---|---|---|---|
| **K** (reduction) | `SIMD` lanes, capped at `36/WW` (≤4 for 8-bit) | `SF = K/SIMD` beats | **No** — SIMD is width-capped; the rest is time. No K-tiling exists. |
| **N** (map) | `PE` lanes, up to `N` (full unroll) | `NF = N/PE` beats | **Yes** — `PE=N` in one core, *or* fan out across `n_tiles` cores. |

So for any 8-bit K > 4 the bulk of K is temporal (`SF = K/4`). The only ways to add
spatial K are narrower weights (raises the SIMD cap) or DSP58; there is **no
K-tiling** (multiple cores summing partial-K products) today — it is the missing
dual of N-tiling.

### N: larger-PE vs N-tiling (same `P_total = n_tiles · PE_tile`)

Reaching a given output parallelism as one big-PE core vs. several tiles is
throughput- and total-DSP-equivalent, but structurally different:

| | larger PE (1 core) | N-tiling (`n_tiles` cores) |
|---|---|---|
| replay_buffer (activation) | 1 | `n_tiles` (activation broadcast + re-buffered) |
| weight ROM | 1 memstream | `n_tiles` memstreams (same bits, split) |
| DSP packing (DSP48) | best | preserved **only if `PE_tile ≥` packing factor** (2 for 8-bit, 4 for 4-bit) |
| control overhead | 1 FSM | `n_tiles` FSMs |
| timing / floorplan at scale | wide fanout, harder | small local cores + trivial 1-bit stitch — easier, SLR-friendly |
| composability | monolithic | independent N-blocks (attention heads, two-operand scaffold) |

**Default to larger PE for a plain dense layer** (what FINN does). **Reach for
N-tiling when you have a reason to partition N** — a semantic block structure, a
layer too big to place/time as one core, or the two-operand path — and keep
`PE_tile` at the packing granularity. There is currently **no guard** enforcing
that, so an odd/`=1` `PE_tile` on a DSP48 part silently loses packing.

---

## 7. N-tiling (`n_tiles`) — the RTL stitch

`n_tiles` splits N into equal column blocks (`n_tile = N/n_tiles`, must divide N),
each an independent `mvu_vvu_axi` folded over `(K, n_tile)`. The stitch lives
entirely in the shim (`../rtl.py`), so from Vitis's view it is still **one**
blackbox: one activation FIFO in, one result FIFO out.

- **activation broadcast** — one shared activation stream fans out to every tile;
  `a_read` advances only when all tiles accept (`&in_tready`).
- **output fan-in** — tile outputs concatenate into one wide beat, tile `ti` at
  `[ti·PB +: PB]`; `p_write = &out_tvalid`.
- **bit contract** — in beat `nf`, tile `ti` lane `pe` (bits `ti·PB + pe·ACCU`) is
  **global** output column `oc = ti·n_tile + nf·PE + pe`. The C twin, both drains,
  and the per-tile weight packer all key off this.
- **weights** — one `.dat` per column slice (`_dat_name`), each packed from
  `B[:, ti·n_tile : (ti+1)·n_tile]`.

The tiles are identical modules fed identical inputs with identical output
backpressure, so they run in provable lockstep — the broadcast/fan-in reductions
are exact. Setting `n_tiles=1` reduces exactly to the original single-core shim.

**Surfacing:** `n_tiles` is live at the gemm-ip-gen layer (Python API, or the
`--config` field). The hls4ml frontend does **not** emit it yet, so the full ATLAS
flow defaults every GEMM to `n_tiles=1` (single core) until the field is plumbed.

---

## 8. Weight-stationary path & the drain

### Weight baking (`weights_in_core` + `weight_file`)

When weights are constant they are baked into the `memstream` ROM instead of
streamed. `gemm_ip.weights.load_weight_dat` reads hls4ml's column-major `.dat` and
returns `B` as `[K][N]`; `../weightpack.py` packs it into the ROM init:

```
orientation :  W[mh][mw] = B[mw][mh]           # MH=N, MW=K, B is [K][N]
line (addr) :  wmem = nf·SF + sf               # nf outer, sf inner
within word :  W[nf·PE+pe][sf·SIMD+s]  at bit (pe·SIMD + s)·WEIGHT_WIDTH   # SIMD=LSB, PE outer
```

This is byte-exact-validated against FINN's own flip-based packer. Without a
`weight_file` the generator bakes a deterministic synthetic matrix (tests /
standalone).

### Requant drain (Keras order: matmul → bias → quantize)

The raw accumulator code carries `2^product_frac` where
`product_frac = input_frac + weight_frac`. The drain:

1. add per-column bias, scaled to the accumulator domain
   (`bias_code = round(bias · 2^product_frac)`);
2. reinterpret the code as fixed-point (`frac = product_frac`);
3. cast to `ap_fixed<outW, outI, AP_RND, AP_SAT>` — round-half-up + saturate — i.e.
   an effective right shift of `req_shift = product_frac − output_frac`.

`accum_precision` / `bias_precision` are not consumed here: the accumulator is
auto-sized and bias lives in the accumulator domain.

---

## 9. Worked examples

Single-core folds for `fixed<8,4>` weights & activations on `xcvu13p`
(`⌈PE/2⌉·SIMD` DSP), from `geometry.fold_plan`:

| shape (M,K,N) | knob | PE | SIMD | SF | NF | DSP | fill lat | note |
|---|---|---|---|---|---|---|---|---|
| (1, 8, 8)     | latency | 8 | 4 | 2 | 1 | 16 | 7 | SIMD capped at 4 |
| (1, 256, 512) | latency | 512 | 4 | 64 | 1 | 1024 | 69 | full unroll (DSP heavy) |
| (1, 256, 512) | `reuse_factor=1024` | 64 | 4 | 64 | 8 | 128 | 69 | folded → 8× fewer DSP |
| (1, 1024, 1024) | latency | 1024 | 4 | 256 | 1 | 2048 | 261 | |
| (1, 254, 257) | latency | 257 | 2 | 127 | 1 | 258 | 132 | N,K prime-ish → stuck at full unroll; SIMD=2 |
| (1, 1024, 1024) | `n_tiles=4`, latency | 256 (×4) | 4 | 256 | 1 | 512 ×4 = 2048 | 261 | same compute as full unroll, 4 placeable cores |

Note SIMD is pinned to 4 wherever `4 | K` (the 8-bit cap), and `n_tiles=4` gives
the same total DSP as the single big-PE core — the difference is floorplan/timing
and modularity, not throughput.

---

## 10. Cheat-sheet

- **Make it fit / go slower:** raise `reuse_factor` (folds PE down, fewer DSP).
- **Go fast (small layer):** `strategy=latency` (full unroll, `PE=N`).
- **Big layer that won't place/time as one core:** set `n_tiles` (direct invoke)
  to spread the same compute across cores — keep `PE_tile` even (8-bit) / ×4 (4-bit).
- **Constant weights:** `weights_in_core=true` + `weight_file` → baked ROM.
- **Awkward dims:** expect divisor snapping; check `plan["reuse_factor_snapped"]`.
- **Don't reach for:** `parallelization_factor` / `accum_precision` (inert today),
  `interface=array` (rejected), >8b precision (rejected).
