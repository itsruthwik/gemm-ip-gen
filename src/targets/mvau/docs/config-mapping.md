# Config mapping — hls4ml / gemm-ip-gen → MVAU RTL

This file explains what each user knob actually does to the generated hardware:
how a gemm config maps onto the `mvu_vvu_axi` / `memstream` parameters documented
in [`mvau-rtl-parameters.md`](mvau-rtl-parameters.md). The authoritative code is
`../geometry.py` (`resolve_fold`, `fold_plan`); this is the narrative. For the
compute core's internal structure (why DSP58 packs 3 K-lanes per DSP, the PCOUT
cascade, `SEGMENTLEN`), see `finn_space/MVU_space/03_compute_cores.md` at the repo
root rather than this file.

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

## Part and core

This target supports **Versal parts only** (`xcv[cpmeh]...`); any other part
raises, naming `--part`. There is a single compute core,
`mvu_vvu_8sx9_dsp58` (DSP58), so there is no part-to-core selection to document.

---

## 1. Knob reference

`M,K,N` come from the GEMM shape (`gemm_m|m`, `gemm_k|k|n_in`, `gemm_n|n|n_out`).
Recall the mapping: **`MW = K`, `MH = N`, `numInputVectors = M`**.

| hls4ml / config field | mvau effect | notes |
|---|---|---|
| `input_precision`  | `ACTIVATION_WIDTH`, `SIGNED_ACTIVATIONS`, activation frac | `u?fixed<W,I>`; frac = `W−I` |
| `weight_precision` | `WEIGHT_WIDTH`, weight frac | |
| `output_precision` | drain output type `ap_fixed<W,I,AP_RND,AP_SAT>` | frac = `W−I` |
| `part` | must be Versal | see above |
| `clock_period_ns` | DSP58 `SEGMENTLEN` (cascade split) | |
| `reuse_factor` | how many times each MAC is used per input vector; picks `(PE, SIMD)` | see §4 |
| `fold_axis` | which dimension `reuse_factor` folds: `n` (default), `k`, or `kn` | see §4 |
| `n_tiles` | split **N** into `n_tiles` stitched MVU cores | see §6 |
| `weights_in_core` + `weight_file` | bake weights into the `memstream` ROM | see §7 |
| `bias` (+ `has_bias`) | per-column bias added in the accumulator domain | see §8 |
| `accum_precision` | **currently inert** — `ACCU_WIDTH` is always auto-sized (§5) | — |
| `interface` | `stream` only; `array` (io_parallel) is rejected | MVU is a streaming core |

`strategy`, `pe`/`simd` (manual bypass), and `k_tiles` also exist; `pe`/`simd` skip
`resolve_fold` entirely (the caller states the fold directly), and `k_tiles` is a
physical/BRAM-shape choice, not something the fold derives.

---

## 2. Precision parsing & the envelope

Precisions are `ac_fixed`/`ap_fixed`-style strings, `u?fixed<W,I>`:

- width `W` → `ACTIVATION_WIDTH` / `WEIGHT_WIDTH` / output width
- `ufixed` ⇒ unsigned (`SIGNED_ACTIVATIONS=0`); `fixed` ⇒ signed
- fractional bits `frac = W − I`; operands are fed as integer **codes** (`value·2^frac`)

**Envelope (`geometry.check_envelope`) — rejected outright, no fallback:**
- `WEIGHT_WIDTH > 8` → error.
- `ACTIVATION_WIDTH > 8` → error, unless `= 9` **and** signed.

There is no bit-plane decomposition and no float path. "Larger GEMM" only ever
means larger `M/K/N`, never wider precision.

---

## 3. ReuseFactor and the fold

`ReuseFactor` (RF) is *how many times each MAC (each `(PE, SIMD)` lane) is used
per input vector*: `RF = K*N / (PE*SIMD)`. It is a resource knob, not a cycle
budget — `resolve_fold` (`../geometry.py`) turns it into `(PE, SIMD)` directly by
picking which dimension pads to honor the request, rather than searching for a
fold that hits a cycle target.

`fold_axis` selects the dimension:

- **`n` (default).** `SIMD = K` — the whole K reduction runs inside one DSP58
  cascade per PE. `N` pads up to the next multiple of RF (`n_pad = ceil(N/RF)*RF`,
  zero-weight output columns, dropped downstream); `PE = n_pad / RF`. RF above `N`
  cannot be honored (`PE` would be below 1) and legalizes to `RF = N` (`PE = 1`),
  with a warning naming the layer.
- **`k`.** `PE = N`; `K` pads up to the next multiple of RF (`k_pad =
  ceil(K/RF)*RF`, zero rows); `SIMD = k_pad / RF`. `SIMD` is never folded below 3
  (the DSP58 core packs 3 K-lanes per DSP) unless `K` itself is below 3, in which
  case `SIMD = K` and `K` never folds. RF beyond `K/3` legalizes to the `SIMD = 3`
  floor, with a warning.
- **`kn`.** Both rules applied independently and simultaneously: `SIMD = K/RF`
  (K-side padding and floor, as above) and `PE = N/RF` (N-side padding and bound,
  as above). Each side is checked and, if needed, legalized on its own. The
  multiplier count under `kn` is `K*N/RF**2` — each MAC is reused `RF**2` times
  per input vector, not `RF` — so the manifest reports the effective reuse
  (`K_pad*N_pad/(PE*SIMD)`) alongside the requested `reuse_factor` so the two are
  never confused.

Padding is the default legalization for an in-range request (a non-dividing RF
pads rather than snapping to a nearby divisor); a warning is only printed at the
bounds above (RF exceeding what the axis can express), never for an in-range,
non-dividing RF.

`RF = 1` is always legal (`SIMD = K`, `PE = N`, no folding) and needs no special
handling; its DSP cost is the caller's responsibility, as with hls4ml.

### Why fold-N is the default

On this core both axes cost the same DSPs for a given RF: fold-N spends
`(N/RF)*ceil(K/3)`, fold-K spends `N*ceil(K/(3*RF))` — both `~= K*N/(3*RF)` when
the division is exact. Fold-N is still the default for two reasons: the whole K
reduction stays inside one DSP58 PCOUT cascade per PE (no fabric adders, no
extra accumulation passes), and the `ceil` rounding waste of a partially filled
cascade is paid once per PE — of which fold-N has RF times fewer than fold-K. See
`finn_space/MVU_space/03_compute_cores.md` for the cascade structure itself.

---

## 4. Manual `(pe, simd)` bypass

Passing both `pe` and `simd` skips `resolve_fold`: the values are used verbatim
(K still pads to a multiple of `simd * k_tiles` so `SF/k_tiles` is integral, and
`pe` must divide `n_tile`). This is a debug path, not part of the `ReuseFactor` /
`fold_axis` contract above.

---

## 5. Derived quantities

```
SF   = K_pad / SIMD                  # K-folds (per input vector)
NF   = n_pad / PE                    # N-folds
WMEM = SF · NF                       # weight beats / vector = memstream DEPTH (per tile)
ACCU = ⌈log2 K_pad⌉ + WEIGHT_WIDTH + ACTIVATION_WIDTH + 1     # auto-sized (guard=1)

weight_stream_width_ba = ceil(PE·SIMD·WEIGHT_WIDTH / 8)·8
input_stream_width_ba  = ceil(SIMD·ACTIVATION_WIDTH / 8)·8
output_stream_width_ba = ceil(PE·ACCU / 8)·8              # per tile

DSP estimate = PE · ceil(SIMD/3)     # DSP58 packs 3 K-lanes per DSP
```

`ACCU_WIDTH` is always computed here (any `accum_precision` knob is ignored), so
the accumulator never overflows for a length-`K_pad` dot product. Fill latency and
output cadence are deterministic RTL properties, not resource knobs; see
`../geometry.py` (`latency_cycles`, `output_ii`) and
`finn_space/MVU_space/03_compute_cores.md` for their derivation.

---

## 6. N-tiling (`n_tiles`) — the RTL stitch

`n_tiles` splits N into equal column blocks (`n_tile = N/n_tiles`, must divide N),
each an independent `mvu_vvu_axi` folded over `(K, n_tile)` by the same
`reuse_factor` / `fold_axis`. The stitch lives entirely in the shim (`../rtl.py`),
so from Vitis's view it is still **one** blackbox: one activation FIFO in, one
result FIFO out.

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

## 7. Weight-stationary path & the drain

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
standalone). Under `fold_axis="n"` the padded N columns are zero-weight; the drain
discards the padded output columns before they reach the caller.

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

## 8. Worked example

`fixed<8,4>` weights & activations on a Versal part, `(M,K,N) = (1, 27, 8)`:

| `fold_axis` | `reuse_factor` | PE | SIMD | K_pad | N_pad | DSP (`PE·ceil(SIMD/3)`) | note |
|---|---|---|---|---|---|---|---|
| `n` | 1 | 8 | 27 | 27 | 8 | 72 | full unroll, no padding |
| `n` | 3 | 3 | 27 | 27 | 9 | 27 | `N=8` doesn't divide by 3; pads to 9 |
| `n` | 16 | 1 | 27 | 27 | 16 | 9 | `RF > N`; legalizes to `RF=8` (`PE=1`), warning |
| `k` | 10 | 8 | 3 | 30 | 8 | 24 | `K/3 = 9`; `RF=10` legalizes to `SIMD=3`, warning |

---

## 9. Cheat-sheet

- **Fewer DSPs / more cycles:** raise `reuse_factor`.
- **Full throughput:** `reuse_factor=1` (`SIMD=K`, `PE=N`, no folding).
- **Big layer that won't place/time as one core:** set `n_tiles` to spread the
  same compute across cores.
- **Fold K instead of N:** `fold_axis="k"` (e.g. when `N` is small/prime and `K`
  has useful divisors).
- **Both axes at once:** `fold_axis="kn"` — remember the multiplier count is
  `RF**2`, not `RF`; check `effective_reuse` in the manifest.
- **Constant weights:** `weights_in_core=true` + `weight_file` → baked ROM.
- **Non-Versal part:** rejected outright — pass a Versal `--part`.
- **Don't reach for:** `accum_precision` (inert), `interface=array` (rejected),
  >8b precision (rejected).
