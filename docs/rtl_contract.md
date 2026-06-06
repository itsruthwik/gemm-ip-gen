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

Vitis AXI-stream wrapper:

```verilog
module {name}_wrapper(
    ap_clk, ap_rst, ap_ce,
    a_tdata, a_tvalid, a_tready,
    b_tdata, b_tvalid, b_tready,
    bias_tdata, bias_tvalid, bias_tready,
    c_tdata, c_tvalid, c_tready
);
```

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

## Synth Protocol

1. Bias arrives first.
2. The wrapper pulses `preload` for one cycle.
3. For each K chunk:
   - pulse `start_mat_mul` on the first A/B beat of that chunk
   - assert `pe_reset` only on chunk 0
   - drive `validity_mask_a_cols_b_rows` and `final_mat_mul_size` for that chunk
   - feed `max(M,N)` row/column beats
4. Intermediate outputs from non-final K chunks are ignored.
5. After the final K chunk, the wrapper aligns slice outputs, concatenates all
   column tiles, and emits full output rows.

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

It must not contain a second GEMM datapath. The Vitis AXI wrapper may use a
one-row skid register so `c_tdata` remains stable while `c_tready` is low.
