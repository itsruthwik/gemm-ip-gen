# Tensor Slice

`tensor_slice_int8` is the hardblock-specific RTL core used by the GEMM wrapper flow.

## Contents

- `tensor_slice_int8.v`: standalone 8×8 int8 systolic slice RTL
- `generate_catapult_rtl.py`: generates Catapult RTL wrappers (behavioral sim, synth, combined core)
- `generate_vitis_rtl.py`: generates Vitis RTL wrappers (behavioral sim, synth)
- `generate_verilog_tb.py`: generates self-checking Verilog testbenches for both protocols
- `_generate_rtl_common.py`: shared helpers (tail mask, cycle counter)
- `tb/`: standalone slice testbenches and regressions

## Interface Summary

The slice is intended for int8 tensor matmul mode:

- `slice_dtype = 2'b00`
- `slice_mode = 1'b0`
- `op = 3'b000`

Control behavior:

- `start_mat_mul` is a 1-cycle launch pulse
- the slice then runs autonomously
- `c_data_available` goes high when `c_data_out[63:0]` holds a valid result row
- `done_mat_mul` pulses when the final row has been emitted

Data behavior:

- `a_data` / `b_data` are the primary top/left boundary inputs
- `a_data_in` / `b_data_in` are chain inputs from neighboring slices
- `a_data_out` / `b_data_out` chain onward to neighboring slices
- `c_data_out[63:0]` contains saturated int8 output values, one row per cycle

Masking support:

- `validity_mask_a_rows`: spatial row mask
- `validity_mask_b_cols`: spatial column mask
- `validity_mask_a_cols_b_rows`: temporal inner-dimension mask

## Grid-Level Latency

The behavioral grid model uses a parameterized formula:

```
beh = k_chunks × max(M,N) + max(0, K+N − k_chunks × max(M,N)) + M  (+1 sync)
wrap = beh + 3  (Catapult wrapper overhead)
II   = k_chunks × max(M,N) + 1  (back-to-back, shadow FIFO)
```

Validated results (RTL sim):

| Shape    | beh | seq II | b2b II |
|----------|-----|--------|--------|
| 8×8×8   |  25 |     29 |      9 |
| 16×8×8  |  33 |     37 |     17 |
| 16×8×16 |  41 |     45 |     17 |
| 8×16×8  |  33 |     37 |     17 |
| 16×16×16|  49 |     53 |     33 |
| 9×17×10 |  40 |     44 |     31 |

## Verification

Standalone slice regressions live in `tb/`.

Wrapper-level verification:

```bash
# All RTL simulation tests (46 tests)
pytest tests/test_rtl_sim.py -v

# Combined core verification (Catapult package test)
pytest tests/test_catapult.py -v
```

Useful slice coverage:

- `tb_tensor_slice_regression.v`: baseline, back-to-back launch, chained-input timing
- `tb_tensor_slice_mask_regression.v`: non-8x8 and mask behavior
- size-specific smoke benches such as `tb_5x5.v`, `tb_12x10.v`, `tb_16x16.v`
