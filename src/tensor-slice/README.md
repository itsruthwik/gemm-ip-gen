# Tensor Slice

`tensor_slice_int8` is the hardblock-specific RTL core used by the GEMM wrapper flow.

## Contents

- `tensor_slice_int8.v`: standalone 8x8 int8 systolic slice RTL
- `generate_verilog_grid.py`: generates tiled RTL wrappers around this slice
- `generate_verilog_tb.py`: generates self-checking wrapper-level Verilog testbenches
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

## Deterministic Latency

For matmul mode, completion is determined from:

`(a_loc + b_loc) * 8 + 7 + K + P - 1 + 8`

where:

- `K = final_mat_mul_size`
- `P = 3`

Validated examples:

- `8x8`, `a_loc=0`, `b_loc=0`: first valid at `launch + 17`, done at `launch + 25`
- `5x5`, `a_loc=0`, `b_loc=0`: first valid at `launch + 14`, done at `launch + 22`
- `8x8`, `a_loc=1`, `b_loc=1`: first valid at `launch + 33`, done at `launch + 41`

## Verification

Standalone regressions live in `tb/`.

Useful coverage:

- `tb_tensor_slice_regression.v`: baseline, back-to-back launch, chained-input timing
- `tb_tensor_slice_mask_regression.v`: non-8x8 and mask behavior
- size-specific smoke benches such as `tb_5x5.v`, `tb_12x10.v`, `tb_16x16.v`

Wrapper-level verification is run from the common layer with:

```bash
python3 verify_generated_wrappers.py
```
