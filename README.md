# gemm-ip-gen

Generates GEMM IP blackbox packages for Catapult HLS and Vitis HLS backends,
targeting the tensor-slice INT8 GEMM hardblock. Each package contains the
RTL core, an hls4ml-native C++ wrapper, and a backend-specific synthesis
script. A combined dispatch header is emitted when multiple packages are
generated together.

## Quick start

```bash
# Install (editable)
pip install -e .

# Catapult (default)
python -m gemm_ip --m 8 --k 8 --n 8 --name gemm_8x8x8 --output_dir ./output

# Vitis
python -m gemm_ip --backend vitis --m 8 --k 8 --n 8 --name gemm_8x8x8 --output_dir ./output_vitis

# From config file
python -m gemm_ip --backend vitis --config gemm_config.json --output_dir ./output_vitis
```

### Using backward-compat shims

If you have existing scripts that call the old entry points, they still work:

```bash
# Same as --backend catapult
python generate_catapult_pkg.py --m 8 --k 8 --n 8 --name gemm_8x8x8

# Same as --backend vitis
python generate_vitis_pkg.py --m 8 --k 8 --n 8 --name gemm_8x8x8

# Legacy config-driven flow
python generate_gemm_ip.py gemm_config.json ./output
```

The config file is written automatically by hls4ml when `GemmIP: True` is
set for a layer. Each entry carries `gemm_m`, `gemm_k`, `gemm_n`,
`interface`, and `protocol` metadata.

## Backend comparison

| Feature | Catapult | Vitis |
|---|---|---|
| Stream type | `ac_channel<ac_int<W>>` | `hls::stream<ap_uint<W>>` |
| Blackbox mechanism | `ac_blackbox` + ccore | JSON descriptor + RTL files |
| C model | `{name}_gemm_ip.h` (templated) | `{name}_wrapper.cpp` (C API) |
| RTL wrapper | `{name}_core.v` | `{name}_wrapper.v` + `{name}_core.v` |
| Build script | `run_catapult.tcl` | `run_vitis.tcl` |
| Interface support | `stream`, `array` | `stream`, `array` |
| Bias handling | In the wrapper capture path, post-rescale, full precision (the core is fed zero bias) | Inside blackbox (preload phase) |
| Core output lane | int16 | `output_precision`-driven |
| Result type | `output_precision` (rescale + bias + round/saturate in wrapper) | `output_precision` (saturated in wrapper) |

## Generated outputs

### Catapult

For a package named `<name>`:

| File | Description |
|---|---|
| `<name>/<name>_core.v` | Tensor-slice RTL core |
| `<name>/<name>_gemm_ip.h` | hls4ml-native C++ wrapper and blackbox binding |
| `<name>/<name>_inst.cpp` | Standalone Catapult synthesis top |
| `<name>/<name>_tb.cpp` | Standalone C-simulation testbench |
| `<name>/run_catapult.tcl` | Catapult synthesis script |
| `gemm_ip_combined.h` | Shape/id dispatch header for multi-package builds |
| `blackbox_files.tcl` | Catapult file-add helper |

### Vitis

For a package named `<name>`:

| File | Description |
|---|---|
| `<name>/<name>_wrapper.cpp` | C API model (`hls::stream<ap_uint<W>>` interface) |
| `<name>/<name>_wrapper.v` | Vitis RTL wrapper (ap_ctrl + AXI-stream FIFO ports) |
| `<name>/<name>_wrapper.json` | Vitis blackbox descriptor |
| `<name>/<name>_gemm_ip.h` | hls4ml-native typed-beat adapter (included by combined header) |
| `<name>/<name>_core.v` | Tensor-slice grid core |
| `<name>/<name>_design.cpp` | Standalone Vitis HLS design top |
| `<name>/<name>_tb.cpp` | Standalone testbench with golden reference |
| `run_vitis.tcl` | Vitis HLS project script |
| `tensor_slice_int8.v` | Tensor-slice RTL core |
| `gemm_ip_combined.h` | Shape/id dispatch header for hls4ml integration |
| `integration_manifest.json` | Package metadata for multi-layer models |

## Vitis interfaces

The generated Vitis package always uses the same packed `ap_uint` stream blackbox
internally, and emits typed hls4ml adapters for both integration styles:

- `nnet::gemm_ip_stream(...)` for stream-fed layers
- `nnet::gemm_ip_array(...)` for array-fed layers

The selected `interface` value is preserved in `integration_manifest.json` and
used by `gemm_ip_combined.h` to dispatch the matching adapter for each generated
shape.

### Stream blackbox contract

The Vitis blackbox uses three input streams and one output stream, all
`hls::stream<ap_uint<W>>`:

```
void <name>_wrapper(
    hls::stream<ap_uint<A_WIDTH>>& a_stream,      // k_chunks × max(M,N) row beats
    hls::stream<ap_uint<B_WIDTH>>& b_stream,      // k_chunks × max(M,N) column beats
    hls::stream<ap_uint<B_WIDTH>>& bias_stream,   // 1 beat
    hls::stream<ap_uint<C_WIDTH>>& c_stream       // M beats
);
```

Protocol sequence (row/col streaming):
1. Read bias from `bias_stream` (1 beat)
2. For each K-chunk (0..k_chunks-1):
   - Feed `max(M,N)` A row beats and `max(M,N)` B column beats
3. Drain M result beats (saturated int32→int8)

## Width formulas (Vitis)

| Packet | Width | Rationale |
|---|---|---|
| A (`a_tdata`) | `ceil(M/8) * 64` | M int8 rows packed 8 per 64-bit tile |
| B (`b_tdata`) | `ceil(N/8) * 64` | N int8 cols packed 8 per 64-bit tile |
| Bias (`bias_tdata`) | `ceil(N/8) * 64` | Same packing as B |
| C (`c_tdata`) | `ceil(N/8) * 64` | N int8 results, saturated from int32 |

## Testing

```bash
# Run all tests (100 tests)
pytest tests/ -v

# RTL simulation only (20 Catapult + 20 Vitis + 6 structural)
pytest tests/test_rtl_sim.py -v

# Catapult package tests (9 tests)
pytest tests/test_catapult.py -v

# Vitis package tests (45 tests)
pytest tests/test_vitis.py -v
```

RTL simulation tests cover all 10 DEFAULT_CASES configs (K≤8 and K>8):
sequential 10-vector, back-to-back 2-vector pipelined (shadow FIFO, II=9
for 8×8×8), and synth structural smoke with `-DSYNTHESIS` iverilog flag.
All tests use the combined core (`ifndef SYNTHESIS` behavioral model, `else`
synth wrapper). Catapult package tests validate C++ header emission,
`CCS_MAIN` SCVerify TB, TCL instantiation names, and bias wiring.

## Catapult SCVerify

For Catapult 2026.1+ with QuestaSIM:

```bash
python -m gemm_ip --m 8 --k 8 --n 8 --name gemm_8x8x8 --output_dir ./pkg
cd pkg/gemm_8x8x8
catapult -shell -f run_catapult.tcl   # synthesis (38 cycles, nangate-45nm)
cd gemm_8x8x8_proj/gemm_8x8x8_sol.v1
echo 'QuestaSIM_Path := /path/to/questasim' > scverify/ccs_env.mk
make -f scverify/Verify_concat_sim_rtl_v_msim.mk sim
```

The combined core uses `ifndef SYNTHESIS` (behavioral grid for simulation)
and `else` (structural synth wrapper for synthesis). SCVerify compares the
C++ golden against the RTL behavioral model automatically.

## Reference

- `docs/rtl_contract.md` — RTL port interfaces, data layout, synth protocol
- `docs/gemm-ip-integration.md` — hls4ml contract, wrapper phases, verification
- `docs/wrapper_timing_model.md` — latency formulas, blackbox binding
- `src/tensor-slice/` — RTL generators, testbench generators, slice RTL
