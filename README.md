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
| Interface support | `stream`, `array` | `stream` only |
| Bias handling | Inside blackbox (preload phase) | Inside blackbox (preload phase) |
| Result type | int8 (saturated from int32) | int8 (saturated from int32) |

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

## Stream interface (Vitis)

The Vitis blackbox uses three input streams and one output stream, all
`hls::stream<ap_uint<W>>`:

```
void <name>_wrapper(
    hls::stream<ap_uint<A_WIDTH>>& a_stream,      // K_steps beats
    hls::stream<ap_uint<B_WIDTH>>& b_stream,      // K_steps beats
    hls::stream<ap_uint<B_WIDTH>>& bias_stream,   // 1 beat
    hls::stream<ap_uint<C_WIDTH>>& c_stream       // M beats
);
```

Protocol sequence inside the wrapper:
1. Read bias from `bias_stream` (1 beat)
2. Preload bias into tensor-slice grid (K_steps cycles)
3. Feed activation/weight data (K_steps cycles)
4. Compute — pipeline latency through grid
5. Drain M result beats (saturated int16→int8)

## Width formulas (Vitis)

| Packet | Width | Rationale |
|---|---|---|
| A (`a_tdata`) | `ceil(M/8) * 64` | M int8 rows packed 8 per 64-bit tile |
| B (`b_tdata`) | `ceil(N/8) * 64` | N int8 cols packed 8 per 64-bit tile |
| Bias (`bias_tdata`) | `ceil(N/8) * 64` | Same packing as B |
| C (`c_tdata`) | `ceil(N/8) * 64` | N int8 results, saturated from int32 |

## Testing

```bash
# Run all tests
pytest test_generate_catapult_pkg.py -v

# Backend-specific tests (when Vitis tests are added)
pytest test_generate_vitis_pkg.py -v
```

Catapult tests cover config normalisation, combined-header dispatch
(including duplicate-shape array layers dispatched by `gemm_ip_id`),
single-package generation, and a sparse-`en` Icarus RTL regression.

## Reference

- `docs/INTEGRATION.md` — hls4ml contract and verification expectations
- `docs/wrapper_timing_model.md` — dead-cycle formula and latency targets
- `tensor-slice/` — RTL hardblock and grid generator
- `vitis_support_plan.md` — Vitis backend implementation plan
- `vitis_support_notes.md` — Implementation progress and decisions
