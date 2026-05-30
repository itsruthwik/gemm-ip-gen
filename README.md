# gemm-ip-gen

Generates Catapult HLS packages for the tensor-slice INT8 GEMM hardblock.
Each package contains the RTL core, an hls4ml-native C++ wrapper, and a
Catapult synthesis script. A combined dispatch header is emitted when
multiple packages are generated together.

## Usage

### Single package

```bash
python generate_catapult_pkg.py \
    --m 8 --k 8 --n 8 \
    --name gemm_8x8x8 \
    --interface stream \       # or array
    --output_dir ./output
```

### From hls4ml gemm_config.json

```bash
python generate_catapult_pkg.py --config gemm_config.json --output_dir ./output
```

The config file is written automatically by hls4ml when `GemmIP: True` is
set for a layer. Each entry carries `gemm_m`, `gemm_k`, `gemm_n`,
`interface`, and `protocol` metadata.

## Interfaces

| Interface | Use case | Call boundary |
|---|---|---|
| `stream` | Einsum / attention score and context | `ac_channel` streams |
| `array` | EinsumDense / projection layers (`io_parallel`) | pre-packed beat arrays |

Both interfaces expose the same blackbox RTL core; only the C++ wrapper
differs.

## Generated outputs

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

## Testing

```bash
pytest test_generate_catapult_pkg.py -q
```

Covers config normalisation, combined-header dispatch (including
duplicate-shape array layers dispatched by `gemm_ip_id`), single-package
generation, and a sparse-`en` Icarus RTL regression.

## Reference

- `docs/INTEGRATION.md` — hls4ml contract and verification expectations
- `docs/wrapper_timing_model.md` — dead-cycle formula and latency targets
- `tensor-slice/` — RTL hardblock and grid generator
