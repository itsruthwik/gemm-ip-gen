# gemm-ip-gen

Generates GEMM IP blackbox packages for Catapult HLS, targeting the tensor-slice
INT8 GEMM hardblock. Each package contains the RTL core, an hls4ml-native C++
wrapper, and a Catapult synthesis script. A combined dispatch header is emitted
when multiple packages are generated together.

## Quick start

```bash
# Fresh machine: create a venv and install everything (Python >= 3.10)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -e .
source .venv/bin/activate

# Already have an environment? Editable install only
pip install -e .

# Generate a single package
python -m gemm_ip --m 8 --k 8 --n 8 --name gemm_8x8x8 --output_dir ./output

# From config file
python -m gemm_ip --config gemm_config.json --output_dir ./output
```

### Using backward-compat shims

If you have existing scripts that call the old entry points, they still work:

```bash
python generate_catapult_pkg.py --m 8 --k 8 --n 8 --name gemm_8x8x8

# Legacy config-driven flow
python generate_gemm_ip.py gemm_config.json ./output
```

The config file is written automatically by hls4ml when `GemmIP: True` is
set for a layer. Each entry carries `gemm_m`, `gemm_k`, `gemm_n`,
`interface`, and `protocol` metadata.

## Package properties

| Feature | Value |
|---|---|
| Stream type | `ac_channel<ac_int<W>>` |
| Blackbox mechanism | `ac_blackbox` + ccore |
| C model | `{name}_gemm_ip.h` (templated) |
| RTL wrapper | `{name}_core.v` |
| Build script | `run_catapult.tcl` |
| Interface support | `stream`, `array` |
| Bias handling | In the wrapper capture path, post-rescale, full precision (the core is fed zero bias) |
| Core output lane | int16 |
| Result type | `output_precision` (rescale + bias + round/saturate in wrapper) |

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
| `integration_manifest.json` | Package metadata for multi-layer models |

## Interfaces

Typed hls4ml adapters are emitted for both integration styles:

- `nnet::gemm_ip_stream(...)` for stream-fed layers
- `nnet::gemm_ip_array(...)` for array-fed layers

The selected `interface` value is preserved in `integration_manifest.json` and
used by `gemm_ip_combined.h` to dispatch the matching adapter for each generated
shape.

## Testing

```bash
# Full suite
pytest tests/ -v

# RTL simulation
pytest tests/test_rtl_sim.py -v

# Catapult package tests
pytest tests/test_catapult.py -v
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
