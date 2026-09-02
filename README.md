# gemm-ip-gen

Generates GEMM IP blackbox packages for hardblock **targets**. A target is one
hardblock welded to a single HLS tool; the only target today is `tensor_slice`
(an INT8 GEMM hardblock, tool: Catapult). Each package contains the RTL core, an
hls4ml-native C++ wrapper, and a Catapult synthesis script. A combined dispatch
header is emitted when multiple packages are generated together.

## Layout

The framework core is thin and target-agnostic; each hardblock is a
self-contained plugin under `src/targets/`:

```text
src/
  gemm_ip/          core: cli, config, common, quant, registry, weights
  targets/
    base.py         the Target contract
    tensor_slice/   geometry, rtl, golden, package, flow, tb/, run_rtl_tests
```

A **target** = one hardblock + its single HLS tool (one tool per hardblock — no
hardblock×tool matrix, no shared protocol/shim layer). A target's `flow.py`
implements the `Target` contract (`geometry`, `emit_rtl`, `emit_behavioral`,
`golden`, `package`, `verify`, `rtl_test`) by delegating to its sibling modules,
and registers under a name in `gemm_ip/registry.py`. The CLI picks one with
`--target` (default: `tensor_slice`). Adding a hardblock is a new
`src/targets/<name>/` directory plus a name in the registry — the core and other
targets are untouched.

## Quick start

```bash
# Fresh machine: create a venv and install everything (Python >= 3.10)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -e .
source .venv/bin/activate

# Already have an environment? Editable install only
pip install -e .

# Generate a single package (default target: tensor_slice)
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

The config file is written automatically by hls4ml when `Strategy: GEMM` is
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
| `catapult_gemm_blackboxes.tcl` | Catapult file-add helper |
| `integration_manifest.json` | Package metadata for multi-layer models |

## Interfaces

Typed hls4ml adapters are emitted for both integration styles:

- `nnet::gemm_ip_stream(...)` for stream-fed layers
- `nnet::gemm_ip_array(...)` for array-fed layers

The selected `interface` value is preserved in `integration_manifest.json` and
used by `gemm_ip_combined.h` to dispatch the matching adapter for each generated
shape.

## Testing

The RTL-level regression generates the behavioral wrapper + a self-checking
testbench for a spread of shapes and runs them under Icarus Verilog:

```bash
# 10 shapes x 3 seeds (behavioral combined core; no external slice IP needed)
src/targets/tensor_slice/run_rtl_tests.sh

# a subset
src/targets/tensor_slice/run_rtl_tests.sh --cases 8x8x8 16x16x16 --seeds 1 7
```

It compiles the combined core without `-DSYNTHESIS`, so the `ifndef SYNTHESIS`
behavioral branch (which does the matmul directly) is the one simulated. The same
regression is reachable through the target contract as `Target.rtl_test(...)`.
The tool-level bar — a real Catapult C-synth + QuestaSim C-vs-RTL cosim — is
covered under **Catapult SCVerify** below.

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
- `src/gemm_ip/` — thin, target-agnostic core (cli, config, quant, registry, …)
- `src/targets/base.py` — the Target contract
- `src/targets/tensor_slice/` — the tensor_slice target: geometry, RTL/testbench generators, package
