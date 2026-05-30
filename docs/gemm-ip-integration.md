# hls4ml-Native GEMM IP Integration

`gemm-ip-gen` generates Catapult GEMM hardblock packages for hls4ml. The public
interface is not the old standalone AXI `Stream_t` wrapper. The generated wrapper
is hls4ml-native and is intended to be included through `GEMM_IP_HEADER`.

## Generated Package Format

For a package named `gemm_8x8x8_batch`, the generator emits:

- `gemm_8x8x8_batch_core.v`: tensor-slice GEMM RTL hardblock
- `gemm_8x8x8_batch_gemm_ip.h`: hls4ml-native C++ wrapper and blackbox binding
- `gemm_8x8x8_batch_inst.cpp`: standalone Catapult top used for package synthesis
- `gemm_8x8x8_batch_tb.cpp`: standalone C-simulation testbench
- `nnet_types.h`: minimal local test types for standalone package validation
- `run_catapult.tcl`: Catapult synthesis script for the standalone package

When generating multiple packages, the output root also contains:

- `gemm_ip_combined.h`: shape dispatch wrapper used by hls4ml
- `gemm_ip_manifest.json`: RTL manifest for downstream integration
- `blackbox_files.tcl`: Catapult file-add helper for generated RTL cores

## Public API

Each generated package provides a shape-specific wrapper:

```cpp
template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void <name>_gemm_ip_stream(
    ac_channel<a_beat_T> &a_beat_stream,
    ac_channel<b_beat_T> &b_beat_stream,
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
);
```

The combined header provides the generic dispatch function:

```cpp
template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void gemm_ip_stream(
    ac_channel<a_beat_T> &a_beat_stream,
    ac_channel<b_beat_T> &b_beat_stream,
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
);
```

## Stream Contract

The wrapper expects one GEMM transaction per call.

For each `kk` from `0` to `CONFIG_T::gemm_k - 1`:

- `a_beat_stream.read()` returns one activation beat containing all active GEMM
  rows for that `kk`
- `b_beat_stream.read()` returns one weight beat containing all active GEMM output
  columns for that `kk`
- the wrapper packs those two beats directly into the hardblock inputs
  `a_rows` and `b_cols`

This is the required low-latency contract. The generated wrapper must not build
temporary local matrices, prepack arrays, or reinterpret a row-major weight stream
inside the wrapper.

The current static checks enforce:

- `CONFIG_T::gemm_m`, `CONFIG_T::gemm_k`, and `CONFIG_T::gemm_n` match the package
- `a_beat_T::size == CONFIG_T::gemm_m`
- `b_beat_T::size == CONFIG_T::gemm_n`
- `res_T::size == CONFIG_T::gemm_n`
- `CONFIG_T::transpose_weights == true`

`transpose_weights == true` is interpreted as: the producer side is responsible
for presenting blackbox-ready per-`kk` weight beats. The generated wrapper should
not do row-major-to-column-beat reshaping.

## Wrapper Phases

The generated wrapper has three hardblock-facing phases:

1. `FEED`: read one activation beat and one weight beat per `kk`, pack directly,
   and call the GEMM blackbox with `in_valid = 1`.
2. `DRAIN_WRITE`: keep stepping the stateful blackbox with `in_valid = 0`, capture
   valid output rows, add bias, and write `res_stream`.
3. `DRAIN_PADDED_ROWS`: drain any padded tile rows that exist because tensor-slice
   tiles are eight rows high.

The blackbox binding uses:

```cpp
ac_blackbox()
    .latency(1)
    .init_delay(1)
    .start_name("en")
    .has_state(true);
```

The true tensor-slice latency is represented by the explicit wrapper drain loops,
not by declaring the full hardblock latency to Catapult.

## Verification Expectations

For each generated package:

1. Standalone C-simulation should pass using `<name>_tb.cpp`.
2. Standalone Catapult synthesis should complete using `run_catapult.tcl`.
3. RTL/SCVerify should pass for representative shapes before using the package as
   an hls4ml integration baseline.
4. The `8x8x8` package should remain close to the earlier standalone low-latency
   behavior. Large latency regressions usually mean wrapper-side packing or
   buffering has re-entered the feed path.
