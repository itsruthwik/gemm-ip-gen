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

The wrapper expects one GEMM transaction per call. Data is fed as row/col
streaming with K-chunks:

- `a_beat_stream` carries `max(M,N)` beats per K-chunk. Each beat is one
  activation row with K elements packed into `grid_rows × 64` bits.
- `b_beat_stream` carries `max(M,N)` beats per K-chunk. Each beat is one
  weight column with K elements packed into `grid_cols × 64` bits.
- Total beats per transaction: `k_chunks × max(M,N)` for each of A and B.
- The wrapper packs beats directly into the blackbox inputs `a_rows` and
  `b_cols` without building temporary local matrices.

Static checks enforce:

- `CONFIG_T::gemm_m`, `CONFIG_T::gemm_k`, and `CONFIG_T::gemm_n` match the package
- `a_beat_T::size == CONFIG_T::gemm_k`
- `b_beat_T::size == CONFIG_T::gemm_k`
- `res_T::size == CONFIG_T::gemm_n`
- `CONFIG_T::transpose_weights == true`

`transpose_weights == true` is interpreted as: the producer side is responsible
for presenting blackbox-ready per-`kk` weight beats. The generated wrapper should
not do row-major-to-column-beat reshaping.

## Wrapper Phases

The generated wrapper has these phases:

1. `BIAS_PACK`: unrolled loop packing bias array into a wide `bias_cols` word.
2. `READ_A_ROWS` / `READ_B_COLS`: pipelined (II=1) reads from `ac_channel` input
   streams into local beat arrays.
3. `FEED`: merged bias-preload + data-feed loop. Step 0 asserts `preload_valid=1,
   in_valid=0` (bias load). Steps 1..`k_chunks×max(M,N)` assert `preload_valid=0,
   in_valid=1` and pack row/col beats into `a_rows` / `b_cols`.
4. `DRAIN`: keeps stepping the stateful blackbox with `in_valid=0`, captures
   valid output rows, and writes to `res_stream`.
5. `DRAIN_PADDED_ROWS`: drains any padded tile rows because tensor-slice tiles are
   eight rows high.

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

1. Standalone RTL simulation via iverilog passes for all DEFAULT_CASES (10 configs,
   sequential + back-to-back).
2. Standalone Catapult synthesis completes using `run_catapult.tcl` (38 cycles at 10ns
   for 8×8×8 on nangate-45nm).
3. SCVerify RTL vs C++ co-simulation passes (0 comparison errors) — the C++ simulation
   model and behavioral Verilog model produce identical output.
4. Synth structural smoke compiles with `-DSYNTHESIS` flag using a stub
   `tensor_slice_int8`.
