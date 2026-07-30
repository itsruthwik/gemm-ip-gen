# hls4ml-Native GEMM IP Integration

`gemm-ip-gen` generates Catapult GEMM hardblock packages for hls4ml. The public
interface is not the old standalone AXI `Stream_t` wrapper. The generated wrapper
is hls4ml-native and is intended to be included through `GEMM_IP_HEADER`.

## Generated Package Format

For a package named `gemm_8x8x8`, the generator emits:

- `gemm_8x8x8_core.v`: tensor-slice GEMM RTL hardblock (combined behavioral +
  structural core, split by `ifndef SYNTHESIS`)
- `gemm_8x8x8_gemm_ip.h`: hls4ml-native C++ wrapper and blackbox binding
- `gemm_8x8x8_inst.cpp`: standalone Catapult top used for package synthesis
- `gemm_8x8x8_tb.cpp`: standalone C-simulation testbench
- `nnet_types.h`: minimal local test types for standalone package validation
- `run_catapult.tcl`: Catapult synthesis script for the standalone package

When generating multiple packages, the output root also contains:

- `gemm_ip_combined.h`: shape dispatch wrapper used by hls4ml
- `integration_manifest.json`: RTL manifest for downstream integration
- `catapult_gemm_blackboxes.tcl`: Catapult file-add helper for generated RTL cores

## Public API

Each generated package provides three shape-specific entry points. The
const-weights variant is the primary one — weights are resident in an array,
not streamed:

```cpp
// Primary: A rows streamed, weights resident.
template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void <name>_gemm_ip_stream_const_weights(
    ac_channel<a_beat_T> &a_stream,
    b_beat_T weight_cols[CONFIG_T::gemm_n],
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
);

// Thin shim: reads the N weight columns from a channel, then calls the above.
template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void <name>_gemm_ip_stream(
    ac_channel<a_beat_T> &a_stream,
    ac_channel<b_beat_T> &b_stream,
    bias_T biases[CONFIG_T::gemm_n],
    ac_channel<res_T> &res_stream
);

// Array in / array out — the einsum / attention interface.
template <class a_beat_T, class b_beat_T, class bias_T, class res_T, typename CONFIG_T>
void <name>_gemm_ip_array(
    a_beat_T a_rows[CONFIG_T::gemm_m],
    b_beat_T weight_cols[CONFIG_T::gemm_n],
    bias_T biases[CONFIG_T::gemm_n],
    res_T results[CONFIG_T::gemm_m]
);
```

The combined header provides the generic shape-dispatching versions of all
three (`nnet::gemm_ip_stream`, `nnet::gemm_ip_stream_const_weights`,
`nnet::gemm_ip_array`), plus `nnet::gemm_ip_stream_sim` — a convenience entry
that takes flat `weights[n_in * n_out]` / `biases[n_out]` in hls4ml dense-layer
form and builds the beat arrays itself.

## Stream Contract

The wrapper processes one GEMM frame per call: M rows of A against the N weight
columns, producing M result rows. (`n_frames > 1` packages feed several frames
in one call — simulation only, see `wrapper_timing_model.md`.)

- `a_stream` carries `M` beats, one K-wide activation/im2col row per beat.
  Each logical row is read from the source exactly **once**.
- `weight_cols` holds `N` K-wide transposed weight columns.
- `res_stream` receives `M` beats, one N-wide result row per beat.

How those beats reach the blackbox depends on the K mode:

- *Chunked* (`gemm_k_spatial == 1`): the feed makes `k_chunks` passes of
  `max(M,N)` beats. On pass 0 each A row is read and its later K chunks are
  pre-packed into `a_replay[k_chunks][max(M,N)]` for replay on subsequent
  passes. B columns are re-packed from `weight_cols` every pass.
- *Full-K-spatial* (`gemm_k_spatial == k_chunks > 1`): a single `max(M,N)`-beat
  pass; each beat carries one A row / B column with **all** K chunks packed
  into a widened `64 * k_chunks`-bit word. No replay storage.

See `rtl_contract.md` for the exact wire-level layout of both modes.

Static checks enforce:

- `CONFIG_T::gemm_m` and `CONFIG_T::gemm_n` match the package
- `a_beat_T::size == CONFIG_T::gemm_k`
- `b_beat_T::size == CONFIG_T::gemm_k`
- `res_T::size == CONFIG_T::gemm_n`
- `CONFIG_T::transpose_weights == true`

`transpose_weights == true` is interpreted as: the producer side is responsible
for presenting blackbox-ready per-`kk` weight beats. The generated wrapper does
not do row-major-to-column-beat reshaping.

## Wrapper Structure

The generated wrapper is a single merged loop, not separate feed and drain
phases:

1. `BIAS_PACK`: unrolled loop that packs a **zero** bias word for the core. The
   core is a pure integer matmul; the real bias is applied at capture time.
2. `RUN` / `RUN_ARRAY`: II=1 pipelined loop, one iteration per core cycle. Each
   iteration decodes its position within the frame (`p = step % period`), packs
   the A/B operand words when `1 <= p <= total_beats`, issues exactly one
   `gemm.run()` call, and polls `out_valid`.
3. Output capture (inside `RUN`): on each `out_valid`, the raw 16-bit integer
   lanes are rescaled by `2^-(frac_a + frac_b)`, the full-precision bias is
   added, and the result is cast to the result type (round/saturate).
4. `DRAIN_PADDED_ROWS`: `MR - M` idle calls, flushing the hardblock's 8-row
   burst granularity.

Because capture is polled from step 0, rows are collected as they emerge —
including while later beats of the same frame are still being fed.
`wrapper_run_loop.md` has the cycle-by-cycle anatomy and timing diagrams;
`wrapper_timing_model.md` has the latency formulas.

The blackbox binding uses:

```cpp
ac_blackbox()
    .entity("<name>_core")
    .verilog_files("<name>_core.v")
    .outputs("c_row out_valid out_last")
    .area(2048.0)
    .delay(<clock-derived>)
    .latency(1)
    .init_delay(1)
    .start_name("en")
    .has_state(true);
```

The true tensor-slice latency is represented by the explicit wrapper call
budget, not by declaring the full hardblock latency to Catapult. Declaring the
full RTL latency makes Catapult schedule excessive pipeline depth around the
blackbox.

## Verification Expectations

For each generated package:

1. Standalone RTL simulation via iverilog passes for all DEFAULT_CASES (10 configs,
   sequential + back-to-back).
2. Standalone Catapult synthesis completes using `run_catapult.tcl`.
3. SCVerify RTL vs C++ co-simulation passes (0 comparison errors) — the C++ simulation
   model and behavioral Verilog model produce identical output.
4. Synth structural smoke compiles with `-DSYNTHESIS` flag using a stub
   `tensor_slice_int8`.

Note that the structural (`SYNTHESIS`) branch is synthesis-only and is not a
cycle-accurate reference; the behavioral branch is the verification authority.
