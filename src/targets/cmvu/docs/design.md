# cmvu target: design rules

Design rules for the `cmvu` target (`gemm-ip-gen/src/targets/cmvu/`), each
with its reason inline. This is the current state of the design; where it
differs from older notes elsewhere, this file wins. See `architecture.md`
and `mode_1_user_guide.md` in this directory for the vendored `cmvu_mode1`
block itself (not the wrapper this target generates around it).

## Numerics

- **32-bit accumulator-scale bias, rounding folded in.** Every bias code is
  a signed int32 value at the product's fractional precision; a layer that
  wants round-half-up (`RND`) output gets `1 << (shift-1)` folded into that
  same code (a bias-less `RND` layer still gets a code carrying only the
  rounding constant) — the block itself never rounds, so rounding has to
  live somewhere upstream of the shift, and the bias add already happens
  once at `acc_first` at the right scale.
- **The block's requant is a plain floor (truncating) arithmetic shift then
  wrap-to-W.** `cmvu_mode1`'s round-half-up adder was removed (see
  `rtl_static/MVU_COMMIT.txt`); this is why rounding moved into the bias.
- **RND and TRN output modes are both accepted; SAT/SAT_SYM are not.** The
  block only wraps (two's-complement), never saturates, so a layer asking
  for saturating overflow can't be represented; RND/TRN are just "was the
  rounding constant folded into the bias or not."
- **Only in/weight/out precisions matter; a narrow accumulator only warns.**
  cmvu accumulates exact int32 products regardless of `accum_precision`, so
  an hls4ml `accum_precision` narrower than `in_frac + weight_frac` doesn't
  change what cmvu computes — it only means hls4ml's own (rounded) reference
  can differ from cmvu's by about an output LSB, which is worth a warning,
  not a hard error.
- **Bias must be exactly representable at the product's fractional
  precision.** hls4ml adds bias into `accum_t` at that precision; a bias
  with bits below that LSB can't be baked exactly into the int32
  accumulator-scale code, so it's rejected at package time instead of
  silently rounded.

## Wrapper boundary and control

- **The public stream ports are exact hls4ml widths; all padding/tiling
  lives in the generated Verilog.** The A input (`a_row`) is `K*8` bits
  (element `k` at `[8k+:8]`, no K-tail padding) and the result output
  (`res_row`) is `N*W` bits (lane `n` at `[W*n+:W]`, no N-tail padding); the
  wrapper zero-extends A into its padded internal row buffer and drops the
  padded tail lanes out of the internal result buffer before the port. This
  keeps the hls4ml-facing interface exactly what hls4ml expects, with the
  block's 4-row/8-lane physical tiling as an implementation detail the
  wrapper alone knows about.
- **Runtime-B's B layout comes from the manifest's `weight_layout`.**
  `column_major` streams one K-high beat per real N column (every block in
  the target block-row has its own write port, so paired k-passes load in
  the same beat when `K_PASSES==2`, and with more K passes the rest are held
  in a small per-block tile store and written after each n-group's columns,
  stalling the B stream 4 cycles per held tile); `row_major` streams one N-wide beat per
  real K row into a block-column write port that holds its tile address
  across a whole 4-beat transaction, so it needs a small residual buffer
  (one tile's worth of rows) plus drain stalls to write more than one
  resident N-pass slot. Each layout is what it is because of which write
  port (row vs. column) the block exposes, not a free choice.
- **B is reloaded for every frame, into one or two slot sets.** hls4ml
  streams a new B ahead of every frame's A rows (a fresh K for attention QK,
  a fresh V for aV), and the wrapper is persistent hardware, so its load FSM
  re-arms whenever the slot set it fills is empty. When two copies of the
  layer's `K_PASSES*N_GROUPS` tiles fit in the block's 8 slots (set 1 starts
  on an even slot, so both sets load identically), the next frame's B loads
  into the idle set while the current frame computes from the other; a set
  is full from the edge its last write lands until the frame's last row
  issues its last pass. A row is accepted only when the set it will read is
  full, so a new frame can start right behind the previous one. With one set
  the load waits for that release.
- **Fixed-cadence, feed-forward control: no blackbox signal drives the HLS
  schedule.** The C++ entry presents inputs on a compile-time-fixed cadence
  and only polls `out_valid`/reads results — it never branches on
  `in_ready`. Catapult's blackbox cosim requires this: a schedule that
  depended on a sampled DUT signal drifted the software call count from the
  RTL's en-qualified cycle count for multi-pass schedules.
- **The clock gate is glitch-free because it's sampled on the falling
  edge.** `en` is combinational output of registered control logic and can
  change while `clk` is high; gating with plain `clk & en` glitches (an
  extra rising edge whenever `en` rises mid-high-phase). Sampling `en` on
  `negedge clk` into `en_n`, then gating with `clk & (en_n | rst)`, gives a
  value that's already stable for the whole high phase — the same value the
  wrapper's own `.ena(en)`-gated registers see.
- **The blackbox reset is synchronous, active-high.** This matches how
  Catapult's own generated design resets in the SCVerify/Catapult build, so
  the vendored block and the wrapper's registers reset in lockstep with the
  rest of the design instead of on a different reset discipline. Catapult
  sees only the wrapper, and every register the wrapper owns (including its
  skew and de-skew delay lines) resets on `posedge clk` only. The vendored
  block keeps its internal async reset, but its `rst` pin is driven only by
  the wrapper's clock-synchronous `rst`, so it behaves as a sync reset from
  outside the wrapper.
- **Pack/unpack is raw-bit, not value.** Packing an A row or unpacking a
  result lane copies each element's bits directly (`.slc<8>()`/`.set_slc()`)
  rather than assigning the `ac_int` by value — a value assignment would
  reinterpret the code as an integer and wrap it into the wrong fixed-point
  range.

## hls4ml integration

- **Four hls4ml GEMM entry points; two are real, two are stubs.**
  `gemm_stream_const_weights` (weight-stationary) and `gemm_stream`
  (runtime-B) dispatch to the matching per-layer generated function;
  `gemm_array`/`gemm_array_const_weights` (io_parallel) are
  `static_assert`-only stubs, since cmvu is stream-only and has no
  io_parallel datapath to service them.
- **One K-wide A row per beat.** hls4ml's stream contract sends one whole
  contraction-dim row per activation beat; a narrower beat would under-read
  the row, so the entry `static_assert`s the beat width up front.
- **`reuse_factor` is accepted and ignored.** hls4ml puts its own
  ReuseFactor on every GEMM layer regardless of target; cmvu's folding is
  set entirely by the `KFold`/`NFold` knobs, which are independent of it.

## Provenance

- **The vendored `cmvu_mode1`/`cmvu_w_mem`/`cmvu_regbank` blocks are
  modified from upstream.** See `rtl_static/MVU_COMMIT.txt` for the exact
  diff (bias width, and moving rounding out of the block's own requant) and
  the commit this target vendors from.
