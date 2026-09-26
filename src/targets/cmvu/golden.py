"""cmvu golden: exact NumPy oracle, bias baking, and LSB-first packers.

The hardware emits a requantized ``y_out`` on every valid pass; the FINAL value
at ``acc_last`` is exactly ``requant(sum(A*B) + bias_code)`` for every fold
configuration, because the int32 accumulation is exact across K passes, N
groups, and cascade stages (the only folding-dependent effect is *when* the
partials appear, not their sum). This module is the single reference for the
golden TBs, the C++ behavioral model, and the baked weight/bias data.

The block itself never rounds: requant is a plain floor (arithmetic) shift.
Rounding, where the layer wants ``RND`` output, is folded into the 32-bit
``bias_code`` at accumulator scale before it is baked in (``+ 1 <<
(shift-1)`` when ``shift > 0``) -- see :func:`bias_codes`. A layer with no
bias and an ``RND`` output still needs a (zero-mean) bias code carrying only
that rounding constant.

Packing conventions (architecture.md §11.1): element/lane 0 in the LSBs; a
canonical weight tile is row-major with ``W[i][j]`` at bit ``(i*8+j)*8``.
"""
import numpy as np

try:
    from . import geometry as _geometry
except ImportError:  # standalone/script import
    import geometry as _geometry


# ── Requantization (architecture.md §2) ───────────────────────────────────────


def wrap_int8(value):
    """Two's-complement wrap to int8 (WRAP overflow policy, no saturation)."""
    return ((int(value) + 128) & 0xFF) - 128


def wrap_int32(value):
    """Two's-complement wrap to int32 (bias/accumulator scale, WRAP policy)."""
    m = 1 << 31
    return ((int(value) + m) & 0xFFFFFFFF) - m


def wrap_to_w(value, width):
    """Two's-complement wrap to ``width`` signed bits (WRAP policy)."""
    width = int(width)
    if not (1 <= width <= _geometry.RESULT_WIDTH):
        raise ValueError(
            f"width must be 1..{_geometry.RESULT_WIDTH}, got {width}")
    v = int(value) & ((1 << width) - 1)
    return v - (1 << width) if (v & (1 << (width - 1))) else v


def requant_w(total, shift, width):
    """Floor (truncating) arithmetic shift, then wrap to ``width`` signed bits.

    Matches ``cmvu_mode1``'s requant (shift -> wrap-to-W): ``y =
    wrap_to_w(total >>> shift, W)``. The block never rounds; a layer that
    wants round-half-up (``RND``) output must fold ``1 << (shift-1)`` into
    its bias code before it reaches here (see :func:`bias_codes`).
    ``width=8`` reproduces the V1 int8 result.
    """
    shift = int(shift)
    if shift < 0 or shift > _geometry.MAX_SHIFT:
        raise ValueError(
            f"shift must be 0..{_geometry.MAX_SHIFT} (SHIFT_WIDTH={_geometry.SHIFT_WIDTH}), "
            f"got {shift}")
    return wrap_to_w(int(total) >> shift, width)


def requant(total, shift):
    """Floor shift + int8 wrap (the ``requant_w(.., 8)`` case)."""
    return requant_w(total, shift, 8)


def requant_rows(totals, shift, width=8):
    """Elementwise :func:`requant_w` over a 1-D/2-D int array."""
    arr = np.asarray(totals, dtype=np.int64)
    flat = [requant_w(v, shift, width) for v in arr.reshape(-1)]
    return np.array(flat, dtype=np.int64).reshape(arr.shape)


# ── Reference oracle ──────────────────────────────────────────────────────────


def reference_rows(A, B, bias_codes=None, shift=0, result_width=8):
    """Exact ``A @ B`` (+ baked bias codes) -> requant -> ``result_width``-bit rows.

    ``A`` is (M, K) and ``B`` is (K, N) integer codes; ``bias_codes`` (if
    given) is one signed int32 code per output lane, at accumulator scale
    with any rounding constant already folded in, added to the raw sum
    before the single (floor) requant (hardware applies it once at
    ``acc_first``). ``result_width`` is the block's effective width W (1..16).
    """
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"shape mismatch: A{A.shape} @ B{B.shape}")
    total = A @ B
    if bias_codes is not None:
        b = np.asarray(bias_codes, dtype=np.int64).reshape(-1)
        if b.size != B.shape[1]:
            raise ValueError(
                f"bias_codes has {b.size} lanes but N={B.shape[1]}")
        total = total + b[None, :]
    return requant_rows(total, shift, result_width)


def bias_codes(bias, scale_shift, rnd=True, requant_shift=None):
    """Bake per-lane bias values into int32 accumulator-scale codes.

    ``code = round(bias * 2**scale_shift) + round_const`` with
    round-half-away-from-zero for the bias itself, where ``round_const =
    1 << (requant_shift-1)`` (else 0) when ``rnd`` is true. ``requant_shift``
    defaults to ``scale_shift`` -- the standalone/test-config convention
    where the bias is expressed directly at the block's requant shift (i.e.
    ``scale_shift`` doubles as the product/accumulator frac for these
    configs, which have no separate in/weight fractional precisions). A code
    outside int32 is two's-complement wrapped, with a warning naming the
    lane (RTL and the C++ model bake the same wrapped code, so hardware wrap
    semantics are what every consumer sees).

    Callers with no bias but an ``RND`` output still need a code: pass
    ``bias=None`` (or an all-zero array) to get just the rounding constant
    on every lane.
    """
    scale_shift = int(scale_shift)
    req_shift = int(scale_shift if requant_shift is None else requant_shift)
    round_const = (1 << (req_shift - 1)) if (rnd and req_shift > 0) else 0
    out = []
    warnings = []
    if bias is None:
        bias = np.zeros(_geometry.N_PHYS, dtype=np.float64)
    for lane, value in enumerate(np.asarray(bias, dtype=np.float64).reshape(-1)):
        scaled = float(value) * (1 << scale_shift)
        rounded = int(np.floor(scaled + 0.5)) if scaled >= 0 else int(np.ceil(scaled - 0.5))
        rounded += round_const
        wrapped = wrap_int32(rounded)
        if wrapped != rounded:
            warnings.append(
                f"WARNING: bias lane {lane} = {value} needs code {rounded} at "
                f"scale_shift {scale_shift} (int32 range), wrapped to {wrapped}.")
        out.append(wrapped)
    return np.array(out, dtype=np.int64), warnings


# ── Packers ───────────────────────────────────────────────────────────────────


def pack_a_row(row, k_chunks_pad):
    """Pack a K-wide activation row into ``k_chunks_pad*4`` LSB-first int8 lanes."""
    lanes = int(k_chunks_pad) * _geometry.K_PHYS
    row = np.asarray(row, dtype=np.int64).reshape(-1)
    if row.size > lanes:
        raise ValueError(f"row has {row.size} lanes, width is {lanes}")
    val = 0
    for i, v in enumerate(row):
        val |= (int(v) & 0xFF) << (8 * i)
    return val


def unpack_res_row(bits, n_chunks_pad, width=8):
    """Unpack ``n_chunks_pad*N_PHYS`` LSB-first ``width``-bit result lanes."""
    lanes = int(n_chunks_pad) * _geometry.N_PHYS
    mask = (1 << int(width)) - 1
    return [wrap_to_w((int(bits) >> (int(width) * i)) & mask, width)
            for i in range(lanes)]


def pack_res_row(lanes, width=8):
    """Pack signed ``width``-bit result lanes LSB-first (lane 0 at bit 0)."""
    width = int(width)
    mask = (1 << width) - 1
    val = 0
    for j, v in enumerate(lanes):
        val |= (int(v) & mask) << (width * j)
    return val


def pad_row(values, n_chunks_pad):
    """Zero-pad a result row to the padded output-lane count."""
    lanes = int(n_chunks_pad) * _geometry.N_PHYS
    out = [int(v) for v in np.asarray(values, dtype=np.int64).reshape(-1)]
    if len(out) > lanes:
        raise ValueError(f"row has {len(out)} lanes, width is {lanes}")
    return out + [0] * (lanes - len(out))


def weight_tiles(B, k, n, k_chunks, n_chunks):
    """Slice ``B`` (K x N) into canonical zero-padded 4x8 tiles ``[kt][nt]``."""
    B = np.asarray(B, dtype=np.int64)
    if B.shape != (int(k), int(n)):
        raise ValueError(f"B shape {B.shape} != (k={k}, n={n})")
    tiles = []
    for kt in range(int(k_chunks)):
        row = []
        for nt in range(int(n_chunks)):
            tile = np.zeros((_geometry.K_PHYS, _geometry.N_PHYS), dtype=np.int64)
            for i in range(_geometry.K_PHYS):
                ki = kt * _geometry.K_PHYS + i
                if ki >= k:
                    break
                for j in range(_geometry.N_PHYS):
                    nj = nt * _geometry.N_PHYS + j
                    if nj >= n:
                        break
                    tile[i, j] = int(B[ki, nj])
            row.append(tile)
        tiles.append(row)
    return tiles


def pack_tile(tile):
    """Canonical tile -> 256-bit int, ``W[i][j]`` at bits ``(i*8+j)*8``."""
    val = 0
    for i in range(_geometry.K_PHYS):
        for j in range(_geometry.N_PHYS):
            val |= (int(tile[i][j]) & 0xFF) << ((i * _geometry.N_PHYS + j) * 8)
    return val


def hex_literal(value, width_bytes):
    """Verilog hex literal body (no prefix) for a ``width_bytes``-wide value."""
    return f"{int(value) & ((1 << (8 * int(width_bytes))) - 1):0{2 * int(width_bytes)}x}"


# ── Self-checking testbench ───────────────────────────────────────────────────


def b_load_beats(B, k, n, k_chunks_pad, n_chunks_pad):
    """Runtime-B stream: 64-bit beats (8 int8) in the wrapper's load order.

    ``ktile`` outer / ``ntile`` inner / tile-row innermost, with
    ``ktile = kp*ks + c`` and ``ntile = np*ns + r``. Padded tiles
    (``ktile >= ceil(k/4)`` or ``ntile >= ceil(n/8)``) are zero, so every block
    slot is written (no X read).

    Superseded by :func:`b_load_beats_col_major` for the runtime-B core's
    column-major ``b_beat`` port; kept for reference/other callers.
    """
    B = np.asarray(B, dtype=np.int64)
    kc = _geometry._ceil_div(int(k), _geometry.K_PHYS)
    nc = _geometry._ceil_div(int(n), _geometry.N_PHYS)
    tiles = weight_tiles(B, k, n, kc, nc)
    zero = np.zeros((_geometry.K_PHYS, _geometry.N_PHYS), dtype=np.int64)
    beats = []
    for kt in range(int(k_chunks_pad)):
        for nt in range(int(n_chunks_pad)):
            tile = tiles[kt][nt] if (kt < kc and nt < nc) else zero
            packed = pack_tile(tile)
            for i in range(_geometry.K_PHYS):
                beats.append((packed >> (i * 64)) & ((1 << 64) - 1))
    return beats


def b_load_beats_col_major(B, k, n):
    """hls4ml's native column-major B stream: one beat per real column.

    Beat ``n`` is column ``n`` of ``B`` (``0..N-1``, real/unpadded ``k``/``n``),
    row ``ki`` at bits ``[8*ki +: 8]`` -- exactly the runtime-B core's
    ``b_beat`` port format. The wrapper itself synthesizes the padding
    columns beyond real ``N``, so this stream carries only the ``n`` real
    columns (no ``k_chunks_pad``/``n_chunks_pad`` padding beats).
    """
    B = np.asarray(B, dtype=np.int64)
    if B.shape != (int(k), int(n)):
        raise ValueError(f"B shape {B.shape} != (k={k}, n={n})")
    beats = []
    for col in range(int(n)):
        val = 0
        for ki in range(int(k)):
            val |= (int(B[ki, col]) & 0xFF) << (8 * ki)
        beats.append(val)
    return beats


def b_load_beats_row_major(B, k, n):
    """hls4ml's row-major B stream: one beat per real row.

    Beat ``k`` is row ``k`` of ``B`` (``0..K-1``, real/unpadded ``k``/``n``),
    column ``ni`` at bits ``[8*ni +: 8]`` -- the runtime-B core's row-major
    ``b_beat`` port format (``N*8`` bits wide). The wrapper synthesizes the
    padding rows beyond real ``K`` itself, so this stream carries only the
    ``k`` real rows.
    """
    B = np.asarray(B, dtype=np.int64)
    if B.shape != (int(k), int(n)):
        raise ValueError(f"B shape {B.shape} != (k={k}, n={n})")
    beats = []
    for row in range(int(k)):
        val = 0
        for ni in range(int(n)):
            val |= (int(B[row, ni]) & 0xFF) << (8 * ni)
        beats.append(val)
    return beats


def _en_gap_decls(seed):
    """``en`` driver + ``step`` task for an en-gap-stressed TB.

    Mirrors the HLS blackbox start qualifier: ``en`` is driven by registered
    control logic that changes just after ``posedge clk``, so a value read
    right after an edge (before that edge's own nonblocking updates land) is
    the value that was stable *during* the cycle ending at that edge -- the
    same value the wrapper's ``else if (en)`` flops and the vendored blocks'
    gated clock sample at that edge. ``step`` advances the driver to the next
    edge where the DUT actually did work, so gap cycles are held through
    rather than raced past.
    """
    return f"""
    // Randomized en gaps: en changes just after posedge clk, like the
    // registered control logic that drives the real start qualifier. About
    // 30% of cycles gate the DUT off; reading `en` right after `@(posedge
    // clk)` (before this edge's own nonblocking update lands) gives the
    // same pre-edge value the DUT's ena-gated flops sampled at this edge.
    reg [31:0] en_seed = 32'd{int(seed)};
    always @(posedge clk or posedge rst) begin
        if (rst) en <= 1'b1;
        else     en <= ($unsigned($random(en_seed)) % 10) >= 3;
    end
    task automatic step;  // block until the next edge where en was asserted
        begin
            @(posedge clk);
            while (!en) @(posedge clk);
        end
    endtask
"""


def _backpressure_decls(seed):
    """``en`` driver + ``step`` task for a sustained-backpressure TB.

    Like :func:`_en_gap_decls` (registered, changes just after posedge clk,
    read pre-edge so it matches the wrapper's own ena-gated sampling), but
    in long bursts instead of a per-cycle coin flip: 20-50 consecutive
    disabled cycles alternating with short (1-5 cycle) enabled bursts. A
    burst this long reliably lands mid-transaction (rows in flight, results
    pending, mid runtime-B load) rather than only ever gating single idle
    cycles, which is what a real backpressured consumer/producer looks like.
    """
    return f"""
    // Sustained backpressure: en alternates between long (20-50 cycle)
    // disabled bursts and short (1-5 cycle) enabled bursts, instead of
    // gating single cycles -- long enough to reliably land mid-transaction
    // (rows in flight, results pending, mid runtime-B load).
    reg [31:0] bp_seed   = 32'd{int(seed)};
    reg [31:0] bp_remain;
    reg        bp_gate;   // 0 = currently in an enabled burst, 1 = disabled
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            en        <= 1'b1;
            bp_gate   <= 1'b0;
            bp_remain <= 32'd3;
        end else begin
            if (bp_remain == 32'd0) begin
                if (bp_gate) begin
                    // was disabled -- switch to a short enabled burst
                    bp_remain <= 32'd1 + ($unsigned($random(bp_seed)) % 32'd5);
                    en        <= 1'b1;
                end else begin
                    // was enabled -- switch to a long disabled burst
                    bp_remain <= 32'd20 + ($unsigned($random(bp_seed)) % 32'd31);
                    en        <= 1'b0;
                end
                bp_gate <= ~bp_gate;
            end else begin
                bp_remain <= bp_remain - 32'd1;
            end
        end
    end
    task automatic step;  // block until the next edge where en was asserted
        begin
            @(posedge clk);
            while (!en) @(posedge clk);
        end
    endtask
"""


def _gap_mode(seed, en_gaps, backpressure):
    """Pick the en-gating mode for an Icarus TB: at most one of ``en_gaps``
    (per-cycle ~30% gate) or ``backpressure`` (long bursts, see
    :func:`_backpressure_decls`) may be set. Returns ``(decls, step_line,
    step_expr, max_cyc)``; ``step_line``/``step_expr`` are the per-row
    ``@(posedge clk)``-or-``step`` forms used by the TB body, and
    ``max_cyc`` is the per-row result-wait bound (backpressure's 20-50 cycle
    disabled bursts need much more headroom than en_gaps' single-cycle
    gates)."""
    if backpressure and en_gaps:
        raise ValueError("en_gaps and backpressure are mutually exclusive")
    if backpressure:
        return (_backpressure_decls(seed + 12345),
                "            step;\n", "step", 2000)
    if en_gaps:
        return (_en_gap_decls(seed + 12345),
                "            step;\n", "step", 600)
    return ("", "            @(posedge clk);\n", "@(posedge clk)", 200)


def generate_runtime_b_tb(m, k, n, kfold, nfold, weight_matrix,
                          bias_codes=None, shift=0, module_name="cmvu_core",
                          seed=42, name=None, result_width=None,
                          en_gaps=False, b_row_major=False,
                          backpressure=False):
    """Self-checking TB for the runtime-B core: load B, then stream A rows.

    ``en_gaps=True`` drives ``en`` with a randomized (~30% low) pattern
    instead of holding it at 1, to catch clock-gating bugs that only show up
    when ``en`` toggles mid-run (standalone TBs otherwise never exercise
    that, since they hold ``en=1`` throughout).

    ``backpressure=True`` (mutually exclusive with ``en_gaps``) drives long
    (20-50 cycle) disabled bursts instead -- see :func:`_backpressure_decls`.

    ``b_row_major`` selects the B-load beat format to match
    :func:`rtl.generate_core`'s ``b_row_major``: column-major (default, one
    beat per real N column) or row-major (one beat per real K row).
    """
    geo = _geometry.resolve_geometry(m, k, n, kfold, nfold, name,
                                     result_width=result_width)
    w = geo["result_width"]
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(int(m), int(k)), dtype=np.int64)
    Y = reference_rows(A, weight_matrix, bias_codes, shift, w)
    if b_row_major:
        beats = b_load_beats_row_major(weight_matrix, k, n)
        beat_bits = int(n) * 8
    else:
        beats = b_load_beats_col_major(weight_matrix, k, n)
        beat_bits = int(k) * 8

    a_vals = [pack_a_row(A[i], geo["k_chunks_pad"]) for i in range(int(m))]
    e_vals = [pack_res_row(Y[i], w) for i in range(int(m))]
    a_bits, res_bits = geo["a_port_bits"], geo["res_port_bits"]
    a_lines = "\n".join(
        f"        a_mem[{i}] = {a_bits}'h{hex_literal(v, a_bits // 8)};"
        for i, v in enumerate(a_vals))
    e_lines = "\n".join(
        f"        e_mem[{i}] = {res_bits}'h{hex_literal(v, res_bits // 8)};"
        for i, v in enumerate(e_vals))
    b_lines = "\n".join(
        f"        b_mem[{i}] = {beat_bits}'h{hex_literal(v, beat_bits // 8)};"
        for i, v in enumerate(beats))

    en_decls, b_step, step_stmt, max_cyc = _gap_mode(seed, en_gaps, backpressure)
    a_step = b_step

    # Row-major: the block's write port holds its address across a whole
    # 4-beat tile transaction, so the wrapper buffers a tile's 4 rows (fill)
    # writes resident slot 0 live off every row, and buffers slots
    # 1..N_PASSES-1's chunks for the row-band's 4 rows, draining them (4
    # beats each) after the band's 4th row -- geo["row_major_drain_cycles"]
    # cycles, 0 when N_PASSES==1 (no buffer, no stall). Real beats arrive
    # back-to-back within a row-band; between bands that both still carry
    # real rows the TB must idle through the rest of that band (self-
    # generated padding rows, no b_valid needed) plus the drain, or it will
    # overwrite/drop a beat the DUT hasn't consumed yet.
    extra_wait = [0] * len(beats)
    if b_row_major:
        drain_cycles = geo["row_major_drain_cycles"]
        idx = 0
        kv = int(k)
        while idx < kv:
            group_size = min(4, kv - idx)
            last = idx + group_size - 1
            if idx + group_size < kv:
                extra_wait[last] = (4 - group_size) + drain_cycles
            idx += group_size
    wait_lines = "\n".join(
        f"        wait_mem[{i}] = 32'd{w};" for i, w in enumerate(extra_wait))

    return f"""// Generated by gemm-ip-gen cmvu target -- do not edit.
// Runtime-B TB: M={m} K={k} N={n} KFold={kfold} NFold={nfold} shift={shift} seed={seed} en_gaps={en_gaps} backpressure={backpressure}
`timescale 1ns/1ps
module {module_name}_tb;
    reg                    clk = 1'b0;
    reg                    rst = 1'b1;
    reg                    en = 1'b1;
    reg                    in_valid = 1'b0;
    reg  [{a_bits}-1:0]    a_row = '0;
    reg                    b_valid = 1'b0;
    reg  [{beat_bits}-1:0] b_beat = '0;
    wire                   in_ready;
    wire                   out_valid;
    wire [{res_bits}-1:0]  res_row;
{en_decls}
    integer errors = 0;
    integer i;
    integer j;
    integer cyc;
    reg [{a_bits}-1:0]     a_mem [0:{int(m)}-1];
    reg [{res_bits}-1:0]   e_mem [0:{int(m)}-1];
    reg [{beat_bits}-1:0]  b_mem [0:{len(beats)}-1];
    reg [31:0]             wait_mem [0:{max(len(beats), 1)}-1];

    {module_name} dut (
        .clk(clk), .rst(rst), .en(en), .in_valid(in_valid),
        .in_ready(in_ready), .a_row(a_row), .b_beat(b_beat), .b_valid(b_valid),
        .out_valid(out_valid), .res_row(res_row)
    );

    always #5 clk = ~clk;

    initial begin
{a_lines}
{e_lines}
{b_lines}
{wait_lines}
    end

    initial begin
        repeat (4) @(posedge clk);
        rst = 1'b0;
        // Phase 1: load B, one beat per real {'row' if b_row_major else 'column'}.
        // Drive on the falling edge so each beat is stable across the
        // sampling posedge (driving on the same posedge races the DUT and
        // drops the beat). {"Row-major blocks buffer a whole tile (4 rows) before writing it, then replay those rows once per resident N_PASSES slot; between tiles that both still carry real rows, idle through that self-generated fill tail + drain replay (wait_mem, precomputed) before presenting the next tile's first beat, or the DUT would drop it." if b_row_major else "Every block in the target block-row has its own write port, so the wrapper consumes one column beat per enabled cycle (all K_SPATIAL blocks of that row write in parallel) -- no per-column hold."}
        for (i = 0; i < {len(beats)}; i = i + 1) begin
            @(negedge clk);
            b_beat  = b_mem[i];
            b_valid = 1'b1;
{b_step}            if (wait_mem[i] != 0) begin
                b_valid <= 1'b0;
                for (j = 0; j < wait_mem[i]; j = j + 1) {step_stmt};
            end
        end
        @(negedge clk);
        b_valid <= 1'b0;
        // Phase 2: stream A rows
        for (i = 0; i < {int(m)}; i = i + 1) begin
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            // Drive the row off the falling edge (like the B beats) so it is
            // seen on exactly one accepting edge; a blocking assignment at the
            // rising edge races the DUT's flops and can present it twice.
            @(negedge clk);
            a_row = a_mem[i];
            in_valid = 1'b1;
{a_step}            in_valid <= 1'b0;
            // A still-frozen out_valid from the previous row can linger
            // through en gaps (an ena-gated register holds, it does not
            // clear, while en is 0) -- drain it before waiting for this
            // row's own pulse so it is not mistaken for a fresh one.
            while (out_valid) @(posedge clk);
            for (cyc = 0; cyc < {max_cyc} && !out_valid; cyc = cyc + 1)
                @(posedge clk);
            if (!out_valid) begin
                $display("FAIL row %0d: no out_valid within {max_cyc} cycles", i);
                errors = errors + 1;
            end else if (res_row !== e_mem[i]) begin
                $display("FAIL row %0d: got %h expected %h", i, res_row, e_mem[i]);
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("ALL_PASS ({int(m)} rows)");
        else             $display("FAILURES=%0d", errors);
        $finish;
    end
endmodule
"""


def generate_runtime_b_multi_call_tb(m, k, n, kfold, nfold, bias_codes=None,
                                     shift=0, module_name="cmvu_core", seed=42,
                                     name=None, result_width=None,
                                     en_gaps=False, b_row_major=False,
                                     n_calls=3, backpressure=False):
    """Multi-call self-checking TB for the runtime-B core.

    The wrapper is persistent hardware, never re-instantiated between calls,
    but hls4ml streams a NEW B ahead of every call's A rows (a fresh K for
    QK, a fresh V for aV). This drives ``n_calls`` consecutive B-load + M-row
    calls at the SAME dut instance, each with an independently random B and
    A, checking every call's results -- the case :func:`generate_runtime_b_tb`
    (one call only) cannot catch, since a load FSM that only ever re-arms at
    reset still passes a single-call TB.
    """
    geo = _geometry.resolve_geometry(m, k, n, kfold, nfold, name,
                                     result_width=result_width)
    w = geo["result_width"]
    rng = np.random.default_rng(seed)
    a_bits, res_bits = geo["a_port_bits"], geo["res_port_bits"]
    beat_bits = (int(n) if b_row_major else int(k)) * 8
    n_beats = int(k) if b_row_major else int(n)

    if b_row_major:
        drain_cycles = geo["row_major_drain_cycles"]
        extra_wait = [0] * n_beats
        idx = 0
        kv = int(k)
        while idx < kv:
            group_size = min(4, kv - idx)
            last = idx + group_size - 1
            if idx + group_size < kv:
                extra_wait[last] = (4 - group_size) + drain_cycles
            idx += group_size
    else:
        extra_wait = [0] * n_beats

    a_init, e_init, b_init, wait_init = [], [], [], []
    for c in range(int(n_calls)):
        B = rng.integers(-128, 128, size=(int(k), int(n)))
        A = rng.integers(-128, 128, size=(int(m), int(k)), dtype=np.int64)
        Y = reference_rows(A, B, bias_codes, shift, w)
        beats = (b_load_beats_row_major(B, k, n) if b_row_major
                else b_load_beats_col_major(B, k, n))
        for i in range(int(m)):
            av = pack_a_row(A[i], geo["k_chunks_pad"])
            ev = pack_res_row(Y[i], w)
            a_init.append(f"        a_mem[{c}][{i}] = "
                          f"{a_bits}'h{hex_literal(av, a_bits // 8)};")
            e_init.append(f"        e_mem[{c}][{i}] = "
                          f"{res_bits}'h{hex_literal(ev, res_bits // 8)};")
        for i, v in enumerate(beats):
            b_init.append(f"        b_mem[{c}][{i}] = "
                          f"{beat_bits}'h{hex_literal(v, beat_bits // 8)};")
        for i, wv in enumerate(extra_wait):
            wait_init.append(f"        wait_mem[{c}][{i}] = 32'd{wv};")

    en_decls, b_step, step_stmt, max_cyc = _gap_mode(seed, en_gaps, backpressure)
    a_step = b_step

    return f"""// Generated by gemm-ip-gen cmvu target -- do not edit.
// Runtime-B multi-call TB: M={m} K={k} N={n} KFold={kfold} NFold={nfold} shift={shift} seed={seed} en_gaps={en_gaps} backpressure={backpressure} n_calls={n_calls} row_major={b_row_major}
`timescale 1ns/1ps
module {module_name}_tb;
    reg                    clk = 1'b0;
    reg                    rst = 1'b1;
    reg                    en = 1'b1;
    reg                    in_valid = 1'b0;
    reg  [{a_bits}-1:0]    a_row = '0;
    reg                    b_valid = 1'b0;
    reg  [{beat_bits}-1:0] b_beat = '0;
    wire                   in_ready;
    wire                   out_valid;
    wire [{res_bits}-1:0]  res_row;
{en_decls}
    integer errors = 0;
    integer i;
    integer j;
    integer cyc;
    integer callc;
    reg [{a_bits}-1:0]     a_mem [0:{int(n_calls)}-1][0:{int(m)}-1];
    reg [{res_bits}-1:0]   e_mem [0:{int(n_calls)}-1][0:{int(m)}-1];
    reg [{beat_bits}-1:0]  b_mem [0:{int(n_calls)}-1][0:{n_beats}-1];
    reg [31:0]             wait_mem [0:{int(n_calls)}-1][0:{max(n_beats, 1)}-1];

    {module_name} dut (
        .clk(clk), .rst(rst), .en(en), .in_valid(in_valid),
        .in_ready(in_ready), .a_row(a_row), .b_beat(b_beat), .b_valid(b_valid),
        .out_valid(out_valid), .res_row(res_row)
    );

    always #5 clk = ~clk;

    initial begin
{chr(10).join(a_init)}
{chr(10).join(e_init)}
{chr(10).join(b_init)}
{chr(10).join(wait_init)}
    end

    initial begin
        repeat (4) @(posedge clk);
        rst = 1'b0;
        for (callc = 0; callc < {int(n_calls)}; callc = callc + 1) begin
            // Phase 1: load B, one beat per real {'row' if b_row_major else 'column'}
            // (see generate_runtime_b_tb -- same per-call protocol, replayed
            // n_calls times against the SAME persistent dut).
            for (i = 0; i < {n_beats}; i = i + 1) begin
                @(negedge clk);
                b_beat  = b_mem[callc][i];
                b_valid = 1'b1;
{b_step}                if (wait_mem[callc][i] != 0) begin
                    b_valid <= 1'b0;
                    for (j = 0; j < wait_mem[callc][i]; j = j + 1) {step_stmt};
                end
            end
            @(negedge clk);
            b_valid <= 1'b0;
            // Phase 2: stream A rows
            for (i = 0; i < {int(m)}; i = i + 1) begin
                @(posedge clk);
                while (!in_ready) @(posedge clk);
                // Drive the row off the falling edge (like the B beats) so it is
                // seen on exactly one accepting edge; a blocking assignment at the
                // rising edge races the DUT's flops and can present it twice.
                @(negedge clk);
                a_row = a_mem[callc][i];
                in_valid = 1'b1;
{a_step}                in_valid <= 1'b0;
                while (out_valid) @(posedge clk);
                for (cyc = 0; cyc < {max_cyc} && !out_valid; cyc = cyc + 1)
                    @(posedge clk);
                if (!out_valid) begin
                    $display("FAIL call %0d row %0d: no out_valid within {max_cyc} cycles", callc, i);
                    errors = errors + 1;
                end else if (res_row !== e_mem[callc][i]) begin
                    $display("FAIL call %0d row %0d: got %h expected %h", callc, i, res_row, e_mem[callc][i]);
                    errors = errors + 1;
                end
            end
            // Between calls: let this call's last out_valid actually clear on
            // an enabled edge before presenting the next call's first B beat.
            // An en gap can hold out_valid high (an ena-gated register just
            // sitting at its last value) past the edge where this TB samples
            // it, so draining here (not just the per-row drain above) makes
            // sure the DUT's own re-arm (call_done -> loading<=1, gated the
            // same way) has actually happened before the next call starts --
            // otherwise the next call's first beat can race a still-loading
            // wrapper and get dropped.
            while (out_valid) @(posedge clk);
        end
        if (errors == 0) $display("ALL_PASS (%0d calls x {int(m)} rows)", {int(n_calls)});
        else             $display("FAILURES=%0d", errors);
        $finish;
    end
endmodule
"""


def generate_streaming_tb(m, k, n, kfold, nfold, weight_matrix, bias_codes=None,
                          shift=0, module_name="cmvu_core", seed=42, name=None,
                          result_width=None, backpressure=False):
    """Self-checking TB that streams rows back to back.

    Unlike :func:`generate_tb` (one row, then wait for its result), this holds
    ``in_valid`` high and advances ``a_row`` on every accepting edge, so the
    wrapper's take-the-next-row-on-the-last-pass path is exercised. Results
    are checked in order. With ``en`` held high it also checks that
    consecutive accepts are exactly ``K_PASSES * N_GROUPS`` cycles apart.
    """
    geo = _geometry.resolve_geometry(m, k, n, kfold, nfold, name,
                                     result_width=result_width)
    w = geo["result_width"]
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(int(m), int(k)), dtype=np.int64)
    Y = reference_rows(A, weight_matrix, bias_codes, shift, w)
    a_bits, res_bits = geo["a_port_bits"], geo["res_port_bits"]
    a_lines = "\n".join(
        f"        a_mem[{i}] = {a_bits}'h"
        f"{hex_literal(pack_a_row(A[i], geo['k_chunks_pad']), a_bits // 8)};"
        for i in range(int(m)))
    e_lines = "\n".join(
        f"        e_mem[{i}] = {res_bits}'h{hex_literal(pack_res_row(Y[i], w), res_bits // 8)};"
        for i in range(int(m)))
    en_decls = _backpressure_decls(seed + 12345) if backpressure else ""
    period = geo["k_passes"] * geo["n_passes"]
    max_cyc = 20000 if backpressure else 200 * int(m) + 200
    cadence_check = "" if backpressure else f"""
            if (in_valid && in_ready) begin
                if (sent > 0 && cyc - last_acc != {period}) begin
                    $display("FAIL accept %0d: %0d cycles after the previous one, expected {period}",
                             sent, cyc - last_acc);
                    errors = errors + 1;
                end
                last_acc <= cyc;
            end"""
    return f"""// Generated by gemm-ip-gen cmvu target -- do not edit.
// Streaming TB: M={m} K={k} N={n} KFold={kfold} NFold={nfold} shift={shift} seed={seed} backpressure={backpressure}
`timescale 1ns/1ps
module {module_name}_tb;
    reg                    clk = 1'b0;
    reg                    rst = 1'b1;
    reg                    en = 1'b1;
    reg                    in_valid = 1'b0;
    reg  [{a_bits}-1:0]    a_row = '0;
    wire                   in_ready;
    wire                   out_valid;
    wire [{res_bits}-1:0]  res_row;
{en_decls}
    integer errors = 0;
    integer sent = 0;
    integer got = 0;
    integer cyc = 0;
    integer last_acc = 0;
    reg [{a_bits}-1:0]   a_mem [0:{int(m)}-1];
    reg [{res_bits}-1:0] e_mem [0:{int(m)}-1];

    {module_name} dut (
        .clk(clk), .rst(rst), .en(en), .in_valid(in_valid),
        .in_ready(in_ready), .a_row(a_row), .out_valid(out_valid),
        .res_row(res_row)
    );

    always #5 clk = ~clk;

    initial begin
{a_lines}
{e_lines}
    end

    // Sample and drive only on edges where en is high: the wrapper's state
    // (and so in_ready/out_valid) only advances on those edges.
    always @(posedge clk) begin
        if (!rst && en) begin
            cyc <= cyc + 1;{cadence_check}
            if (in_valid && in_ready) begin
                sent <= sent + 1;
                if (sent + 1 < {int(m)}) a_row <= a_mem[sent + 1];
                else                     in_valid <= 1'b0;
            end
            if (out_valid) begin
                if (got >= {int(m)}) begin
                    $display("FAIL: extra result %h", res_row);
                    errors = errors + 1;
                end else if (res_row !== e_mem[got]) begin
                    $display("FAIL row %0d: got %h expected %h", got, res_row, e_mem[got]);
                    errors = errors + 1;
                end
                got <= got + 1;
            end
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;
        a_row = a_mem[0];
        in_valid = 1'b1;
        begin : wait_done
            integer t;
            for (t = 0; t < {max_cyc}; t = t + 1) begin
                @(posedge clk);
                if (got == {int(m)}) disable wait_done;
            end
        end
        repeat (40) @(posedge clk);  // catch any extra result
        if (got != {int(m)}) begin
            $display("FAIL: %0d of {int(m)} results (%0d rows sent)", got, sent);
            errors = errors + 1;
        end
        if (errors == 0) $display("ALL_PASS ({int(m)} rows streamed)");
        else             $display("FAILURES=%0d", errors);
        $finish;
    end
endmodule
"""


def generate_tb(m, k, n, kfold, nfold, weight_matrix, bias_codes=None, shift=0,
                module_name="cmvu_core", seed=42, name=None, result_width=None,
                en_gaps=False, backpressure=False):
    """Emit a self-checking TB for the single-block temporal core.

    Drives one activation row per M beat (``in_ready`` handshake), waits for
    each ``out_valid`` pulse, and compares ``res_row`` against the padded
    reference row from :func:`reference_rows`.

    ``en_gaps=True`` drives ``en`` with a randomized (~30% low) pattern
    instead of holding it at 1 -- see :func:`generate_runtime_b_tb`.
    ``backpressure=True`` (mutually exclusive) drives long disabled bursts
    instead -- see :func:`_backpressure_decls`.
    """
    geo = _geometry.resolve_geometry(m, k, n, kfold, nfold, name,
                                     result_width=result_width)
    w = geo["result_width"]
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(int(m), int(k)), dtype=np.int64)
    Y = reference_rows(A, weight_matrix, bias_codes, shift, w)

    a_vals = [pack_a_row(A[i], geo["k_chunks_pad"]) for i in range(int(m))]
    e_vals = [pack_res_row(Y[i], w) for i in range(int(m))]

    a_bits, res_bits = geo["a_port_bits"], geo["res_port_bits"]
    a_lines = "\n".join(
        f"        a_mem[{i}] = {a_bits}'h{hex_literal(v, a_bits // 8)};"
        for i, v in enumerate(a_vals))
    e_lines = "\n".join(
        f"        e_mem[{i}] = {res_bits}'h{hex_literal(v, res_bits // 8)};"
        for i, v in enumerate(e_vals))

    en_decls, a_step, _step_stmt, max_cyc = _gap_mode(seed, en_gaps, backpressure)

    return f"""// Generated by gemm-ip-gen cmvu target -- do not edit.
// Self-checking TB: M={m} K={k} N={n} KFold={kfold} NFold={nfold} shift={shift} seed={seed} en_gaps={en_gaps} backpressure={backpressure}
`timescale 1ns/1ps
module {module_name}_tb;
    reg                    clk = 1'b0;
    reg                    rst = 1'b1;
    reg                    en = 1'b1;
    reg                    in_valid = 1'b0;
    reg  [{a_bits}-1:0]    a_row = '0;
    wire                   in_ready;
    wire                   out_valid;
    wire [{res_bits}-1:0]  res_row;
{en_decls}
    integer errors = 0;
    integer i;
    integer cyc;
    reg [{a_bits}-1:0]   a_mem [0:{int(m)}-1];
    reg [{res_bits}-1:0] e_mem [0:{int(m)}-1];

    {module_name} dut (
        .clk(clk), .rst(rst), .en(en), .in_valid(in_valid),
        .in_ready(in_ready), .a_row(a_row), .out_valid(out_valid),
        .res_row(res_row)
    );

    always #5 clk = ~clk;

    initial begin
{a_lines}
{e_lines}
    end

    initial begin
        repeat (4) @(posedge clk);
        rst = 1'b0;
        for (i = 0; i < {int(m)}; i = i + 1) begin
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            // Drive the row off the falling edge (like the B beats) so it is
            // seen on exactly one accepting edge; a blocking assignment at the
            // rising edge races the DUT's flops and can present it twice.
            @(negedge clk);
            a_row = a_mem[i];
            in_valid = 1'b1;
{a_step}            in_valid <= 1'b0;
            // A still-frozen out_valid from the previous row can linger
            // through en gaps (an ena-gated register holds, it does not
            // clear, while en is 0) -- drain it before waiting for this
            // row's own pulse so it is not mistaken for a fresh one.
            while (out_valid) @(posedge clk);
            for (cyc = 0; cyc < {max_cyc} && !out_valid; cyc = cyc + 1)
                @(posedge clk);
            if (!out_valid) begin
                $display("FAIL row %0d: no out_valid within {max_cyc} cycles", i);
                errors = errors + 1;
            end else if (res_row !== e_mem[i]) begin
                $display("FAIL row %0d: got %h expected %h", i, res_row, e_mem[i]);
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("ALL_PASS ({int(m)} rows)");
        else             $display("FAILURES=%0d", errors);
        $finish;
    end
endmodule
"""
