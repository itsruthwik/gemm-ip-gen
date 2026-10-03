"""cmvu RTL generation: generated wrapper around vendored ``cmvu_mode1`` blocks.

One core file per GEMM layer, used by every simulator (no ``ifndef SYNTHESIS``
datapath split):

* Public stream: one K-wide activation row per M row in (``a_row``/``in_valid``
  with an ``in_ready`` accept), one N-wide result row out (``res_row``/
  ``out_valid``). The wrapper buffers the row and replays the spec's N-outer /
  K-inner pass schedule itself -- the sequencer lives in the generated
  Verilog (tensor_slice pattern), never in C++.
* Spatial composition is generated here, never taken from ``cmvu_array``:
  ``n_spatial`` broadcast rows x ``k_spatial`` cascade columns of
  ``cmvu_mode1``. Cascade hop and broadcast hop are both 1 cycle; column ``c``
  is skewed by ``c`` cycles; row ``r``'s tail emission is de-skewed so every
  row's lane group lands together at
  ``L + (n_spatial-1) + (k_spatial-1)`` (architecture.md sections 5, 7).
  Every cascade stage stays ``valid`` on every pass; K/N tail tiles are
  zero-padded (never idle a stage).
* Bias is injected exactly once per output lane at the cascade tail (the
  temporal accumulator) of each broadcast row, on that group's first pass;
  upstream cascade stages run ``acc_first = valid`` (acc_term = 0) so the
  cascade carries only the current pass's partial and only the tail
  accumulates across K passes (architecture.md sections 6, 10: sum-cascade
  in space first, then single-slot accumulation in time).
* Weights are baked into each block's real ``cmvu_w_mem`` banks by a sim-only
  ``initial`` block, vendor-conditional: direct hierarchical assignment on
  Icarus, ``force``/``release`` on VCS/Questa (both reject a second driver into
  the ``always_ff`` banks; Icarus cannot force array words). Zero cycles, no
  load FSM, no runtime ``.dat``.

* Reset is synchronous, active-high, at the wrapper boundary: every register
  the wrapper owns resets on ``posedge clk`` only, and Catapult sees just the
  wrapper. The vendored blocks keep their internal async reset, but their
  ``rst`` pin is driven only by the wrapper's clock-synchronous ``rst``.

The single-block temporal core (no spatial replication) is the 1x1 grid
degenerate case of this same generator.
"""
import numpy as np

try:
    from . import geometry as _geometry
    from . import golden as _golden
except ImportError:  # standalone/script import
    import geometry as _geometry
    import golden as _golden


def _const_mux(add, name, width, sel, n_vals, expr):
    """Declare `name` as a mux over the runtime counter `sel` (values 0..n_vals-1), each arm a
    constant-offset slice from expr(v). A single value needs no mux."""
    if n_vals == 1:
        add(f"    wire [{width}-1:0] {name} = {expr(0)};")
        return
    add(f"    reg  [{width}-1:0] {name};")
    add("    always_comb begin")
    add(f"        case ({sel})")
    for v in range(n_vals):
        add(f"            {v}: {name} = {expr(v)};")
    add(f"            default: {name} = '0;")
    add("        endcase")
    add("    end")


# ── Baked weights / init block (zero-cycle sim-only write, no load FSM) ───────


def _block_slot_values(geo, weight_matrix, row, col):
    """Baked per-slot canonical tiles for block ``(row, col)``.

    Slot ``s = n_group*k_passes + k_pass`` (the schedule order every block
    follows); the tile is zero when the block's logical tile is padding.
    """
    tiles = _golden.weight_tiles(
        np.asarray(weight_matrix, dtype=np.int64),
        geo["k"], geo["n"], geo["k_chunks"], geo["n_chunks"])
    by_slot = {}
    for kp, np_ in geo["slot_sequence"]:
        kt = _geometry.ktile_of(kp, col, geo["k_spatial"])
        nt = _geometry.ntile_of(np_, row, geo["n_spatial"])
        if kt < geo["k_chunks"] and nt < geo["n_chunks"]:
            s = _geometry.slot_of(kp, np_, geo["k_passes"])
            by_slot[s] = _golden.pack_tile(tiles[kt][nt])
    return by_slot


def _init_block(block_slots):
    """Sim-only init: one literal per physical bank word for every block."""
    direct, forced, released = [], [], []
    for inst, by_slot in block_slots:
        for s in range(_geometry.MEM_TILES):
            bank = 0 if (s % 2 == 0) else 1
            word = s // 2
            value = by_slot.get(s, 0)
            ref = f"{inst}.u_w_mem.bank{bank}[{word}]"
            literal = f"256'h{_golden.hex_literal(value, 32)}"
            direct.append(f"            {ref} = {literal};")
            forced.append(f"            force {ref} = {literal};")
            released.append(f"            release {ref};")
    return "\n".join([
        "`ifndef SYNTHESIS",
        "    // Baked weight tiles (slot s = n_group*K_PASSES + k_pass);",
        "    // unused slots are zeroed so no read can hit X.",
        "`ifdef __ICARUS__",
        "    initial begin",
        *direct,
        "    end",
        "`else",
        "    initial begin",
        *forced,
        "        #0;",
        *released,
        "    end",
        "`endif",
        "`endif",
    ])


# ── Generated wrapper ─────────────────────────────────────────────────────────


def _sync_reg(add, reg, width, d, q):
    """One en-gated, sync-reset pipeline register ``d -> q``.

    The wrapper's own delay lines use this instead of the vendored
    ``cmvu_regbank``, whose reset is asynchronous: every register the wrapper
    owns resets synchronously, and the async reset stays inside the blocks.
    """
    add(f"    reg [{width}-1:0] {reg};")
    add("    always_ff @(posedge clk) begin")
    add(f"        if (rst)     {reg} <= '0;")
    add(f"        else if (en) {reg} <= {d};")
    add("    end")
    add(f"    assign {q} = {reg};")


def generate_core(m, k, n, kfold, nfold, weight_matrix, bias_codes=None, shift=0,
                  a_signed=True, b_signed=True, module_name="cmvu_core",
                  name=None, runtime_b=False, result_width=None,
                  b_row_major=False):
    """Emit the generated cmvu core Verilog for shape/config.

    ``weight_matrix`` is (K, N) integer codes; ``bias_codes`` one signed int32
    code per output lane at accumulator scale, rounding constant already
    folded in (or None); ``shift`` the requant shift (0..31).

    ``runtime_b=False`` bakes the weights into each block's ``cmvu_w_mem`` via
    the sim-only init block. ``runtime_b=True`` instead exposes a
    ``b_beat``/``b_valid`` stream and drives the block's write port
    (``b_in``/``w_we``/``w_load_start``/``w_col_major``/``w_dual_tile``/
    ``w_tile_sel``) directly; ``weight_matrix`` is ignored in this mode.

    ``b_row_major=False`` (default) is hls4ml's column-major format: beat
    ``n`` is column ``n`` of B, row ``k`` at bits ``[8k +: 8]``, ``n =
    0..N-1``. Every block in the target block-row has its own write port, so
    one column beat is consumed per cycle and written to all K_SPATIAL
    columns of that row in parallel (paired k-passes when ``K_PASSES == 2``);
    once the real ``N`` columns are exhausted the wrapper synthesizes the
    remaining zero columns of the tail/padding n-tiles itself. With
    ``K_PASSES > 2`` the passes beyond the live pair (or single, for an odd
    base slot) are collected in a per-block tile store as the group's 8
    columns arrive and written after them as 4-beat row-major transactions,
    stalling the B stream 4 cycles per staged tile
    (``geometry.runtime_b_load_schedule``).

    ``b_row_major=True`` is hls4ml's row-major format: beat ``k`` is row
    ``k`` of B (real ``K`` beats), column ``n`` at bits ``[8n +: 8]``, port
    width ``N*8``. Every block in the target block-COLUMN has its own write
    port, so a row beat is written to all N_SPATIAL rows of that column in
    parallel; the block's write port is byte-addressed by tile row and
    unlike column-major it holds its target slot address across a whole
    4-beat tile transaction, so beats for different N_PASSES slots of the
    same block cannot be interleaved cycle-by-cycle. The wrapper buffers one
    full tile (4 rows, zero-padded to N_SPATIAL*N_PASSES*8 lanes) as it
    streams in, then replays those 4 rows once per resident N_PASSES slot
    (slot 0 first, ...), so only one tile's worth of data is ever buffered,
    never all of B; when ``N_PASSES == 1`` this degenerates to writing each
    row on arrival with no extra replay pass.
    """
    geo = _geometry.resolve_geometry(m, k, n, kfold, nfold, name,
                                     result_width=result_width)
    if not (0 <= int(shift) <= _geometry.MAX_SHIFT):
        raise ValueError(f"shift must be 0..{_geometry.MAX_SHIFT}, got {shift}")

    ks, ns = geo["k_spatial"], geo["n_spatial"]
    kp_n, np_n = geo["k_passes"], geo["n_passes"]
    a_slice_w = _geometry.K_PHYS * _geometry.IN_WIDTH      # 32
    rw = geo["result_width"]                                # effective W (1..RESULT_WIDTH)
    # The block's y_out lanes are RESULT_WIDTH (32) bits (sign-extended W-bit
    # value); bias lanes are BIAS_WIDTH (32), accumulator scale. The wrapper
    # keeps the y path at RESULT_WIDTH and slices each lane to W on the
    # emitted res_row.
    res_group_w = _geometry.N_PHYS * _geometry.RESULT_WIDTH   # 256
    bias_group_w = _geometry.N_PHYS * _geometry.BIAS_WIDTH    # 256
    out_lanes = geo["n_chunks_pad"] * _geometry.N_PHYS
    a_bits, res_bits = geo["a_row_bits"], geo["res_row_bits"]
    a_port_bits = geo["a_port_bits"]
    res_port_bits = geo["res_port_bits"]
    res_buf_bits = geo["n_chunks_pad"] * res_group_w
    L = _geometry.L
    done_depth = L + (ks - 1) + (ns - 1)

    kp_w = max(1, (kp_n - 1).bit_length())
    np_w = max(1, (np_n - 1).bit_length())

    # Bus layout: {bias_group, tsel, acc_last, acc_first, valid, a_slice}
    b_valid = a_slice_w
    b_first = b_valid + 1
    b_last = b_first + 1
    b_tsel = b_last + 1
    b_bias = b_tsel + 3
    bgw = ns * bias_group_w
    bus_w = b_bias + bgw

    if bias_codes is None:
        bias_val = 0
    else:
        bias_val = _golden.pack_res_row(
            _golden.pad_row(bias_codes, geo["n_chunks_pad"]),
            width=_geometry.BIAS_WIDTH)
    bias_bits = geo["n_chunks_pad"] * bias_group_w
    bias_hex = _golden.hex_literal(bias_val, bias_bits // 8)

    block_slots = [] if runtime_b else [
        (f"u_blk_r{r}_c{c}", _block_slot_values(geo, weight_matrix, r, c))
        for r in range(ns) for c in range(ks)
    ]

    ln = []
    add = ln.append
    add("// Generated by gemm-ip-gen cmvu target -- do not edit.")
    add(f"// M={m} K={k} N={n} KFold={kfold} NFold={nfold} shift={int(shift)}")
    add(f"// grid: {ns} broadcast rows x {ks} cascade cols ({geo['blocks']} blocks); "
        f"k_passes={kp_n}, n_passes={np_n}, slots/block={geo['slots_per_block']}")
    add(f"module {module_name} (")
    add("    input  wire                        clk,")
    add("    input  wire                        rst,")
    add("    input  wire                        en,  // blackbox start (idle/clock-gate hint)")
    add("    input  wire                        in_valid,")
    add("    output wire                        in_ready,")
    add(f"    input  wire [{a_port_bits}-1:0] a_row,  // unpadded: element k at [8k +: 8]")
    if runtime_b and not b_row_major:
        add(f"    input  wire [{k * 8}-1:0]           b_beat,  // column-major: B[:,n], row k at bits [8k +: 8]")
        add("    input  wire                        b_valid,")
    elif runtime_b:
        add(f"    input  wire [{n * 8}-1:0]           b_beat,  // row-major: B[k,:], col n at bits [8n +: 8]")
        add("    input  wire                        b_valid,")
    add("    output wire                        out_valid,")
    add(f"    output wire [{res_port_bits}-1:0] res_row  // unpadded: lane n at [W*n +: W]")
    add(");")
    add("")
    add(f"    localparam int unsigned K_PASSES  = {kp_n};")
    add(f"    localparam int unsigned N_GROUPS  = {np_n};")
    add(f"    localparam int unsigned K_SPATIAL = {ks};")
    add(f"    localparam int unsigned N_SPATIAL = {ns};")
    add(f"    localparam int unsigned L         = {L};")
    add(f"    localparam int unsigned A_SLICE_W = {a_slice_w};")
    add(f"    localparam int unsigned GROUP_W   = {res_group_w};")
    add(f"    localparam int unsigned BIAS_W    = {bias_group_w};")
    add(f"    localparam int unsigned OUT_W     = {rw};")
    ow_bits = max(1, (_geometry.RESULT_WIDTH - 1).bit_length())   # $clog2(RESULT_WIDTH): the block's out_w port
    add(f"    localparam [{ow_bits - 1}:0]        OUT_W_ENC = {ow_bits}'d{rw - 1};")
    add(f"    localparam int unsigned BUS_W     = {bus_w};")
    add(f"    localparam int unsigned B_VALID   = {b_valid};")
    add(f"    localparam int unsigned B_FIRST   = {b_first};")
    add(f"    localparam int unsigned B_LAST    = {b_last};")
    add(f"    localparam int unsigned B_TSEL_LO = {b_tsel};")
    add(f"    localparam int unsigned B_BIAS_LO = {b_bias};")
    add(f"    localparam [4:0] SHIFT_AMT = 5'd{int(shift)};")
    add(f"    localparam [{bias_bits}-1:0] BIAS_WORD = {bias_bits}'h{bias_hex};")
    add("")
    add("    reg                       busy;")
    add(f"    reg  [{kp_w}-1:0]         kp;")
    add(f"    reg  [{np_w}-1:0]         np;")
    add(f"    reg  [{a_bits}-1:0] a_row_buf;")
    add(f"    reg  [{res_buf_bits}-1:0] res_buf;")
    add("    reg                       out_valid_r;")
    add("")
    add("    wire valid = busy;")
    add("    // Last pass of the current row. The next row is taken on this same")
    add("    // edge, so rows issue back to back with no idle cycle between them:")
    add("    // the blocks sample this pass's operands from a_row_buf on the edge")
    add("    // that loads the new row into it.")
    add("    wire row_last = busy && (kp == K_PASSES - 1) && (np == N_GROUPS - 1);")
    add("    // HLS blackbox call-qualified clock: the C++ model advances only")
    add("    // when run() is called (en=1), so the vendored blocks must see the")
    add("    // same clocks or their pipeline drifts across en gaps.")
    add("    // `en` is combinational output of registered control logic and can")
    add("    // change while clk is high; a plain `clk & en` AND gate then glitches")
    add("    // (an extra rising edge on gclk whenever en rises mid-high-phase),")
    add("    // over-advancing the vendored blocks relative to the wrapper's own")
    add("    // `.ena(en)`-gated registers, which only ever sample en at posedge")
    add("    // clk. Sample en on the falling edge instead (a negedge-triggered")
    add("    // register, not a level-sensitive latch, so it is DFT/lint clean)")
    add("    // so it is already stable low-to-low across the whole high phase of")
    add("    // clk -- the standard glitch-free ICG. Force the gate open during")
    add("    // rst so the blocks see clock edges while rst is held.")
    add("    reg en_n;")
    add("    always @(negedge clk) begin")
    add("        if (rst) en_n <= 1'b0;")
    add("        else     en_n <= en;")
    add("    end")
    add("    wire gclk = clk & (en_n | rst);")
    if runtime_b:
        m_w = max(1, (int(m) - 1).bit_length())
        add(f"    localparam int unsigned M_ROWS = {int(m)};")
        slots = kp_n * np_n
        # Set 1 starts on an even slot (slots rounded up to even) so every
        # set's base is even: paired writes and the load schedule are then
        # the same for both sets.
        set_base = slots + (slots % 2)
        dbuf = set_base + slots <= _geometry.MEM_TILES
        add("    // B is reloaded for every frame (hls4ml streams a new B ahead of")
        add("    // each frame's A rows). The block's weight slots hold one or two")
        add("    // sets of this layer's K_PASSES*N_GROUPS tiles: set s starts at")
        add("    // slot s*SET_BASE (even). With two sets (double buffering) the next")
        add("    // frame's B loads into the idle set while the current frame")
        add("    // computes from the other; with one set it loads once the current")
        add("    // frame's last row has issued its last pass. A set is full from")
        add("    // the edge its last write lands until that release.")
        add(f"    localparam int unsigned SET_BASE = {set_base};")
        add(f"    localparam bit DBUF = 1'b{1 if dbuf else 0};")
        add("    reg                       rd_set;   // set the current frame reads")
        add("    reg                       wr_set;   // set the load FSM fills")
        add("    reg  [1:0]                set_full;")
        add("    reg                       loading_d;")
        add("    wire                      arm;      // start loading wr_set")
        add("    wire                      rb_ready; // next row's set is full")
        add("    // Rows accepted so far in this frame (mod M); the frame's last row")
        add("    // is in flight once it wraps back to 0.")
        add(f"    reg  [{m_w}-1:0]          acc_count;")
        add("    wire frame_end = row_last && (acc_count == '0);")
        add("    always_ff @(posedge clk) begin")
        add("        if (rst) begin")
        add("            acc_count <= '0;")
        add("        end else if (en && in_valid && in_ready) begin")
        add("            acc_count <= (acc_count == M_ROWS - 1) ? '0 : "
            "acc_count + 1'b1;")
        add("        end")
        add("    end")
    if runtime_b and not b_row_major:
        # A column beat carries every K pass's slice for its block row, but a
        # block's write port takes at most two tiles per beat (paired column
        # write) and holds its slot across the 8-beat transaction. With more
        # than two K passes the rest are staged: each group's extra tiles are
        # collected in a per-block tile store as the 8 columns stream in, then
        # written as 4-beat row-major transactions (the "drain") while the B
        # stream waits, 4 cycles per staged tile.
        stg = kp_n > 2
        # Live pair (k passes 0,1) needs an even base slot np*K_PASSES; with an
        # odd K_PASSES the odd n-groups start on an odd slot and write pass 0
        # alone, staging one tile more.
        nb_pair = kp_n - 2
        nb_single = kp_n - 1
        nb_max = nb_single if kp_n % 2 else nb_pair
        c_w = max(1, (ks - 1).bit_length())
        r_w = max(1, (ns - 1).bit_length())
        pad_k = geo["k_chunks_pad"] * _geometry.K_PHYS
        zero_pad_bits = pad_k * 8 - k * 8
        add("    reg                       loading;")
        add(f"    reg  [{np_w}-1:0]         ld_np;")
        add(f"    reg  [{r_w}-1:0]          ld_r;")
        add("    reg [2:0]                ld_col;")
        if stg:
            add("    reg                       draining;")
        add("    // hls4ml sends N column beats in order; column n belongs to")
        add("    // ntile nt=n/8 (block row r=nt%N_SPATIAL, group np=nt/N_SPATIAL),")
        add("    // tile-column col=n%8. n_is_real gates the wait on b_valid: once")
        add("    // the real N columns are exhausted the wrapper keeps stepping on")
        add("    // its own, feeding zero columns for the tail/padding n-tiles.")
        add("    // Every block in the target block-row has its own write port, so")
        add("    // all K_SPATIAL blocks of that row take their chunk of the column")
        add("    // beat in the SAME cycle -- one column beat per cycle, no serial")
        add("    // per-block (ld_c) loop.")
        add("    wire [31:0] ld_nt  = ld_np * N_SPATIAL + ld_r;")
        add("    wire [31:0] cur_n  = ld_nt * 32'd8 + {29'd0, ld_col};")
        add(f"    localparam int unsigned N_VAL = {n};")
        add("    wire n_is_real     = cur_n < N_VAL;")
        add("    wire beat_ready    = !n_is_real || b_valid;")
        add("    wire ld_active     = loading && beat_ready"
            + (" && !draining;" if stg else ";"))
        add("    // Register each load beat together with its target block-row/slot")
        add("    // so the blocks' write ports sample a stable (row, slot, beat)")
        add("    // triple (driving it combinationally from counters that change on")
        add("    // the same edge as the beat is a race that drops that beat).")
        add("    reg                       b_valid_r;")
        add(f"    reg  [{k * 8}-1:0]        b_beat_r;")
        add(f"    reg  [{np_w}-1:0]         tgt_np;")
        add(f"    reg  [{r_w}-1:0]          tgt_r;")
        add("    reg                       tgt_first;")
        if stg:
            j_w = max(1, (nb_max - 1).bit_length())
            add("    reg  [2:0]                tgt_col;")
            add(f"    localparam bit KP_ODD = 1'b{kp_n % 2};")
            add("    // Staged-tile count for this group: K_PASSES-2 when the live")
            add("    // write is a pair, K_PASSES-1 when it is a single tile.")
            add("    // A live pair needs an even base slot; set bases are even, so")
            add("    // that is n_group*K_PASSES being even.")
            add("    wire ld_pair  = ~(ld_np[0] & KP_ODD);")
            add("    wire tgt_pair = ~(tgt_np[0] & KP_ODD);")
            add(f"    wire [{j_w}-1:0] nb_last = ld_pair ? {j_w}'d{nb_pair - 1} "
                f": {j_w}'d{nb_single - 1};")
            add(f"    reg  [{j_w}-1:0]          dr_j;")
            add("    reg  [1:0]                dr_row;")
            add("    reg                       dw_valid_r;")
            add(f"    reg  [{j_w}-1:0]          dw_j_r;")
            add("    reg  [1:0]                dw_row_r;")
            add(f"    reg  [{r_w}-1:0]          dw_r_r;")
            add("    reg  [2:0]                dw_slot_r;")
            add("    // Tile store: one canonical row-major 4x8 tile per staged")
            add("    // pass per cascade column, byte (row*8 + col).")
            for c in range(ks):
                for j in range(nb_max):
                    add(f"    reg  [255:0]              stg_c{c}_j{j};")
        if zero_pad_bits > 0:
            add(f"    wire [{pad_k * 8}-1:0] b_col_pad = "
                f"{{{zero_pad_bits}'d0, b_beat_r}};")
        else:
            add(f"    wire [{pad_k * 8}-1:0] b_col_pad = b_beat_r;")
        stg_clr = (" draining <= 1'b0; dr_j <= '0; dr_row <= '0;"
                   " dw_valid_r <= 1'b0;") if stg else ""
        add("    always_ff @(posedge clk) begin")
        add("        if (rst) begin")
        add("            loading <= 1'b1; b_valid_r <= 1'b0;")
        add("            ld_np <= '0; ld_r <= '0; ld_col <= '0;" + stg_clr)
        add("        end else if (en) begin")
        add("            // Re-arm for the next frame's B as soon as its set is")
        add("            // empty (`arm`, see the set control below).")
        add("            if (arm) begin")
        add("                loading <= 1'b1; b_valid_r <= 1'b0;")
        add("                ld_np <= '0; ld_r <= '0; ld_col <= '0;" + stg_clr)
        add("            end else begin")
        add("                b_valid_r <= ld_active;")
        if stg:
            add("                dw_valid_r <= loading && draining;")
            add("                if (loading && draining) begin")
            add("                    // One staged-tile row per cycle into slot")
            add("                    // np*K_PASSES + (live tiles) + j, all blocks")
            add("                    // of block row ld_r in parallel.")
            add("                    dw_j_r <= dr_j; dw_row_r <= dr_row; dw_r_r <= ld_r;")
            add("                    dw_slot_r <= wr_set * SET_BASE + ld_np * K_PASSES"
                " + (ld_pair ? 3'd2 : 3'd1) + dr_j;")
            add("                    if (dr_row != 2'd3) begin")
            add("                        dr_row <= dr_row + 1'b1;")
            add("                    end else begin")
            add("                        dr_row <= '0;")
            add("                        if (dr_j != nb_last) begin")
            add("                            dr_j <= dr_j + 1'b1;")
            add("                        end else begin")
            add("                            dr_j <= '0;")
            add("                            draining <= 1'b0;")
            add("                            if (ld_r != N_SPATIAL - 1) begin")
            add("                                ld_r <= ld_r + 1'b1;")
            add("                            end else begin")
            add("                                ld_r <= '0;")
            add("                                if (ld_np != N_GROUPS - 1) begin")
            add("                                    ld_np <= ld_np + 1'b1;")
            add("                                end else begin")
            add("                                    loading <= 1'b0;")
            add("                                end")
            add("                            end")
            add("                        end")
            add("                    end")
            add("                end")
        add("                if (ld_active) begin")
        add("                    b_beat_r <= n_is_real ? b_beat : '0;")
        add("                    tgt_np <= ld_np; tgt_r <= ld_r;")
        add("                    tgt_first <= (ld_col == '0);")
        if stg:
            add("                    tgt_col <= ld_col;")
        add("                    if (ld_col != 3'd7) begin")
        add("                        ld_col <= ld_col + 1'b1;")
        add("                    end else begin")
        add("                        ld_col <= '0;")
        if stg:
            add("                        // The group's 8 columns are in: drain its")
            add("                        // staged tiles before moving on.")
            add("                        draining <= 1'b1;")
            add("                    end")
            add("                end")
            add("            end")
            add("        end")
            add("    end")
        if not stg:
            add("                        if (ld_r != N_SPATIAL - 1) begin")
            add("                            ld_r <= ld_r + 1'b1;")
            add("                        end else begin")
            add("                            ld_r <= '0;")
            add("                            if (ld_np != N_GROUPS - 1) begin")
            add("                                ld_np <= ld_np + 1'b1;")
            add("                            end else begin")
            add("                                loading <= 1'b0;")
            add("                            end")
            add("                        end")
            add("                    end")
            add("                end")
            add("            end")
            add("        end")
            add("    end")
            add("    assign in_ready = rb_ready;")
        else:
            # The last drain write lands the cycle after loading clears.
            add("    assign in_ready = rb_ready;")
            # Fill the tile store as each registered column beat is written:
            # staged pass p of cascade column c is chunk kt = p*K_SPATIAL + c.
            add("    // Tile store fill: column tgt_col of every staged tile, rows")
            add("    // 0-3 from the pass's 32-bit column slice.")
            add("    always_ff @(posedge clk) begin")
            add("        if (en && b_valid_r) begin")
            for c in range(ks):
                for j in range(nb_max):
                    for pair, first in ((True, 2), (False, 1)):
                        p_ = first + j
                        if p_ > kp_n - 1:
                            continue
                        cond = "tgt_pair" if pair else "!tgt_pair"
                        add(f"            if ({cond}) begin")
                        for row in range(4):
                            add(f"                stg_c{c}_j{j}[({row}*8 + tgt_col)*8 +: 8] <= "
                                f"b_col_pad[{(p_ * ks + c) * 32 + row * 8} +: 8];")
                        add("            end")
            add("        end")
            add("    end")
            add("    // Drain beat: staged tile dw_j_r, row dw_row_r, per cascade column.")
            for c in range(ks):
                last = f"stg_c{c}_j{nb_max - 1}[dw_row_r*64 +: 64]"
                if nb_max == 1:
                    add(f"    wire [63:0] dw_in_c{c} = {last};")
                else:
                    add(f"    wire [63:0] dw_in_c{c} =")
                    for j in range(nb_max - 1):
                        add(f"        (dw_j_r == {j}) ? stg_c{c}_j{j}[dw_row_r*64 +: 64] :")
                    add(f"        {last};")
        add("    // Write-side slot base: slot s = n_group*K_PASSES + k_pass, so")
        add("    // np*K_PASSES is the (even, when paired) base slot for this column.")
        _const_mux(add, "tgt_slot_base", "3", "tgt_np", np_n, lambda v: f"3'd{(v * kp_n) & 7}")
        add("    wire [2:0] ld_tsel = wr_set * SET_BASE + tgt_slot_base;")
        add("    // Per-column write data: kt = kp*K_SPATIAL+c. K_PASSES==1 loads a")
        add("    // single tile per column beat; K_PASSES==2 pairs kp=0/1 into one")
        add("    // beat (b_in[31:0]=kp0, b_in[63:32]=kp1).")
        for c in range(ks):
            lo = f"b_col_pad[{c * 32} +: 32]"
            if kp_n == 1:
                add(f"    wire [63:0] b_in_c{c} = {{32'd0, {lo}}};")
            else:
                hi = f"b_col_pad[{(ks + c) * 32} +: 32]"
                add(f"    wire [63:0] b_in_c{c} = {{{hi}, {lo}}};")
    elif runtime_b:
        c_w = max(1, (ks - 1).bit_length())
        kc_pad = geo["k_chunks_pad"]
        kc_w = max(1, (kc_pad - 1).bit_length())
        pad_n_lanes = geo["n_chunks_pad"] * _geometry.N_PHYS
        pad_bits = pad_n_lanes * 8
        zero_pad_bits_n = pad_bits - n * 8
        add(f"    localparam int unsigned K_CHUNKS_PAD = {kc_pad};")
        add("    // hls4ml sends K real row beats in order; row k belongs to")
        add("    // ktile kt=k/4 (block column c=kt%K_SPATIAL, k_pass kp=kt/K_SPATIAL),")
        add("    // tile-row i=k%4. Every block in the target block-column has its own")
        add("    // write port, and that port holds its write address across a whole")
        add("    // 4-beat tile transaction (unlike column-major's per-column ports).")
        add(f"    reg                       loading;")
        add(f"    reg  [{kc_w}-1:0]         ld_kt;")
        add("    reg  [1:0]                ld_i;     // row within tile (0..3)")
        add("    wire [31:0] c_of_kt  = ld_kt % K_SPATIAL;")
        add("    wire [31:0] kp_of_kt = ld_kt / K_SPATIAL;")
        add("    wire [31:0] ld_k     = ld_kt * 32'd4 + {30'd0, ld_i};")
        add(f"    localparam int unsigned K_VAL = {k};")
        add("    wire k_is_real   = ld_k < K_VAL;")
        add("    wire beat_ready  = !k_is_real || b_valid;")
        if zero_pad_bits_n > 0:
            add(f"    wire [{pad_bits}-1:0] b_row_pad = "
                f"{{{zero_pad_bits_n}'d0, b_beat}};")
        else:
            add(f"    wire [{pad_bits}-1:0] b_row_pad = b_beat;")
        if np_n == 1:
            add("    // N_PASSES==1: every block's only resident slot (kp) is")
            add("    // written live off each incoming row -- no buffer at all,")
            add("    // just the registered-beat/target pipeline stage every")
            add("    // runtime-B load path uses to avoid a same-edge write race.")
            add("    wire ld_active = loading && beat_ready;")
            add("    reg                       b_valid_r;")
            add(f"    reg  [{pad_bits}-1:0]     b_beat_r;")
            add(f"    reg  [{c_w}-1:0]          tgt_c;")
            add(f"    reg  [{kp_w}-1:0]         tgt_kp;")
            add("    reg                       tgt_first;")
            add("    always_ff @(posedge clk) begin")
            add("        if (rst) begin")
            add("            loading <= 1'b1; b_valid_r <= 1'b0;")
            add("            ld_kt <= '0; ld_i <= '0;")
            add("        end else if (en) begin")
            add("            // Re-arm for the next frame's B as soon as its set is")
            add("            // empty (`arm`, see the set control below).")
            add("            if (arm) begin")
            add("                loading <= 1'b1; b_valid_r <= 1'b0;")
            add("                ld_kt <= '0; ld_i <= '0;")
            add("            end else begin")
            add("                b_valid_r <= ld_active;")
            add("                if (ld_active) begin")
            add("                    b_beat_r <= k_is_real ? b_row_pad : '0;")
            add(f"                    tgt_c <= c_of_kt[{c_w}-1:0];")
            add(f"                    tgt_kp <= kp_of_kt[{kp_w}-1:0];")
            add("                    tgt_first <= (ld_i == '0);")
            add("                    if (ld_i != 2'd3) begin")
            add("                        ld_i <= ld_i + 1'b1;")
            add("                    end else begin")
            add("                        ld_i <= '0;")
            add("                        if (ld_kt != K_CHUNKS_PAD - 1) begin")
            add("                            ld_kt <= ld_kt + 1'b1;")
            add("                        end else begin")
            add("                            loading <= 1'b0;")
            add("                        end")
            add("                    end")
            add("                end")
            add("            end")
            add("        end")
            add("    end")
            add("    assign in_ready = rb_ready;")
            add("    wire [2:0] ld_tsel = wr_set * SET_BASE + tgt_kp;")
            add("    // nt = r when N_PASSES==1 (n_chunks_pad == N_SPATIAL).")
            for r in range(ns):
                add(f"    wire [63:0] b_in_r{r} = b_beat_r[{r}*64 +: 64];")
        else:
            extra_bits = (np_n - 1) * ns * 64
            add("    // N_PASSES>1: the block's write port can't interleave its")
            add("    // N_PASSES resident slots beat-by-beat (address-hold, see")
            add("    // above), so slot 0 is written live off every incoming row")
            add("    // (like N_PASSES==1) while slots 1..N_PASSES-1's chunks for")
            add("    // this row are buffered; after the row-band's 4th row the")
            add("    // buffered slots are drained (4 beats each), stalling new")
            add("    // input -- geo['row_major_drain_cycles'] = 4*(N_PASSES-1)")
            add("    // cycles, the same schedule golden.py's TB precomputes.")
            add("    reg                       ld_phase; // 0=fill(+live slot0), 1=drain")
            add(f"    reg  [{np_w}-1:0]         ld_npd;   // slot being drained (1..N_PASSES-1)")
            add("    wire fill_active  = loading && !ld_phase && beat_ready;")
            add("    wire drain_active = loading && ld_phase;")
            add(f"    reg  [{extra_bits}-1:0]   buf_row0, buf_row1, buf_row2, buf_row3;")
            add("    reg                       b_valid_r;")
            add(f"    reg  [{pad_bits}-1:0]     b_beat_r;")
            add(f"    reg  [{c_w}-1:0]          tgt_c;")
            add(f"    reg  [{kp_w}-1:0]         tgt_kp;")
            add(f"    reg  [{np_w}-1:0]         tgt_npd;")
            add("    reg  [1:0]                tgt_i;")
            add("    reg                       tgt_first;")
            add("    always_ff @(posedge clk) begin")
            add("        if (rst) begin")
            add("            loading <= 1'b1; b_valid_r <= 1'b0;")
            add("            ld_kt <= '0; ld_phase <= 1'b0; ld_i <= '0; "
                f"ld_npd <= {np_w}'d1;")
            add("        end else if (en) begin")
            add("            // Re-arm for the next frame's B as soon as its set is")
            add("            // empty (`arm`, see the set control below).")
            add("            if (arm) begin")
            add("                loading <= 1'b1; b_valid_r <= 1'b0;")
            add(f"                ld_kt <= '0; ld_phase <= 1'b0; ld_i <= '0; "
                f"ld_npd <= {np_w}'d1;")
            add("            end else begin")
            add("                b_valid_r <= fill_active || drain_active;")
            add("                if (fill_active) begin")
            add("                    b_beat_r <= k_is_real ? b_row_pad : '0;")
            add("                    case (ld_i)")
            add(f"                        2'd0: buf_row0 <= k_is_real ? "
                f"b_row_pad[{ns}*64 +: {extra_bits}] : '0;")
            add(f"                        2'd1: buf_row1 <= k_is_real ? "
                f"b_row_pad[{ns}*64 +: {extra_bits}] : '0;")
            add(f"                        2'd2: buf_row2 <= k_is_real ? "
                f"b_row_pad[{ns}*64 +: {extra_bits}] : '0;")
            add(f"                        default: buf_row3 <= k_is_real ? "
                f"b_row_pad[{ns}*64 +: {extra_bits}] : '0;")
            add("                    endcase")
            add(f"                    tgt_c <= c_of_kt[{c_w}-1:0];")
            add(f"                    tgt_kp <= kp_of_kt[{kp_w}-1:0];")
            add(f"                    tgt_npd <= {np_w}'d0; tgt_i <= ld_i;")
            add("                    tgt_first <= (ld_i == '0);")
            add("                    if (ld_i != 2'd3) begin")
            add("                        ld_i <= ld_i + 1'b1;")
            add("                    end else begin")
            add("                        ld_i <= '0; ld_phase <= 1'b1;")
            add("                    end")
            add("                end else if (drain_active) begin")
            add(f"                    tgt_c <= c_of_kt[{c_w}-1:0];")
            add(f"                    tgt_kp <= kp_of_kt[{kp_w}-1:0];")
            add("                    tgt_npd <= ld_npd; tgt_i <= ld_i;")
            add("                    tgt_first <= (ld_i == '0);")
            add("                    if (ld_i != 2'd3) begin")
            add("                        ld_i <= ld_i + 1'b1;")
            add("                    end else begin")
            add("                        ld_i <= '0;")
            add("                        if (ld_npd != N_GROUPS - 1) begin")
            add("                            ld_npd <= ld_npd + 1'b1;")
            add("                        end else begin")
            add(f"                            ld_npd <= {np_w}'d1; ld_phase <= 1'b0;")
            add("                            if (ld_kt != K_CHUNKS_PAD - 1) begin")
            add("                                ld_kt <= ld_kt + 1'b1;")
            add("                            end else begin")
            add("                                loading <= 1'b0;")
            add("                            end")
            add("                        end")
            add("                    end")
            add("                end")
            add("            end")
            add("        end")
            add("    end")
            add("    assign in_ready = rb_ready;")
            _const_mux(add, "tgt_slot_base", "3", "tgt_npd", np_n, lambda v: f"3'd{(v * kp_n) & 7}")
            add("    wire [2:0] ld_tsel = wr_set * SET_BASE + tgt_slot_base + tgt_kp;")
            add(f"    wire [{extra_bits}-1:0] buf_sel = (tgt_i == 2'd0) ? buf_row0 :")
            add("                              (tgt_i == 2'd1) ? buf_row1 :")
            add("                              (tgt_i == 2'd2) ? buf_row2 : buf_row3;")
            add("    // Slot 0 reads the live registered beat; slots 1..N_PASSES-1")
            add("    // read the buffered (drain) data selected by tgt_i.")
            for r in range(ns):
                add(f"    wire [63:0] b_in_r{r} = (tgt_npd == {np_w}'d0) ? "
                    f"b_beat_r[{r}*64 +: 64] :")
                add(f"                            buf_sel[(tgt_npd-{np_w}'d1)*{ns}*64 "
                    f"+ {r}*64 +: 64];")
    else:
        add("    assign in_ready = ~busy | row_last;")
    if runtime_b:
        add("    // Arm the load FSM when its set is empty. loading_d keeps it off")
        add("    // for the cycle the last write lands (set_full not yet set).")
        add("    assign arm = !loading && !loading_d && !set_full[wr_set];")
        add("    always_ff @(posedge clk) begin")
        add("        if (rst) begin")
        add("            rd_set <= 1'b0; wr_set <= 1'b0; set_full <= 2'b00;")
        add("            loading_d <= 1'b0;")
        add("        end else if (en) begin")
        add("            loading_d <= loading;")
        add("            // Load finished: the last write lands on this edge.")
        add("            if (loading_d && !loading) begin")
        add("                set_full[wr_set] <= 1'b1;")
        add("                if (DBUF) wr_set <= ~wr_set;")
        add("            end")
        add("            // The frame's last row issues its last pass: release its set.")
        add("            if (busy && frame_end) begin")
        add("                set_full[rd_set] <= 1'b0;")
        add("                if (DBUF) rd_set <= ~rd_set;")
        add("            end")
        add("        end")
        add("    end")
        add("    // The next row reads rd_set, or at a frame boundary the next")
        add("    // frame's set (the other one; with one set it must reload first).")
        add("    wire next_set_full = (busy && frame_end)")
        add("        ? (DBUF && set_full[~rd_set]) : set_full[rd_set];")
        add("    assign rb_ready = (~busy | row_last) & next_set_full;")
    add("    // Framing MUST be gated by valid: cmvu_mode1's `done` is an")
    add("    // unconditional delay of acc_last, so an idle-time acc_last would")
    add("    // latch `done` permanently (the cmvu_array.sv single-pass bug).")
    add("    wire first_pulse = busy && (kp == '0);")
    add("    wire last_pulse  = busy && (kp == K_PASSES - 1);")
    # Slot base np*K_PASSES as a constant per group (see _const_mux): with K_PASSES not a power
    # of two the multiply becomes a DSP on the tile_sel path.
    _const_mux(add, "np_slot_base", "3", "np", np_n, lambda v: f"3'd{(v * kp_n) & 7}")
    if runtime_b:
        add("    wire [2:0] tsel_now = (rd_set*SET_BASE + np_slot_base + kp) & 3'h7;")
    else:
        add("    wire [2:0] tsel_now = (np_slot_base + kp) & 3'h7;")
    add("")
    add("    // Per-column entry buses + cascade-alignment skew (c cycles).")
    for c in range(ks):
        # Pass kp selects a constant slice, so emit a case over kp with literal offsets rather
        # than a_row_buf[(kp*K_SPATIAL + c)*A_SLICE_W +: ...]: with K_SPATIAL not a power of two
        # synthesis builds a real multiplier for kp*K_SPATIAL (VTR puts it on a DSP), and that
        # multiply feeds a wide variable part-select, the critical path of those folds.
        _const_mux(add, f"a_slice_c{c}", "A_SLICE_W", "kp", kp_n,
                   lambda v, c=c: f"a_row_buf[{(v * ks + c) * _geometry.K_PHYS * _geometry.IN_WIDTH} +: A_SLICE_W]")
        if c == ks - 1:
            # Bias is owned by the cascade tail (the temporal accumulator);
            # upstream blocks run acc_term=0 every pass so their cascade_out
            # is that pass's local partial only.
            _const_mux(add, f"bias_c{c}", str(bgw), "np", np_n,
                       lambda v: f"BIAS_WORD[{v * geo['n_spatial'] * _geometry.N_PHYS * _geometry.BIAS_WIDTH} +: {bgw}]")
        else:
            add(f"    wire [{bgw}-1:0] bias_c{c} = {bgw}'h0;")
        entry = ("{" + f"bias_c{c}, tsel_now, last_pulse, first_pulse, valid, "
                 f"a_slice_c{c}" + "}")
        add(f"    wire [BUS_W-1:0] entry_c{c} = {entry};")
        if c == 0:
            add(f"    wire [BUS_W-1:0] col_c0 = entry_c0;")
        else:
            add(f"    wire [BUS_W-1:0] col_skew_c{c} [0:{c}];")
            add(f"    assign col_skew_c{c}[0] = entry_c{c};")
            for i in range(c):
                _sync_reg(add, f"u_skew_c{c}_s{i}", "BUS_W",
                          f"col_skew_c{c}[{i}]", f"col_skew_c{c}[{i+1}]")
            add(f"    wire [BUS_W-1:0] col_c{c} = col_skew_c{c}[{c}];")
    add("")
    add("    // Broadcast rows: one forwarding register per row hop.")
    for c in range(ks):
        for r in range(ns):
            add(f"    wire [BUS_W-1:0] bus_r{r}_c{c};")
        add(f"    assign bus_r0_c{c} = col_c{c};")
        for r in range(1, ns):
            _sync_reg(add, f"u_bcast_r{r}_c{c}", "BUS_W",
                      f"bus_r{r-1}_c{c}", f"bus_r{r}_c{c}")
    add("")
    add("    // Block grid: cascade along columns. cascade_out may only drive the")
    add("    // next block's cascade_in; unused endpoint pins (head cascade_in,")
    add("    // tail cascade_out) are left dangling -- never tied to 0/1. The head")
    add("    // sets CASCADE_EN=0 so its dangling cascade_in is ignored.")
    for c in range(ks):
        for r in range(ns):
            bus = f"bus_r{r}_c{c}"
            add(f"    wire [{res_group_w}-1:0] y_blk_r{r}_c{c};")
            add(f"    wire done_blk_r{r}_c{c};")
            add(f"    wire yv_blk_r{r}_c{c};")
            if c < ks - 1:
                add(f"    wire [{_geometry.N_PHYS * _geometry.ACC_WIDTH}-1:0] "
                    f"casc_blk_r{r}_c{c};")
            tail = (c == ks - 1)
            params = " #(.CASCADE_EN(1'b0))" if c == 0 else ""
            add(f"    cmvu_mode1{params} u_blk_r{r}_c{c} (")
            # The block's reset is async internally; the wrapper's rst is
            # synchronous to clk, so it behaves as a sync reset from outside.
            add("        .clk(gclk), .rst(rst),")
            if tail:
                # Tail: temporal accumulation (reset at each group's pass 0)
                # and the row's bias lanes (added exactly once).
                add(f"        .valid({bus}[B_VALID]), "
                    f".acc_first({bus}[B_FIRST]), .acc_last({bus}[B_LAST]),")
            else:
                # Upstream cascade stage: reset the accumulator every pass so
                # cascade_out carries only this pass's local partial.
                add(f"        .valid({bus}[B_VALID]), "
                    f".acc_first({bus}[B_VALID]), .acc_last({bus}[B_VALID]),")
            if runtime_b and not b_row_major and kp_n > 2:
                # Live column writes (paired when the base slot is even) and
                # staged-tile drain writes (row-major) share the write port;
                # they are never in the same cycle.
                hit = f"(b_valid_r && (tgt_r == {r}))"
                dhit = f"(dw_valid_r && (dw_r_r == {r}))"
                add(f"        .a_in({bus}[0 +: A_SLICE_W]), "
                    f".b_in(dw_valid_r ? dw_in_c{c} : b_in_c{c}),")
                add(f"        .w_we({hit} || {dhit}),")
                add(f"        .w_load_start(({hit} && tgt_first) || "
                    f"({dhit} && dw_row_r == 2'd0)),")
                add("        .w_col_major(~dw_valid_r), "
                    ".w_dual_tile(~dw_valid_r & tgt_pair),")
                add(f"        .tile_sel({bus}[B_TSEL_LO +: 3]), "
                    f".w_tile_sel(dw_valid_r ? dw_slot_r : ld_tsel),")
            elif runtime_b and not b_row_major:
                add(f"        .a_in({bus}[0 +: A_SLICE_W]), .b_in(b_in_c{c}),")
                add(f"        .w_we(b_valid_r && (tgt_r == {r})),")
                add(f"        .w_load_start(b_valid_r && (tgt_r == {r}) "
                    f"&& tgt_first),")
                add(f"        .w_col_major(1'b1), "
                    f".w_dual_tile(1'b{1 if kp_n == 2 else 0}),")
                add(f"        .tile_sel({bus}[B_TSEL_LO +: 3]), .w_tile_sel(ld_tsel),")
            elif runtime_b:
                add(f"        .a_in({bus}[0 +: A_SLICE_W]), .b_in(b_in_r{r}),")
                add(f"        .w_we(b_valid_r && (tgt_c == {c})),")
                add(f"        .w_load_start(b_valid_r && (tgt_c == {c}) "
                    f"&& tgt_first),")
                add("        .w_col_major(1'b0), .w_dual_tile(1'b0),")
                add(f"        .tile_sel({bus}[B_TSEL_LO +: 3]), .w_tile_sel(ld_tsel),")
            else:
                add(f"        .a_in({bus}[0 +: A_SLICE_W]), .b_in(64'd0),")
                add("        .w_we(1'b0), .w_load_start(1'b0), "
                    ".w_col_major(1'b0), .w_dual_tile(1'b0),")
                add(f"        .tile_sel({bus}[B_TSEL_LO +: 3]), .w_tile_sel(3'd0),")
            add(f"        .a_signed(1'b{1 if a_signed else 0}), "
                f".b_signed(1'b{1 if b_signed else 0}), .shift_amt(SHIFT_AMT),")
            add("        .out_w(OUT_W_ENC),")
            # cascade_out -> next block's cascade_in only; the chain head's
            # cascade_in and the chain tail's cascade_out are left dangling.
            if c == 0:
                add("        .cascade_in(),")
            else:
                add(f"        .cascade_in(casc_blk_r{r}_c{c-1}),")
            if tail:
                add(f"        .bias_in({bus}[B_BIAS_LO + {r * bias_group_w} "
                    f"+: BIAS_W]),")
            else:
                add("        .bias_in('0),")
            if tail:
                add(f"        .y_out(y_blk_r{r}_c{c}), .cascade_out(),")
            else:
                add(f"        .y_out(y_blk_r{r}_c{c}), "
                    f".cascade_out(casc_blk_r{r}_c{c}),")
            add(f"        .y_valid(yv_blk_r{r}_c{c}), .done(done_blk_r{r}_c{c})")
            add("    );")
    add("")
    add("    // Output de-skew: row r's tail emits r cycles after row 0's; add")
    add(f"    // (N_SPATIAL-1-r) hops so all rows align at L + (ns-1) + (ks-1).")
    add(f"    wire [{res_group_w}-1:0] y_align [0:{ns - 1}];")
    for r in range(ns):
        hops = ns - 1 - r
        add(f"    wire [{res_group_w}-1:0] y_tail_r{r} = y_blk_r{r}_c{ks - 1};")
        if hops == 0:
            add(f"    assign y_align[{r}] = y_tail_r{r};")
        else:
            add(f"    wire [{res_group_w}-1:0] y_ds_r{r} [0:{hops}];")
            add(f"    assign y_ds_r{r}[0] = y_tail_r{r};")
            for i in range(hops):
                _sync_reg(add, f"u_yds_r{r}_s{i}", "GROUP_W",
                          f"y_ds_r{r}[{i}]", f"y_ds_r{r}[{i+1}]")
            add(f"    assign y_align[{r}] = y_ds_r{r}[{hops}];")
    hops_d = ns - 1
    add(f"    wire done_tail0 = done_blk_r0_c{ks - 1};")
    if hops_d == 0:
        add("    wire done_align = done_tail0;")
    else:
        add(f"    wire done_ds [0:{hops_d}];")
        add("    assign done_ds[0] = done_tail0;")
        for i in range(hops_d):
            _sync_reg(add, f"u_dds_s{i}", "1",
                      f"done_ds[{i}]", f"done_ds[{i+1}]")
        add(f"    wire done_align = done_ds[{hops_d}];")
    add("")
    add(f"    // Group-index pipe: aligned done is {done_depth} cycles after its")
    add("    // last pass is issued (L + column skew + row de-skew).")
    add(f"    reg [{np_w}-1:0] grp_pipe [0:{done_depth - 1}];")
    add("    integer gi;")
    add("    always_ff @(posedge clk) begin")
    add("        if (rst) begin")
    add(f"            for (gi = 0; gi < {done_depth}; gi = gi + 1) grp_pipe[gi] <= '0;")
    add("        end else if (en) begin")
    add("            grp_pipe[0] <= busy ? np : '0;")
    add(f"            for (gi = 1; gi < {done_depth}; gi = gi + 1) "
        "grp_pipe[gi] <= grp_pipe[gi-1];")
    add("        end")
    add("    end")
    add("")
    add("    // Accept one row, then replay every (n_group, k_pass) pair")
    add("    // back-to-back; the next row is taken on the last pass (in_ready).")
    add("    // Single always_ff per variable: VCS rejects")
    add("    // always_ff variables with more than one procedural driver.")
    add("    integer ri;")
    add("    always_ff @(posedge clk) begin")
    add("        if (rst) begin")
    add("            busy        <= 1'b0;")
    add("            kp          <= '0;")
    add("            np          <= '0;")
    add("            a_row_buf   <= '0;")
    add("            out_valid_r <= 1'b0;")
    add("            res_buf     <= '0;")
    add("        end else if (en) begin")
    add("            out_valid_r <= 1'b0;")
    add("            if (in_valid && in_ready) begin")
    if a_port_bits < a_bits:
        add(f"                a_row_buf <= {{{a_bits - a_port_bits}'d0, a_row}};")
    else:
        add("                a_row_buf <= a_row;")
    add("                kp        <= '0;")
    add("                np        <= '0;")
    add("                busy      <= 1'b1;")
    add("            end else if (busy && kp == K_PASSES - 1) begin")
    add("                kp <= '0;")
    add("                if (np == N_GROUPS - 1) begin")
    add("                    np   <= '0;")
    add("                    busy <= 1'b0;")
    add("                end else begin")
    add("                    np <= np + 1'b1;")
    add("                end")
    add("            end else if (busy) begin")
    add("                kp <= kp + 1'b1;")
    add("            end")
    add("            if (done_align) begin")
    # Constant base per group (case over the group index), not grp*N_SPATIAL: see _const_mux.
    add(f"                case (grp_pipe[{done_depth - 1}])")
    for g in range(np_n):
        add(f"                    {g}: for (ri = 0; ri < N_SPATIAL; ri = ri + 1)")
        add(f"                        res_buf[({g * geo['n_spatial']} + ri)*GROUP_W +: GROUP_W] <= y_align[ri];")
    add("                    default: ;")
    add("                endcase")
    add(f"                if (grp_pipe[{done_depth - 1}] == N_GROUPS - 1) "
        "out_valid_r <= 1'b1;")
    add("            end")
    add("        end")
    add("    end")
    add("")
    add("    assign out_valid = out_valid_r;")
    add("    // Emit the effective W-bit value per lane: slice each RESULT_WIDTH-bit")
    add("    // (sign-extended) result lane down to its low OUT_W bits. Only the")
    add(f"    // first {n} (real) lanes are exposed at the port -- the padded")
    add(f"    // tail lanes ({out_lanes - int(n)} of them, up to n_chunks_pad*N_PHYS)")
    add("    // live only in the internal res_buf.")
    add("    genvar gs;")
    add("    generate")
    add(f"        for (gs = 0; gs < {int(n)}; gs = gs + 1) begin : g_res_slice")
    add(f"            assign res_row[gs*OUT_W +: OUT_W] = "
        f"res_buf[gs*{_geometry.RESULT_WIDTH} +: OUT_W];")
    add("        end")
    add("    endgenerate")
    add("")
    if not runtime_b:
        add(_init_block(block_slots))
    add("endmodule")
    return "\n".join(ln) + "\n"
