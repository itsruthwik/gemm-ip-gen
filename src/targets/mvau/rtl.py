"""mvau RTL emission: the thin shim around FINN's ``mvu_vvu_axi``.

Generalizes the cosim-validated ``temp_space/mvau-spike/mvau_bb.v``. One shim =
one MVU tile: it instantiates ``mvu_vvu_axi`` with the tile's folding parameters
and adapts the FINN AXIS interface to the Vitis-HLS RTL-blackbox contract:

  * active-high ``ap_rst``  ->  FINN's active-low ``ap_rst_n`` (inverted)
  * ``ap_ce`` global stall gating the three handshakes (safe on a fully
    backpressured module: no beat transfers when ce=0, dsp_zero injects bubbles)
  * ap_fifo ports the JSON binds (``*_dout/*_empty_n/*_read``, ``*_din/*_full_n/*_write``)
    -- AXIS ``TVALID/TREADY`` bind directly to ap_fifo ``empty_n/read`` & ``full_n/write``

The module name MUST equal the JSON ``c_function_name`` -- Vitis instantiates the
blackbox by the C function name, not ``rtl_top_module_name`` (learned in cosim).

Self-contained (only ``geometry`` as a sibling import).
"""

from . import geometry as _geom
from . import weightpack as _wpack


def _tile(shape, tile=None, **plan_kwargs):
    """Return the per-instance MVU tile params for *shape*, computing the plan
    if a precomputed *tile* dict is not supplied."""
    if tile is not None:
        return tile
    return _geom.resolve_plan(shape, **plan_kwargs)["tile"]


def _requant_lanes(t, n_lanes, raw_exprs, bias_codes, reg_prefix, bias_index_exprs=None):
    """Return Verilog for a bank of *n_lanes* per-column requantize stages: one
    pipeline stage each, sitting after the K-tile partial-sum adder (if the tile's
    lane has more than one raw partial) and producing the narrow, final,
    ``out_width``-bit-per-lane registers this bank's caller concatenates into
    ``p_din``.

    ``raw_exprs[i]`` is either one Verilog expression (a signed ``ACCU_WIDTH``-bit
    value, k_tiles==1) or a list of such expressions to sum first (k_tiles>1,
    mirrors the C twin's raw += loop order). ``bias_codes`` (or None) is the FULL
    bias ROM this bank shares -- at product_frac scale, from
    ``weightpack.bias_acc_codes``/``bias_codes_for_tile`` (the same source of truth
    the C twin bakes) -- addressed per lane by ``bias_index_exprs[i]`` (a Verilog
    expression, e.g. a runtime ``nf_cnt*PE+pe`` when a shim's PE lanes carry
    different output columns on different cycles; a compile-time literal ``i`` when
    they don't). Defaults to the lane's own static index when *bias_index_exprs* is
    omitted. Padded lanes' bias entries may be 0 -- their requantized value is
    unused downstream (dropped by the HLS-side unpack), consistent with today's
    raw-beat behavior for padding.

    Returns ``(decls, reg_names)``: the Verilog text (bias ROM if any + one
    combinational sum/round/shift + one register per lane) and the list of
    per-lane register names (each ``out_width`` bits wide) to slice into ``p_din``.
    """
    accu = t["accu_width"]
    out_width = t["output_width"]
    shift = t["product_frac"] - t["output_frac"]
    has_bias = bias_codes is not None
    codes = list(bias_codes) if has_bias else []
    bias_width = _geom.rv_width(accu, has_bias)
    lines = []
    rom_name = f"{reg_prefix}_bias_rom"
    if has_bias:
        lines.append(_wpack.bias_verilog_rom(rom_name, codes, bias_width))
    reg_names = []
    for i in range(n_lanes):
        raw = raw_exprs[i]
        parts = raw if isinstance(raw, (list, tuple)) else [raw]
        sum_expr = " + ".join(f"$signed({p})" for p in parts) if len(parts) > 1 else f"$signed({parts[0]})"
        idx_expr = (bias_index_exprs[i] if bias_index_exprs is not None else str(i))
        bias_term = f" + $signed({rom_name}[{idx_expr}])" if has_bias else ""
        biased = f"({sum_expr}){bias_term}"
        reg = f"{reg_prefix}_{i}"
        reg_names.append(reg)
        lines.append(f"    wire signed [{bias_width - 1}:0] {reg}_biased = {biased};")
        if shift > 0:
            # RHS operands are self-determined to this wire's declared width, so the
            # signed {reg}_biased is sign-extended automatically -- no manual
            # sign-extension concatenation needed.
            rnd_width = bias_width + 1
            lines.append(f"    wire signed [{rnd_width - 1}:0] {reg}_rnd = "
                         f"{reg}_biased + {rnd_width}'sd{1 << (shift - 1)};")
            shifted = f"({reg}_rnd >>> {shift})"
        elif shift < 0:
            shifted = f"({reg}_biased <<< {-shift})"
        else:
            shifted = f"{reg}_biased"
        # Combinational (not registered): p_din must be valid the same cycle the
        # underlying tile asserts out_tvalid/drives p_write, so this narrow code
        # cannot lag the raw beat by a pipeline stage without also retiming the
        # AXI-style valid/ready handshake around it -- out of scope here. Assigning
        # a wider signed expression to this out_width-bit wire truncates to the low
        # out_width bits, which is exactly the intended wrap.
        lines.append(f"    wire signed [{out_width - 1}:0] {reg} = {shifted};   // wrap: low {out_width} bits")
    return "\n".join(lines) + "\n", reg_names


def generate_shim(shape, module_name="mvau_core", force_behavioral=True,
                  tile=None, weights_in_core=False, init_file=None,
                  init_files=None, n_tiles=1, k_tiles=1, bias_codes=None,
                  raw_k=None, raw_n=None, max_inflight=None, **plan_kwargs):
    """Emit the shim Verilog wrapping ``mvu_vvu_axi`` for one MVU tile.

    ``module_name`` must match the blackbox C function name. ``force_behavioral``
    bakes ``FORCE_BEHAVIORAL`` into the instance (1 for unisim-free cosim, 0 for
    real DSP primitives downstream).

    ``weights_in_core`` selects the weight-stationary shim: each of ``n_tiles``
    FINN ``memstream``s (init'd from ``init_files[i]``, one per N-column slice)
    bakes its weights and drives its tile's ``s_axis_weights`` internally -- no
    external weight FIFO port. With ``n_tiles>1`` the tiles are stitched in RTL:
    the activation FIFO fans out to all tiles and a per-lane requantize stage (bias
    add, shift + round-half-up + wrap) narrows each tile's raw ``PE*ACCU_WIDTH``
    output to ``PE*out_width`` before the tiles concatenate into one result beat
    (tile ``i`` occupies ``[i*PB +: PB]``, ``PB`` now the narrow per-tile width).
    ``init_file`` is a single-tile shorthand for ``init_files=[init_file]``. The
    default (streamed) shim keeps the ``w_*`` port and is reserved for the
    two-operand case; it gets the same per-lane requantize stage.
    """
    t = _tile(shape, tile=tile, **plan_kwargs)
    wbits = t["weight_stream_width_ba"]
    abits = t["input_stream_width_ba"]
    pbits = t["output_stream_width_ba"]     # narrow, post-requant: PE*out_width, byte-aligned
    fb = 1 if force_behavioral else 0
    n_tiles = int(n_tiles)
    k_tiles = int(k_tiles)
    m = int(shape[0])

    def _w(n):                            # bit-width to hold values 0..n-1
        return max(1, (n - 1).bit_length())
    if weights_in_core:
        if init_files is None:
            init_files = [init_file] if init_file else None
        if k_tiles > 1:
            return _generate_kt_shim(t, module_name, fb, wbits, abits, pbits,
                                     init_files, n_tiles, k_tiles, m, bias_codes=bias_codes,
                                     raw_k=raw_k, raw_n=raw_n, max_inflight=max_inflight)
        return _generate_ws_shim(t, module_name, fb, wbits, abits, pbits,
                                 init_files, n_tiles, m, bias_codes=bias_codes,
                                 raw_k=raw_k, raw_n=raw_n, max_inflight=max_inflight)

    in_total = m * t['sf']     # activation beats/node (matches feed_a's m*SF reads)
    run_total = m * t['nf']    # output beats/node (matches requant's m*NF reads)
    accu = t["accu_width"]
    pe = t["pe"]
    raw_bits = pe * accu
    req_decls, req_regs = _requant_lanes(
        t, pe, [f"out_tdata_raw[{i * accu} +: {accu}]" for i in range(pe)],
        bias_codes, "rq")
    return f"""// Generated by gemm-ip-gen (mvau target). Shim around FINN's mvu_vvu_axi.
// Tile: MW(K)={t['mw']} MH(N)={t['mh']} PE={t['pe']} SIMD={t['simd']} \
core={t['compute_core']} ACCU={t['accu_width']} out_width={t['output_width']}
// Module name MUST equal the JSON c_function_name (Vitis instantiates by it).
module {module_name} (
    input  wire                 ap_clk,
    input  wire                 ap_rst,     // active-high
    input  wire                 ap_ce,      // active-high clock enable / stall
    input  wire                 ap_start,   // ap_ctrl_chain: caller holds high until ap_ready
    input  wire                 ap_continue,// ap_ctrl_chain: caller pulses to clear ap_done
    output wire                 ap_ready,   // ap_ctrl_chain: 1-cycle start-token consumption
    output wire                 ap_done,    // ap_ctrl_chain: held until ap_continue
    output wire                 ap_idle,    // ap_ctrl_chain: no invocation in flight/pending

    // weight FIFO (input)    {wbits} = ceil(PE*SIMD*WEIGHT_WIDTH/8)*8
    input  wire [{wbits - 1}:0] w_dout,
    input  wire                 w_empty_n,
    output wire                 w_read,
    // activation FIFO (input)  {abits} = ceil(SIMD*ACTIVATION_WIDTH/8)*8
    input  wire [{abits - 1}:0] a_dout,
    input  wire                 a_empty_n,
    output wire                 a_read,
    // output FIFO (output)     {pbits} = ceil(PE*out_width/8)*8 (post-requant)
    output wire [{pbits - 1}:0] p_din,
    input  wire                 p_full_n,
    output wire                 p_write
);
    wire rst_n = ~ap_rst;

    wire         wgt_tvalid, wgt_tready;
    wire         in_tvalid,  in_tready;
    wire         out_tvalid, out_tready;
    wire [{raw_bits - 1}:0] out_tdata_raw;   // raw PE*ACCU_WIDTH beat straight off the core

    // per-lane requantize stage: bias add + shift/round-half-up/wrap to out_width
{req_decls}
    assign p_din = {{{', '.join(reversed(req_regs))}}};

    // ---- ap_ctrl_chain invocation-level FSM: IDLE <-> RUN, one node/invocation ----
    localparam IDLE = 1'd0, RUN = 1'd1;
    reg                        state = IDLE;
    reg                        done_r = 0;
    reg  [{_w(in_total + 1) - 1}:0]  icnt = 0;   // activation beats accepted this node
    reg  [{_w(run_total + 1) - 1}:0]  ocnt = 0;   // output beats produced this node

    assign ap_ready = (state == IDLE) & ~done_r & ap_start & ap_ce;
    assign ap_done  = done_r;
    assign ap_idle  = (state == IDLE) & ~done_r;

    always @(posedge ap_clk) begin
        if (ap_rst) begin
            state <= IDLE; done_r <= 0; icnt <= 0; ocnt <= 0;
        end else if (ap_ce) begin
            if (ap_ready) begin
                state <= RUN; icnt <= 0; ocnt <= 0;
            end else if (state == RUN) begin
                if (in_tvalid & in_tready) icnt <= icnt + 1'b1;
                if (p_write) begin
                    if (ocnt == {run_total - 1}) begin
                        state <= IDLE; done_r <= 1'b1;
                    end else ocnt <= ocnt + 1'b1;
                end
            end
            if (done_r & ap_continue) done_r <= 1'b0;
        end
    end

    // gate the activation handshake on icnt too: stop pulling A beats once this
    // node's {in_total} beats are accepted, even if the next node's rows are queued.
    wire in_open = (state == RUN) & (icnt != {in_total});

    // ce-gated ap_fifo <-> AXIS binding (empty_n==TVALID, read==TREADY,
    //                                    full_n==TREADY, write==TVALID)
    assign wgt_tvalid = ap_ce & w_empty_n;
    assign w_read     = ap_ce & wgt_tready;
    assign in_tvalid  = ap_ce & in_open & a_empty_n;
    assign a_read     = ap_ce & in_open & in_tready;
    assign out_tready = ap_ce & p_full_n;
    assign p_write    = ap_ce & out_tvalid & p_full_n;   // count only real transfers

    mvu_vvu_axi #(
        .IS_MVU({t['is_mvu']}),
        .VERSION(3),
        .MW({t['mw']}), .MH({t['mh']}), .PE({t['pe']}), .SIMD({t['simd']}),
        .ACTIVATION_WIDTH({t['activation_width']}), .WEIGHT_WIDTH({t['weight_width']}),
        .ACCU_WIDTH({t['accu_width']}),
        .NARROW_WEIGHTS({t['narrow_weights']}), .SIGNED_ACTIVATIONS({t['signed_activations']}),
        .SEGMENTLEN({t['segmentlen']}), .PUMPED_COMPUTE(0), .FORCE_BEHAVIORAL({fb})
    ) inst (
        .ap_clk(ap_clk),
        .ap_clk2x(1'b0),
        .ap_rst_n(rst_n),
        .s_axis_weights_tdata(w_dout),
        .s_axis_weights_tvalid(wgt_tvalid),
        .s_axis_weights_tready(wgt_tready),
        .s_axis_input_tdata(a_dout),
        .s_axis_input_tvalid(in_tvalid),
        .s_axis_input_tready(in_tready),
        .m_axis_output_tdata(out_tdata_raw),
        .m_axis_output_tvalid(out_tvalid),
        .m_axis_output_tready(out_tready)
    );
endmodule
"""


def _nf_counter(nf, reg_name="nf_cnt", advance_cond="p_write", split=False):
    """A free-running ``0..nf-1`` counter tracking which of the ``nf`` per-vector
    column blocks is on the output beat this cycle, advancing on *advance_cond*
    (a real, backpressure-gated transfer) and resetting to 0 when a new invocation
    starts (``ap_ready``). Needed only when a requantize stage's bias ROM must be
    addressed dynamically (``nf>1`` and the layer has a bias): the same PE lanes
    carry different output columns on different cycles, so a compile-time lane
    index alone cannot pick the right bias code.

    If *split* is True, returns a ``(decl, always_block)`` pair instead of one
    combined string: the register declaration alone (so callers referencing
    *reg_name* combinationally can declare it ahead of their own wires), and the
    sequential update block separately (so it can be placed after *advance_cond*
    is itself declared, when that's a wire computed later in the module, e.g.
    ``accept_beat``).
    """
    def _w(n):
        return max(1, (n - 1).bit_length())
    decl = f"    reg [{_w(nf) - 1}:0] {reg_name} = 0;   // which of the {nf} per-vector column blocks is on the beat\n"
    # Self-wrapping purely off *advance_cond* (a real output-side transfer): with the
    # decoupled handshake, ap_ready is now an input-side event (fires when the current
    # node's LAST input beat is taken, which can be cycles ahead of -- or, with enough
    # inflight headroom, even after -- this node's output phase finishes), so it is no
    # longer a valid moment to zero this output-phase counter. Every {nf} accepted
    # output beats already brings it back to 0 on its own.
    always_block = f"""    always @(posedge ap_clk) begin
        if (ap_rst) {reg_name} <= 0;
        else if (ap_ce) begin
            if ({advance_cond}) {reg_name} <= ({reg_name} == {nf - 1}) ? 0 : {reg_name} + 1'b1;
        end
    end
"""
    if split:
        return decl, always_block
    return decl + always_block



def _inflight_for(t, run_total, max_inflight=None):
    """Nodes the wrapper may hold between input accepted and output drained.
    Explicit value wins; otherwise enough that a node's fill latency (plus the
    requant/drain pipeline) never stalls admission: with one-row nodes (fc
    layers, run_total == 1) the core takes ~latency cycles per node, so the cap
    must cover latency / run_total nodes plus one. Clamped to [2, 8]."""
    if max_inflight is not None:
        return int(max_inflight)
    import math
    need = math.ceil((t["latency_cycles"] + 6) / max(1, run_total)) + 1
    return max(2, min(8, need))


def _decoupled_ctrl(in_total, run_total, in_advance, out_advance, max_inflight=2):
    """Shared ap_ctrl_chain decoupled handshake/FSM, one node = *in_total* input
    beats in / *run_total* output beats out. Used by both the weight-stationary
    (``_generate_ws_shim``) and K-tiled (``_generate_kt_shim``) wrappers so this
    logic is written once; the 2-op emitters keep their own (unrelated) FSMs.

    Unlike the old IDLE/RUN FSM (which only re-armed ap_ready after the current
    node's last output beat was written AND ap_continue had cleared ap_done --
    so consecutive nodes could never overlap), this has no state register at
    all: the input and output sides run off their own free-running per-node
    beat counters (*icnt*/*ocnt*, each wrapping 0..total-1) and a small
    ``inflight`` counter caps how many nodes may have their input accepted
    before their output has fully drained (``MAX_INFLIGHT``, default 2 --
    enough to keep the core fed back-to-back with zero idle cycles between
    nodes, since the next node's first input beat can be accepted the very
    cycle the current node's last input beat is taken).

    Protocol (Vitis ap_ctrl_chain-legal: ap_ready is a pulse and may occur
    before ap_done of the same invocation, ap_start held high by the caller
    for as long as invocations remain):
      * ``in_open  = ap_start & ap_ce & (inflight < MAX_INFLIGHT)`` -- no FSM
        state gating; callers AND this into their own a_empty_n/backpressure
        gating exactly as before.
      * on the input beat where ``icnt == in_total-1`` and *in_advance* (a real,
        backpressure-gated input transfer) fires: pulse ``ap_ready`` that same
        cycle (combinational -- Vitis requires it land no later than the beat
        that completes the invocation) and bump ``inflight``.
      * on the output beat where ``ocnt == run_total-1`` and *out_advance*
        fires: drop ``inflight`` and bump ``done_pending``.
      * ``ap_done = (done_pending != 0)``; ``ap_continue`` while ``ap_done``
        drops one pending completion. ``ap_idle`` = nothing in flight or
        pending.
      * the same-cycle increment/decrement case (an *in_advance* and
        *out_advance* completion landing together, or a new completion and an
        ``ap_continue`` landing together) is a net no-op, not two separate
        +1/-1 updates.

    xvlog enforces declare-before-use even for plain wires, so this can't be one
    self-contained block: *in_advance*/*out_advance* (e.g. ``can_load``/``p_write``)
    are themselves declared by the caller in between the counters and the rest of
    the handshake. Returns ``(early_decls, late_decls, in_open_name)``:
    *early_decls* (localparams + the icnt/ocnt/inflight/done_pending regs + the
    ``in_open`` wire -- everything callers' own *in_advance* wiring needs) goes
    where the old IDLE/RUN FSM used to sit; *late_decls* (the ap_ready/ap_done/
    ap_idle assigns + the always block) goes after *in_advance* and *out_advance*
    are themselves declared (where the old FSM's always block used to sit).
    ``in_open_name`` is always ``"in_open"``, returned for documentation at call
    sites.
    """
    def _w(n):
        return max(1, (n - 1).bit_length())
    icnt_w = _w(in_total)
    ocnt_w = _w(run_total)
    io_w = _w(max_inflight + 1)
    early_decls = f"""    // ---- ap_ctrl_chain decoupled handshake: input and output sides run off
    // independent free-running per-node beat counters; up to MAX_INFLIGHT nodes
    // may have their input accepted before their output has fully drained, so
    // consecutive nodes overlap with no idle gap. See _decoupled_ctrl's docstring.
    localparam MAX_INFLIGHT = {max_inflight};
    reg [{icnt_w - 1}:0] icnt = 0;             // input beats accepted this node, wraps at {in_total}
    reg [{ocnt_w - 1}:0] ocnt = 0;             // output beats produced this node, wraps at {run_total}
    reg [{io_w - 1}:0] inflight = 0;           // nodes with input accepted, output not yet fully drained
    reg [{io_w - 1}:0] done_pending = 0;       // completed nodes awaiting ap_continue

    wire in_open = ap_start & ap_ce & (inflight < MAX_INFLIGHT);
"""
    late_decls = f"""    // ---- ap_ctrl_chain decoupled handshake, part 2 (needs {in_advance!r}/{out_advance!r}
    // declared above): see _decoupled_ctrl's docstring.
    wire in_last_beat  = ({in_advance}) & (icnt == {in_total - 1});
    wire out_last_beat = ({out_advance}) & (ocnt == {run_total - 1});
    wire continue_ack  = ap_done & ap_continue;

    assign ap_ready = in_last_beat;
    assign ap_done  = (done_pending != 0);
    assign ap_idle  = (inflight == 0) & (done_pending == 0);

    always @(posedge ap_clk) begin
        if (ap_rst) begin
            icnt <= 0; ocnt <= 0; inflight <= 0; done_pending <= 0;
        end else if (ap_ce) begin
            if ({in_advance}) icnt <= in_last_beat  ? {icnt_w}'d0 : icnt + 1'b1;
            if ({out_advance}) ocnt <= out_last_beat ? {ocnt_w}'d0 : ocnt + 1'b1;
            case ({{in_last_beat, out_last_beat}})
                2'b10: inflight <= inflight + 1'b1;
                2'b01: inflight <= inflight - 1'b1;
                default: ; // 2'b00 no-op, 2'b11 (same-cycle in+out) net no-op
            endcase
            case ({{out_last_beat, continue_ack}})
                2'b10: done_pending <= done_pending + 1'b1;
                2'b01: done_pending <= done_pending - 1'b1;
                default: ; // 2'b00 no-op, 2'b11 (new completion + ack) net no-op
            endcase
        end
    end
"""
    return early_decls, late_decls, "in_open"


def _mvu_inst(t, fb, i, act_expr="a_dout"):
    """One ``mvu_vvu_axi`` instance for tile ``i`` with the tile's fold params,
    wired to that tile's memstream (weights) and its activation input ``act_expr``
    (the whole ``a_dout`` for N-tiling/broadcast, or a per-tile slice for K-tiling).
    """
    return f"""    mvu_vvu_axi #(
        .IS_MVU({t['is_mvu']}),
        .VERSION(3),
        .MW({t['mw']}), .MH({t['mh']}), .PE({t['pe']}), .SIMD({t['simd']}),
        .ACTIVATION_WIDTH({t['activation_width']}), .WEIGHT_WIDTH({t['weight_width']}),
        .ACCU_WIDTH({t['accu_width']}),
        .NARROW_WEIGHTS({t['narrow_weights']}), .SIGNED_ACTIVATIONS({t['signed_activations']}),
        .SEGMENTLEN({t['segmentlen']}), .PUMPED_COMPUTE(0), .FORCE_BEHAVIORAL({fb})
    ) inst_{i} (
        .ap_clk(ap_clk),
        .ap_clk2x(1'b0),
        .ap_rst_n(rst_n),
        .s_axis_weights_tdata(w_odat_{i}),
        .s_axis_weights_tvalid(wgt_tvalid_{i}),
        .s_axis_weights_tready(wgt_tready_{i}),
        .s_axis_input_tdata({act_expr}),
        .s_axis_input_tvalid(in_tvalid),
        .s_axis_input_tready(in_tready[{i}]),
        .m_axis_output_tdata(out_tdata_{i}),
        .m_axis_output_tvalid(out_tvalid[{i}]),
        .m_axis_output_tready(out_tready)
    );
"""


def _ws_tile_block(t, fb, wbits, wmem, i, init_file, act_expr="a_dout", label=None):
    """RTL for tile ``i``: its own memstream (baked weights for the tile's slice),
    the ce-gated weight handshake, and the ``mvu_vvu_axi`` instance fed activation
    ``act_expr``. Exposes the tile's raw ``PE*ACCU_WIDTH`` output wire
    (``out_tdata_i``) -- this block does no arithmetic and never touches ``p_din``;
    the caller (``_generate_ws_shim`` / ``_generate_kt_shim``) builds the
    requantize stage once it can see every tile sharing an output column (K-tiled
    grids must sum K-partials across tiles before requantizing, not per-tile)."""
    raw_pb = t["pe"] * t["accu_width"]
    col0, col1 = i * t["mh"], (i + 1) * t["mh"] - 1
    zero = "{%d{1'b0}}" % wbits
    tag = label if label is not None else f"output columns [{col0}, {col1}]"
    return f"""    // ---- tile {i}: {tag} ----
    wire [{wbits - 1}:0] w_odat_{i};
    wire                 w_ovld_{i}, w_ordy_{i}, wgt_tvalid_{i}, wgt_tready_{i};
    wire [{raw_pb - 1}:0] out_tdata_{i};   // raw PE*ACCU_WIDTH beat straight off this tile's core
    memstream #(
        .DEPTH({wmem}), .WIDTH({wbits}),
        .INIT_FILE("{init_file}"), .RAM_STYLE("auto")
    ) wmem_{i} (
        .clk(ap_clk), .rst(ap_rst),
        .config_ce(1'b0), .config_we(1'b0), .config_address(32'd0), .config_d0({zero}),
        .config_rack(), .config_q0(),
        .ordy(w_ordy_{i}), .ovld(w_ovld_{i}), .odat(w_odat_{i})
    );
    assign wgt_tvalid_{i} = ap_ce & w_ovld_{i};   // ce-gated memstream -> weight AXIS
    assign w_ordy_{i}     = ap_ce & wgt_tready_{i};
{_mvu_inst(t, fb, i, act_expr)}"""


def _generate_kt_shim(t, module_name, fb, wbits, abits, pbits, init_files, n_tiles, k_tiles, m,
                      bias_codes=None, raw_k=None, raw_n=None, max_inflight=None):
    """K-tiled (and combined N+K) weight-stationary grid shim: an ``n_tiles``x``k_tiles`` grid
    of MVU cores. Tile (j,i) reduces K-slice i (MW=K_pad/k_tiles) for N-slice j (MH=N_tile),
    each a baked ``memstream`` (DEPTH = wmem = SF_tile*NF) holding that (j,i) weight block.

    Boundary ports are UNPADDED and match hls4ml's own TDATA widths exactly, same contract
    as ``_generate_ws_shim``: ``a_dout`` is ``raw_k*ACTIVATION_WIDTH`` bits (one full hls4ml
    row per beat) and ``p_din`` is ``raw_n*out_width`` bits (no PE/NF padding, no N-tile pad
    tail). ALL K-padding (zero-fill to the grid's total ``k_pad`` = MW_tile*k_tiles) and the
    SF_tile-way SIMD fan-out (one per-K-tile SIMD-wide slice per cycle, broadcast identically
    to every N-slice) now happen in an ``arow_reg`` shift register here, exactly mirroring
    ``_generate_ws_shim``'s activation side -- the only difference is each of the ``k_tiles``
    slices routes to a *different* physical tile's lane instead of the single tile's lane.
    Per output column, the ``k_tiles`` raw partials are summed, bias-added, and
    shift/round-half-up/wrapped to ``out_width`` (mirrors the C twin's raw += loop order);
    the ``NF`` per-vector beats and ``n_tiles`` N-slices latch into one ``orow_reg`` exactly as
    in ``_generate_ws_shim``, so the beat this shim emits is one unpadded ``N*out_width`` row,
    once per external vector. Reduces to the pure K-tiled shim at n_tiles=1."""
    ntiles = n_tiles * k_tiles
    if not init_files or any(not f for f in init_files) or len(init_files) != ntiles:
        raise ValueError(f"grid shim needs {ntiles} (= n_tiles*k_tiles) memstream init file(s)")
    wmem = t["wmem"]
    accu = t["accu_width"]
    aw, outw = t["activation_width"], t["output_width"]
    pe, nf, simd, mw = t["pe"], t["nf"], t["simd"], t["mw"]
    sf_tile = mw // simd                 # per-tile SF: SIMD-wide beats to drain one tile's MW
    k_pad = mw * k_tiles                 # grid's total padded K (all k_tiles concatenated)
    K = raw_k if raw_k is not None else k_pad
    N = raw_n if raw_n is not None else n_tiles * t["mh"]
    ntile_real = N // n_tiles            # unpadded per-tile column count
    ABR = K * aw                         # raw, unpadded activation PORT width
    APAD = k_pad * aw                    # zero-padded internal row-register width
    PBR = N * outw                       # raw, unpadded result port width

    # tile (j,i)'s activation slice: SIMD lanes [i*MW + sf_cnt*SIMD, ...) of the padded
    # row register -- the grid twin of _generate_ws_shim's single "cur_slice" wire, just
    # indexed per K-tile instead of shared by every tile.
    def _tile_act_expr(i):
        base = f"{i} * {mw} * {aw} + sf_cnt * {simd * aw}"
        return f"arow_reg[({base}) +: {simd * aw}]"
    blocks = "\n".join(
        _ws_tile_block(t, fb, wbits, wmem, j * k_tiles + i, init_files[j * k_tiles + i],
                       act_expr=_tile_act_expr(i),
                       label=f"N-slice {j} K-slice {i} partial ({t['mh']} cols)")
        for j in range(n_tiles) for i in range(k_tiles))

    # nf_cnt is needed whenever NF>1: to pick the right bias code (dynamic column) AND to
    # know, in the output accumulator below, which of the NF beats is on the wire this cycle.
    need_nf_cnt = nf > 1
    if need_nf_cnt:
        nf_cnt_decl, nf_cnt_seq = _nf_counter(nf, advance_cond="accept_beat", split=True)
    else:
        nf_cnt_decl, nf_cnt_seq = "", ""
    nf_expr = "nf_cnt" if need_nf_cnt else "0"

    # Per output column (N-slice j, lane pe): sum the k_tiles raw partials, then
    # requantize once -- never requantize a partial before it is summed.
    raw_exprs = []
    for j in range(n_tiles):
        for pe_i in range(pe):
            raw_exprs.append([
                f"out_tdata_{j * k_tiles + i}[{pe_i * accu} +: {accu}]" for i in range(k_tiles)])
    if bias_codes:
        full_codes = []
        for j in range(n_tiles):
            full_codes += _wpack.bias_codes_for_tile(bias_codes, j, ntile_real, t["mh"])
        bias_idx = [f"{j} * {t['mh']} + {nf_expr} * {pe} + {pe_i}"
                   for j in range(n_tiles) for pe_i in range(pe)]
    else:
        full_codes, bias_idx = None, None
    req_decls, req_regs = _requant_lanes(t, n_tiles * pe, raw_exprs, full_codes, "rq",
                                         bias_index_exprs=bias_idx)

    def _w(n):
        return max(1, (n - 1).bit_length())

    # ---- output side: latch the NF*n_tiles per-column requant registers into one
    # unpadded N*out_width beat, dropping each tile's pad tail columns (local_oc
    # >= ntile_real), and fire p_write once per external row (the last NF beat) --
    # identical to _generate_ws_shim's output accumulator. ----
    oc_src = {}
    for j in range(n_tiles):
        for nf_i in range(nf):
            for pe_i in range(pe):
                local_oc = nf_i * pe + pe_i
                if local_oc < ntile_real:
                    oc = j * ntile_real + local_oc
                    oc_src[oc] = (req_regs[j * pe + pe_i], nf_i)
    orow_bits = []
    for oc in range(N):
        reg, nf_i = oc_src[oc]
        live = f"(nf_cnt == {nf_i})" if need_nf_cnt else "1'b1"
        orow_bits.append((oc, reg, nf_i, live))
    p_din_assign = "\n".join(
        f"    assign p_din[{oc * outw} +: {outw}] = {live} ? {reg} : orow_reg[{oc * outw} +: {outw}];"
        for oc, reg, nf_i, live in orow_bits)
    orow_latch = "\n".join(
        f"        if (accept_beat && {nf_expr} == {nf_i}) orow_reg[{oc * outw} +: {outw}] <= {reg};"
        for oc, reg, nf_i, live in orow_bits if nf_i != nf - 1)   # last phase never needs latching

    in_total = m          # one external activation beat (one full row) per vector
    run_total = m          # one external result beat (one full row) per vector
    sf_bits = _w(sf_tile)
    pad_bits = (k_pad - K) * aw
    arow_pad_expr = ("a_dout" if pad_bits == 0
                     else "{" + "{%d{1'b0}}" % pad_bits + ", a_dout}")
    ctrl_decls, ctrl_late, _ = _decoupled_ctrl(in_total, run_total, in_advance="can_load",
                                               out_advance="p_write",
                                               max_inflight=_inflight_for(t, run_total, max_inflight))
    return f"""// Generated by gemm-ip-gen (mvau target). Weight-stationary grid shim:
// {n_tiles}x{k_tiles} (N-tiles x K-tiles) MVU cores; tile (j,i) reduces K-slice i (MW={mw})
// for N-slice j ({t['mh']} cols). Boundary is UNPADDED (matches hls4ml's own TDATA widths):
// a_dout is one raw K={K}*ACTIVATION_WIDTH row/beat (K-padding to k_pad={k_pad} + the per-tile
// SF_tile={sf_tile}-way SIMD fan-out happen here); p_din is one raw N={N}*out_width beat
// (K-tile partials summed, N-tile stitching + pad-column drop happen here too). See
// _generate_kt_shim's docstring.
// Tile: MW(K/tile)={mw} MH(N/tile)={t['mh']} PE={t['pe']} SIMD={simd} \
core={t['compute_core']} ACCU={t['accu_width']} WMEM={wmem} N_TILES={n_tiles} K_TILES={k_tiles}
// Module name MUST equal the JSON c_function_name (Vitis instantiates by it).
module {module_name} (
    input  wire                 ap_clk,
    input  wire                 ap_rst,     // active-high
    input  wire                 ap_ce,      // active-high clock enable / stall
    input  wire                 ap_start,   // ap_ctrl_chain: caller holds high until ap_ready
    input  wire                 ap_continue,// ap_ctrl_chain: caller pulses to clear ap_done
    output wire                 ap_ready,   // ap_ctrl_chain: 1-cycle start-token consumption
    output wire                 ap_done,    // ap_ctrl_chain: held until ap_continue
    output wire                 ap_idle,    // ap_ctrl_chain: no invocation in flight/pending

    // activation FIFO (input)  {ABR} = raw K*ACTIVATION_WIDTH, UNPADDED (one hls4ml row/beat)
    input  wire [{ABR - 1}:0] a_dout,
    input  wire                 a_empty_n,
    output wire                 a_read,
    // output FIFO (output)     {PBR} = raw N*out_width, UNPADDED (K-summed, no N-tile pad tail)
    output wire [{PBR - 1}:0] p_din,
    input  wire                 p_full_n,
    output wire                 p_write
);
    wire rst_n = ~ap_rst;

{ctrl_decls}
    // ---- activation side: buffer one external (unpadded) row, zero-filled to the
    // grid's total k_pad={k_pad}, and fan it out to the {sf_tile} SIMD-wide beats every
    // tile needs, one per cycle (every K-tile advances in lockstep, reading its own
    // MW-wide band of the same padded row).
    reg  [{APAD - 1}:0] arow_reg;
    reg                  row_valid = 0;
    reg  [{max(sf_bits - 1, 0)}:0] sf_cnt = 0;
    // per-tile handshakes: every tile in the grid shares the same sf_cnt-selected
    // activation slice + output backpressure, so they all run in lockstep.
    wire [{ntiles - 1}:0] in_tready;
    wire [{ntiles - 1}:0] out_tvalid;
    wire [{APAD - 1}:0] arow_pad = {arow_pad_expr};
    // look ahead to the cycle a row is about to fully drain (its last SIMD-wide
    // slice accepted by every tile) so the next row can be loaded the SAME
    // cycle, back-to-back -- without this, SF=1 configs would waste one bubble
    // cycle/row (need_load only true the cycle AFTER row_valid clears).
    wire row_draining = row_valid & (&in_tready) & (sf_cnt == {sf_tile - 1});
    wire need_load = ~row_valid | row_draining;
    wire can_load  = ap_ce & in_open & need_load & a_empty_n;
    assign a_read = can_load;
    wire in_tvalid = ap_ce & row_valid;
    // nf_cnt (0..{nf - 1}, advancing on accept_beat) -- needed whenever NF>1, both
    // to pick a dynamic bias code (in the per-lane requantize section below) and
    // to know, here, which of the NF beats/vector is on the wire this cycle.
    // Declared here, ahead of its first use in out_tready/p_write below.
{nf_cnt_decl}    wire out_tready  = ap_ce & (({nf_expr} != {nf - 1}) | p_full_n);
    wire accept_beat = ap_ce & (&out_tvalid) & out_tready;
    assign p_write = accept_beat & ({nf_expr} == {nf - 1});   // one row/beat, unpadded
{nf_cnt_seq}
{ctrl_late}
    // activation row buffer sequencing -- independent of the ctrl FSM above (a
    // node's row buffer only cares whether ITS beats are still arriving; it does
    // not need to know how many other nodes are inflight/pending).
    always @(posedge ap_clk) begin
        if (ap_rst) begin
            row_valid <= 0; sf_cnt <= 0;
        end else if (ap_ce) begin
            if (can_load) begin
                arow_reg <= arow_pad; row_valid <= 1'b1; sf_cnt <= 0;
            end else if (row_valid & (&in_tready)) begin
                if (sf_cnt == {sf_tile - 1}) begin row_valid <= 0; sf_cnt <= 0; end
                else sf_cnt <= sf_cnt + 1'b1;
            end
        end
    end

{blocks}
    // per-column requantize stage: sum the K_TILES raw partials, bias add, shift +
    // round-half-up + wrap to out_width (declared after the tile blocks so it
    // never references an out_tdata_i wire, or its nf_cnt counter's accept_beat/
    // ap_ready, before their declaration)
{req_decls}
    // output accumulator: latch every phase but the last (which is driven live,
    // combinationally, on the very cycle p_write fires) into the unpadded p_din beat.
    reg [{PBR - 1}:0] orow_reg;
    always @(posedge ap_clk) begin
        if (accept_beat) begin
{orow_latch if orow_latch else "            // NF == 1: every column is driven live, nothing to latch"}
        end
    end
{p_din_assign}
endmodule
"""


def generate_two_operand_shim(shape, module_name="mvau_core", force_behavioral=True,
                              tile=None, plan=None, mode=0, **plan_kwargs):
    """Emit the two-operand (``gemm_stream``) shim: A and B both runtime activation
    streams. B takes the MVU weight port, loaded at runtime into the forked
    ``dynamic_load_2op`` module (2-bank ping-pong, see
    ``rtl_static/dynamic_load_2op.sv``), then replayed across the M rows of A
    (B-stationary). ``mode`` selects the B layout: 0 = row-major (Mode A, the
    module's PE-wide input beat, one SIMD lane per beat) or 1 = col-major
    (Mode B, the module's SIMD-wide input beat, one PE lane per beat). The
    module's ``odat`` is the exact PE*SIMD*WEIGHT_WIDTH MVU weight word --
    feeds ``s_axis_weights_tdata`` directly, no hand packing needed.

    The B FIFO here carries the module's own NARROW input beat (PE-wide for
    Mode A / SIMD-wide for Mode B) -- the wide-beat-to-narrow-beat gearbox
    (``feed_b``) lives on the HLS side (see package.py), not in this shim.

    ap_ctrl bookkeeping (``ocnt``/``run_total``) is retained per plan V1: no
    overlap/always-live loader (that's V2) -- each node's A/weight-consumption
    handshakes are gated on a ``run_r`` register set by ``ap_start`` and cleared
    once ``run_total`` output beats have been produced. B beats may stream into
    the loader at any time (gated only by the module's own ``irdy``/guard
    logic), which is a side effect of the module's design, not new shim logic.

    Scope: single tile, any (PE, SIMD, SF, NF) including the fully-spatial
    DEPTH==1 case (one weight word/vector, SF=NF=1); no N/K-tiling (2-op only
    ever folds within one MVU tile -- see
    jojo-track/defer/mvau-two-operand-dynamic-load/plan.md's "Cleanup: collapse
    2-op to a single dynamic_load_2op tile")."""
    t = tile if tile is not None else _geom.resolve_plan(shape, **plan_kwargs)["tile"]
    p = plan if plan is not None else _geom.resolve_plan(shape, **plan_kwargs)
    fb = 1 if force_behavioral else 0
    N = p["n"]
    M = p["num_input_vectors"]
    PE, SIMD, SF, NF = t["pe"], t["simd"], t["sf"], t["nf"]
    WW, ACCU = t["weight_width"], t["accu_width"]
    WB = t["weight_stream_width_ba"]     # MVU weight-port word width (PE*SIMD*WW, byte-aligned)
    AB = t["input_stream_width_ba"]      # activation beat (SIMD*AW)
    PB = t["output_stream_width_ba"]     # output beat (PE*ACCU)
    DEPTH = NF * SF
    run_total = M * NF                    # output beats per node (NF beats per vector)

    # Narrow B beat: dynamic_load_2op's idat width -- PE-wide (Mode A) or SIMD-wide
    # (Mode B), byte-aligned for the ap_fifo/AXIS boundary. The raw (unaligned) width
    # is what actually connects to the module; any byte-alignment pad bits above it
    # are don't-cares (feed_b never sets them) and are simply dropped here.
    LANES_RAW = PE if mode == 0 else SIMD
    BW_RAW = LANES_RAW * WW
    BW = ((BW_RAW + 7) // 8) * 8

    def _w(n):                            # bit-width to hold values 0..n-1
        return max(1, (n - 1).bit_length())
    raw_bits = PE * ACCU
    # pad odat (exactly PE*SIMD*WW bits) up to the byte-aligned MVU weight-port width WB
    odat_pad = WB - PE * SIMD * WW
    w_odat_expr = ("w_odat_raw" if odat_pad == 0
                   else f"{{{odat_pad}'b0, w_odat_raw}}")
    req_decls, req_regs = _requant_lanes(
        t, PE, [f"out_tdata_raw[{i * ACCU} +: {ACCU}]" for i in range(PE)], None, "rq")
    return f"""// Generated by gemm-ip-gen (mvau target). Two-operand (gemm_stream) shim:
// A + B both runtime streams; B loaded at runtime into dynamic_load_2op (2-bank
// ping-pong), replayed across M. MODE={mode} (0=row-major/A, 1=col-major/B).
// Tile: MW(K)={t['mw']} MH(N)={N} PE={PE} SIMD={SIMD} SF={SF} NF={NF} \
core={t['compute_core']} ACCU={ACCU} DEPTH={DEPTH} M={M}
// Module name MUST equal the JSON c_function_name (Vitis instantiates by it).
module {module_name} (
    input  wire                 ap_clk,
    input  wire                 ap_rst,     // active-high
    input  wire                 ap_ce,      // active-high clock enable / stall
    input  wire                 ap_start,   // ap_ctrl_chain: caller holds high until ap_ready
    input  wire                 ap_continue,// ap_ctrl_chain: caller pulses to clear ap_done
    output wire                 ap_ready,   // ap_ctrl_chain: 1-cycle start-token consumption
    output wire                 ap_done,    // ap_ctrl_chain: held until ap_continue
    output wire                 ap_idle,    // ap_ctrl_chain: no invocation in flight/pending

    // activation FIFO (input)  {AB} = ceil(SIMD*ACTIVATION_WIDTH/8)*8
    input  wire [{AB - 1}:0] a_dout,
    input  wire                 a_empty_n,
    output wire                 a_read,
    // B FIFO (input)           {BW} = ceil({LANES_RAW}*WEIGHT_WIDTH/8)*8 (dynamic_load_2op's
    // narrow input beat -- the wide-beat gearbox lives in the HLS feed_b process)
    input  wire [{BW - 1}:0] b_dout,
    input  wire                 b_empty_n,
    output wire                 b_read,
    // output FIFO (output)     {PB} = ceil(PE*out_width/8)*8 (post-requant; no bias -- two-operand)
    output wire [{PB - 1}:0] p_din,
    input  wire                 p_full_n,
    output wire                 p_write
);
    wire rst_n = ~ap_rst;

    reg                    run_r = 0;       // node RUN phase active
    reg                    done_r = 0;      // ap_done pending, cleared by ap_continue
    reg  [{_w(run_total) - 1}:0]  ocnt = 0;   // output beats seen this node
    reg  [{_w(M * SF + 1) - 1}:0]  icnt = 0;       // A beats accepted this node (M*SF)

    assign ap_ready = ~run_r & ~done_r & ap_start & ap_ce;
    assign ap_done  = done_r;
    assign ap_idle  = ~run_r & ~done_r;

    // ---- weight loader (runtime-loaded, double-buffered), MVU core, run-phase handshakes ----
    wire                          ld_ivld, ld_irdy;
    wire [{BW_RAW - 1}:0]         ld_idat = b_dout[{BW_RAW - 1}:0];
    wire                          w_ovld, w_ordy;
    wire [{PE * SIMD * WW - 1}:0] w_odat_raw;
    wire                          wgt_tvalid, wgt_tready;
    wire                          in_tvalid, in_tready, out_tvalid, out_tready;
    wire [{raw_bits - 1}:0] out_tdata_raw;   // raw PE*ACCU_WIDTH beat straight off the core

    // per-lane requantize stage (no bias -- two-operand GEMM never has one): shift +
    // round-half-up + wrap to out_width
{req_decls}
    assign p_din = {{{', '.join(reversed(req_regs))}}};

    always @(posedge ap_clk) begin
        if (ap_rst) begin
            run_r <= 0; done_r <= 0; ocnt <= 0; icnt <= 0;
        end else if (ap_ce) begin
            if (ap_ready) begin
                run_r <= 1'b1; ocnt <= 0; icnt <= 0;
            end else if (run_r) begin
                if (in_tvalid & in_tready) icnt <= icnt + 1'b1;
                if (p_write) begin
                    if (ocnt == {run_total - 1}) begin run_r <= 1'b0; done_r <= 1'b1; end
                    else ocnt <= ocnt + 1'b1;
                end
            end
            if (done_r & ap_continue) done_r <= 1'b0;
        end
    end

    // b handshake: dynamic_load_2op accepts B beats whenever it has room (its own
    // writer FSM/guard, not gated by ap_ctrl -- filling ahead of a node's RUN phase
    // is safe by construction and simply a side effect of always-open b_read).
    assign ld_ivld = ap_ce & b_empty_n;
    assign b_read  = ld_ivld & ld_irdy;

    dynamic_load_2op #(
        .PE({PE}), .SIMD({SIMD}), .WEIGHT_WIDTH({WW}),
        .MH({N}), .MW({t['mw']}), .N_REPS({M}), .MODE({mode}),
        .RAM_STYLE("distributed")
    ) loader (
        .ap_clk(ap_clk), .ap_rst_n(rst_n),
        .ivld(ld_ivld), .irdy(ld_irdy), .idat(ld_idat),
        .ovld(w_ovld), .ordy(w_ordy), .odat(w_odat_raw)
    );

    // RUN-phase AXIS binding (frozen outside run_r so the core stays idle)
    // gate the A handshake on icnt too: stop pulling A beats once this node's M*SF
    // beats are accepted, even if the next node's rows are already queued in the FIFO.
    wire in_open = run_r & (icnt != {M * SF});
    assign wgt_tvalid = ap_ce & run_r & w_ovld;
    assign w_ordy     = ap_ce & run_r & wgt_tready;
    assign in_tvalid  = ap_ce & in_open & a_empty_n;
    assign a_read     = ap_ce & in_open & in_tready;
    assign out_tready = ap_ce & p_full_n;
    assign p_write    = ap_ce & out_tvalid & p_full_n;   // count only real transfers

    mvu_vvu_axi #(
        .IS_MVU({t['is_mvu']}),
        .VERSION(3),
        .MW({t['mw']}), .MH({N}), .PE({t['pe']}), .SIMD({SIMD}),
        .ACTIVATION_WIDTH({t['activation_width']}), .WEIGHT_WIDTH({WW}),
        .ACCU_WIDTH({ACCU}),
        .NARROW_WEIGHTS({t['narrow_weights']}), .SIGNED_ACTIVATIONS({t['signed_activations']}),
        .SEGMENTLEN({t['segmentlen']}), .PUMPED_COMPUTE(0), .FORCE_BEHAVIORAL({fb})
    ) inst (
        .ap_clk(ap_clk),
        .ap_clk2x(1'b0),
        .ap_rst_n(rst_n),
        .s_axis_weights_tdata({w_odat_expr}),
        .s_axis_weights_tvalid(wgt_tvalid),
        .s_axis_weights_tready(wgt_tready),
        .s_axis_input_tdata(a_dout),
        .s_axis_input_tvalid(in_tvalid),
        .s_axis_input_tready(in_tready),
        .m_axis_output_tdata(out_tdata_raw),
        .m_axis_output_tvalid(out_tvalid),
        .m_axis_output_tready(out_tready)
    );
endmodule
"""





def _generate_ws_shim(t, module_name, fb, wbits, abits, pbits, init_files, n_tiles, m,
                      bias_codes=None, raw_k=None, raw_n=None, max_inflight=None):
    """Weight-stationary shim: ``n_tiles`` FINN ``memstream`` + ``mvu_vvu_axi``
    tiles stitched in RTL. Each memstream (baked from ``init_files[i]``) replaces
    the external weight FIFO for its N-column slice and drives its tile's
    ``s_axis_weights`` directly.

    memstream ``WIDTH`` = the byte-aligned weight-stream width (the packed
    PE*SIMD*WEIGHT_WIDTH weights sit in the low bits, high bits zero); ``DEPTH`` =
    WMEM = NF*SF words per tile, re-streamed cyclically so every input vector
    re-reads the same weights.

    Boundary ports are UNPADDED and match hls4ml's own TDATA widths exactly:
    ``a_dout`` is ``raw_k*ACTIVATION_WIDTH`` bits (one full hls4ml row per beat,
    no SIMD padding) and ``p_din`` is ``raw_n*out_width`` bits (no PE padding, no
    N-tile-pad tail columns). All K-padding (zero-fill up to ``k_pad`` and the
    SF-way fan-out into one SIMD-wide beat/cycle) and N-padding/stitching
    (dropping each tile's pad tail columns and concatenating the ``n_tiles``
    slices) now happen in this wrapper, in an ``arow_reg``/``orow_reg`` shift
    register pair, instead of in the HLS-side glue (see ``package.py``'s
    ``_gemm_ip_header``, now a pure bit-reinterpretation with no cycles of its
    own). ``raw_k``/``raw_n`` default to the padded widths (no-op) for any
    caller that does not supply them.

    Stitching (all tiles are identical modules fed identical activations with
    identical output backpressure, so they run in lockstep):
      * one activation register (loaded once per external row, zero-padded to
        ``k_pad``) fans its ``SF`` SIMD-wide slices out to every tile one per
        cycle; ``a_read`` pulls the next external row only once the current one
        is fully drained.
      * each tile's raw ``PE*ACCU_WIDTH`` output goes through a per-lane requantize
        stage (bias add, shift + round-half-up + wrap) narrowing it to
        ``out_width``/lane; the ``NF`` per-vector beats and ``n_tiles`` N-slices
        latch into one ``orow_reg`` (this cycle's last-``NF`` lanes bypass the
        register and read the live combinational requant value, since ``p_din``
        must be valid the same cycle ``p_write`` fires); ``p_write`` fires once
        per external row, on the last of the ``NF`` beats. Generalizes
        temp_space/mvau-ws (cosim PASS)."""
    if not init_files or any(not f for f in init_files):
        raise ValueError("weight-stationary shim requires an init_file per tile "
                         "(memstream $readmemh path)")
    if len(init_files) != n_tiles:
        raise ValueError(f"expected {n_tiles} init file(s), got {len(init_files)}")
    wmem = t["wmem"]
    accu = t["accu_width"]
    aw, outw = t["activation_width"], t["output_width"]
    pe, nf, sf, simd = t["pe"], t["nf"], t["sf"], t["simd"]
    k_pad = t["mw"]                      # K padded to a multiple of SIMD (per-tile MW)
    K = raw_k if raw_k is not None else k_pad
    N = raw_n if raw_n is not None else n_tiles * t["mh"]
    ntile_real = N // n_tiles            # unpadded per-tile column count
    ABR = K * aw                         # raw, unpadded activation PORT width
    APAD = k_pad * aw                    # zero-padded internal row-register width
    PBR = N * outw                       # raw, unpadded result port width

    blocks = "\n".join(
        _ws_tile_block(t, fb, wbits, wmem, i, init_files[i], act_expr="cur_slice")
        for i in range(n_tiles))

    # nf_cnt is needed whenever NF>1: to pick the right bias code (dynamic column,
    # same lanes carry different columns across cycles) AND to know, in the output
    # accumulator below, which of the NF beats is on the wire this cycle.
    need_nf_cnt = nf > 1
    if need_nf_cnt:
        nf_cnt_decl, nf_cnt_seq = _nf_counter(nf, advance_cond="accept_beat", split=True)
    else:
        nf_cnt_decl, nf_cnt_seq = "", ""
    nf_expr = "nf_cnt" if need_nf_cnt else "0"
    raw_exprs = [f"out_tdata_{i}[{pe_i * accu} +: {accu}]"
                for i in range(n_tiles) for pe_i in range(pe)]
    if bias_codes:
        # Bucket the N-long baked bias into each tile's own n_pad (=NF*PE) local
        # lanes (bias_codes is real-column length; ntile_real derives from it since
        # all n_tiles are equal-width).
        full_codes = []
        for i in range(n_tiles):
            full_codes += _wpack.bias_codes_for_tile(bias_codes, i, ntile_real, t["mh"])
        bias_idx = [f"{i} * {t['mh']} + {nf_expr} * {pe} + {pe_i}"
                   for i in range(n_tiles) for pe_i in range(pe)]
    else:
        full_codes, bias_idx = None, None
    req_decls, req_regs = _requant_lanes(t, n_tiles * pe, raw_exprs, full_codes, "rq",
                                         bias_index_exprs=bias_idx)

    def _w(n):
        return max(1, (n - 1).bit_length())

    # ---- output side: latch the NF*n_tiles per-column requant registers into one
    # unpadded N*out_width beat, dropping each tile's pad tail columns (local_oc
    # >= ntile_real), and fire p_write once per external row (the last NF beat). ----
    oc_src = {}
    for i in range(n_tiles):
        for nf_i in range(nf):
            for pe_i in range(pe):
                local_oc = nf_i * pe + pe_i
                if local_oc < ntile_real:
                    oc = i * ntile_real + local_oc
                    oc_src[oc] = (req_regs[i * pe + pe_i], nf_i)
    orow_bits = []
    for oc in range(N):
        reg, nf_i = oc_src[oc]
        live = f"(nf_cnt == {nf_i})" if need_nf_cnt else "1'b1"
        orow_bits.append((oc, reg, nf_i, live))
    p_din_assign = "\n".join(
        f"    assign p_din[{oc * outw} +: {outw}] = {live} ? {reg} : orow_reg[{oc * outw} +: {outw}];"
        for oc, reg, nf_i, live in orow_bits)
    orow_latch = "\n".join(
        f"        if (accept_beat && {nf_expr} == {nf_i}) orow_reg[{oc * outw} +: {outw}] <= {reg};"
        for oc, reg, nf_i, live in orow_bits if nf_i != nf - 1)   # last phase never needs latching

    in_total = m         # one external activation beat (one full row) per vector
    run_total = m         # one external result beat (one full row) per vector
    sf_bits = _w(sf)
    pad_bits = (k_pad - K) * aw
    arow_pad_expr = ("a_dout" if pad_bits == 0
                     else "{" + "{%d{1'b0}}" % pad_bits + ", a_dout}")
    ctrl_decls, ctrl_late, _ = _decoupled_ctrl(in_total, run_total, in_advance="can_load",
                                               out_advance="p_write",
                                               max_inflight=_inflight_for(t, run_total, max_inflight))
    return f"""// Generated by gemm-ip-gen (mvau target). Weight-stationary shim: {n_tiles} FINN
// MVU tile(s) -- each a memstream (baked weights) + mvu_vvu_axi -- stitched in RTL.
// Boundary is UNPADDED (matches hls4ml's own TDATA widths): a_dout is one raw
// K={K}*ACTIVATION_WIDTH row/beat (K-padding + the SF={sf}-way SIMD fan-out happen
// here); p_din is one raw N={N}*out_width beat (N-tile stitching + pad-column drop
// happen here too). See _generate_ws_shim's docstring.
// Tile: MW(K)={t['mw']} MH(N/tile)={t['mh']} PE={t['pe']} SIMD={t['simd']} \
core={t['compute_core']} ACCU={t['accu_width']} out_width={t['output_width']} WMEM={wmem} N_TILES={n_tiles}
// Module name MUST equal the JSON c_function_name (Vitis instantiates by it).
module {module_name} (
    input  wire                 ap_clk,
    input  wire                 ap_rst,     // active-high
    input  wire                 ap_ce,      // active-high clock enable / stall
    input  wire                 ap_start,   // ap_ctrl_chain: caller holds high until ap_ready
    input  wire                 ap_continue,// ap_ctrl_chain: caller pulses to clear ap_done
    output wire                 ap_ready,   // ap_ctrl_chain: 1-cycle start-token consumption
    output wire                 ap_done,    // ap_ctrl_chain: held until ap_continue
    output wire                 ap_idle,    // ap_ctrl_chain: no invocation in flight/pending

    // activation FIFO (input)  {ABR} = raw K*ACTIVATION_WIDTH, UNPADDED (one hls4ml row/beat)
    input  wire [{ABR - 1}:0] a_dout,
    input  wire                 a_empty_n,
    output wire                 a_read,
    // output FIFO (output)     {PBR} = raw N*out_width, UNPADDED (no N-tile pad tail)
    output wire [{PBR - 1}:0] p_din,
    input  wire                 p_full_n,
    output wire                 p_write
);
    wire rst_n = ~ap_rst;

{ctrl_decls}
    // ---- activation side: buffer one external (unpadded) row, zero-filled to
    // k_pad={k_pad}, and fan it out to the {sf}-deep SIMD-wide beats the MVU tiles
    // need, one per cycle.
    reg  [{APAD - 1}:0] arow_reg;
    reg                  row_valid = 0;
    reg  [{max(sf_bits - 1, 0)}:0] sf_cnt = 0;
    // per-tile handshakes: every tile shares the same activation slice + output
    // backpressure, so they run in lockstep (tiles are identical modules).
    wire [{n_tiles - 1}:0] in_tready;
    wire [{n_tiles - 1}:0] out_tvalid;
    wire [{APAD - 1}:0] arow_pad = {arow_pad_expr};
    // look ahead to the cycle a row is about to fully drain (its last SIMD-wide
    // slice accepted by every tile) so the next row can be loaded the SAME
    // cycle, back-to-back -- without this, SF=1 configs would waste one bubble
    // cycle/row (need_load only true the cycle AFTER row_valid clears).
    wire row_draining = row_valid & (&in_tready) & (sf_cnt == {sf - 1});
    wire need_load = ~row_valid | row_draining;
    wire can_load  = ap_ce & in_open & need_load & a_empty_n;
    assign a_read = can_load;
    wire [{abits - 1}:0] cur_slice = arow_reg[sf_cnt * {abits} +: {abits}];
    wire in_tvalid = ap_ce & row_valid;
    // nf_cnt (0..{nf - 1}, advancing on accept_beat) -- needed whenever NF>1, both
    // to pick a dynamic bias code (in the per-lane requantize section below) and
    // to know, here, which of the NF beats/vector is on the wire this cycle.
    // Declared here, ahead of its first use in out_tready/p_write below.
{nf_cnt_decl}    wire out_tready  = ap_ce & (({nf_expr} != {nf - 1}) | p_full_n);
    wire accept_beat = ap_ce & (&out_tvalid) & out_tready;
    assign p_write = accept_beat & ({nf_expr} == {nf - 1});   // one row/beat, unpadded
{nf_cnt_seq}
{ctrl_late}
    // activation row buffer sequencing -- independent of the ctrl FSM above (a
    // node's row buffer only cares whether ITS beats are still arriving; it does
    // not need to know how many other nodes are inflight/pending).
    always @(posedge ap_clk) begin
        if (ap_rst) begin
            row_valid <= 0; sf_cnt <= 0;
        end else if (ap_ce) begin
            if (can_load) begin
                arow_reg <= arow_pad; row_valid <= 1'b1; sf_cnt <= 0;
            end else if (row_valid & (&in_tready)) begin
                if (sf_cnt == {sf - 1}) begin row_valid <= 0; sf_cnt <= 0; end
                else sf_cnt <= sf_cnt + 1'b1;
            end
        end
    end

{blocks}
    // per-lane requantize stage: bias add + shift/round-half-up/wrap to out_width
    // (declared after the tile blocks so it never references an out_tdata_i wire,
    // or its nf_cnt counter's accept_beat/ap_ready, before their declaration --
    // the same declaration-order bug this target already had to fix once)
{req_decls}
    // output accumulator: latch every phase but the last (which is driven live,
    // combinationally, on the very cycle p_write fires) into the unpadded p_din beat.
    reg [{PBR - 1}:0] orow_reg;
    always @(posedge ap_clk) begin
        if (accept_beat) begin
{orow_latch if orow_latch else "            // NF == 1: every column is driven live, nothing to latch"}
        end
    end
{p_din_assign}
endmodule
"""
