// cmvu_mode1 — CMVU Mode-1 (inner-product / adder-tree) datapath, single block.
//
// Computes, for a k x n physical grid (default k=4, n=8):
//   y[j] = requant( sum_{i=0..k-1} a[i] * W[i][j]  +  cascade_in[j] )   for j = 0 .. n-1
//
// Output lanes are physically RESULT_WIDTH (default 32) bits; the runtime `out_w` selects an
// effective width W = out_w+1 in 1..RESULT_WIDTH, and the requantized W-bit value is
// sign-extended into the full lane (see the requant block near the end of this file).
//
// The entire pipeline is built out of cmvu_regbank instances; with every bank PRESENT (the
// default) the end-to-end latency is L = 1(REG_IN) + 1(REG_MULT) + 2(REG_TREE) + 1(REG_RED) +
// 1(REG_OUT) = 6 cycles. cascade_out taps REG_RED (latency L-1 = 5); y_out taps REG_OUT
// (latency L = 6). See docs/architecture.md sections 1, 2, 5, 6, 8, 11.
//
// cascade_in timing contract (docs/architecture.md sections 5, 7): cascade_in is consumed
// DIRECTLY (combinationally) at the reduce node -- it is NOT re-delayed through a copy of this
// block's own REG_IN/REG_MULT/REG_TREE pipeline. This is what makes a cascade hop cost exactly
// 1 cycle (the entire reason cascade_out taps REG_RED at L-1 rather than REG_OUT at L). The
// caller must present cascade_in D cycles after the corresponding a_in, where
// D = 1(REG_IN, always present) + REG_MULT_PRESENT + sum(REG_TREE_PRESENT[]) (= 4 at the
// all-present default) -- the depth of the local pipeline that produces tree_partial, which cascade_in is
// added to combinationally at the reduce node.
//
// This increment (P1b) adds the accumulator buffer (docs section 6) and temporal-K folding
// control (docs section 8): `acc_first`/`acc_last` framing and single-slot accumulation,
// and the accbuf read-modify-write at the reduce node (acc_term). No multi-block
// cascade/broadcast chaining logic beyond the plain cascade_in/cascade_out ports (section
// 11.4). (`start`/`mode` ports removed 2026-09-17.)
module cmvu_mode1 #(
    parameter int unsigned IN_WIDTH        = 8,
    parameter int unsigned COEF_WIDTH      = 8,
    parameter int unsigned ACC_WIDTH       = 32,
    parameter int unsigned BIAS_WIDTH      = 32,
    parameter int unsigned RESULT_WIDTH    = 32,
    parameter int unsigned SHIFT_WIDTH     = 5,
    parameter int unsigned K               = 4,
    parameter int unsigned N               = 8,
    parameter int unsigned M_MEM_TILES     = 8,    // resident weight-tile slots (docs 4: both banks, Mode 1)

    parameter bit REG_MULT_PRESENT = 1'b1,
    parameter bit REG_RED_PRESENT  = 1'b1,
    parameter bit REG_OUT_PRESENT  = 1'b1,

    parameter bit CASCADE_EN       = 1'b1,
    parameter bit REQUANT_PRESENT  = 1'b1
) (
    input  wire                          clk,
    input  wire                          rst,

    input  wire                          valid,
    // input  wire                       start,        // REMOVED 2026-09-17: no logic use
    input  wire                          acc_first,    // first K-pass touch (acc_term = bias_in lane)
    input  wire                          acc_last,     // final K-pass touch (marks emit/done)
    // input  wire                       mode,         // REMOVED 2026-09-17: single-mode block; no logic use
    input  wire [K*IN_WIDTH-1:0]         a_in,         // activation slice, element i @ [i*IN_WIDTH +: IN_WIDTH]
    input  wire [63:0]                   b_in,         // row-major: 8 int8; column-major: K int8 in low bits
    input  wire                          w_we,         // accepted weight-load beat
    input  wire                          w_load_start, // optional explicit transaction start/restart (with w_we)
    input  wire                          w_col_major,  // format sampled only at transaction start: 0=row, 1=column
    input  wire                          w_dual_tile,  // with column format: load logical pair {w_tile_sel,w_tile_sel+1}
    input  wire [$clog2(M_MEM_TILES < 2 ? 2 : M_MEM_TILES)-1:0] tile_sel,   // active (read) tile slot
    input  wire [$clog2(M_MEM_TILES < 2 ? 2 : M_MEM_TILES)-1:0] w_tile_sel, // write-target tile slot
    input  wire                          a_signed,
    input  wire                          b_signed,
    input  wire [SHIFT_WIDTH-1:0]        shift_amt,
    input  wire [$clog2(RESULT_WIDTH < 2 ? 2 : RESULT_WIDTH)-1:0] out_w,  // effective result width minus 1 (W = out_w+1); uniform all lanes
    input  wire [N*ACC_WIDTH-1:0]        cascade_in,
    input  wire [N*BIAS_WIDTH-1:0]       bias_in,      // per-lane signed bias, sign-extended on acc_first

    output wire [N*RESULT_WIDTH-1:0]     y_out,
    output wire [N*ACC_WIDTH-1:0]        cascade_out,
    output wire                          y_valid,
    output wire                          done          // acc_last delayed by L (final-emit marker)
);
    // The tree has ceil(log2(K)) levels; this increment assumes K is a power of two (the V1
    // reference grid, K=4, is). Generalizing to non-power-of-two K is out of scope here.
    localparam int unsigned TREE_LEVELS = $clog2(K);

    parameter bit [TREE_LEVELS-1:0] REG_TREE_PRESENT = {TREE_LEVELS{1'b1}};

    // ---------------------------------------------------------------------
    // Operand memory (docs/architecture.md section 4): M_MEM_TILES resident weight tiles, each
    // k*n*COEF_WIDTH bits (256 b at V1 defaults), row-major per section 11.1 (W[i][j] at linear
    // index i*N+j). Implemented by cmvu_w_mem, a 1R1W synchronous memory -- ONE read port (the
    // active tile, addressed by `tile_sel`; the address is registered internally, so the tile
    // word is valid the cycle AFTER `tile_sel` is presented) and ONE write port (the write-target
    // tile, addressed by `w_tile_sel`, filled 8 int8/cycle over LOAD_CHUNKS beats via the existing
    // b_in/w_we protocol -- section 8).
    //
    // Bubble-free tile switching (section 4): because there are many slots, the caller can
    // preload the next tile into a FREE slot (any slot other than the one `tile_sel` currently
    // reads) while this block keeps computing against the active tile, then swing `tile_sel`
    // over once the load completes. This is a plain 1R1W array, not a double-buffer -- there is
    // exactly one copy of each tile's data.
    //
    // CALLER CONTRACT (section 4): on a valid compute cycle, no write destination may equal
    // `tile_sel`. With valid=0 it is safe to initialize the selected slot because no computation
    // consumes that read. Same-address read/write returns the prior contents for that cycle.
    // ---------------------------------------------------------------------
    localparam int unsigned W_TILE_BITS  = K * N * COEF_WIDTH;
    localparam int unsigned ELEMS_PER_LOAD = 64 / COEF_WIDTH;               // 8
    localparam int unsigned LOAD_CHUNKS  = (K * N) / ELEMS_PER_LOAD;        // 4
    localparam int unsigned LOAD_BEATS_MAX = (N > LOAD_CHUNKS) ? N : LOAD_CHUNKS;
    localparam int unsigned LOAD_CNT_W   = (LOAD_BEATS_MAX <= 1) ? 1 : $clog2(LOAD_BEATS_MAX);
    localparam int unsigned TILE_SEL_W   = (M_MEM_TILES < 2) ? 1 : $clog2(M_MEM_TILES);

    localparam int unsigned W_TILE_BYTES = W_TILE_BITS / 8;
    logic [LOAD_CNT_W-1:0]    w_load_cnt;
    logic                     w_loading;
    logic                     w_load_fmt_col, w_load_fmt_dual;
    logic [TILE_SEL_W-1:0]    w_load_addr;
    wire                      w_new_transaction = !w_loading || w_load_start;
    wire                      w_active_col_major = w_new_transaction ? w_col_major : w_load_fmt_col;
    wire                      w_active_dual = w_new_transaction ? w_dual_tile : w_load_fmt_dual;
    wire [TILE_SEL_W-1:0]     w_active_addr = w_new_transaction ? w_tile_sel : w_load_addr;
    wire [LOAD_CNT_W-1:0]     w_active_idx = w_new_transaction ? '0 : w_load_cnt;
    logic [W_TILE_BITS-1:0]   w_wr_data;
    logic [W_TILE_BYTES-1:0]  w_wr_byte_en;
    logic [W_TILE_BITS-1:0]   w_wr_data_pair;
    logic [W_TILE_BYTES-1:0]  w_wr_byte_en_pair;
    wire  [W_TILE_BITS-1:0]   w_tile;

    // A transaction begins on the first accepted beat after completion, or explicitly on
    // w_load_start. Address and format are then captured, so changing their pins mid-tile has
    // no effect. w_load_start deterministically discards an incomplete tile and restarts at
    // beat zero. Row-major has LOAD_CHUNKS 64-bit beats; column-major has N beats, one byte per
    // row at each column (K=4/N=8 at V1 defaults).
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            w_load_cnt <= '0;
            w_loading  <= 1'b0;
            w_load_fmt_col <= 1'b0;
            w_load_fmt_dual <= 1'b0;
            w_load_addr <= '0;
        end else if (w_we) begin
            if (w_new_transaction) begin
                w_load_fmt_col <= w_col_major;
                w_load_fmt_dual <= w_dual_tile;
                w_load_addr    <= w_tile_sel;
            end
            if (w_active_idx == (w_active_col_major ? N - 1 : LOAD_CHUNKS - 1)) begin
                w_load_cnt <= '0;
                w_loading  <= 1'b0;
            end else begin
                w_load_cnt <= w_active_idx + 1'b1;
                w_loading  <= 1'b1;
            end
        end
    end

    always_comb begin
        integer w_row, w_byte;
        w_row = 0;
        w_byte = 0;
        w_wr_data = '0;
        w_wr_byte_en = '0;
        if (w_we) begin
            if (w_active_col_major) begin
                for (w_row = 0; w_row < K; w_row = w_row + 1) begin
                    w_wr_data[(w_row*N + w_active_idx)*COEF_WIDTH +: COEF_WIDTH] =
                        b_in[w_row*COEF_WIDTH +: COEF_WIDTH];
                    for (w_byte = 0; w_byte < (COEF_WIDTH / 8); w_byte = w_byte + 1)
                        w_wr_byte_en[(w_row*N + w_active_idx)*(COEF_WIDTH / 8) + w_byte] = 1'b1;
                end
            end else begin
                w_wr_data[w_active_idx * 64 +: 64] = b_in;
                for (w_byte = 0; w_byte < 8; w_byte = w_byte + 1)
                    w_wr_byte_en[w_active_idx*8 + w_byte] = 1'b1;
            end
        end
    end

    // The upper 32 bits are only consumed by paired column transactions.  Keeping this in a
    // parameter guard avoids out-of-range selects for non-V1 shapes; those shapes reject paired
    // mode below rather than silently truncating a payload.
    generate if (K * COEF_WIDTH <= 32) begin : g_paired_payload
        always_comb begin : p_pair_data
            integer p_row, p_byte;
            w_wr_data_pair = '0;
            w_wr_byte_en_pair = '0;
            if (w_we && w_active_dual) begin
                for (p_row = 0; p_row < K; p_row = p_row + 1) begin
                    w_wr_data_pair[(p_row*N + w_active_idx)*COEF_WIDTH +: COEF_WIDTH] =
                        b_in[32 + p_row*COEF_WIDTH +: COEF_WIDTH];
                    for (p_byte = 0; p_byte < (COEF_WIDTH / 8); p_byte = p_byte + 1)
                        w_wr_byte_en_pair[(p_row*N + w_active_idx)*(COEF_WIDTH / 8) + p_byte] = 1'b1;
                end
            end
        end
    end else begin : g_no_paired_payload
        always_comb begin
            w_wr_data_pair = '0;
            w_wr_byte_en_pair = '0;
        end
    end endgenerate

    cmvu_w_mem #(
        .WORDS      (M_MEM_TILES),
        .WORD_BITS  (W_TILE_BITS)
    ) u_w_mem (
        .clk          (clk),
        .rst          (rst),
        .rd_addr      (tile_sel),
        .rd_data      (w_tile),
        .wr0_addr     (w_active_addr), .wr0_en(w_we),
        .wr0_byte_en  (w_wr_byte_en), .wr0_data(w_wr_data),
        .wr1_addr     (w_active_addr + TILE_SEL_W'(1)), .wr1_en(w_we && w_active_dual && w_active_col_major),
        .wr1_byte_en  (w_wr_byte_en_pair), .wr1_data(w_wr_data_pair)
    );

`ifndef SYNTHESIS
    initial assert (M_MEM_TILES >= 2 && (M_MEM_TILES % 2) == 0)
        else $error("cmvu_mode1: paired storage requires an even tile count >= 2");
    initial assert ((K * N * COEF_WIDTH) % 8 == 0 && (COEF_WIDTH % 8) == 0)
        else $error("cmvu_mode1: byte-addressed loader requires byte-wide coefficients/tile");
    always_ff @(posedge clk) if (!rst && w_we && w_new_transaction) begin
        assert (!(w_active_dual && !w_active_col_major)) else $error("cmvu_mode1: dual-tile row-major format is invalid");
        assert (!w_active_dual || (w_active_addr[0] == 1'b0)) else $error("cmvu_mode1: paired load base must be even");
        assert (!w_active_dual || (K * COEF_WIDTH <= 32)) else $error("cmvu_mode1: paired payload exceeds b_in half");
    end
    // tile_sel is live: check every write beat that coincides with a valid computation.
    always_ff @(posedge clk) if (!rst && w_we && valid) begin
        assert (!(w_active_dual && w_active_col_major && ((tile_sel == w_active_addr) || (tile_sel == (w_active_addr + TILE_SEL_W'(1))))))
            else $error("cmvu_mode1: paired preload targets active tile");
        assert (!(!(w_active_dual && w_active_col_major) && tile_sel == w_active_addr))
            else $error("cmvu_mode1: preload targets active tile");
    end
`endif

    // ---------------------------------------------------------------------
    // REG_IN: register the activation slice + runtime signedness flags.
    //
    // REG_IN is now MANDATORY (not bypassable): cmvu_w_mem's read address is itself registered
    // internally, so its data is valid one cycle after `tile_sel` is presented on the pin. REG_IN
    // registering `a_in` by that same one cycle is what keeps the activation aligned with the
    // weight tile at the multiply stage below -- `tile_sel` now drives the memory's read address
    // DIRECTLY (no longer carried through REG_IN itself; the memory's own address register
    // supplies the delay that used to be provided by carrying tile_sel through this bank). This
    // adds no extra latency: REG_IN already existed on the datapath (L stays 6, see file header).
    // ---------------------------------------------------------------------
    localparam int unsigned AIN_BITS = K * IN_WIDTH;
    localparam int unsigned REG_IN_W = AIN_BITS + 2;

    wire [AIN_BITS-1:0] a_slice = a_in;
    wire [REG_IN_W-1:0] reg_in_d = {b_signed, a_signed, a_slice};
    wire [REG_IN_W-1:0] reg_in_q;

    cmvu_regbank #(.W(REG_IN_W), .PRESENT(1'b1)) u_reg_in (
        .clk(clk), .rst(rst), .ena(1'b1), .d(reg_in_d), .q(reg_in_q)
    );

    wire [AIN_BITS-1:0]        a_stage       = reg_in_q[AIN_BITS-1:0];
    wire                       a_signed_stg  = reg_in_q[AIN_BITS];
    wire                       b_signed_stg  = reg_in_q[AIN_BITS+1];

    // w_tile (from cmvu_w_mem, instantiated above) is already aligned with a_stage: `tile_sel`
    // was registered by the memory's own read-address register on the same cycle REG_IN
    // registered a_in, so both arrive at the multiply stage one cycle later, in step.

    // ---------------------------------------------------------------------
    // Multiply: extend each operand to IN_WIDTH+1 bits per its runtime signedness, multiply
    // signed, then sign-extend the product to ACC_WIDTH. See docs/architecture.md 11.2.
    // ---------------------------------------------------------------------
    localparam int unsigned EXT_W = IN_WIDTH + 1;

    genvar gj, gi, gl, gidx;

    logic signed [ACC_WIDTH-1:0] mult_comb [0:N-1][0:K-1];
    logic signed [ACC_WIDTH-1:0] mult_q    [0:N-1][0:K-1];

    generate
        for (gj = 0; gj < N; gj = gj + 1) begin : g_mult_lane
            for (gi = 0; gi < K; gi = gi + 1) begin : g_mult_row
                logic [IN_WIDTH-1:0]      a_val;
                logic [COEF_WIDTH-1:0]    w_val;
                logic signed [EXT_W-1:0] ext_a;
                logic signed [EXT_W-1:0] ext_w;
                logic signed [2*EXT_W-1:0] prod;

                assign a_val = a_stage[gi * IN_WIDTH +: IN_WIDTH];
                assign w_val = w_tile[(gi * N + gj) * COEF_WIDTH +: COEF_WIDTH];
                assign ext_a = a_signed_stg ? {{1{a_val[IN_WIDTH-1]}}, a_val}
                                            : {1'b0, a_val};
                assign ext_w = b_signed_stg ? {{1{w_val[COEF_WIDTH-1]}}, w_val}
                                            : {1'b0, w_val};
                assign prod = ext_a * ext_w;
                assign mult_comb[gj][gi] = {{(ACC_WIDTH-2*EXT_W){prod[2*EXT_W-1]}}, prod};

                cmvu_regbank #(.W(ACC_WIDTH), .PRESENT(REG_MULT_PRESENT)) u_reg_mult (
                    .clk(clk), .rst(rst), .ena(1'b1),
                    .d(mult_comb[gj][gi]), .q(mult_q[gj][gi])
                );
            end
        end
    endgenerate

    // ---------------------------------------------------------------------
    // Adder trees: N independent trees, each a balanced depth TREE_LEVELS tree over K terms,
    // registered per level via REG_TREE_PRESENT[level].
    // ---------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] tree_val [0:TREE_LEVELS][0:N-1][0:K-1];

    generate
        for (gj = 0; gj < N; gj = gj + 1) begin : g_tree_seed
            for (gi = 0; gi < K; gi = gi + 1) begin : g_tree_seed_i
                assign tree_val[0][gj][gi] = mult_q[gj][gi];
            end
        end

        for (gl = 0; gl < TREE_LEVELS; gl = gl + 1) begin : g_tree_level
            localparam int unsigned COUNT_IN  = K >> gl;
            localparam int unsigned COUNT_OUT = COUNT_IN / 2;
            for (gj = 0; gj < N; gj = gj + 1) begin : g_tree_lane
                for (gidx = 0; gidx < COUNT_OUT; gidx = gidx + 1) begin : g_tree_node
                    logic signed [ACC_WIDTH-1:0] sum_comb;
                    assign sum_comb = tree_val[gl][gj][2*gidx] + tree_val[gl][gj][2*gidx+1];
                    cmvu_regbank #(.W(ACC_WIDTH), .PRESENT(REG_TREE_PRESENT[gl])) u_reg_tree (
                        .clk(clk), .rst(rst), .ena(1'b1),
                        .d(sum_comb), .q(tree_val[gl+1][gj][gidx])
                    );
                end
            end
        end
    endgenerate

    wire signed [ACC_WIDTH-1:0] tree_partial [0:N-1];
    generate
        for (gj = 0; gj < N; gj = gj + 1) begin : g_tree_out
            assign tree_partial[gj] = tree_val[TREE_LEVELS][gj][0];
        end
    endgenerate

    // ---------------------------------------------------------------------
    // Single-slot K-first accumulation: only acc_first/acc_last control accumulation.
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // Accumulation side-band delay: {acc_last, acc_first} travels with the pipeline,
    // delayed through the same register stages (REG_IN, REG_MULT, REG_TREE[0..TREE_LEVELS-1])
    // as tree_partial, so it lands aligned with the data at the reduce node's combinational
    // input (cascade_in itself needs no such delay -- see the cascade_in timing contract above).
    // ---------------------------------------------------------------------
    localparam int unsigned SIDE_BITS = 2;

    wire [SIDE_BITS-1:0] side_d0 = {acc_last, acc_first};
    wire [SIDE_BITS-1:0] side_d1, side_d2;
    wire [SIDE_BITS-1:0] side_tree [0:TREE_LEVELS];

    cmvu_regbank #(.W(SIDE_BITS), .PRESENT(1'b1))   u_side_in   (.clk(clk), .rst(rst), .ena(1'b1), .d(side_d0), .q(side_d1));
    cmvu_regbank #(.W(SIDE_BITS), .PRESENT(REG_MULT_PRESENT)) u_side_mult (.clk(clk), .rst(rst), .ena(1'b1), .d(side_d1), .q(side_d2));

    assign side_tree[0] = side_d2;
    generate
        for (gl = 0; gl < TREE_LEVELS; gl = gl + 1) begin : g_side_tree
            cmvu_regbank #(.W(SIDE_BITS), .PRESENT(REG_TREE_PRESENT[gl])) u_side_lvl (
                .clk(clk), .rst(rst), .ena(1'b1),
                .d(side_tree[gl]), .q(side_tree[gl+1])
            );
        end
    endgenerate

    wire [SIDE_BITS-1:0]  side_aligned    = side_tree[TREE_LEVELS];
    wire                  acc_first_aligned = side_aligned[0];
    wire                  acc_last_aligned  = side_aligned[1];

    // ---------------------------------------------------------------------
    // Bias matched-delay path (docs section 6, 8): bias_in must land at the reduce node aligned
    // with acc_first_aligned above, so it is delayed through the SAME register
    // stages (REG_IN, REG_MULT, REG_TREE[0..TREE_LEVELS-1]) as the side-band {acc_last,
    // acc_first} bundle -- a feed-forward side input, not part of the datapath itself
    // (adds no stage, does not change L or II).
    // ---------------------------------------------------------------------
    wire [N*BIAS_WIDTH-1:0] bias_d1, bias_d2;
    wire [N*BIAS_WIDTH-1:0] bias_tree [0:TREE_LEVELS];

    cmvu_regbank #(.W(N*BIAS_WIDTH), .PRESENT(1'b1))   u_bias_in   (.clk(clk), .rst(rst), .ena(1'b1), .d(bias_in), .q(bias_d1));
    cmvu_regbank #(.W(N*BIAS_WIDTH), .PRESENT(REG_MULT_PRESENT)) u_bias_mult (.clk(clk), .rst(rst), .ena(1'b1), .d(bias_d1), .q(bias_d2));

    assign bias_tree[0] = bias_d2;
    generate
        for (gl = 0; gl < TREE_LEVELS; gl = gl + 1) begin : g_bias_tree
            cmvu_regbank #(.W(N*BIAS_WIDTH), .PRESENT(REG_TREE_PRESENT[gl])) u_bias_lvl (
                .clk(clk), .rst(rst), .ena(1'b1),
                .d(bias_tree[gl]), .q(bias_tree[gl+1])
            );
        end
    endgenerate

    wire [N*BIAS_WIDTH-1:0] bias_aligned = bias_tree[TREE_LEVELS];

    // ---------------------------------------------------------------------
    // Cascade-in timing contract (docs/architecture.md sections 5, 7): cascade_out is tapped
    // at REG_RED (latency L-1), specifically so that chaining a cascade hop costs exactly 1
    // cycle -- NOT a re-delay through this block's own input pipeline. Accordingly, cascade_in
    // is consumed DIRECTLY (combinationally) at the reduce node below, alongside tree_partial;
    // it is NOT re-delayed through REG_IN/REG_MULT/REG_TREE copies.
    //
    // Caller/timing contract: cascade_in must be PRESENTED to this block D cycles after the
    // a_in slice it corresponds to, where D = 1(REG_IN, always present) + REG_MULT_PRESENT +
    // sum(REG_TREE_PRESENT[0..TREE_LEVELS-1]) (= 4 at the all-present default). That is exactly
    // the depth of the pipeline that produces tree_partial, so an upstream block's cascade_out
    // (available at its own L-1) needs only a further +1 cycle of input skew to align here --
    // the 1-cycle cascade hop the architecture calls for (see cmvu_array.sv for the multi-block
    // skew derivation).
    // ---------------------------------------------------------------------

    // ---------------------------------------------------------------------
    // valid delay (data-present) chain, declared here so v_tree[TREE_LEVELS] (valid aligned
    // with the reduce node's combinational input) is available to gate the accbuf write below.
    // v_red/y_valid/done are driven further down, after REG_RED/REG_OUT exist.
    // ---------------------------------------------------------------------
    wire v_in, v_mult, v_tree [0:TREE_LEVELS], v_red;

    cmvu_regbank #(.W(1), .PRESENT(1'b1))   u_v_in   (.clk(clk), .rst(rst), .ena(1'b1), .d(valid), .q(v_in));
    cmvu_regbank #(.W(1), .PRESENT(REG_MULT_PRESENT)) u_v_mult (.clk(clk), .rst(rst), .ena(1'b1), .d(v_in),  .q(v_mult));

    assign v_tree[0] = v_mult;
    generate
        for (gl = 0; gl < TREE_LEVELS; gl = gl + 1) begin : g_v_tree
            cmvu_regbank #(.W(1), .PRESENT(REG_TREE_PRESENT[gl])) u_v_lvl (
                .clk(clk), .rst(rst), .ena(1'b1),
                .d(v_tree[gl]), .q(v_tree[gl+1])
            );
        end
    endgenerate

    // ---------------------------------------------------------------------
    // Accumulator buffer: one running int32 accumulator register per lane. The reduce node
    // reads its prior value combinationally and writes the new value on the clock edge.
    // classic read-modify-write: the combinational read (acc_term below) samples the OLD
    // value, and the always_ff write below lands the new `sum` on the same clock edge, giving
    // the "read-before-write" semantics an RMW accumulate needs.
    // ---------------------------------------------------------------------

    // ---------------------------------------------------------------------
    // Reduce node: single-cycle combinational 3-input add (tree_partial + cascade_term +
    // acc_term), registered by REG_RED. sum_q (post REG_RED) is cascade_out (latency L-1).
    // acc_term is bias on a first touch, else the accumulator's running partial.
    // ---------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] sum_comb [0:N-1];
    logic signed [ACC_WIDTH-1:0] sum_q    [0:N-1];

    generate
        for (gj = 0; gj < N; gj = gj + 1) begin : g_reduce
            logic signed [ACC_WIDTH-1:0] accbuf;

            wire signed [ACC_WIDTH-1:0] casc_term =
                CASCADE_EN ? $signed(cascade_in[gj*ACC_WIDTH +: ACC_WIDTH]) : '0;
            wire signed [BIAS_WIDTH-1:0] bias_lane = $signed(bias_aligned[gj*BIAS_WIDTH +: BIAS_WIDTH]);
            // Sign-extend bias_lane (BIAS_WIDTH) into ACC_WIDTH. When BIAS_WIDTH == ACC_WIDTH
            // (the default, both 32) this is a straight pass-through: a zero-width replication
            // {{0{...}}, bias_lane} is not legal, so the two cases are generated separately.
            wire signed [ACC_WIDTH-1:0] bias_lane_aligned;
            if (ACC_WIDTH > BIAS_WIDTH) begin : g_bias_ext
                assign bias_lane_aligned =
                    {{(ACC_WIDTH-BIAS_WIDTH){bias_lane[BIAS_WIDTH-1]}}, bias_lane};
            end else begin : g_bias_noext
                assign bias_lane_aligned = $signed(bias_lane[ACC_WIDTH-1:0]);
            end
            wire signed [ACC_WIDTH-1:0] acc_term =
                acc_first_aligned ? bias_lane_aligned : accbuf;
            assign sum_comb[gj] = tree_partial[gj] + casc_term + acc_term;

            cmvu_regbank #(.W(ACC_WIDTH), .PRESENT(REG_RED_PRESENT)) u_reg_red (
                .clk(clk), .rst(rst), .ena(1'b1),
                .d(sum_comb[gj]), .q(sum_q[gj])
            );

            assign cascade_out[gj*ACC_WIDTH +: ACC_WIDTH] = sum_q[gj];

            // accbuf write-back: every accumulating cycle (any valid data reaching the
            // reduce node's combinational input, aligned via v_tree[TREE_LEVELS] below)
            // writes the just-computed running partial back to the lane accumulator.
            always_ff @(posedge clk or posedge rst) begin
                if (rst) accbuf <= '0;
                else if (v_tree[TREE_LEVELS]) accbuf <= sum_comb[gj];
            end
        end
    endgenerate

    // ---------------------------------------------------------------------
    // Requant control delay: {out_w, shift_amt} aligned with sum_q, i.e. delayed through REG_IN,
    // REG_MULT, REG_TREE[0..TREE_LEVELS-1], REG_RED. out_w selects the effective per-lane
    // result width W = out_w + 1 (1..RESULT_WIDTH).
    // ---------------------------------------------------------------------
    localparam int unsigned OUT_W_WIDTH = (RESULT_WIDTH < 2) ? 1 : $clog2(RESULT_WIDTH);
    localparam int unsigned CTRL_W     = SHIFT_WIDTH + OUT_W_WIDTH;

    wire [CTRL_W-1:0] ctrl_tree [0:TREE_LEVELS];
    wire [CTRL_W-1:0] ctrl_d1, ctrl_d2, ctrl_delayed;
    wire [SHIFT_WIDTH-1:0] shift_delayed   = ctrl_delayed[SHIFT_WIDTH-1:0];
    wire [OUT_W_WIDTH-1:0] out_w_delayed   = ctrl_delayed[SHIFT_WIDTH +: OUT_W_WIDTH];

    cmvu_regbank #(.W(CTRL_W), .PRESENT(1'b1))   u_ctrl_in   (.clk(clk), .rst(rst), .ena(1'b1), .d({out_w, shift_amt}), .q(ctrl_d1));
    cmvu_regbank #(.W(CTRL_W), .PRESENT(REG_MULT_PRESENT)) u_ctrl_mult (.clk(clk), .rst(rst), .ena(1'b1), .d(ctrl_d1), .q(ctrl_d2));

    assign ctrl_tree[0] = ctrl_d2;
    generate
        for (gl = 0; gl < TREE_LEVELS; gl = gl + 1) begin : g_ctrl_tree
            cmvu_regbank #(.W(CTRL_W), .PRESENT(REG_TREE_PRESENT[gl])) u_ctrl_lvl (
                .clk(clk), .rst(rst), .ena(1'b1),
                .d(ctrl_tree[gl]), .q(ctrl_tree[gl+1])
            );
        end
    endgenerate

    cmvu_regbank #(.W(CTRL_W), .PRESENT(REG_RED_PRESENT)) u_ctrl_red (
        .clk(clk), .rst(rst), .ena(1'b1),
        .d(ctrl_tree[TREE_LEVELS]), .q(ctrl_delayed)
    );

    // ---------------------------------------------------------------------
    // Requant (combinational, between REG_RED and REG_OUT): plain arithmetic (floor/TRN) shift,
    // then WRAP to the runtime effective width W = out_w+1 and sign-extend that W-bit value
    // into the full RESULT_WIDTH lane. See docs/architecture.md section 2.
    //
    // No rounding happens in this block: the round-half-up adder and its ACC_WIDTH+1 guard bit
    // are gone. Rounding is now the integrator's responsibility, folded into the 32-bit
    // `bias_in` (accumulator scale) on acc_first -- e.g. RND output adds 2^(shift_amt-1) into
    // the bias before this block ever sees it (see docs/architecture.md section 8 and the cmvu
    // generator's golden model). This one requant block therefore serves both RND and TRN
    // hls4ml output modes exactly; it does not itself distinguish them.
    // ---------------------------------------------------------------------
    logic [N*RESULT_WIDTH-1:0] y_pre;

    generate
        for (gj = 0; gj < N; gj = gj + 1) begin : g_requant
            logic signed [ACC_WIDTH-1:0] q;
            logic [RESULT_WIDTH-1:0] q_trunc;
            logic [OUT_W_WIDTH-1:0]  sh;
            if (REQUANT_PRESENT) begin : g_req_on
                assign q = sum_q[gj] >>> shift_delayed;
                // Left-shift q by sh = RESULT_WIDTH - W so q[W-1] lands in the lane MSB (low sh
                // bits become zero), truncate to RESULT_WIDTH, then arithmetic-shift back: the
                // net effect is wrap-to-W followed by sign-extension to RESULT_WIDTH. sh = 0
                // reproduces a plain RESULT_WIDTH wrap.
                assign sh = (RESULT_WIDTH - 1) - out_w_delayed;
                assign q_trunc = q << sh;
                assign y_pre[gj*RESULT_WIDTH +: RESULT_WIDTH] = $signed(q_trunc) >>> sh;
            end else begin : g_req_off
                assign y_pre[gj*RESULT_WIDTH +: RESULT_WIDTH] = sum_q[gj][RESULT_WIDTH-1:0];
            end
        end
    endgenerate

    cmvu_regbank #(.W(N*RESULT_WIDTH), .PRESENT(REG_OUT_PRESENT)) u_reg_out (
        .clk(clk), .rst(rst), .ena(1'b1), .d(y_pre), .q(y_out)
    );

    // ---------------------------------------------------------------------
    // valid delay chain: y_valid = valid delayed by L; matches the data-path register presence
    // stage-for-stage so it stays correct even when a stage is configured as bypassed.
    // ---------------------------------------------------------------------
    cmvu_regbank #(.W(1), .PRESENT(REG_RED_PRESENT)) u_v_red (.clk(clk), .rst(rst), .ena(1'b1), .d(v_tree[TREE_LEVELS]), .q(v_red));
    cmvu_regbank #(.W(1), .PRESENT(REG_OUT_PRESENT)) u_v_out (.clk(clk), .rst(rst), .ena(1'b1), .d(v_red), .q(y_valid));

    // ---------------------------------------------------------------------
    // done: acc_last delayed by L (same full chain as y_valid, starting from the
    // reduce-aligned acc_last_aligned so it lines up with the completed/emitted sum).
    // ---------------------------------------------------------------------
    wire done_red;
    cmvu_regbank #(.W(1), .PRESENT(REG_RED_PRESENT)) u_done_red (.clk(clk), .rst(rst), .ena(1'b1), .d(acc_last_aligned), .q(done_red));
    cmvu_regbank #(.W(1), .PRESENT(REG_OUT_PRESENT)) u_done_out (.clk(clk), .rst(rst), .ena(1'b1), .d(done_red),         .q(done));

endmodule
