// Final architecture (v3) recreation of the tensor_slice_int8_atlas hardblock.
// Spec: ts-rtl-work/tracker/tensor-slice-int8-final-architecture.md
//
// Reused: 8x8 grid, a shifts east, b shifts south, signed int8 multiply,
// 32-bit accumulators, pe_reset clears.
// New: A row / B column staged per boundary tile, serial shift into the grid
// edge (first byte bypassed from the pin on the load cycle), chain taps taken
// directly from the last column/row register (8-cycle hop), progressive
// per-row shadow snapshots, combinational readout gated by op[0].
// Rev-3 (spec §3A): overlapped back-to-back waves. `bc` free-runs on en
// edges; each start allocates a wave slot (t0, op[2] commit tag) and all
// load/capture compares key off per-slot age = bc - t0. A committed slot
// captures cell (i,j) at age loc+i+j+8 (same-edge k7 product) into shadow,
// clearing the cell's accumulator; row-ready is 64 per-cell cell_flags; emission
// is one row per edge gated by op[0]; `done_mat_mul` is a per-frame pulse at
// cell (0,7). op[1] truncates the current tail and re-arms at the next
// committed row-0 capture. `pe_reset` cold-clears only when provably idle.
// All sequential state advances only while en==1 (the core-level clock
// enable); en==0 freezes the schedule, snapshots, and the array so wrapper
// and slices stay in lockstep across wait states.
// Stage-1 shift is a 4-bit field driven directly by the `shift_amount` pin,
// latched at start. Each c_data_out lane is 32 bits, so with a zero shift the
// row leaves the slice as the full accumulator.

// Hard block: VTR/parmys (yosys) must treat this definition as a black box and
// map instances to the arch model of the same name. The `blackbox` attribute
// makes yosys discard the body; simulators (iverilog/Questa) ignore it and
// use the behavioural model below.
(* blackbox *)
module tensor_slice_int8_atlas (
    input  wire        clk,
    input  wire        reset,
    input  wire        en,
    input  wire        pe_reset,
    input  wire        start_mat_mul,
    output wire        done_mat_mul,

    input  wire [63:0] a_data,
    input  wire [63:0] b_data,
    input  wire [63:0] a_data_in,
    input  wire [63:0] b_data_in,
    output wire [63:0] a_data_out,
    output wire [63:0] b_data_out,

    output wire [255:0] c_data_out,
    output wire         c_data_available,

    input  wire [7:0]  validity_mask_a_rows,
    input  wire [7:0]  validity_mask_a_cols_b_rows,
    input  wire [7:0]  validity_mask_b_cols,

    input  wire [2:0]  op,
    input  wire [7:0]  final_mat_mul_size,

    input  wire [4:0]  a_loc,
    input  wire [4:0]  b_loc,

    input  wire [3:0]  shift_amount
);

    // ---------------------------------------------------------------- shift
    wire [3:0] shift_ports = shift_amount;
    reg  [3:0] shift_lat;

    // --------------------------------------------------------- shadow/readout
    // Declared before the schedule because the idle detector reads `cell_flags`.
    reg signed [31:0] shadow [0:7][0:7];
    // cell_flags[i*8+j] = cell (i,j) has been captured for the committed wave.
    reg [63:0] cell_flags;
    reg [2:0]  ptr;            // next row to emit; walks 0..7 and wraps
    reg        trunc;
    reg        done_r;

    // ------------------------------------------------------------ schedule
    wire [15:0] loca = {11'd0, a_loc} << 3;
    wire [15:0] locb = {11'd0, b_loc} << 3;
    wire [15:0] loc  = loca + locb;

    // Rev-3 can keep at most eight waves in flight. The final wrapper uses
    // tile locations that need no more than six concurrent waves at F=8; any
    // additional concurrent wave is a protocol violation.
    localparam integer NSLOT = 8;
    reg  [15:0] bc;
    reg  [15:0] slot_t0 [0:NSLOT-1];
    reg         slot_commit [0:NSLOT-1];
    reg  [NSLOT-1:0] slot_live;

    // Start events only count on enabled edges. A new slot uses the current
    // bc as age 0 during the start edge.
    wire start_ev = start_mat_mul && en;
    wire busy_pre = (|slot_live) || (|cell_flags);
    wire idle_clear = pe_reset && !busy_pre && !start_mat_mul;

    // A slot retiring on this edge may immediately be reused for a new start.
    wire [NSLOT-1:0] slot_retire;
    wire [NSLOT-1:0] slot_available;
    reg [2:0] alloc_idx;
    reg alloc_full;
    integer sa;
    always @* begin
        alloc_idx = 3'd0;
        alloc_full = 1'b1;
        for (sa = 0; sa < NSLOT; sa = sa + 1) begin
            if (alloc_full && slot_available[sa]) begin
                // sa ranges 0..NSLOT-1; NSLOT is 8.
                alloc_idx = sa[2:0];
                alloc_full = 1'b0;
            end
        end
    end

    // Effective live/commit/age on this edge, including the slot that is
    // being allocated by the current start event. Load and capture windows
    // must use these values, not only the stored registers.
    wire [NSLOT-1:0] slot_eff_live;
    wire [NSLOT-1:0] slot_eff_commit;
    wire [15:0] slot_age [0:NSLOT-1];
    genvar gs;
    generate
        for (gs = 0; gs < NSLOT; gs = gs + 1) begin: g_slots
            assign slot_eff_live[gs] =
                slot_live[gs] ||
                (start_ev && !alloc_full && (gs == alloc_idx));
            assign slot_eff_commit[gs] =
                slot_commit[gs] ||
                (start_ev && !alloc_full && (gs == alloc_idx) && op[2]);
            assign slot_age[gs] =
                bc - ((start_ev && !alloc_full && (gs == alloc_idx)) ?
                      bc : slot_t0[gs]);
            assign slot_retire[gs] =
                slot_live[gs] &&
                ((bc - slot_t0[gs]) > (loc + 16'd23));
        end
    endgenerate
    assign slot_available = ~slot_live | slot_retire;

    // Slot state: allocate on a start, retire when the wave's schedule is
    // over (last capture at age loc+22), cold-clear on an idle pe_reset. A
    // slot retiring on the current edge can be reallocated by the same start.
    always @(posedge clk) begin
        if (reset) begin
            slot_live <= {NSLOT{1'b0}};
            for (sa = 0; sa < NSLOT; sa = sa + 1) begin
                slot_t0[sa]     <= 16'd0;
                slot_commit[sa] <= 1'b0;
            end
        end else if (en) begin
            for (sa = 0; sa < NSLOT; sa = sa + 1)
                if (slot_retire[sa])
                    slot_live[sa] <= 1'b0;
            if (start_ev && !alloc_full) begin
                slot_live[alloc_idx]   <= 1'b1;
                slot_t0[alloc_idx]     <= bc;
                slot_commit[alloc_idx] <= op[2];
            end
            if (idle_clear)
                slot_live <= {NSLOT{1'b0}};
        end
    end

    // At most one live wave can load a shared staging register on an edge:
    // every per-wave load window is eight edges wide and start spacing is at
    // least eight, so overlapping matches are a protocol violation (checked
    // by the directed benches).
    reg a_hit;
    reg [2:0] a_hit_row;
    reg b_hit;
    reg [2:0] b_hit_col;
    reg [15:0] la;
    integer ls;
    always @* begin
        a_hit = 1'b0;
        a_hit_row = 3'd0;
        b_hit = 1'b0;
        b_hit_col = 3'd0;
        for (ls = 0; ls < NSLOT; ls = ls + 1) begin
            if (slot_eff_live[ls]) begin
                la = slot_age[ls];
                if ((b_loc == 5'd0) && (la >= loca) && (la < (loca + 16'd8))) begin
                    a_hit = 1'b1;
                    a_hit_row = la[2:0];
                end
                if ((a_loc == 5'd0) && (la >= locb) && (la < (locb + 16'd8))) begin
                    b_hit = 1'b1;
                    b_hit_col = la[2:0];
                end
            end
        end
    end

    wire        a_ld_win = (b_loc == 5'd0) && a_hit;
    wire        b_ld_win = (a_loc == 5'd0) && b_hit;
    wire [2:0]  a_ld_row = a_hit_row;
    wire [2:0]  b_ld_col = b_hit_col;

    wire [7:0]  kmask = validity_mask_a_cols_b_rows;

    // ------------------------------------------------- boundary staging (A)
    // a_row_q[i] holds the row word for row i, LSB = next K lane. On the load
    // cycle the k=0 byte is bypassed straight from the pin and the register
    // captures the word pre-shifted by one byte (k=1 at LSB).
    reg [63:0] a_row_q [0:7];
    reg [63:0] b_col_q [0:7];

    wire a_ld_i_0 = a_ld_win && (a_ld_row == 3'd0);
    wire a_ld_i_1 = a_ld_win && (a_ld_row == 3'd1);
    wire a_ld_i_2 = a_ld_win && (a_ld_row == 3'd2);
    wire a_ld_i_3 = a_ld_win && (a_ld_row == 3'd3);
    wire a_ld_i_4 = a_ld_win && (a_ld_row == 3'd4);
    wire a_ld_i_5 = a_ld_win && (a_ld_row == 3'd5);
    wire a_ld_i_6 = a_ld_win && (a_ld_row == 3'd6);
    wire a_ld_i_7 = a_ld_win && (a_ld_row == 3'd7);

    wire b_ld_j_0 = b_ld_win && (b_ld_col == 3'd0);
    wire b_ld_j_1 = b_ld_win && (b_ld_col == 3'd1);
    wire b_ld_j_2 = b_ld_win && (b_ld_col == 3'd2);
    wire b_ld_j_3 = b_ld_win && (b_ld_col == 3'd3);
    wire b_ld_j_4 = b_ld_win && (b_ld_col == 3'd4);
    wire b_ld_j_5 = b_ld_win && (b_ld_col == 3'd5);
    wire b_ld_j_6 = b_ld_win && (b_ld_col == 3'd6);
    wire b_ld_j_7 = b_ld_win && (b_ld_col == 3'd7);

    wire [63:0] a_preshift_0 = {8'd0, (validity_mask_a_rows[0] && kmask[7]) ? a_data[63:56] : 8'd0,
                                      (validity_mask_a_rows[0] && kmask[6]) ? a_data[55:48] : 8'd0,
                                      (validity_mask_a_rows[0] && kmask[5]) ? a_data[47:40] : 8'd0,
                                      (validity_mask_a_rows[0] && kmask[4]) ? a_data[39:32] : 8'd0,
                                      (validity_mask_a_rows[0] && kmask[3]) ? a_data[31:24] : 8'd0,
                                      (validity_mask_a_rows[0] && kmask[2]) ? a_data[23:16] : 8'd0,
                                      (validity_mask_a_rows[0] && kmask[1]) ? a_data[15:8]  : 8'd0};
    wire [63:0] a_preshift_1 = {8'd0, (validity_mask_a_rows[1] && kmask[7]) ? a_data[63:56] : 8'd0,
                                      (validity_mask_a_rows[1] && kmask[6]) ? a_data[55:48] : 8'd0,
                                      (validity_mask_a_rows[1] && kmask[5]) ? a_data[47:40] : 8'd0,
                                      (validity_mask_a_rows[1] && kmask[4]) ? a_data[39:32] : 8'd0,
                                      (validity_mask_a_rows[1] && kmask[3]) ? a_data[31:24] : 8'd0,
                                      (validity_mask_a_rows[1] && kmask[2]) ? a_data[23:16] : 8'd0,
                                      (validity_mask_a_rows[1] && kmask[1]) ? a_data[15:8]  : 8'd0};
    wire [63:0] a_preshift_2 = {8'd0, (validity_mask_a_rows[2] && kmask[7]) ? a_data[63:56] : 8'd0,
                                      (validity_mask_a_rows[2] && kmask[6]) ? a_data[55:48] : 8'd0,
                                      (validity_mask_a_rows[2] && kmask[5]) ? a_data[47:40] : 8'd0,
                                      (validity_mask_a_rows[2] && kmask[4]) ? a_data[39:32] : 8'd0,
                                      (validity_mask_a_rows[2] && kmask[3]) ? a_data[31:24] : 8'd0,
                                      (validity_mask_a_rows[2] && kmask[2]) ? a_data[23:16] : 8'd0,
                                      (validity_mask_a_rows[2] && kmask[1]) ? a_data[15:8]  : 8'd0};
    wire [63:0] a_preshift_3 = {8'd0, (validity_mask_a_rows[3] && kmask[7]) ? a_data[63:56] : 8'd0,
                                      (validity_mask_a_rows[3] && kmask[6]) ? a_data[55:48] : 8'd0,
                                      (validity_mask_a_rows[3] && kmask[5]) ? a_data[47:40] : 8'd0,
                                      (validity_mask_a_rows[3] && kmask[4]) ? a_data[39:32] : 8'd0,
                                      (validity_mask_a_rows[3] && kmask[3]) ? a_data[31:24] : 8'd0,
                                      (validity_mask_a_rows[3] && kmask[2]) ? a_data[23:16] : 8'd0,
                                      (validity_mask_a_rows[3] && kmask[1]) ? a_data[15:8]  : 8'd0};
    wire [63:0] a_preshift_4 = {8'd0, (validity_mask_a_rows[4] && kmask[7]) ? a_data[63:56] : 8'd0,
                                      (validity_mask_a_rows[4] && kmask[6]) ? a_data[55:48] : 8'd0,
                                      (validity_mask_a_rows[4] && kmask[5]) ? a_data[47:40] : 8'd0,
                                      (validity_mask_a_rows[4] && kmask[4]) ? a_data[39:32] : 8'd0,
                                      (validity_mask_a_rows[4] && kmask[3]) ? a_data[31:24] : 8'd0,
                                      (validity_mask_a_rows[4] && kmask[2]) ? a_data[23:16] : 8'd0,
                                      (validity_mask_a_rows[4] && kmask[1]) ? a_data[15:8]  : 8'd0};
    wire [63:0] a_preshift_5 = {8'd0, (validity_mask_a_rows[5] && kmask[7]) ? a_data[63:56] : 8'd0,
                                      (validity_mask_a_rows[5] && kmask[6]) ? a_data[55:48] : 8'd0,
                                      (validity_mask_a_rows[5] && kmask[5]) ? a_data[47:40] : 8'd0,
                                      (validity_mask_a_rows[5] && kmask[4]) ? a_data[39:32] : 8'd0,
                                      (validity_mask_a_rows[5] && kmask[3]) ? a_data[31:24] : 8'd0,
                                      (validity_mask_a_rows[5] && kmask[2]) ? a_data[23:16] : 8'd0,
                                      (validity_mask_a_rows[5] && kmask[1]) ? a_data[15:8]  : 8'd0};
    wire [63:0] a_preshift_6 = {8'd0, (validity_mask_a_rows[6] && kmask[7]) ? a_data[63:56] : 8'd0,
                                      (validity_mask_a_rows[6] && kmask[6]) ? a_data[55:48] : 8'd0,
                                      (validity_mask_a_rows[6] && kmask[5]) ? a_data[47:40] : 8'd0,
                                      (validity_mask_a_rows[6] && kmask[4]) ? a_data[39:32] : 8'd0,
                                      (validity_mask_a_rows[6] && kmask[3]) ? a_data[31:24] : 8'd0,
                                      (validity_mask_a_rows[6] && kmask[2]) ? a_data[23:16] : 8'd0,
                                      (validity_mask_a_rows[6] && kmask[1]) ? a_data[15:8]  : 8'd0};
    wire [63:0] a_preshift_7 = {8'd0, (validity_mask_a_rows[7] && kmask[7]) ? a_data[63:56] : 8'd0,
                                      (validity_mask_a_rows[7] && kmask[6]) ? a_data[55:48] : 8'd0,
                                      (validity_mask_a_rows[7] && kmask[5]) ? a_data[47:40] : 8'd0,
                                      (validity_mask_a_rows[7] && kmask[4]) ? a_data[39:32] : 8'd0,
                                      (validity_mask_a_rows[7] && kmask[3]) ? a_data[31:24] : 8'd0,
                                      (validity_mask_a_rows[7] && kmask[2]) ? a_data[23:16] : 8'd0,
                                      (validity_mask_a_rows[7] && kmask[1]) ? a_data[15:8]  : 8'd0};

    wire [63:0] b_preshift_0 = {8'd0, (validity_mask_b_cols[0] && kmask[7]) ? b_data[63:56] : 8'd0,
                                      (validity_mask_b_cols[0] && kmask[6]) ? b_data[55:48] : 8'd0,
                                      (validity_mask_b_cols[0] && kmask[5]) ? b_data[47:40] : 8'd0,
                                      (validity_mask_b_cols[0] && kmask[4]) ? b_data[39:32] : 8'd0,
                                      (validity_mask_b_cols[0] && kmask[3]) ? b_data[31:24] : 8'd0,
                                      (validity_mask_b_cols[0] && kmask[2]) ? b_data[23:16] : 8'd0,
                                      (validity_mask_b_cols[0] && kmask[1]) ? b_data[15:8]  : 8'd0};
    wire [63:0] b_preshift_1 = {8'd0, (validity_mask_b_cols[1] && kmask[7]) ? b_data[63:56] : 8'd0,
                                      (validity_mask_b_cols[1] && kmask[6]) ? b_data[55:48] : 8'd0,
                                      (validity_mask_b_cols[1] && kmask[5]) ? b_data[47:40] : 8'd0,
                                      (validity_mask_b_cols[1] && kmask[4]) ? b_data[39:32] : 8'd0,
                                      (validity_mask_b_cols[1] && kmask[3]) ? b_data[31:24] : 8'd0,
                                      (validity_mask_b_cols[1] && kmask[2]) ? b_data[23:16] : 8'd0,
                                      (validity_mask_b_cols[1] && kmask[1]) ? b_data[15:8]  : 8'd0};
    wire [63:0] b_preshift_2 = {8'd0, (validity_mask_b_cols[2] && kmask[7]) ? b_data[63:56] : 8'd0,
                                      (validity_mask_b_cols[2] && kmask[6]) ? b_data[55:48] : 8'd0,
                                      (validity_mask_b_cols[2] && kmask[5]) ? b_data[47:40] : 8'd0,
                                      (validity_mask_b_cols[2] && kmask[4]) ? b_data[39:32] : 8'd0,
                                      (validity_mask_b_cols[2] && kmask[3]) ? b_data[31:24] : 8'd0,
                                      (validity_mask_b_cols[2] && kmask[2]) ? b_data[23:16] : 8'd0,
                                      (validity_mask_b_cols[2] && kmask[1]) ? b_data[15:8]  : 8'd0};
    wire [63:0] b_preshift_3 = {8'd0, (validity_mask_b_cols[3] && kmask[7]) ? b_data[63:56] : 8'd0,
                                      (validity_mask_b_cols[3] && kmask[6]) ? b_data[55:48] : 8'd0,
                                      (validity_mask_b_cols[3] && kmask[5]) ? b_data[47:40] : 8'd0,
                                      (validity_mask_b_cols[3] && kmask[4]) ? b_data[39:32] : 8'd0,
                                      (validity_mask_b_cols[3] && kmask[3]) ? b_data[31:24] : 8'd0,
                                      (validity_mask_b_cols[3] && kmask[2]) ? b_data[23:16] : 8'd0,
                                      (validity_mask_b_cols[3] && kmask[1]) ? b_data[15:8]  : 8'd0};
    wire [63:0] b_preshift_4 = {8'd0, (validity_mask_b_cols[4] && kmask[7]) ? b_data[63:56] : 8'd0,
                                      (validity_mask_b_cols[4] && kmask[6]) ? b_data[55:48] : 8'd0,
                                      (validity_mask_b_cols[4] && kmask[5]) ? b_data[47:40] : 8'd0,
                                      (validity_mask_b_cols[4] && kmask[4]) ? b_data[39:32] : 8'd0,
                                      (validity_mask_b_cols[4] && kmask[3]) ? b_data[31:24] : 8'd0,
                                      (validity_mask_b_cols[4] && kmask[2]) ? b_data[23:16] : 8'd0,
                                      (validity_mask_b_cols[4] && kmask[1]) ? b_data[15:8]  : 8'd0};
    wire [63:0] b_preshift_5 = {8'd0, (validity_mask_b_cols[5] && kmask[7]) ? b_data[63:56] : 8'd0,
                                      (validity_mask_b_cols[5] && kmask[6]) ? b_data[55:48] : 8'd0,
                                      (validity_mask_b_cols[5] && kmask[5]) ? b_data[47:40] : 8'd0,
                                      (validity_mask_b_cols[5] && kmask[4]) ? b_data[39:32] : 8'd0,
                                      (validity_mask_b_cols[5] && kmask[3]) ? b_data[31:24] : 8'd0,
                                      (validity_mask_b_cols[5] && kmask[2]) ? b_data[23:16] : 8'd0,
                                      (validity_mask_b_cols[5] && kmask[1]) ? b_data[15:8]  : 8'd0};
    wire [63:0] b_preshift_6 = {8'd0, (validity_mask_b_cols[6] && kmask[7]) ? b_data[63:56] : 8'd0,
                                      (validity_mask_b_cols[6] && kmask[6]) ? b_data[55:48] : 8'd0,
                                      (validity_mask_b_cols[6] && kmask[5]) ? b_data[47:40] : 8'd0,
                                      (validity_mask_b_cols[6] && kmask[4]) ? b_data[39:32] : 8'd0,
                                      (validity_mask_b_cols[6] && kmask[3]) ? b_data[31:24] : 8'd0,
                                      (validity_mask_b_cols[6] && kmask[2]) ? b_data[23:16] : 8'd0,
                                      (validity_mask_b_cols[6] && kmask[1]) ? b_data[15:8]  : 8'd0};
    wire [63:0] b_preshift_7 = {8'd0, (validity_mask_b_cols[7] && kmask[7]) ? b_data[63:56] : 8'd0,
                                      (validity_mask_b_cols[7] && kmask[6]) ? b_data[55:48] : 8'd0,
                                      (validity_mask_b_cols[7] && kmask[5]) ? b_data[47:40] : 8'd0,
                                      (validity_mask_b_cols[7] && kmask[4]) ? b_data[39:32] : 8'd0,
                                      (validity_mask_b_cols[7] && kmask[3]) ? b_data[31:24] : 8'd0,
                                      (validity_mask_b_cols[7] && kmask[2]) ? b_data[23:16] : 8'd0,
                                      (validity_mask_b_cols[7] && kmask[1]) ? b_data[15:8]  : 8'd0};

    // --------------------------------------------------------- grid entries
    wire signed [7:0] edge_a_0 = (b_loc != 5'd0) ? a_data_in[7:0] :
                                 a_ld_i_0 ? ((validity_mask_a_rows[0] && kmask[0]) ? a_data[7:0]  : 8'd0) : a_row_q[0][7:0];
    wire signed [7:0] edge_a_1 = (b_loc != 5'd0) ? a_data_in[15:8] :
                                 a_ld_i_1 ? ((validity_mask_a_rows[1] && kmask[0]) ? a_data[7:0]  : 8'd0) : a_row_q[1][7:0];
    wire signed [7:0] edge_a_2 = (b_loc != 5'd0) ? a_data_in[23:16] :
                                 a_ld_i_2 ? ((validity_mask_a_rows[2] && kmask[0]) ? a_data[7:0]  : 8'd0) : a_row_q[2][7:0];
    wire signed [7:0] edge_a_3 = (b_loc != 5'd0) ? a_data_in[31:24] :
                                 a_ld_i_3 ? ((validity_mask_a_rows[3] && kmask[0]) ? a_data[7:0]  : 8'd0) : a_row_q[3][7:0];
    wire signed [7:0] edge_a_4 = (b_loc != 5'd0) ? a_data_in[39:32] :
                                 a_ld_i_4 ? ((validity_mask_a_rows[4] && kmask[0]) ? a_data[7:0]  : 8'd0) : a_row_q[4][7:0];
    wire signed [7:0] edge_a_5 = (b_loc != 5'd0) ? a_data_in[47:40] :
                                 a_ld_i_5 ? ((validity_mask_a_rows[5] && kmask[0]) ? a_data[7:0]  : 8'd0) : a_row_q[5][7:0];
    wire signed [7:0] edge_a_6 = (b_loc != 5'd0) ? a_data_in[55:48] :
                                 a_ld_i_6 ? ((validity_mask_a_rows[6] && kmask[0]) ? a_data[7:0]  : 8'd0) : a_row_q[6][7:0];
    wire signed [7:0] edge_a_7 = (b_loc != 5'd0) ? a_data_in[63:56] :
                                 a_ld_i_7 ? ((validity_mask_a_rows[7] && kmask[0]) ? a_data[7:0]  : 8'd0) : a_row_q[7][7:0];

    wire signed [7:0] edge_b_0 = (a_loc != 5'd0) ? b_data_in[7:0] :
                                 b_ld_j_0 ? ((validity_mask_b_cols[0] && kmask[0]) ? b_data[7:0]  : 8'd0) : b_col_q[0][7:0];
    wire signed [7:0] edge_b_1 = (a_loc != 5'd0) ? b_data_in[15:8] :
                                 b_ld_j_1 ? ((validity_mask_b_cols[1] && kmask[0]) ? b_data[7:0]  : 8'd0) : b_col_q[1][7:0];
    wire signed [7:0] edge_b_2 = (a_loc != 5'd0) ? b_data_in[23:16] :
                                 b_ld_j_2 ? ((validity_mask_b_cols[2] && kmask[0]) ? b_data[7:0]  : 8'd0) : b_col_q[2][7:0];
    wire signed [7:0] edge_b_3 = (a_loc != 5'd0) ? b_data_in[31:24] :
                                 b_ld_j_3 ? ((validity_mask_b_cols[3] && kmask[0]) ? b_data[7:0]  : 8'd0) : b_col_q[3][7:0];
    wire signed [7:0] edge_b_4 = (a_loc != 5'd0) ? b_data_in[39:32] :
                                 b_ld_j_4 ? ((validity_mask_b_cols[4] && kmask[0]) ? b_data[7:0]  : 8'd0) : b_col_q[4][7:0];
    wire signed [7:0] edge_b_5 = (a_loc != 5'd0) ? b_data_in[47:40] :
                                 b_ld_j_5 ? ((validity_mask_b_cols[5] && kmask[0]) ? b_data[7:0]  : 8'd0) : b_col_q[5][7:0];
    wire signed [7:0] edge_b_6 = (a_loc != 5'd0) ? b_data_in[55:48] :
                                 b_ld_j_6 ? ((validity_mask_b_cols[6] && kmask[0]) ? b_data[7:0]  : 8'd0) : b_col_q[6][7:0];
    wire signed [7:0] edge_b_7 = (a_loc != 5'd0) ? b_data_in[63:56] :
                                 b_ld_j_7 ? ((validity_mask_b_cols[7] && kmask[0]) ? b_data[7:0]  : 8'd0) : b_col_q[7][7:0];

    reg signed [7:0] ga [0:7][0:7];
    reg signed [7:0] gb [0:7][0:7];
    reg signed [31:0] acc [0:7][0:7];

    // Per-cell capture. A live committed wave owns a one-hot capture
    // diagonal: at age loc+8+ci+cj its cell (ci,cj) forms its last (k7)
    // product. The pre-edge acc holds k0..k6 and ga*gb is the k7 term, so the
    // same-edge sum is the complete value. The capture writes shadow and
    // clears the accumulator in the same edge, freeing the cell for the next
    // wave (F>=8). With F>=8 at most two capture windows overlap, so ORing
    // the per-slot diagonals is exact.
    reg [14:0] slot_diag [0:NSLOT-1];
    integer ds;
    always @* begin
        for (ds = 0; ds < NSLOT; ds = ds + 1) begin
            slot_diag[ds] = 15'd0;
            if (slot_eff_live[ds] && slot_eff_commit[ds] &&
                (slot_age[ds] >= (loc + 16'd8)) &&
                (slot_age[ds] <= (loc + 16'd22)))
                slot_diag[ds][slot_age[ds] - loc - 16'd8] = 1'b1;
        end
    end

    wire [63:0] cap;
    genvar gi, gj;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin: g_cap_i
            for (gj = 0; gj < 8; gj = gj + 1) begin: g_cap_j
                assign cap[gi*8+gj] = slot_diag[0][gi+gj] |
                                      slot_diag[1][gi+gj] |
                                      slot_diag[2][gi+gj] |
                                      slot_diag[3][gi+gj] |
                                      slot_diag[4][gi+gj] |
                                      slot_diag[5][gi+gj] |
                                      slot_diag[6][gi+gj] |
                                      slot_diag[7][gi+gj];
            end
        end
    endgenerate

    wire [7:0] row_ready;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin: g_row_ready
            assign row_ready[gi] = &cell_flags[gi*8 +: 8];
        end
    endgenerate

    wire emit = (op[0] == 1'b0) && row_ready[ptr] && (!trunc);

    function signed [31:0] stage1(input signed [31:0] x);
        stage1 = (shift_lat > 0) ? ((x + (32'sd1 << (shift_lat - 1))) >>> shift_lat) : x;
    endfunction

    wire signed [31:0] lane0 = stage1(shadow[ptr][0]);
    wire signed [31:0] lane1 = stage1(shadow[ptr][1]);
    wire signed [31:0] lane2 = stage1(shadow[ptr][2]);
    wire signed [31:0] lane3 = stage1(shadow[ptr][3]);
    wire signed [31:0] lane4 = stage1(shadow[ptr][4]);
    wire signed [31:0] lane5 = stage1(shadow[ptr][5]);
    wire signed [31:0] lane6 = stage1(shadow[ptr][6]);
    wire signed [31:0] lane7 = stage1(shadow[ptr][7]);

    assign c_data_out = {lane7, lane6, lane5, lane4, lane3, lane2, lane1, lane0};
    assign c_data_available = emit;
    assign done_mat_mul = done_r;

    // chain taps: taken directly from the last column/row register
    assign a_data_out = {ga[7][7], ga[6][7], ga[5][7], ga[4][7], ga[3][7], ga[2][7], ga[1][7], ga[0][7]};
    assign b_data_out = {gb[7][7], gb[7][6], gb[7][5], gb[7][4], gb[7][3], gb[7][2], gb[7][1], gb[7][0]};

    // ------------------------------------------------------------- sequencing
    integer i, j;

    always @(posedge clk) begin
        if (reset) begin
            bc         <= 16'd0;
            shift_lat  <= 4'd0;
            cell_flags      <= 64'd0;
            ptr        <= 3'd0;
            trunc      <= 1'b0;
            done_r     <= 1'b0;
        end else if (en) begin
            // emit advance first: start / pe_reset / op[1] below override it
            if (emit) begin
                ptr <= ptr + 3'd1;              // wraps 7 -> 0
                cell_flags[ptr*8 +: 8] <= 8'd0;      // row taken: release its cell_flags
            end

            // Captured cells set their cell_flags AFTER the take clear above, so a
            // same-edge take/capture tie (F=8) leaves the flag set for the
            // new wave; otherwise the row would never complete.
            for (i = 0; i < 8; i = i + 1)
                for (j = 0; j < 8; j = j + 1)
                    if (cap[i*8+j])
                        cell_flags[i*8+j] <= 1'b1;

            done_r <= cap[7];   // row-0 complete pulse for a committed wave

            // Free-running schedule clock; ages are bc - slot_t0. Start no
            // longer touches bc/cell_flags/ptr/done, so waves stay independent.
            bc <= bc + 16'd1;

            if (start_mat_mul)
                shift_lat <= shift_ports;

            // A truncated tail re-arms when the NEXT committed wave
            // completes its row 0 (cap[7] echoes done). That is one edge
            // before that wave's first emit, so the tail stays suppressed
            // without delaying the next wave even when the next start
            // already happened (F>=8 overlap).
            if (cap[7])
                trunc <= 1'b0;

            if (op[1]) begin
                trunc <= 1'b1;
                ptr   <= 3'd0;
            end

            // Cold clear only when provably idle (no live slots, no cell_flags).
            if (idle_clear) begin
                done_r <= 1'b0;
                cell_flags  <= 64'd0;
                ptr    <= 3'd0;
                trunc  <= 1'b0;
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < 8; i = i + 1) begin
                a_row_q[i] <= 64'd0;
                b_col_q[i] <= 64'd0;
            end
        end else if (en) begin
            if (a_ld_i_0) a_row_q[0] <= a_preshift_0; else a_row_q[0] <= {8'd0, a_row_q[0][63:8]};
            if (a_ld_i_1) a_row_q[1] <= a_preshift_1; else a_row_q[1] <= {8'd0, a_row_q[1][63:8]};
            if (a_ld_i_2) a_row_q[2] <= a_preshift_2; else a_row_q[2] <= {8'd0, a_row_q[2][63:8]};
            if (a_ld_i_3) a_row_q[3] <= a_preshift_3; else a_row_q[3] <= {8'd0, a_row_q[3][63:8]};
            if (a_ld_i_4) a_row_q[4] <= a_preshift_4; else a_row_q[4] <= {8'd0, a_row_q[4][63:8]};
            if (a_ld_i_5) a_row_q[5] <= a_preshift_5; else a_row_q[5] <= {8'd0, a_row_q[5][63:8]};
            if (a_ld_i_6) a_row_q[6] <= a_preshift_6; else a_row_q[6] <= {8'd0, a_row_q[6][63:8]};
            if (a_ld_i_7) a_row_q[7] <= a_preshift_7; else a_row_q[7] <= {8'd0, a_row_q[7][63:8]};

            if (b_ld_j_0) b_col_q[0] <= b_preshift_0; else b_col_q[0] <= {8'd0, b_col_q[0][63:8]};
            if (b_ld_j_1) b_col_q[1] <= b_preshift_1; else b_col_q[1] <= {8'd0, b_col_q[1][63:8]};
            if (b_ld_j_2) b_col_q[2] <= b_preshift_2; else b_col_q[2] <= {8'd0, b_col_q[2][63:8]};
            if (b_ld_j_3) b_col_q[3] <= b_preshift_3; else b_col_q[3] <= {8'd0, b_col_q[3][63:8]};
            if (b_ld_j_4) b_col_q[4] <= b_preshift_4; else b_col_q[4] <= {8'd0, b_col_q[4][63:8]};
            if (b_ld_j_5) b_col_q[5] <= b_preshift_5; else b_col_q[5] <= {8'd0, b_col_q[5][63:8]};
            if (b_ld_j_6) b_col_q[6] <= b_preshift_6; else b_col_q[6] <= {8'd0, b_col_q[6][63:8]};
            if (b_ld_j_7) b_col_q[7] <= b_preshift_7; else b_col_q[7] <= {8'd0, b_col_q[7][63:8]};
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < 8; i = i + 1) begin
                for (j = 0; j < 8; j = j + 1) begin
                    ga[i][j]  <= 8'sd0;
                    gb[i][j]  <= 8'sd0;
                    acc[i][j] <= 32'sd0;
                end
            end
        end else if (en) begin
            if (idle_clear) begin
                for (i = 0; i < 8; i = i + 1)
                    for (j = 0; j < 8; j = j + 1)
                        acc[i][j] <= 32'sd0;
            end else begin
                for (i = 0; i < 8; i = i + 1)
                    for (j = 0; j < 8; j = j + 1)
                        if (cap[i*8+j]) begin
                            shadow[i][j] <= acc[i][j] + (ga[i][j] * gb[i][j]);
                            acc[i][j]    <= 32'sd0;
                        end else begin
                            acc[i][j] <= acc[i][j] + (ga[i][j] * gb[i][j]);
                        end
            end

            ga[0][0] <= edge_a_0;
            ga[1][0] <= edge_a_1;
            ga[2][0] <= edge_a_2;
            ga[3][0] <= edge_a_3;
            ga[4][0] <= edge_a_4;
            ga[5][0] <= edge_a_5;
            ga[6][0] <= edge_a_6;
            ga[7][0] <= edge_a_7;

            gb[0][0] <= edge_b_0;
            gb[0][1] <= edge_b_1;
            gb[0][2] <= edge_b_2;
            gb[0][3] <= edge_b_3;
            gb[0][4] <= edge_b_4;
            gb[0][5] <= edge_b_5;
            gb[0][6] <= edge_b_6;
            gb[0][7] <= edge_b_7;

            for (i = 0; i < 8; i = i + 1)
                for (j = 1; j < 8; j = j + 1)
                    ga[i][j] <= ga[i][j-1];

            for (i = 1; i < 8; i = i + 1)
                for (j = 0; j < 8; j = j + 1)
                    gb[i][j] <= gb[i-1][j];
        end
    end

endmodule
