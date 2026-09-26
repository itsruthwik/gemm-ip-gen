/******************************************************************************
 * Forked from FINN finn-rtllib/dynload/hdl/dynamic_load.sv (AMD, BSD-3-Clause)
 * to support two storage/streaming layout modes for the two-operand MVAU
 * dynamic weight loader: two storage/streaming layout modes, chosen so feed_b
 * never needs more than a single arriving wide beat (no reorder buffer).
 *
 * ONE FULL B ROW/COLUMN PER BEAT: unlike the original dynamic_load (and this
 * module's earlier revision), the writer never sub-divides an arriving B beat
 * -- ``idat`` carries an ENTIRE B row (Mode A) or column (Mode B) at once, and
 * the writer commits it to every relevant RAM group in the same cycle.  This
 * cuts the load-side beat count from k_pad*NF (Mode A) / n*SF (Mode B) down
 * to k_pad / n -- the aV node of an attention layer, for example, used to be
 * load-bound (load cost == compute cost); now the load finishes in a small
 * fraction of the compute time.
 *
 * MODE=0 ("row_major", Mode A): storage is NF groups of SIMD physical RAMs,
 *   each RAM word PE-wide, depth SF (address = sf). idat is the whole row:
 *   NF*PE elements, arranged idat[nf][pe] (element index nf*PE+pe, matching
 *   hls4ml's own row-major element order -- see generate_two_operand_shim's
 *   b_wide). A row arriving at lane=simd, address=sf writes slice nf into
 *   group nf's RAM at [lane][sf], for every nf in the same cycle. The writer
 *   counter nest is lane (=simd) fastest, sf slowest -- no nf counter at all.
 *
 * MODE=1 ("col_major", Mode B): the mirror -- storage is SF groups of PE
 *   physical RAMs, each RAM word SIMD-wide, depth NF (address = nf). idat is
 *   the whole column: SF*SIMD elements, arranged idat[sf][simd] (element
 *   index sf*SIMD+simd). A column arriving at lane=pe, address=nf writes
 *   slice sf into group sf's RAM at [lane][nf], for every sf in the same
 *   cycle. Writer counter nest is lane (=pe) fastest, nf slowest -- no sf
 *   counter.
 *
 * Both modes produce the identical output word odat[PE-1:0][SIMD-1:0]
 * [WEIGHT_WIDTH-1:0] and identical replay semantics (N_TLS = SF*NF distinct
 * words replayed N_REPS times each, 2-bank ping-pong double buffering). The
 * reader's consumption counter (cons_sfnf, nf-slow/sf-fast) and the 2-bank
 * handshake (ST_WR_*_WAIT vs state_rd) are unchanged from the original
 * dynamic_load; only the write-side counter/RAM-group structure and the
 * overtake guard were reworked for the one-beat-per-row/column write.
 *****************************************************************************/

`timescale 1ns/1ps

module dynamic_load_2op #(
    int unsigned  PE,
    int unsigned  SIMD,
    int unsigned  WEIGHT_WIDTH,
    int unsigned  MH,
    int unsigned  MW,
    int unsigned  N_REPS,
    int unsigned  MODE = 0,               // 0 = row_major (Mode A), 1 = col_major (Mode B)
    parameter  RAM_STYLE = "distributed"
)(
    input	logic  ap_clk,
    input	logic  ap_rst_n,

    input   logic  ivld,
    output  logic  irdy,
    // Mode A: one whole B row per beat, idat[nf][pe] (NF*PE elements).
    // Mode B: one whole B column per beat, idat[sf][simd] (SF*SIMD elements).
    input   logic  [(MODE == 0 ? MH/PE : MW/SIMD)-1:0][(MODE == 0 ? PE : SIMD)-1:0][WEIGHT_WIDTH-1:0] idat,

    output  logic  ovld,
    input   logic  ordy,
    output  logic  [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat
);

// ----------------------------------------------------------------------------
// Consts and types
// ----------------------------------------------------------------------------

localparam int unsigned  SF = MW/SIMD;
localparam int unsigned  NF = MH/PE;
localparam int unsigned  N_TLS = SF*NF;

// GROUPS: number of parallel RAM groups (one full set of LANES RAMs per group).
//   Mode A (MODE==0): GROUPS = NF (one group per output-fold slice of the row)
//   Mode B (MODE==1): GROUPS = SF (one group per K-fold slice of the column)
// LANES: writer's fast (innermost) counter range == the RAM count per group.
//   Mode A: LANES = SIMD (one physical RAM per SIMD lane, PE-wide word)
//   Mode B: LANES = PE   (one physical RAM per PE lane,   SIMD-wide word)
// DEPTH: per-RAM depth == the writer's slow (outermost) counter range.
//   Mode A: DEPTH = SF (address = sf)
//   Mode B: DEPTH = NF (address = nf)
localparam int unsigned  GROUPS = (MODE == 0) ? NF : SF;
localparam int unsigned  LANES  = (MODE == 0) ? SIMD : PE;
localparam int unsigned  WORDW  = (MODE == 0) ? PE : SIMD;   // per-RAM word width in elements
localparam int unsigned  DEPTH  = (MODE == 0) ? SF : NF;

localparam int unsigned LANES_BITS = (LANES == 1) ? 1 : $clog2(LANES);
localparam int unsigned DEPTH_BITS = (DEPTH == 1) ? 1 : $clog2(DEPTH);
localparam int unsigned GROUPS_BITS = (GROUPS == 1) ? 1 : $clog2(GROUPS);
localparam int unsigned NF_BITS = (NF == 1) ? 1 : $clog2(NF);
localparam int unsigned SF_BITS = (SF == 1) ? 1 : $clog2(SF);
localparam int unsigned N_TLS_BITS = (N_TLS == 1) ? 1 : $clog2(N_TLS);
localparam int unsigned N_REPS_BITS = (N_REPS == 1) ? 1 : $clog2(N_REPS);

typedef enum logic[1:0]  {ST_WR_0, ST_WR_0_WAIT, ST_WR_1, ST_WR_1_WAIT} state_wr_t;
typedef enum logic  {ST_RD_0, ST_RD_1} state_rd_t;

// ----------------------------------------------------------------------------
// Writer
//
// One whole row (Mode A) / column (Mode B) arrives per beat and is committed
// to every GROUP's RAM in the same cycle -- there is no per-group counter at
// all; only LANES (fast) and DEPTH (slow) advance, LANES*DEPTH beats total
// (== k_pad in Mode A, == n in Mode B).
// ----------------------------------------------------------------------------

// -- Regs
state_wr_t state_wr_C = ST_WR_0, state_wr_N;
state_rd_t state_rd_C = ST_RD_0, state_rd_N;

logic[LANES_BITS-1:0] curr_lane_C = '0, curr_lane_N;
logic[DEPTH_BITS-1:0] curr_addr_C = '0, curr_addr_N;

// -- Signals
logic [1:0][LANES-1:0] a_we;
logic [1:0][DEPTH_BITS-1:0] a_addr;

// -- REG
always_ff @( posedge ap_clk ) begin : REG_PROC_WR
    if(~ap_rst_n) begin
        state_wr_C <= ST_WR_0;

        curr_lane_C <= 0;
        curr_addr_C <= 0;
    end
    else begin
        state_wr_C <= state_wr_N;

        curr_lane_C <= curr_lane_N;
        curr_addr_C <= curr_addr_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_WR
    state_wr_N = state_wr_C;

    unique case (state_wr_C)
        ST_WR_0:
            if ((curr_lane_C == LANES - 1) && (curr_addr_C == DEPTH - 1) && ivld) begin
                state_wr_N = (state_rd_C == ST_RD_0) ? ST_WR_1 : ST_WR_0_WAIT;
            end

        ST_WR_0_WAIT:
            state_wr_N = (state_rd_C == ST_RD_0) ? ST_WR_1 : ST_WR_0_WAIT;

        ST_WR_1:
            if ((curr_lane_C == LANES - 1) && (curr_addr_C == DEPTH - 1) && ivld) begin
                state_wr_N = (state_rd_C == ST_RD_1) ? ST_WR_0 : ST_WR_1_WAIT;
            end

        ST_WR_1_WAIT:
            state_wr_N = (state_rd_C == ST_RD_1) ? ST_WR_0 : ST_WR_1_WAIT;

    endcase
end

// -- DP
always_comb begin : DP_PROC_WR
    curr_lane_N = curr_lane_C;
    curr_addr_N = curr_addr_C;

    // Input
    irdy = 1'b0;

    // Buffers
    a_we = '0;
    for(int i = 0; i < 2; i++)
        a_addr[i] = curr_addr_C;

    // Write and count
    case (state_wr_C)
        ST_WR_0, ST_WR_1: begin
            irdy = 1'b1;

            if(ivld) begin
                a_we[state_wr_C == ST_WR_1][curr_lane_C] = 1;

                curr_lane_N = (curr_lane_C == LANES-1) ? 0 : curr_lane_C + 1;
                curr_addr_N = (curr_lane_C == LANES-1) ? ((curr_addr_C == DEPTH-1) ? 0 : curr_addr_C + 1) : curr_addr_C;
            end
        end
    endcase

end

// ----------------------------------------------------------------------------
// Reader
//
// cons_sfnf (0..N_TLS-1, nf-slow/sf-fast) is unchanged from the original
// dynamic_load; it is decomposed into cons_nf = cons_sfnf/SF and
// cons_sf = cons_sfnf%SF so it can address the split (group, depth-address)
// RAM organization above:
//   Mode A: RAM address = cons_sf (into every group's depth-SF RAM);
//            selected group = cons_nf.
//   Mode B: RAM address = cons_nf (into every group's depth-NF RAM);
//            selected group = cons_sf.
//
// Overtake guard: since the writer no longer has a per-group counter, an
// address is simply safe once the writer's single slow counter (curr_addr,
// which IS the RAM address dimension in both modes) has passed the address
// being read:
//   Mode A: curr_addr_C (=sf) > cons_sf
//   Mode B: curr_addr_C (=nf) > cons_nf
// ----------------------------------------------------------------------------

logic [NF_BITS-1:0] cons_nf;
logic [SF_BITS-1:0] cons_sf;

logic guard_ok;
logic [DEPTH_BITS-1:0] b_addr_sel;
logic [GROUPS_BITS-1:0] sel_grp;
generate
if (MODE == 0) begin : genModeSelA
    assign b_addr_sel = cons_sf;
    assign sel_grp = cons_nf;
    assign guard_ok = curr_addr_C > cons_sf;
end : genModeSelA
else begin : genModeSelB
    assign b_addr_sel = cons_nf;
    assign sel_grp = cons_sf;
    assign guard_ok = curr_addr_C > cons_nf;
end : genModeSelB
endgenerate

// -- Regs
logic [N_TLS_BITS-1:0] cons_sfnf_C = '0, cons_sfnf_N;
logic [N_REPS_BITS-1:0] cons_r_C = '0, cons_r_N;

logic [1:0] vld_s0_C = '0, vld_s0_N;
logic [1:0] vld_s1_C = '0, vld_s1_N;

logic vld_C = '0, vld_N;
logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat_C = '0, odat_N;

assign cons_nf = cons_sfnf_C / SF;
assign cons_sf = cons_sfnf_C % SF;

// -- Signals
logic [1:0][DEPTH_BITS-1:0] b_addr;
logic [1:0][GROUPS_BITS-1:0] grp_sel_C = '0, grp_sel_N;
logic [1:0][PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat_ram;

// -- REG
always_ff @( posedge ap_clk ) begin : REG_PROC_RD
    if(~ap_rst_n) begin
        state_rd_C <= ST_RD_0;

        cons_sfnf_C <= 0;
        cons_r_C  <= 0;

        vld_s0_C <= 0;
        vld_s1_C <= 0;
        vld_C <= 0;
        odat_C <= 0;

        grp_sel_C <= '0;
    end
    else begin
        state_rd_C <= state_rd_N;

        cons_sfnf_C <= cons_sfnf_N;
        cons_r_C  <= cons_r_N;

        vld_s0_C <= vld_s0_N;
        vld_s1_C <= vld_s1_N;
        vld_C <= vld_N;
        odat_C <= odat_N;

        grp_sel_C <= grp_sel_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_RD
    state_rd_N = state_rd_C;

    case (state_rd_C)
        ST_RD_0:
            if(ordy && ((state_wr_C != ST_WR_0) || (guard_ok))) begin
                if((cons_sfnf_C == N_TLS-1) && (cons_r_C == N_REPS-1)) begin
                    state_rd_N = ST_RD_1;
                end
            end

        ST_RD_1:
            if(ordy && ((state_wr_C != ST_WR_1) || (guard_ok))) begin
                if((cons_sfnf_C == N_TLS-1) && (cons_r_C == N_REPS-1)) begin
                    state_rd_N = ST_RD_0;
                end
            end

    endcase
end

// -- DP
always_comb begin : DP_PROC_RD
    cons_sfnf_N = cons_sfnf_C;
    cons_r_N = cons_r_C;

    for(int i = 0; i < 2; i++) begin
        vld_s0_N[i] = ordy ? 1'b0 : vld_s0_C[i];
        vld_s1_N[i] = ordy ? vld_s0_C[i] : vld_s1_C[i];
    end

    vld_N = ordy ? |vld_s1_C : vld_C;
    odat_N = ordy ? (vld_s1_C[0] ? odat_ram[0] : odat_ram[1]) : odat_C;

    for(int i = 0; i < 2; i++) begin
        b_addr[i] = b_addr_sel;
    end

    // grp_sel tracks b_addr through the same 1-cycle "issued this cycle" timing
    // (both are combinationally derived from cons_sfnf_C); it feeds the RAM
    // read's group mux with the matching pipeline delay -- see the RAM block
    // below, where grp_sel_C (its PRE-edge/old value, read on the same edge
    // that samples Ram[b_addr] into RdReg) selects RdReg's old content.
    grp_sel_N = grp_sel_C;

    case(state_rd_C)
        ST_RD_0: begin
            if(ordy) begin
                if((state_wr_C == ST_WR_0) ? (guard_ok) : 1'b1) begin
                    vld_s0_N[0] = 1'b1;
                    grp_sel_N[0] = sel_grp;

                    cons_sfnf_N = (cons_sfnf_C == N_TLS-1) ? 0 : cons_sfnf_C + 1;
                    cons_r_N = (cons_sfnf_C == N_TLS-1) ? ((cons_r_C == N_REPS-1) ? 0 : cons_r_C + 1) : cons_r_C;
                end
            end
        end

        ST_RD_1: begin
            if(ordy) begin
                if((state_wr_C == ST_WR_1) ? (guard_ok) : 1'b1) begin

                    vld_s0_N[1] = 1'b1;
                    grp_sel_N[1] = sel_grp;

                    cons_sfnf_N = (cons_sfnf_C == N_TLS-1) ? 0 : cons_sfnf_C + 1;
                    cons_r_N = (cons_sfnf_C == N_TLS-1) ? ((cons_r_C == N_REPS-1) ? 0 : cons_r_C + 1) : cons_r_C;
                end
            end
        end

    endcase

end

assign ovld = vld_C;
assign odat = odat_C;

// ----------------------------------------------------------------------------
// Weight RAMs
//
// GROUPS x LANES physical RAMs per bank, each WORDW elements wide, depth
// DEPTH. Every group is read at the same address (b_addr) every cycle; the
// group actually feeding odat_ram is chosen by grp_sel_C, registered in
// lockstep with RdReg (both updated only when ordy) so the group index
// applied when RdReg's OLD (nonblocking pre-edge) content is copied into
// odat_ram is the one that was in effect when THAT content was fetched from
// Ram, one cycle earlier -- i.e. the same read-latency pipeline as the
// original dynamic_load, just with an extra group dimension muxed in.
//
// Mode A (MODE==0): NF groups x SIMD lanes, each RAM PE-wide -- identical
//   physical shape to the original dynamic_load's SIMD RAMs, just split into
//   NF independent depth-SF copies instead of one depth-SF*NF RAM.
// Mode B (MODE==1): SF groups x PE lanes, each RAM SIMD-wide -- the
//   transpose, split into SF independent depth-NF copies.
// ----------------------------------------------------------------------------

generate
for(genvar i = 0; i < 2; i++) begin : genBank
    for(genvar k = 0; k < LANES; k++) begin : genLane
        (* RAM_STYLE = RAM_STYLE *)
        logic [WORDW-1:0][WEIGHT_WIDTH-1:0]  Ram[GROUPS][2**DEPTH_BITS];
        logic [GROUPS-1:0][WORDW-1:0][WEIGHT_WIDTH-1:0]  RdReg;

        always_ff @(posedge ap_clk) begin
            if(a_we[i][k]) begin
                for(int g = 0; g < GROUPS; g++)
                    Ram[g][a_addr[i]] <= idat[g];
            end
            if(ordy) begin
                for(int g = 0; g < GROUPS; g++)
                    RdReg[g] <= Ram[g][b_addr[i]];
                for(int p = 0; p < WORDW; p++)
                    odat_ram[i][ (MODE == 0) ? p : k ][ (MODE == 0) ? k : p ] <= RdReg[grp_sel_C[i]][p];
            end
        end
    end : genLane
end : genBank
endgenerate

endmodule : dynamic_load_2op
