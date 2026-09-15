/******************************************************************************
 * Forked from FINN finn-rtllib/dynload/hdl/dynamic_load.sv (AMD, BSD-3-Clause)
 * to support two storage/streaming layout modes for the two-operand MVAU
 * dynamic weight loader. See jojo-track/defer/mvau-two-operand-dynamic-load/
 * plan.md ("Customization: two layout modes") for the design rationale.
 *
 * MODE=0 ("row_major", Mode A): identical organization to the original
 *   dynamic_load -- SIMD parallel physical RAMs (one per SIMD lane), each RAM
 *   word PE-wide. Input beat (idat) is PE-wide; writer fills one SIMD lane
 *   per beat (curr_lane == curr_simd, SIMD*N_TLS beats total).
 *
 * MODE=1 ("col_major", Mode B): the transpose -- PE parallel physical RAMs
 *   (one per PE lane), each RAM word SIMD-wide. Input beat (idat) is
 *   SIMD-wide; writer fills one PE lane per beat (curr_lane == curr_pe,
 *   PE*N_TLS beats total).
 *
 * Both modes produce the identical output word odat[PE-1:0][SIMD-1:0]
 * [WEIGHT_WIDTH-1:0] and identical replay semantics (N_TLS distinct words
 * replayed N_REPS times each, 2-bank ping-pong double buffering). The writer
 * FSM/counter structure and the reader overtake guard are shared verbatim
 * between modes by parameterizing on LANES (= SIMD for Mode A, PE for Mode
 * B) instead of hardcoding SIMD; only the physical RAM shape/count and the
 * write-address-to-RAM mapping differ per mode (generate-selected).
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
    // Mode A: PE-wide input beat (one SIMD lane per beat).
    // Mode B: SIMD-wide input beat (one PE lane per beat).
    input   logic  [(MODE == 0 ? PE : SIMD)-1:0][WEIGHT_WIDTH-1:0] idat,

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

// LANES: number of physical RAM banks per ping-pong bank, and the range of
// the innermost writer serial-fill counter (curr_lane).
//   Mode A (MODE==0): LANES = SIMD (one physical RAM per SIMD lane, PE-wide word)
//   Mode B (MODE==1): LANES = PE   (one physical RAM per PE lane,   SIMD-wide word)
localparam int unsigned  LANES = (MODE == 0) ? SIMD : PE;
// WORDW: per-RAM word width in elements (the "other" dimension from LANES).
localparam int unsigned  WORDW = (MODE == 0) ? PE : SIMD;

localparam int unsigned LANES_BITS = (LANES == 1) ? 1 : $clog2(LANES);
localparam int unsigned WGT_ADDR_BITS = (N_TLS == 1) ? 1 : $clog2(N_TLS);
localparam int unsigned NF_BITS = (NF == 1) ? 1 : $clog2(NF);
localparam int unsigned SF_BITS = (SF == 1) ? 1 : $clog2(SF);
localparam int unsigned N_TLS_BITS = (N_TLS == 1) ? 1 : $clog2(N_TLS);
localparam int unsigned N_REPS_BITS = (N_REPS == 1) ? 1 : $clog2(N_REPS);

logic [NF-1:0][WGT_ADDR_BITS-1:0] offsets;

typedef enum logic[1:0]  {ST_WR_0, ST_WR_0_WAIT, ST_WR_1, ST_WR_1_WAIT} state_wr_t;
typedef enum logic  {ST_RD_0, ST_RD_1} state_rd_t;

// ----------------------------------------------------------------------------
// Writer
//
// Mode A (MODE==0, row-major): curr_nf innermost, then curr_lane (=curr_simd),
// then curr_sf outermost/slowest -- matches row-major B arrival (a full N-wide
// row per beat; the whole row = all nf at fixed sf).
//
// Mode B (MODE==1, col-major): TRANSPOSED nesting -- curr_sf innermost, then
// curr_lane (=curr_pe), then curr_nf outermost/slowest -- matches col-major B
// arrival (a full K-wide column per beat; the whole column = all sf at fixed
// nf). This is the true mirror of Mode A, required so feed_b only ever needs
// to hold the single arriving column (no reorder buffer) -- see
// jojo-track/defer/mvau-two-operand-dynamic-load/plan.md, "Mode B ordering
// resolution".
//
// Write address = offsets[nf] + sf = nf*SF + sf in BOTH modes (unchanged);
// one lane (a_we[bank][curr_lane]) written per beat with the full WORDW-wide
// idat.
// ----------------------------------------------------------------------------

// -- Regs
state_wr_t state_wr_C = ST_WR_0, state_wr_N;
state_rd_t state_rd_C = ST_RD_0, state_rd_N;

logic[NF_BITS-1:0] curr_nf_C = '0, curr_nf_N;
logic[N_TLS_BITS-1:0] curr_sf_C = '0, curr_sf_N;
logic[LANES_BITS-1:0] curr_lane_C = '0, curr_lane_N;

// -- Signals
logic [1:0][LANES-1:0] a_we;
logic [1:0][WGT_ADDR_BITS-1:0] a_addr;

// -- Offsets
for(genvar i = 0; i < NF; i++) begin
    assign offsets[i] = i * SF;
end

// -- REG
always_ff @( posedge ap_clk ) begin : REG_PROC_WR
    if(~ap_rst_n) begin
        state_wr_C <= ST_WR_0;

        curr_nf_C <= 0;
        curr_sf_C <= 0;
        curr_lane_C <= 0;
    end
    else begin
        state_wr_C <= state_wr_N;

        curr_nf_C <= curr_nf_N;
        curr_sf_C <= curr_sf_N;
        curr_lane_C <= curr_lane_N;
    end
end

// -- NSL
always_comb begin : NSL_PROC_WR
    state_wr_N = state_wr_C;

    unique case (state_wr_C)
        ST_WR_0:
            if ((curr_lane_C == LANES - 1) && (curr_sf_C == SF - 1) && (curr_nf_C == NF - 1) && ivld) begin
                state_wr_N = (state_rd_C == ST_RD_0) ? ST_WR_1 : ST_WR_0_WAIT;
            end

        ST_WR_0_WAIT:
            state_wr_N = (state_rd_C == ST_RD_0) ? ST_WR_1 : ST_WR_0_WAIT;

        ST_WR_1:
            if ((curr_lane_C == LANES - 1) && (curr_sf_C == SF - 1) && (curr_nf_C == NF - 1) && ivld) begin
                state_wr_N = (state_rd_C == ST_RD_1) ? ST_WR_0 : ST_WR_1_WAIT;
            end

        ST_WR_1_WAIT:
            state_wr_N = (state_rd_C == ST_RD_1) ? ST_WR_0 : ST_WR_1_WAIT;

    endcase
end

// -- DP
always_comb begin : DP_PROC_WR
    curr_nf_N = curr_nf_C;
    curr_sf_N = curr_sf_C;
    curr_lane_N = curr_lane_C;

    // Input
    irdy = 1'b0;

    // Buffers
    a_we = '0;
    for(int i = 0; i < 2; i++)
        a_addr[i] = offsets[curr_nf_C] + curr_sf_C;

    // Write and count
    case (state_wr_C)
        ST_WR_0, ST_WR_1: begin
            irdy = 1'b1;

            if(ivld) begin
                a_we[state_wr_C == ST_WR_1][curr_lane_C] = 1;

                if (MODE == 0) begin
                    // Mode A: nf fastest, lane middle, sf slowest.
                    curr_nf_N   = (curr_nf_C == NF-1) ? 0 : curr_nf_C + 1;
                    curr_lane_N = (curr_nf_C == NF-1) ? ((curr_lane_C == LANES-1) ? 0 : curr_lane_C + 1) : curr_lane_C;
                    curr_sf_N   = (curr_nf_C == NF-1) ? ((curr_lane_C == LANES-1) ? ((curr_sf_C == SF-1) ? 0 : curr_sf_C + 1) : curr_sf_C) : curr_sf_C;
                end
                else begin
                    // Mode B: sf fastest, lane middle, nf slowest (transposed).
                    curr_sf_N   = (curr_sf_C == SF-1) ? 0 : curr_sf_C + 1;
                    curr_lane_N = (curr_sf_C == SF-1) ? ((curr_lane_C == LANES-1) ? 0 : curr_lane_C + 1) : curr_lane_C;
                    curr_nf_N   = (curr_sf_C == SF-1) ? ((curr_lane_C == LANES-1) ? ((curr_nf_C == NF-1) ? 0 : curr_nf_C + 1) : curr_nf_C) : curr_nf_C;
                end
            end
        end
    endcase

end

// ----------------------------------------------------------------------------
// Reader
//
// Overtake guard, re-derived per mode (the writer's *slow* (outermost) phase
// differs between modes, so the counter that bounds safe early-reads differs
// too):
//
// Mode A: sf is the slow phase. `curr_sf_C > cons_sfnf_C` is safe/correct for
// the low address range (address < SF, i.e. the nf=0 block, where
// cons_sfnf_C IS the sf-component of the address) because a full sf pass
// writes every nf at that sf across all lanes before sf advances -- this is
// the original dynamic_load guard, unchanged.
//
// Mode B: nf is now the slow phase, but -- unlike Mode A's sf-slow/nf-fast
// nesting, where a fixed sf's *inner* nf sweep covers every nf for every
// lane within one sf tick -- Mode B's nesting is nf-slow/lane-mid/sf-fast:
// for a fixed nf, each lane pass sweeps *all* sf again, so a given address
// (nf,sf) only receives its final (all-LANES) write partway through the
// *last* lane's sweep for that nf, not linearly across the whole nf block.
// So the address must be decomposed (cons_nf = cons_sfnf_C/SF, cons_sf =
// cons_sfnf_C%SF) and the guard is:
//   - safe once the writer has moved past this address's nf entirely
//     (curr_nf_C > cons_nf), OR
//   - the writer is still on this nf but on its LAST lane pass
//     (curr_lane_C == LANES-1) and has advanced sf past this address's sf
//     (curr_sf_C > cons_sf).
// ----------------------------------------------------------------------------

logic guard_ok;
generate
if (MODE == 0) begin : genGuardA
    assign guard_ok = curr_sf_C > cons_sfnf_C;
end : genGuardA
else begin : genGuardB
    logic [NF_BITS-1:0] cons_nf;
    logic [SF_BITS-1:0] cons_sf;
    assign cons_nf = cons_sfnf_C / SF;
    assign cons_sf = cons_sfnf_C % SF;
    assign guard_ok = (curr_nf_C > cons_nf) ||
        ((curr_nf_C == cons_nf) && (curr_lane_C == LANES-1) && (curr_sf_C > cons_sf));
end : genGuardB
endgenerate

// -- Regs
logic [N_TLS_BITS-1:0] cons_sfnf_C = '0, cons_sfnf_N;
logic [N_REPS_BITS-1:0] cons_r_C = '0, cons_r_N;

logic [1:0] vld_s0_C = '0, vld_s0_N;
logic [1:0] vld_s1_C = '0, vld_s1_N;

logic vld_C = '0, vld_N;
logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] odat_C = '0, odat_N;

// -- Signals
logic [1:0][WGT_ADDR_BITS-1:0] b_addr;
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
    end
    else begin
        state_rd_C <= state_rd_N;

        cons_sfnf_C <= cons_sfnf_N;
        cons_r_C  <= cons_r_N;

        vld_s0_C <= vld_s0_N;
        vld_s1_C <= vld_s1_N;
        vld_C <= vld_N;
        odat_C <= odat_N;
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
        b_addr[i] = cons_sfnf_C;
    end

    case(state_rd_C)
        ST_RD_0: begin
            if(ordy) begin
                if((state_wr_C == ST_WR_0) ? (guard_ok) : 1'b1) begin
                    vld_s0_N[0] = 1'b1;

                    cons_sfnf_N = (cons_sfnf_C == N_TLS-1) ? 0 : cons_sfnf_C + 1;
                    cons_r_N = (cons_sfnf_C == N_TLS-1) ? ((cons_r_C == N_REPS-1) ? 0 : cons_r_C + 1) : cons_r_C;
                end
            end
        end

        ST_RD_1: begin
            if(ordy) begin
                if((state_wr_C == ST_WR_1) ? (guard_ok) : 1'b1) begin

                    vld_s0_N[1] = 1'b1;

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
// Mode A (MODE==0): SIMD physical RAMs per bank (genSimd), each word
//   PE-wide -- identical layout/shape to the original dynamic_load.
// Mode B (MODE==1): PE physical RAMs per bank (genPe), each word SIMD-wide
//   -- the transpose. Either way, odat_ram[bank][pe][simd] ends up populated
//   identically for the shared reader logic above.
// ----------------------------------------------------------------------------

generate
if (MODE == 0) begin : genModeA
    for(genvar i = 0; i < 2; i++) begin : genBank
        for(genvar k = 0; k < SIMD; k++) begin : genSimd
            (* RAM_STYLE = RAM_STYLE *)
            logic [PE-1:0][WEIGHT_WIDTH-1:0]  Ram[2**WGT_ADDR_BITS];
            logic [PE-1:0][WEIGHT_WIDTH-1:0]  RdReg;

            always_ff @(posedge ap_clk) begin
                if(a_we[i][k])  Ram[a_addr[i]] <= idat;
                if(ordy) begin
                    RdReg <= Ram[b_addr[i]];
                    foreach(RdReg[p])  odat_ram[i][p][k] <= RdReg[p];
                end
            end
        end : genSimd
    end : genBank
end : genModeA
else begin : genModeB
    for(genvar i = 0; i < 2; i++) begin : genBank
        for(genvar k = 0; k < PE; k++) begin : genPe
            (* RAM_STYLE = RAM_STYLE *)
            logic [SIMD-1:0][WEIGHT_WIDTH-1:0]  Ram[2**WGT_ADDR_BITS];
            logic [SIMD-1:0][WEIGHT_WIDTH-1:0]  RdReg;

            always_ff @(posedge ap_clk) begin
                if(a_we[i][k])  Ram[a_addr[i]] <= idat;
                if(ordy) begin
                    RdReg <= Ram[b_addr[i]];
                    foreach(RdReg[s])  odat_ram[i][k][s] <= RdReg[s];
                end
            end
        end : genPe
    end : genBank
end : genModeB
endgenerate

endmodule : dynamic_load_2op
