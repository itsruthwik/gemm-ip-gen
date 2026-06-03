// tensor_slice_int8 — row/column streaming, direct systolic feed.
// No buffering.  A rows and B columns feed directly into per-K-position
// staggering delay lines.  Byte k of each beat is delayed k cycles so
// A[i][k] and B[k][j] meet at cell (i,j) at cycle i+j+k+1 (+1 = launch offset).
//
// For 8×8×8 (M=8,K=8,N=8):  input cycles 0-7, first output cycle 17,
// last output cycle 24, done cycle 26.  ~25 cycles first-in to last-out.
//
// Port contract (same as original):
//   a_data byte k = A[current_row][k]   — one row of A per cycle
//   b_data byte k = B[k][current_col]   — one col of B per cycle
//   (row t and col t arrive simultaneously at cycle t)

`define BB_DWIDTH 8
`define BB_MAT_MUL_SIZE 8
`define NUM_CYCLES_IN_MAC 3

module tensor_slice_int8(
    input clk, input reset, input pe_reset,
    input start_mat_mul, output done_mat_mul,
    input [63:0] a_data, input [63:0] b_data,
    input [63:0] a_data_in, input [63:0] b_data_in,
    output [63:0] a_data_out, output [63:0] b_data_out,
    output [127:0] c_data_out, output c_data_available,
    input [7:0] validity_mask_a_rows,
    input [7:0] validity_mask_a_cols_b_rows,
    input [7:0] validity_mask_b_cols,
    input [1:0] slice_dtype, input slice_mode,
    input [2:0] op, input preload, input no_rounding,
    input [7:0] final_mat_mul_size,
    input [4:0] a_loc, input [4:0] b_loc,
    input rowcol_chain_mode  // 0=direct feed, 1=location-aware chained
);

    wire [7:0] matN = (op == 3'b100) ? b_data[31:24] : final_mat_mul_size;
    // Readout timing: direct mode starts after launch + 7 + matN + MAC.
    // Chained mode adds (a_loc+b_loc)*8 cycles for tile location offset.
    wire [7:0] direct_l_config = 7 + matN + `NUM_CYCLES_IN_MAC;
    wire [7:0] chain_l_config  = ((a_loc + b_loc) << 3) + 7 + 8 + `NUM_CYCLES_IN_MAC;
    wire [7:0] l_config = rowcol_chain_mode ? chain_l_config : direct_l_config;
    assign done_mat_mul = (clk_cnt == l_config + 8);

    // ── Per-K-position staggering ──────────────────────────────────────────
    // sA[k][d] : byte k of each A beat, delayed d cycles (d = 0..k)
    // sB[k][d] : byte k of each B beat, delayed d cycles (d = 0..k)
    // All NBA — shift reads old values, capture writes new beat, both
    // take effect at end of timestep.  Injection reads OLD sA/sB values.
    reg [7:0] sA [0:7][0:7];
    reg [7:0] sB [0:7][0:7];

    // ── Systolic array ─────────────────────────────────────────────────────
    reg signed [7:0] a_reg [0:7][0:7];
    reg signed [7:0] b_reg [0:7][0:7];
    reg signed [31:0] acc [0:7][0:7];
    reg preload_done;
    reg [63:0] a_chain_pipe [0:7];
    reg [63:0] b_chain_pipe [0:7];

    reg [7:0] clk_cnt;
    reg [2:0] readout_ptr;
    reg op_active, readout_active, c_avail;
    assign c_data_available = c_avail;
    assign a_data_out = a_chain_pipe[7];
    assign b_data_out = b_chain_pipe[7];

    wire [7:0] sat_C [0:7];
    genvar gi;
    generate
        for (gi=0; gi<8; gi=gi+1) begin
            assign sat_C[gi] = (acc[readout_ptr][gi] > 31'sd127) ? 8'h7F :
                               ((acc[readout_ptr][gi] < -31'sd128) ? 8'h80 :
                                acc[readout_ptr][gi][7:0]);
        end
    endgenerate
    assign c_data_out = {64'b0, sat_C[7], sat_C[6], sat_C[5], sat_C[4],
                                       sat_C[3], sat_C[2], sat_C[1], sat_C[0]};

    integer r, c, k;
    wire launch_op = start_mat_mul & (~op_active) & (~readout_active);
    wire step_cycle = op_active | launch_op;
    wire [7:0] cur_cycle = launch_op ? 8'd0 : clk_cnt;

    // Location-aware chaining support.
    // In chained mode, tile (r,c) captures data during its local window
    // starting at loc_delay = (a_loc + b_loc) * 8 cycles after launch.
    wire [7:0] loc_delay = (a_loc + b_loc) << 3;
    wire [7:0] local_idx = cur_cycle - loc_delay;
    wire chain_in_phase = rowcol_chain_mode &&
        (cur_cycle >= loc_delay) && (local_idx < 8'd8);
    wire direct_in_phase = (cur_cycle < matN);
    wire in_phase = rowcol_chain_mode ? chain_in_phase : direct_in_phase;

    // Input selection: boundary tiles use primary data; off-boundary use chain.
    wire [63:0] eff_a_in = (b_loc == 0) ? a_data : a_data_in;
    wire [63:0] eff_b_in = (a_loc == 0) ? b_data : b_data_in;

    // Injection cycle — offset by loc_delay in chained mode.
    // Guarded against unsigned underflow when cur_cycle < loc_delay.
    wire [7:0] inject_cycle = rowcol_chain_mode && (cur_cycle >= loc_delay)
        ? (cur_cycle - loc_delay) : (~rowcol_chain_mode ? cur_cycle : 8'd0);

    always @(posedge clk) begin
        if (reset || pe_reset) begin
            preload_done <= 1'b0;
            clk_cnt <= 0; op_active <= 0; readout_active <= 0;
            c_avail <= 0; readout_ptr <= 0;
            for (k=0; k<8; k=k+1) begin
                a_chain_pipe[k] <= 64'b0; b_chain_pipe[k] <= 64'b0;
                for (r=0; r<=k; r=r+1) begin sA[k][r] <= 8'b0; sB[k][r] <= 8'b0; end
            end
            for (r=0; r<8; r=r+1) begin
                for (c=0; c<8; c=c+1) begin
                    a_reg[r][c] <= 0; b_reg[r][c] <= 0; acc[r][c] <= 0;
                end
            end
        end else if (preload && !launch_op && !op_active) begin
            preload_done <= 1'b1;
            for (r=0; r<8; r=r+1) begin
                for (c=0; c<8; c=c+1) begin
                    if (validity_mask_a_rows[r] && validity_mask_b_cols[c])
                        acc[r][c] <= {{24{b_data[c*8 + 7]}}, b_data[c*8 +: 8]};
                    else acc[r][c] <= 32'sd0;
                end
            end
        end else if (launch_op) begin
            // ── Launch: init control, capture beat 0 into stage 0 (NBA).
            // No shift, no injection, no propagation — matches original pattern.
            op_active <= 1'b1; readout_active <= 1'b0;
            c_avail <= 1'b0; readout_ptr <= 3'd0; clk_cnt <= 8'd1;

            for (r=0; r<8; r=r+1) begin
                for (c=0; c<8; c=c+1) begin
                    a_reg[r][c] <= 0; b_reg[r][c] <= 0;
                    if (!preload_done) acc[r][c] <= 0;
                end
            end
            preload_done <= 1'b0;
            for (k=0; k<8; k=k+1) begin
                a_chain_pipe[k] <= 64'b0; b_chain_pipe[k] <= 64'b0;
                for (r=0; r<=k; r=r+1) begin sA[k][r] <= 8'b0; sB[k][r] <= 8'b0; end
            end

            // Capture beat 0 into stage 0 (NBA — all writes from same old state)
            a_chain_pipe[0] <= eff_a_in;
            b_chain_pipe[0] <= eff_b_in;
            for (k=0; k<8; k=k+1) begin
                sA[k][0] <= (validity_mask_a_cols_b_rows[k]) ?
                            eff_a_in[k*8 +: 8] : 8'b0;
                sB[k][0] <= (validity_mask_a_cols_b_rows[k]) ?
                            eff_b_in[k*8 +: 8] : 8'b0;
            end
            // No injection, propagation, or accumulation on launch cycle
        end else if (step_cycle) begin
            clk_cnt <= clk_cnt + 1'b1;

            // ── 1. Shift + capture staggering.  All NBA — shift reads old
            // values, capture writes new beat.  Injection reads OLD sA/sB.
            for (k=1; k<8; k=k+1) begin
                for (r=k; r>=1; r=r-1) begin
                    sA[k][r] <= sA[k][r-1];
                    sB[k][r] <= sB[k][r-1];
                end
            end
            for (k=0; k<8; k=k+1) begin
                sA[k][0] <= (in_phase && validity_mask_a_cols_b_rows[k]) ?
                            eff_a_in[k*8 +: 8] : 8'b0;
                sB[k][0] <= (in_phase && validity_mask_a_cols_b_rows[k]) ?
                            eff_b_in[k*8 +: 8] : 8'b0;
            end

            // ── 2. Chain pipeline
            a_chain_pipe[0] <= eff_a_in;
            b_chain_pipe[0] <= eff_b_in;
            for (k=1; k<8; k=k+1) begin
                a_chain_pipe[k] <= a_chain_pipe[k-1];
                b_chain_pipe[k] <= b_chain_pipe[k-1];
            end

            // ── 3. Inject staggered values into systolic edges
            // A[i][k] emerges from sA[k][k] after i+k+1 cycles
            // At cycle t: old sA[k][k] = A[t-1-k][k] → inject into a_reg[t-1-k][0]
            // Clear edge registers to 0 (matches original: a_stagger[r][r]
            // naturally goes to 0 after wavefront passes).  Injection loop
            // selectively overwrites matching positions.
            for (r=0; r<8; r=r+1) a_reg[r][0] <= 8'b0;
            for (c=0; c<8; c=c+1) b_reg[0][c] <= 8'b0;
            // Inject from staggering outputs
            for (k=0; k<8; k=k+1) begin
                if (k+1 <= inject_cycle) begin
                    if ((inject_cycle - 1 - k) < 8) begin
                        r = inject_cycle - 1 - k;
                        if (validity_mask_a_rows[r])
                            a_reg[r][0] <= sA[k][k];
                    end
                    if ((inject_cycle - 1 - k) < 8) begin
                        c = inject_cycle - 1 - k;
                        if (validity_mask_b_cols[c])
                            b_reg[0][c] <= sB[k][k];
                    end
                end
            end

            // ── 4. Systolic propagation
            for (r=0; r<8; r=r+1) begin
                for (c=1; c<8; c=c+1) a_reg[r][c] <= a_reg[r][c-1];
            end
            for (c=0; c<8; c=c+1) begin
                for (r=1; r<8; r=r+1) b_reg[r][c] <= b_reg[r-1][c];
            end

            // ── 5. Accumulation
            for (r=0; r<8; r=r+1) begin
                for (c=0; c<8; c=c+1) begin
                    acc[r][c] <= acc[r][c] + (a_reg[r][c] * b_reg[r][c]);
                end
            end

            // ── 6. Readout
            if (cur_cycle == l_config - 1) begin
                readout_active <= 1'b1; c_avail <= 1'b1; readout_ptr <= 0;
            end else if (readout_active) begin
                readout_ptr <= readout_ptr + 1;
                if (readout_ptr == 7) begin readout_active <= 1'b0; c_avail <= 1'b0; end
            end

            if (cur_cycle == l_config + 8) begin
                op_active <= 1'b0; clk_cnt <= 8'd0;
            end
        end
    end
endmodule
