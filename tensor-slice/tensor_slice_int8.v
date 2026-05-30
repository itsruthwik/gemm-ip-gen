
`define BB_DWIDTH 8
`define BB_MAT_MUL_SIZE 8
`define NUM_CYCLES_IN_MAC 3

module tensor_slice_int8(
    input clk, input reset, input pe_reset,
    input start_mat_mul, output done_mat_mul,
    input [63:0] a_data,      // Primary data input
    input [63:0] b_data,      // Primary data input
    input [63:0] a_data_in,   // Systolic chain input
    input [63:0] b_data_in,   // Systolic chain input
    output [63:0] a_data_out, // Systolic chain output
    output [63:0] b_data_out, // Systolic chain output
    output [127:0] c_data_out, 
    output c_data_available,
    input [7:0] validity_mask_a_rows,
    input [7:0] validity_mask_a_cols_b_rows,
    input [7:0] validity_mask_b_cols,
    input [1:0] slice_dtype, input slice_mode,
    input [2:0] op, input preload, input no_rounding,
    input [7:0] final_mat_mul_size,
    input [4:0] a_loc, input [4:0] b_loc
);

    // Dynamic N calculation for Mat-Vec mode
    wire [7:0] matN = (op == 3'b100) ? b_data[31:24] : final_mat_mul_size;

    // --- INTERNAL LATENCY LOGIC ---
    // Formula: (Global Row 0 index) + (Global Col 7 index) + N + Pipeline.
    // The final registered MAC contribution is visible one cycle after the
    // multiply/accumulate assignment that consumes the last diagonal operand.
    wire [7:0] l_config = ((a_loc + b_loc) << 3) + 7 + matN + `NUM_CYCLES_IN_MAC;

    // Chain-fed slices receive delayed operands from neighboring slices.
    // Boundary-fed slices must internally buffer their K input beats so they can
    // be replayed later to align with the chained operand arriving from the
    // orthogonal direction.
    reg [63:0] a_hist [0:255];
    reg [63:0] b_hist [0:255];

    reg signed [7:0] a_reg [0:7][0:7];
    reg signed [7:0] b_reg [0:7][0:7];
    reg signed [31:0] acc [0:7][0:7];
    
    // Internal staggering registers
    reg [7:0] a_stagger [0:7][0:7];
    reg [7:0] b_stagger [0:7][0:7];
    reg [63:0] a_chain_pipe [0:7];
    reg [63:0] b_chain_pipe [0:7];

    reg [7:0] clk_cnt;
    reg [2:0] readout_ptr;
    reg op_active, readout_active, c_avail;
    assign c_data_available = c_avail;
    
    assign done_mat_mul = (clk_cnt == l_config + 8);

    wire [7:0] sat_C [0:7];
    genvar i;
    generate
        for (i=0; i<8; i=i+1) begin
            assign sat_C[i] = (acc[readout_ptr][i] > 31'sd127) ? 8'h7F : 
                              ((acc[readout_ptr][i] < -31'sd128) ? 8'h80 : acc[readout_ptr][i][7:0]);
        end
    endgenerate

    assign c_data_out = { 64'b0, sat_C[7], sat_C[6], sat_C[5], sat_C[4], sat_C[3], sat_C[2], sat_C[1], sat_C[0] };
    
    // Output chains: relay full 64-bit tile words delayed by one slice width.
    assign a_data_out = a_chain_pipe[7];
    assign b_data_out = b_chain_pipe[7];

    integer r, c, k;
    wire launch_op = start_mat_mul & (~op_active) & (~readout_active);
    wire step_cycle = op_active | launch_op;
    wire [7:0] cur_cycle = launch_op ? 8'd0 : clk_cnt;
    wire [7:0] a_direct_delay = {a_loc, 3'b000};
    wire [7:0] b_direct_delay = {b_loc, 3'b000};
    wire a_direct_feed_phase =
        (cur_cycle >= a_direct_delay) && (cur_cycle <= (a_direct_delay + matN));
    wire b_direct_feed_phase =
        (cur_cycle >= b_direct_delay) && (cur_cycle <= (b_direct_delay + matN));
    wire [7:0] a_direct_idx = cur_cycle - a_direct_delay;
    wire [7:0] b_direct_idx = cur_cycle - b_direct_delay;
    wire a_temporal_mask_valid =
        (a_direct_idx < 8) ? validity_mask_a_cols_b_rows[a_direct_idx[2:0]] : 1'b1;
    wire b_temporal_mask_valid =
        (b_direct_idx < 8) ? validity_mask_a_cols_b_rows[b_direct_idx[2:0]] : 1'b1;
    // Boundary-fed inputs need a tile-location delay so they align with the
    // chained operand arriving from the orthogonal direction. Chain-fed inputs
    // arrive already delayed by upstream propagation and should not be gated.
    wire a_input_valid = (b_loc == 0) ? (a_direct_feed_phase & a_temporal_mask_valid) : 1'b1;
    wire b_input_valid = (a_loc == 0) ? (b_direct_feed_phase & b_temporal_mask_valid) : 1'b1;
    wire [63:0] eff_a_in =
        (b_loc == 0) ? ((a_loc == 0) ? a_data : a_hist[a_direct_idx]) : a_data_in;
    wire [63:0] eff_b_in =
        (a_loc == 0) ? ((b_loc == 0) ? b_data : b_hist[b_direct_idx]) : b_data_in;

    always @(posedge clk) begin
        if (reset || pe_reset) begin
            clk_cnt <= 0; op_active <= 0; readout_active <= 0; c_avail <= 0; readout_ptr <= 0;
            for (k=0; k<256; k=k+1) begin
                a_hist[k] <= 64'b0;
                b_hist[k] <= 64'b0;
            end
            for (k=0; k<8; k=k+1) begin
                a_chain_pipe[k] <= 64'b0;
                b_chain_pipe[k] <= 64'b0;
            end
            for (r=0; r<8; r=r+1) begin
                for (c=0; c<8; c=c+1) begin
                    a_reg[r][c] <= 0; b_reg[r][c] <= 0; acc[r][c] <= 0;
                    a_stagger[r][c] <= 0; b_stagger[r][c] <= 0;
                end
            end
        end else if (launch_op) begin
            op_active <= 1'b1;
            readout_active <= 1'b0;
            c_avail <= 1'b0;
            readout_ptr <= 3'd0;
            clk_cnt <= 8'd1;

            for (k=0; k<256; k=k+1) begin
                a_hist[k] <= 64'b0;
                b_hist[k] <= 64'b0;
            end
            a_hist[0] <= a_data;
            b_hist[0] <= b_data;
            a_chain_pipe[0] <= eff_a_in;
            b_chain_pipe[0] <= eff_b_in;
            for (k=1; k<8; k=k+1) begin
                a_chain_pipe[k] <= 64'b0;
                b_chain_pipe[k] <= 64'b0;
            end

            for (r=0; r<8; r=r+1) begin
                for (c=0; c<8; c=c+1) begin
                    a_reg[r][c] <= 0; b_reg[r][c] <= 0; acc[r][c] <= 0;
                    a_stagger[r][c] <= 0; b_stagger[r][c] <= 0;
                end
            end

            // 1. Input Staggering and Masking
            for (r=0; r<8; r=r+1) begin
                a_stagger[r][0] <= (validity_mask_a_rows[r] && a_input_valid) ? eff_a_in[r*8 +: 8] : 8'b0;
                for (k=1; k<8; k=k+1) a_stagger[r][k] <= 8'b0;
            end
            for (c=0; c<8; c=c+1) begin
                b_stagger[c][0] <= (validity_mask_b_cols[c] && b_input_valid) ? eff_b_in[c*8 +: 8] : 8'b0;
                for (k=1; k<8; k=k+1) b_stagger[c][k] <= 8'b0;
            end

            // No propagation/accumulation/readout on launch cycle beyond latching the
            // first input beat into the stage-0 staggering registers.
        end else if (step_cycle) begin
            clk_cnt <= clk_cnt + 1'b1;
            if (cur_cycle < matN) begin
                a_hist[cur_cycle] <= a_data;
                b_hist[cur_cycle] <= b_data;
            end
            a_chain_pipe[0] <= eff_a_in;
            b_chain_pipe[0] <= eff_b_in;
            for (k=1; k<8; k=k+1) begin
                a_chain_pipe[k] <= a_chain_pipe[k-1];
                b_chain_pipe[k] <= b_chain_pipe[k-1];
            end

            // 1. Input Staggering and Masking
            for (r=0; r<8; r=r+1) begin
                a_stagger[r][0] <= (validity_mask_a_rows[r] && a_input_valid) ? eff_a_in[r*8 +: 8] : 8'b0;
                for (k=1; k<8; k=k+1) a_stagger[r][k] <= a_stagger[r][k-1];
            end
            for (c=0; c<8; c=c+1) begin
                b_stagger[c][0] <= (validity_mask_b_cols[c] && b_input_valid) ? eff_b_in[c*8 +: 8] : 8'b0;
                for (k=1; k<8; k=k+1) b_stagger[c][k] <= b_stagger[c][k-1];
            end

            // 2. Systolic Propagation
            for (r=0; r<8; r=r+1) begin
                a_reg[r][0] <= a_stagger[r][r];
                for (c=1; c<8; c=c+1) a_reg[r][c] <= a_reg[r][c-1];
            end
            for (c=0; c<8; c=c+1) begin
                b_reg[0][c] <= b_stagger[c][c];
                for (r=1; r<8; r=r+1) b_reg[r][c] <= b_reg[r-1][c];
            end

            // 3. Accumulation
            for (r=0; r<8; r=r+1) begin
                for (c=0; c<8; c=c+1) begin
                    acc[r][c] <= acc[r][c] + (a_reg[r][c] * b_reg[r][c]);
                end
            end

            // 4. Readout Logic
            if (cur_cycle == l_config - 1) begin
                readout_active <= 1'b1; c_avail <= 1'b1; readout_ptr <= 0;
            end else if (readout_active) begin
                readout_ptr <= readout_ptr + 1;
                if (readout_ptr == 7) begin readout_active <= 1'b0; c_avail <= 1'b0; end
            end

            if (cur_cycle == l_config + 8) begin
                op_active <= 1'b0;
                clk_cnt <= 8'd0;
            end
        end
    end
endmodule
