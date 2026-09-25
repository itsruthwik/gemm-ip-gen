// cmvu_w_mem — logical 1R1W tile memory with two interleaved physical 1R1W banks.
//
// A generic, self-contained, synthesizable memory of WORDS entries, each WORD_BITS wide,
// implemented as two interleaved physical 1R1W banks. It accepts up to two logical writes per
// cycle only when their addresses select distinct banks (logical address bit 0).
//
// READ: the read address is REGISTERED internally (single-cycle synchronous read) -- there is
// no asynchronous/combinational read path. Present `rd_addr` on cycle N; the corresponding
// WORD_BITS-wide word appears on `rd_data` on cycle N+1.
//
// WRITE: byte enables make the write port layout-neutral.  A row-major load enables one
// contiguous group of bytes; a column-major load enables one byte in every row.  This keeps
// the stored/read representation canonical row-major without placing a transpose on the
// compute path.  `wr_data` is WORD_BITS wide and only bytes selected by `wr_byte_en` update.
//
// Collision rule: on a cycle where a write lands at the same address the read address is
// registering, the read returns the PRIOR (pre-write) contents of that address on the
// following cycle -- i.e. read-before-write, same-address collision resolves to old data.
//
// Reset: the read-data output register clears to 0 on `rst` (async reset, matching the
// convention used elsewhere in this pipeline); the memory array contents themselves are not
// reset (only the read data register is defined at reset).
module cmvu_w_mem #(
    parameter int unsigned WORDS      = 64,   // number of WORD_BITS-wide entries
    parameter int unsigned WORD_BITS  = 256,  // width of one entry
    parameter int unsigned ADDR_W     = (WORDS  < 2) ? 1 : $clog2(WORDS),
    parameter int unsigned WORD_BYTES = WORD_BITS / 8,
    parameter int unsigned BANK_WORDS = (WORDS < 2) ? 1 : WORDS / 2
) (
    input  wire                    clk,
    input  wire                    rst,   // async reset (read-data register only)

    // Read port: registered address -> data valid next cycle.
    input  wire [ADDR_W-1:0]       rd_addr,
    output reg  [WORD_BITS-1:0]    rd_data,

    // Two logical write requests can be accepted only when they map to separate banks.
    input wire [ADDR_W-1:0] wr0_addr, input wire wr0_en,
    input wire [WORD_BYTES-1:0] wr0_byte_en, input wire [WORD_BITS-1:0] wr0_data,
    input wire [ADDR_W-1:0] wr1_addr, input wire wr1_en,
    input wire [WORD_BYTES-1:0] wr1_byte_en, input wire [WORD_BITS-1:0] wr1_data
);
    // Logical address bit 0 selects bank; address[ADDR_W-1:1] is the shared pair index.
    logic [WORD_BITS-1:0] bank0 [0:BANK_WORDS-1];
    logic [WORD_BITS-1:0] bank1 [0:BANK_WORDS-1];
    localparam int unsigned BANK_ADDR_W = (ADDR_W <= 1) ? 1 : ADDR_W - 1;

    // Pre-route logical requests before the arrays. Each physical array below has exactly one
    // write-enable/address/data/mask bundle and one write process, making its 1R1W shape
    // explicit to memory inference. A same-bank dual request is an asserted caller error.
    logic bank0_wr_en, bank1_wr_en;
    logic [BANK_ADDR_W-1:0] bank0_wr_addr, bank1_wr_addr;
    logic [WORD_BYTES-1:0] bank0_wr_byte_en, bank1_wr_byte_en;
    logic [WORD_BITS-1:0] bank0_wr_data, bank1_wr_data;
    always_comb begin
        bank0_wr_en='0; bank0_wr_addr='0; bank0_wr_byte_en='0; bank0_wr_data='0;
        bank1_wr_en='0; bank1_wr_addr='0; bank1_wr_byte_en='0; bank1_wr_data='0;
        if (wr0_en) begin
            if (wr0_addr[0]) begin bank1_wr_en=1'b1; bank1_wr_addr=wr0_addr >> 1; bank1_wr_byte_en=wr0_byte_en; bank1_wr_data=wr0_data; end
            else begin bank0_wr_en=1'b1; bank0_wr_addr=wr0_addr >> 1; bank0_wr_byte_en=wr0_byte_en; bank0_wr_data=wr0_data; end
        end
        if (wr1_en) begin
            if (wr1_addr[0]) begin bank1_wr_en=1'b1; bank1_wr_addr=wr1_addr >> 1; bank1_wr_byte_en=wr1_byte_en; bank1_wr_data=wr1_data; end
            else begin bank0_wr_en=1'b1; bank0_wr_addr=wr1_addr >> 1; bank0_wr_byte_en=wr1_byte_en; bank0_wr_data=wr1_data; end
        end
    end
    integer bank0_byte, bank1_byte;
    always_ff @(posedge clk) if (bank0_wr_en)
        for (bank0_byte=0; bank0_byte<WORD_BYTES; bank0_byte=bank0_byte+1)
            if (bank0_wr_byte_en[bank0_byte])
                bank0[bank0_wr_addr][bank0_byte*8 +: 8] <= bank0_wr_data[bank0_byte*8 +: 8];
    always_ff @(posedge clk) if (bank1_wr_en)
        for (bank1_byte=0; bank1_byte<WORD_BYTES; bank1_byte=bank1_byte+1)
            if (bank1_wr_byte_en[bank1_byte])
                bank1[bank1_wr_addr][bank1_byte*8 +: 8] <= bank1_wr_data[bank1_byte*8 +: 8];

    // Read, synchronous, registered address/data: same-address read/write collision resolves
    // to the PRIOR contents (this always-block samples `mem` before the write always-block's
    // nonblocking assignment above takes effect this same edge, giving read-before-write).
    always_ff @(posedge clk or posedge rst) begin
        if (rst) rd_data <= '0;
        else if (rd_addr[0]) rd_data <= bank1[rd_addr >> 1];
        else                 rd_data <= bank0[rd_addr >> 1];
    end

`ifndef SYNTHESIS
    initial assert (WORDS >= 2 && (WORDS % 2) == 0)
        else $error("cmvu_w_mem: interleaved storage requires an even WORDS >= 2");
    always_ff @(posedge clk) if (!rst && wr0_en && wr1_en)
        assert (wr0_addr[0] != wr1_addr[0]) else $error("cmvu_w_mem: two writes targeted one physical bank");
`endif

endmodule
