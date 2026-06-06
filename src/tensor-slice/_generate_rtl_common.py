"""Shared helpers for generate_catapult_rtl.py and generate_vitis_rtl.py."""
# NOTE: grid_rows / grid_cols / stream-width helpers are canonical in
#       gemm_ip.metadata — import from there for those.


def tail_mask_hex(total, chunk_index):
    remain = total - chunk_index * 8
    if remain <= 0:
        return 0x00
    if remain >= 8:
        return 0xFF
    return (0xFF >> (8 - remain)) & 0xFF


def vm(val):
    return f"8'h{val:02X}"


def total_cycles(m, k, n, P=3, feed_mode="chained"):
    """Conservative total cycle count for the grid counter.

    Row/col: preload(1) + collect(max(M,N)) + transition(2) +
    feed(max_loc_delay + max(M,N)) + compute + output_align + margin.
    """
    gr = (m + 7) // 8
    gc = (n + 7) // 8
    last_loc = (gr - 1 + gc - 1) * 8
    beats = max(m, n)
    local_compute = 7 + k + P
    output_align = (gc - 1) * 8 + (gr - 1) * gc * 8
    margin = 60
    return 1 + beats + 2 + (last_loc + beats) + last_loc + local_compute + output_align + 8 + margin


def tile_comment(m, k, n, gr, gc):
    return (
        f"// Dimensions: M={m}, K={k}, N={n}\n"
        f"// GRID_ROWS={gr}, GRID_COLS={gc}\n"
        f"// A tail (rows after tile {gr - 1}): {vm(tail_mask_hex(m, gr - 1))}\n"
        f"// B tail (cols after tile {gc - 1}): {vm(tail_mask_hex(n, gc - 1))}\n"
    )
