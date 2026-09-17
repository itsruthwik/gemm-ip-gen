"""tensor_slice geometry: tile counts, stream widths, latency, and validity masks.

This is the canonical home for the 8-lane tiling geometry of the tensor_slice
hardblock. It is deliberately self-contained (no ``gemm_ip`` imports) so the RTL
generation path — ``rtl.py`` / ``golden.py`` and ``run_rtl_tests.py`` — can import
it with only the target directory on ``sys.path``.
"""
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────

LANE_WIDTH = 8  # tensor_slice processes 8 int8 lanes per tile per cycle

# Standalone slice IP RTL for this target (external; not required for the
# behavioral RTL path, only for structural synthesis).
TENSOR_SLICE_SRC = Path(__file__).resolve().parent / "tensor_slice_int8.v"


# ── Helpers ────────────────────────────────────────────────────────────────────
# _ceil_div is kept local (rather than imported from gemm_ip.common) so this
# module stays import-free of the core — see the module docstring.


def _ceil_div(a, b):
    return (a + b - 1) // b


# ── Width and tile calculations ────────────────────────────────────────────────


def grid_rows(m):
    """Number of 8-lane row tiles needed to cover *m* rows."""
    return _ceil_div(m, LANE_WIDTH)


def grid_cols(n):
    """Number of 8-lane column tiles needed to cover *n* columns."""
    return _ceil_div(n, LANE_WIDTH)


def k_chunks(k):
    """Number of 8-lane K chunks needed to cover *k* reduction lanes."""
    return _ceil_div(k, LANE_WIDTH)


def k_chunk_size(k, chunk_index):
    """Valid K lanes in one 8-lane chunk."""
    remain = int(k) - int(chunk_index) * LANE_WIDTH
    if remain <= 0:
        return 0
    return min(LANE_WIDTH, remain)


def k_chunk_mask(k, chunk_index):
    """8-bit validity mask for one K chunk."""
    return tail_mask_hex(k, chunk_index)


def k_passes(k, k_spatial):
    """Number of sequential passes over K needed with *k_spatial* parallel chunks.

    This is the legalized ReuseFactor: the number of times each input vector's
    K reduction is fed through the array (MAC uses per input vector), not a
    cycle count and not II.
    """
    kc = k_chunks(k)
    return _ceil_div(kc, int(k_spatial))


def k_chunks_padded(k, k_spatial):
    """Total K-chunk slots across all passes, including zero/masked padding."""
    return k_passes(k, k_spatial) * int(k_spatial)


def multipliers(m, n, k_spatial):
    """INT8 multiplier count for a grid with *k_spatial* parallel K partitions."""
    return 64 * grid_rows(m) * grid_cols(n) * int(k_spatial)


def resolve_fold_m(m, reuse_factor, name=None):
    """Legalize a requested ReuseFactor into a row-tile-group (fold-M) plan.

    ``reuse_factor`` (RF) is the number of row-tile groups (frames) the M
    dimension is folded into: ``grid_rows = ceil(m/8)`` row tiles are covered
    by ``mg = ceil(grid_rows / RF)`` row tiles per group, in
    ``m_passes = ceil(grid_rows / mg)`` back-to-back frames; the legalized RF
    is ``m_passes`` (which may land lower than requested -- silently).
    Requests above ``grid_rows`` (RF > grid_rows) legalize down to
    ``grid_rows`` (one row tile per frame) with a warning. RF=1 legalizes to
    ``mg=grid_rows``, ``m_passes=1`` -- today's single-frame hardware. The
    legal range is ``1..grid_rows``.

    Returns a dict: mg, m_passes, grid_rows, grid_rows_pad,
    reuse_factor_requested, reuse_factor (legalized), warnings (list[str]).
    """
    f = _resolve_axis_fold(m, reuse_factor, name)
    return {
        "mg": f["group"],
        "m_passes": f["passes"],
        "grid_rows": f["chunks"],
        "grid_rows_pad": f["chunks_pad"],
        "reuse_factor_requested": f["reuse_factor_requested"],
        "reuse_factor": f["reuse_factor"],
        "warnings": f["warnings"],
    }


def resolve_fold_n(n, reuse_factor, name=None):
    """Legalize a requested ReuseFactor into a column-tile-group (fold-N) plan.

    ``reuse_factor`` (RF) is the number of column-tile groups (frames) the N
    dimension is folded into: ``grid_cols = ceil(n/8)`` column tiles are
    covered by ``cg = ceil(grid_cols / RF)`` column tiles per group, in
    ``n_passes = ceil(grid_cols / cg)`` back-to-back frames; the legalized RF
    is ``n_passes`` (which may land lower than requested -- silently).
    Requests above ``grid_cols`` (RF > grid_cols) legalize down to
    ``grid_cols`` (one column tile per frame) with a warning. RF=1 legalizes
    to ``cg=grid_cols``, ``n_passes=1`` -- today's single-frame hardware. The
    legal range is ``1..grid_cols``.

    Returns a dict: cg, n_passes, grid_cols, grid_cols_pad,
    reuse_factor_requested, reuse_factor (legalized), warnings (list[str]).
    """
    f = _resolve_axis_fold(n, reuse_factor, name)
    return {
        "cg": f["group"],
        "n_passes": f["passes"],
        "grid_cols": f["chunks"],
        "grid_cols_pad": f["chunks_pad"],
        "reuse_factor_requested": f["reuse_factor_requested"],
        "reuse_factor": f["reuse_factor"],
        "warnings": f["warnings"],
    }


def _resolve_axis_fold(dim, reuse_factor, name=None):
    """Legalize a requested pass count for one axis's tile-group folding.

    Generic version of :func:`resolve_fold_m` / :func:`resolve_fold_n`:
    ``chunks = ceil(dim/8)`` tiles are covered by ``group = ceil(chunks/RF)``
    tiles per group, in ``passes = ceil(chunks/group)`` back-to-back groups;
    the legalized RF is ``passes``. RF=1 legalizes to ``group=chunks``,
    ``passes=1`` (fully spatial). Legal range ``1..chunks``.

    Returns a dict: group, passes, chunks, chunks_pad, reuse_factor_requested,
    reuse_factor (legalized), warnings.
    """
    chunks = _ceil_div(int(dim), LANE_WIDTH)
    rf_req = int(reuse_factor)
    warnings = []
    rf_use = rf_req
    if rf_use < 1:
        rf_use = 1
    if rf_use > chunks:
        who = f" for layer {name}" if name is not None else ""
        warnings.append(
            f"WARNING: Invalid ReuseFactor={rf_req}{who}. "
            f"Using ReuseFactor={chunks} instead. Valid ReuseFactor(s): 1..{chunks}."
        )
        rf_use = chunks
    group = _ceil_div(chunks, rf_use)
    passes = _ceil_div(chunks, group)
    chunks_pad = passes * group
    return {
        "group": group,
        "passes": passes,
        "chunks": chunks,
        "chunks_pad": chunks_pad,
        "reuse_factor_requested": rf_req,
        "reuse_factor": passes,
        "warnings": warnings,
    }


def resolve_mkn_geometry(m, k, n, m_reuse_factor=1, k_reuse_factor=1, n_reuse_factor=1,
                          name=None):
    """General independent M/K/N tensor_slice geometry resolver.

    Each axis is legalized independently via the same tile-group-fold rule
    (:func:`_resolve_axis_fold`): axis ``reuse_factor`` RF=1 means "fully
    spatial, one pass"; RF>1 folds the axis's 8-lane tile count into ``RF``
    (legalized) back-to-back groups. There is no cross-axis coupling -- the M,
    K, and N legalizations are independent of each other. This is the general
    form the single-axis ``fold_axis`` resolvers below degenerate to.

    Returns a dict with, per axis, the spatial tile-group size and pass count
    (``m_spatial``/``m_passes``, ``k_spatial``/``k_passes``, ``n_spatial``/
    ``n_passes``), the raw tile counts (``grid_rows``, ``k_chunks``,
    ``grid_cols``) and their group-padded totals (``grid_rows_pad``,
    ``k_chunks_pad``, ``grid_cols_pad``), the derived slice count
    (``slices = k_spatial * m_spatial * n_spatial``, the number of
    concurrently-active 8x8 tensor-slice tiles), ``multipliers`` (64 per
    slice), stream widths (``a_bits``/``b_bits``/``c_bits``), and per-axis
    ``warnings``.
    """
    fm = _resolve_axis_fold(m, m_reuse_factor, name)
    fk = _resolve_axis_fold(k, k_reuse_factor, name)
    fn = _resolve_axis_fold(n, n_reuse_factor, name)

    m_spatial = fm["group"]
    m_passes = fm["passes"]
    k_spatial = fk["group"]
    k_passes_ = fk["passes"]
    n_spatial = fn["group"]
    n_passes = fn["passes"]

    slices = k_spatial * m_spatial * n_spatial
    mult = 64 * slices

    return {
        "m_spatial": m_spatial,
        "m_passes": m_passes,
        "grid_rows": fm["chunks"],
        "grid_rows_pad": fm["chunks_pad"],
        "m_reuse_factor_requested": fm["reuse_factor_requested"],
        "m_reuse_factor": fm["reuse_factor"],

        "k_spatial": k_spatial,
        "passes": k_passes_,
        "k_passes": k_passes_,
        "k_chunks": fk["chunks"],
        "k_chunks_pad": fk["chunks_pad"],
        "k_reuse_factor_requested": fk["reuse_factor_requested"],
        "k_reuse_factor": fk["reuse_factor"],
        "reuse_factor": fk["reuse_factor"],
        "effective_reuse": fk["reuse_factor"],

        "n_spatial": n_spatial,
        "n_passes": n_passes,
        "grid_cols": fn["chunks"],
        "grid_cols_pad": fn["chunks_pad"],
        "n_reuse_factor_requested": fn["reuse_factor_requested"],
        "n_reuse_factor": fn["reuse_factor"],

        "slices": slices,
        "multipliers": mult,
        "warnings": fm["warnings"] + fk["warnings"] + fn["warnings"],
    }


def resolve_reuse_factor(k, reuse_factor, name=None, fold_axis="k", m=None, n=None):
    """Legalize a requested ReuseFactor for the tensor_slice target.

    ``fold_axis`` selects which dimension ReuseFactor folds: ``"k"`` (default)
    is the phase 1 K-partition legalization above; ``"m"`` folds row tiles
    instead (see :func:`resolve_fold_m`) and keeps K fully spatial
    (``k_spatial = k_chunks``, one K pass -- the rf=1 K fields); ``"n"`` folds
    column tiles instead (see :func:`resolve_fold_n`) and also keeps K and M
    fully spatial (``k_spatial = k_chunks``, one K pass -- the rf=1 K fields).

    Under ``fold_axis="m"``/``"n"`` this returns the same dict shape as the
    ``k`` path, with the K fields pinned to their rf=1 values and the fold
    fields added; ``reuse_factor`` is the legalized fold pass count.
    """
    if fold_axis == "m":
        if m is None:
            raise ValueError("resolve_reuse_factor(fold_axis='m') requires m")
        kc = k_chunks(k)
        fm = resolve_fold_m(m, reuse_factor, name)
        return {
            "k_spatial": kc,
            "passes": 1,
            "k_chunks": kc,
            "k_chunks_pad": kc,
            "reuse_factor_requested": fm["reuse_factor_requested"],
            "reuse_factor": fm["reuse_factor"],
            "effective_reuse": 1,
            "warnings": fm["warnings"],
            "mg": fm["mg"],
            "m_passes": fm["m_passes"],
            "grid_rows": fm["grid_rows"],
            "grid_rows_pad": fm["grid_rows_pad"],
        }
    if fold_axis == "n":
        if n is None:
            raise ValueError("resolve_reuse_factor(fold_axis='n') requires n")
        kc = k_chunks(k)
        fn = resolve_fold_n(n, reuse_factor, name)
        return {
            "k_spatial": kc,
            "passes": 1,
            "k_chunks": kc,
            "k_chunks_pad": kc,
            "reuse_factor_requested": fn["reuse_factor_requested"],
            "reuse_factor": fn["reuse_factor"],
            "effective_reuse": fn["reuse_factor"],
            "warnings": fn["warnings"],
            "cg": fn["cg"],
            "n_passes": fn["n_passes"],
            "grid_cols": fn["grid_cols"],
            "grid_cols_pad": fn["grid_cols_pad"],
        }
    return _resolve_reuse_factor_k(k, reuse_factor, name)


def _resolve_reuse_factor_k(k, reuse_factor, name=None):
    """Legalize a requested ReuseFactor into a K-partition count.

    ``reuse_factor`` (RF) is the number of passes over K each input vector's
    reduction takes -- i.e. how many times each MAC in the array is reused
    per input vector. It is never a cycle count or an initiation interval.

    ``k_spatial`` parallel K partitions cover ``k_chunks = ceil(k/8)`` chunks
    in ``passes = ceil(k_chunks / k_spatial)`` passes; the legalized RF is
    ``passes`` (which may land lower than requested -- silently). Requests
    above ``k_chunks`` (RF > k_chunks) legalize down to ``k_chunks`` (today's
    chunked, ks=1) with a warning. RF=1 legalizes to ``k_spatial=k_chunks``
    (today's full-K). The legal range is ``1..k_chunks``.

    Returns a dict: k_spatial, passes, k_chunks, k_chunks_pad,
    reuse_factor_requested, reuse_factor (legalized), effective_reuse
    (== passes), warnings (list[str]).
    """
    f = _resolve_axis_fold(k, reuse_factor, name)
    return {
        "k_spatial": f["group"],
        "passes": f["passes"],
        "k_chunks": f["chunks"],
        "k_chunks_pad": f["chunks_pad"],
        "reuse_factor_requested": f["reuse_factor_requested"],
        "reuse_factor": f["reuse_factor"],
        "effective_reuse": f["passes"],
        "warnings": f["warnings"],
    }


def a_stream_width(m, k_spatial=1):
    """Bit-width of the activation stream packet for *m* rows.

    ``k_spatial == 1`` keeps today's grid-padded word (one 64-bit lane group
    per row tile). ``k_spatial > 1`` is the narrow K-spatial word: one 64-bit
    lane group per K partition, independent of row-tile count.
    """
    if k_spatial and int(k_spatial) > 1:
        return 64 * int(k_spatial)
    return grid_rows(m) * 64


def b_stream_width(n, k_spatial=1):
    """Bit-width of the weight (and bias) stream packet for *n* columns.

    See :func:`a_stream_width` for the ``k_spatial`` word-width rule.
    """
    if k_spatial and int(k_spatial) > 1:
        return 64 * int(k_spatial)
    return grid_cols(n) * 64


def c_stream_width(n, out_bits=8):
    """Bit-width of the result stream packet for *n* columns.

    ``out_bits`` is the per-lane result width from ``output_precision`` (default
    8 = legacy int8). 8 lanes per column tile, so width = grid_cols * 8 * out_bits.
    """
    return grid_cols(n) * 8 * out_bits


def bias_stream_width(n):
    """Bit-width of the bias stream packet for *n* columns."""
    return b_stream_width(n)


# ── Validity masks (for tail dimensions) ───────────────────────────────────────


def tail_mask_hex(total, chunk_index):
    """Return an 8-bit validity mask for the *chunk_index*-th 8-lane tile."""
    remain = total - chunk_index * LANE_WIDTH
    if remain <= 0:
        return 0x00
    if remain >= LANE_WIDTH:
        return 0xFF
    return (0xFF >> (LANE_WIDTH - remain)) & 0xFF


def vm(val):
    """Format a mask byte as a Verilog literal."""
    return f"8'h{val:02X}"


# ── Latency / cycle counts ─────────────────────────────────────────────────────


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


def latency_cycles(k_val, grid_rows_val, grid_cols_val, m=None, n=None,
                   k=None, feed_mode="direct"):
    """First output available at this cycle (0-based).  Conservative estimate;
    exact transaction control uses total_cycles().
    """
    if m is not None and n is not None and k is not None:
        # Conservative: total_cycles minus output FIFO drain margin (~20 cycles)
        return max(0, total_cycles(m, k, n, feed_mode=feed_mode) - 20 - (grid_rows_val * 8))
    # Fallback
    return (grid_cols_val - 1) * 8 + k_val + 10


def latency_first_out(m, k, n, k_spatial):
    """First-output beat offset for the K-spatial behavioral sim model.

    ``total_beats = passes * max(m, n)``; the remaining wave latency is
    ``max(0, K' + n - total_beats)`` where ``K'`` is *k* itself whenever there
    is no K-chunk padding (``k_spatial == 1``, today's chunked endpoint, or
    ``k_chunks_pad == k_chunks``, today's full-K endpoint and any other
    passes==1 case) and is ``8 * k_chunks_pad`` otherwise (padded partial
    passes: the masked pad chunks still occupy a beat's worth of K in the
    wave-latency term). This reproduces the existing chunked and full-K
    formulas exactly at their endpoints.
    """
    ks = int(k_spatial)
    kc = k_chunks(k)
    passes = k_passes(k, ks)
    kc_pad = passes * ks
    total_beats = passes * max(m, n)
    if ks == 1 or kc_pad == kc:
        k_term = k
    else:
        k_term = 8 * kc_pad
    latency = max(0, k_term + n - total_beats)
    return total_beats + latency


def dead_cycles_raw(grid_cols_val):
    """Cycles from first output to drain completion (pre-trim)."""
    return (grid_cols_val - 1) * 8 + 10


def dead_cycles(grid_cols_val):
    """Same as dead_cycles_raw plus 1 for pipeline register."""
    return dead_cycles_raw(grid_cols_val) + 1


# ── Verilog comment helpers ─────────────────────────────────────────────────────


def tile_comment(m, k, n, gr, gc):
    return (
        f"// Dimensions: M={m}, K={k}, N={n}\n"
        f"// GRID_ROWS={gr}, GRID_COLS={gc}\n"
        f"// A tail (rows after tile {gr - 1}): {vm(tail_mask_hex(m, gr - 1))}\n"
        f"// B tail (cols after tile {gc - 1}): {vm(tail_mask_hex(n, gc - 1))}\n"
    )


def verilog_tile_comment(item):
    """Return a Verilog comment summarising grid dimensions and tail masks."""
    gr = item["grid_rows"]
    gc = item["grid_cols"]
    return (
        f"// Dimensions: n_in={item['k']}, n_out={item['n']}, m={item['m']}\n"
        f"// GRID_ROWS={gr}, GRID_COLS={gc}\n"
        f"// A tail mask (rows after tile {gr - 1}): {vm(tail_mask_hex(item['m'], gr - 1))}\n"
        f"// B tail mask (cols after tile {gc - 1}): {vm(tail_mask_hex(item['n'], gc - 1))}\n"
    )
