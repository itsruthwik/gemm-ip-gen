"""
Shared GEMM metadata, width calculations, and config normalisation for gemm-ip-gen.

Both the Catapult and Vitis backend generators use this module so that stream
widths, tile counts, latency formulas, and validity masks are computed
identically regardless of backend.
"""

import re
import sys as _sys
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────

LANE_WIDTH = 8  # tensor-slice processes 8 int8 lanes per tile per cycle
TENSOR_SLICE_SRC = Path(__file__).resolve().parent.parent / "tensor-slice" / "tensor_slice_int8.v"
SUPPORTED_BACKENDS = ("catapult", "vitis")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _safe_name(name):
    """Sanitise an arbitrary string into a valid C++/Verilog identifier."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(name))
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "dense_layer"


def _ceil_div(a, b):
    return (a + b - 1) // b


def _is_ac_integer_type(type_name):
    """Check if a type name string represents an integer type (rather than fixed-point)."""
    if not isinstance(type_name, str):
        return False
    compact = type_name.replace(" ", "")
    return (
        compact.startswith("int<")
        or compact.startswith("uint<")
        or compact.startswith("ac_int<")
        or compact.startswith("ac_uint<")
    )


# ── Width and tile calculations ────────────────────────────────────────────────


def grid_rows(m):
    """Number of 8-lane row tiles needed to cover *m* rows."""
    return _ceil_div(m, LANE_WIDTH)


def grid_cols(n):
    """Number of 8-lane column tiles needed to cover *n* columns."""
    return _ceil_div(n, LANE_WIDTH)


def a_stream_width(m):
    """Bit-width of the activation stream packet for *m* rows."""
    return grid_rows(m) * 64


def b_stream_width(n):
    """Bit-width of the weight (and bias) stream packet for *n* columns."""
    return grid_cols(n) * 64


def c_stream_width(n):
    """Bit-width of the result stream packet for *n* columns (int8)."""
    return grid_cols(n) * 64


def bias_stream_width(n):
    """Bit-width of the bias stream packet for *n* columns."""
    return b_stream_width(n)


def k_steps(k):
    """Number of K-slice iterations (inner dimension steps)."""
    return _ceil_div(k, LANE_WIDTH)


# ── Latency formulas ───────────────────────────────────────────────────────────


def latency_cycles(k_steps_val, grid_rows_val, grid_cols_val):
    """First output available at this cycle (0-based)."""
    return (grid_cols_val - 1) * 8 + k_steps_val * 8 + 10


def dead_cycles_raw(grid_cols_val):
    """Cycles from first output to drain completion (pre-trim)."""
    return (grid_cols_val - 1) * 8 + 10


def dead_cycles(grid_cols_val):
    """Same as dead_cycles_raw plus 1 for pipeline register."""
    return dead_cycles_raw(grid_cols_val) + 1


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


# ── Config normalisation ───────────────────────────────────────────────────────


def normalize_gemm_config(cfg):
    """Normalise a config dict or list into a list of GEMM items.

    Accepts list, single-item shorthand dict (with m/k/n/name), or
    hls4ml-style named dict.  Backward compatible with all existing formats.
    """
    if isinstance(cfg, list):
        out = []
        for item in cfg:
            item.setdefault("interface", "stream")
            item.setdefault("backend", "catapult")
            item.setdefault("protocol", {})
            out.append(_normalize_item(item))
        return out

    if isinstance(cfg, dict):
        # Single-item shorthand
        if "m" in cfg and "k" in cfg and "n" in cfg and "name" in cfg:
            cfg.setdefault("interface", "stream")
            cfg.setdefault("backend", "catapult")
            cfg.setdefault("protocol", {})
            return [_normalize_item(cfg)]

        # hls4ml-style named dict
        out = []
        for name, item in cfg.items():
            item["name"] = name  # set BEFORE normalize so emit_name uses it
            item.setdefault("interface", "stream")
            item.setdefault("backend", "catapult")
            item.setdefault("protocol", {})
            normalized = _normalize_item(item)
            normalized.setdefault("name", name)
            out.append(normalized)
        return out

    raise TypeError(f"Unsupported config format: {type(cfg)}")


def _normalize_item(item):
    """Ensure a single GEMM config item has all required keys."""
    item.setdefault("gemm_ip_id", item.get("name"))
    item.setdefault("gemm_ip_index", None)

    m = item.get("gemm_m") or item.get("m", 8)
    k = item.get("gemm_k") or item.get("k", item.get("n_in", 8))
    n = item.get("gemm_n") or item.get("n", item.get("n_out", 8))

    item["m"] = int(m)
    item["k"] = int(k)
    item["n"] = int(n)

    item["grid_rows"] = grid_rows(item["m"])
    item["grid_cols"] = grid_cols(item["n"])
    item["k_steps"] = k_steps(item["k"])
    item["emit_name"] = _safe_name(item.get("name", f"gemm_{m}x{k}x{n}"))

    item.setdefault("interface", "stream")
    item.setdefault("backend", "catapult")
    return item


# ── Verilog helpers ────────────────────────────────────────────────────────────


def verilog_tile_comment(item):
    """Return a Verilog comment summarising grid dimensions and tail masks."""
    gr = item["grid_rows"]
    gc = item["grid_cols"]
    ks = item["k_steps"]
    return (
        f"// Dimensions: n_in={item['k']}, n_out={item['n']}, m={item['m']}\n"
        f"// GRID_ROWS={gr}, GRID_COLS={gc}, K_STEPS={ks}\n"
        f"// A tail mask (rows after tile {gr - 1}): {vm(tail_mask_hex(item['m'], gr - 1))}\n"
        f"// B tail mask (cols after tile {gc - 1}): {vm(tail_mask_hex(item['n'], gc - 1))}\n"
        f"// K tail mask (inner dim tile {ks - 1}):   {vm(tail_mask_hex(item['k'], ks - 1))}\n"
    )


def load_catapult_rtl_generator():
    """Import generate_grid_verilog from tensor-slice/generate_catapult_rtl.py."""
    ts_dir = str(Path(__file__).resolve().parent.parent / "tensor-slice")
    if ts_dir not in _sys.path:
        _sys.path.insert(0, ts_dir)
    from generate_catapult_rtl import generate_grid_verilog  # noqa: F811
    return generate_grid_verilog


def load_vitis_rtl_generator():
    """Import generate_vitis_rtl from tensor-slice/generate_vitis_rtl.py."""
    ts_dir = str(Path(__file__).resolve().parent.parent / "tensor-slice")
    if ts_dir not in _sys.path:
        _sys.path.insert(0, ts_dir)
    from generate_vitis_rtl import generate_vitis_rtl  # noqa: F811
    return generate_vitis_rtl
