"""GEMM config normalisation for gemm-ip-gen.

Accepts the several config shapes the CLI and hls4ml emit (list, single-item
shorthand dict, or hls4ml-style named dict) and normalises them into a uniform
list of GEMM items. Target-specific tile geometry (``grid_rows`` / ``grid_cols``)
is sourced from the active target's geometry module.
"""

from gemm_ip.common import _safe_name

from targets.tensor_slice.geometry import grid_rows, grid_cols


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
    item["emit_name"] = _safe_name(item.get("name", f"gemm_{m}x{k}x{n}"))

    item.setdefault("interface", "stream")
    item.setdefault("backend", "catapult")

    # Weight-stationary (const-weight) variant: the wrapper bakes the weights into
    # an internal ROM (from weight_file) and takes no external weight port; the csim
    # header takes A only (no weight argument). Selected solely by weights_in_core;
    # weight_file is the raw-int .dat hls4ml emits, packed per weight_layout
    # (column_major [n][k] default; row_major [k][n] under SecondOperandRowMajor).
    item["weights_in_core"] = bool(item.get("weights_in_core", False))
    item.setdefault("weight_file", item.get("weight_file"))
    item.setdefault("weight_layout", item.get("weight_layout") or "column_major")
    return item


def _normalize_config_items(cfg):
    if isinstance(cfg, list):
        for item in cfg:
            item.setdefault("interface", "stream")
            item.setdefault("protocol", {})
            item.setdefault("gemm_ip_id", item.get("name"))
            item.setdefault("gemm_ip_index", None)
            # ReuseFactor hls4ml emits per GEMM (honored by the generic behavioral
            # target; the RTL targets have their own tiling).
            item.setdefault("reuse_factor", 1)
        return cfg
    if isinstance(cfg, dict):
        if "m" in cfg and "k" in cfg and "n" in cfg and "name" in cfg:
            cfg.setdefault("interface", "stream")
            cfg.setdefault("protocol", {})
            cfg.setdefault("gemm_ip_id", cfg.get("name"))
            cfg.setdefault("gemm_ip_index", None)
            cfg.setdefault("reuse_factor", 1)
            return [cfg]
        items = []
        for name, item in cfg.items():
            interface = item.get("interface", "stream")
            items.append({
                "name": name,
                "m": item.get("gemm_m", 1),
                "k": item.get("gemm_k", item["n_in"]),
                "n": item.get("gemm_n", item["n_out"]),
                "interface": interface,
                "protocol": item.get("protocol", {}),
                "gemm_ip_id": item.get("gemm_ip_id", name),
                "gemm_ip_index": item.get("gemm_ip_index"),
                "reuse_factor": item.get("reuse_factor", 1),
                "fold_axis": item.get("fold_axis"),
                "output_precision": item.get("output_precision"),
                "input_precision": item.get("input_precision"),
                "weight_precision": item.get("weight_precision"),
                # accum_precision drives tensor_slice's S1 (in-slice pre-round)
                # derivation; has_bias/bias are the single source of truth for
                # whether/what compile-time bias to bake (see
                # jojo-track/open/tensor-slice-bias-in-rtl).
                "accum_precision": item.get("accum_precision"),
                "bias_precision": item.get("bias_precision"),
                "has_bias": item.get("has_bias"),
                "bias": item.get("bias"),
                "clock_period_ns": item.get("clock_period_ns"),
                "part": item.get("part"),
                # DEBUG: mvau user-directed fold/tiling knobs injected via ATLASConfig
                # (bypassing hls4ml). TODO: Ruthwik change this.
                "pe": item.get("pe"),
                "simd": item.get("simd"),
                "k_tiles": item.get("k_tiles"),
                "n_tiles": item.get("n_tiles"),
                "second_operand_row_major": item.get("second_operand_row_major"),
                # Weight-stationary (const-weight) selection + weights source.
                "weights_in_core": bool(item.get("weights_in_core", False)),
                "weight_file": item.get("weight_file"),
                "weight_layout": item.get("weight_layout") or "column_major",
            })
        return items
    raise TypeError("Unsupported config format")
