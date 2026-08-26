"""Generic, target-agnostic helpers shared across the gemm-ip-gen core."""

import re


def _safe_name(name):
    """Sanitise an arbitrary string into a valid C++/Verilog identifier."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(name))
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "dense_layer"


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
