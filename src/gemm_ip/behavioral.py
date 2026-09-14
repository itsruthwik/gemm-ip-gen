"""Tool-neutral rules shared by every behavioral-HLS GEMM target.

Both implementations of the ``generic`` target (Vitis and Catapult) consume
a manifest of the same shape and legalize the same hls4ml ReuseFactor rules
against it; neither owns a hardblock, so this module (not either target
directory) is the single place these rules live. Only the RF-snapping and
geometry logic belongs here -- everything downstream of it (the emitted C++
idiom, the package layout, the tool invocation) stays tool-specific in each
target's own directory.
"""

import math
from bisect import bisect_left


def valid_reuse_factors(n_in, n_out):
    """hls4ml's reuse-factor rules (fpga_backend._validate_reuse_factor), verbatim:
    rf must divide n_in*n_out; below n_in the multiplier count must be a multiple of
    n_out; above n_in, rf must be a multiple of n_in."""
    valid = []
    for rf in range(1, n_in * n_out + 1):
        multfactor = min(n_in, rf)
        multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
        ok = ((multiplier_limit % n_out) == 0) or (rf >= n_in)
        ok = ok and (((rf % n_in) == 0) or (rf < n_in))
        ok = ok and (((n_in * n_out) % rf) == 0)
        if ok:
            valid.append(rf)
    return valid


def closest_reuse_factor(valid_rf, chosen_rf):
    """hls4ml's get_closest_reuse_factor: nearest valid value, smaller on ties."""
    pos = bisect_left(valid_rf, chosen_rf)
    if pos == 0:
        return valid_rf[0]
    if pos == len(valid_rf):
        return valid_rf[-1]
    before, after = valid_rf[pos - 1], valid_rf[pos]
    return before if (after - chosen_rf) >= (chosen_rf - before) else after


def snap_reuse_factor(item):
    """Legalize a manifest item's ReuseFactor against hls4ml's validation rules.

    ReuseFactor is a pure pass-through from hls4ml's HLSConfig into the manifest
    (no ATLASConfig knob for it) -- but hls4ml's init_dense skips
    set_closest_reuse_factor for Strategy=GEMM layers, so the raw HLSConfig value
    reaches this manifest unvalidated; a resource core then builds the remainder
    regime with a non-dividing block factor (several x the area of the snapped
    point) if handed it as-is. So every behavioral target legalizes it itself
    here: snap, print hls4ml's own warning, keep the request as
    reuse_factor_requested. The combined header reads the snapped value from the
    manifest (gemm_rf<CONFIG_T>) instead of hls4ml's CONFIG_T::reuse_factor.
    """
    name = item.get("name", "?")
    n_in = int(item.get("gemm_k", item.get("k", item.get("n_in", 0))) or 0)
    n_out = int(item.get("gemm_n", item.get("n", item.get("n_out", 0))) or 0)
    chosen = int(item.get("reuse_factor", 1) or 1)
    item["reuse_factor_requested"] = chosen
    if n_in <= 0 or n_out <= 0:
        return
    valid = valid_reuse_factors(n_in, n_out)
    if chosen in valid:
        return
    closest = closest_reuse_factor(valid, chosen)
    print(f'WARNING: Invalid ReuseFactor={chosen} in layer "{name}".'
          f'Using ReuseFactor={closest} instead. Valid ReuseFactor(s): '
          f'{",".join(map(str, valid))}.')
    item["reuse_factor"] = closest


def geometry(shape):
    """Tile geometry for a resource-only behavioral GEMM: shape passes straight
    through as (gemm_m, gemm_k, gemm_n) plus the hls4ml-facing n_in/n_out
    aliases -- no grid/fold plan, unlike an RTL hardblock target."""
    m, k, n = shape
    return {"gemm_m": m, "gemm_k": k, "gemm_n": n, "n_in": k, "n_out": n}
