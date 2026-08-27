"""mvau target geometry: folding search, core/ACCU/SEGMENTLEN/NARROW resolution,
and the hls4ml-knob -> FINN-fold mapping.

Self-contained (no ``gemm_ip`` imports) so ``rtl.py`` / ``golden.py`` / ``package.py``
can import it with only the target directory on ``sys.path`` -- mirrors the
convention of ``targets/tensor_slice/geometry.py``.

The compute is FINN's RTL MVU (``mvu_vvu_axi`` + the DSP-packing cores). A GEMM
``C[M,N] = A[M,K] . B[K,N]`` maps to one MVU as ``W := B^T`` (MH=N, MW=K), rows of
A streamed as activations (numInputVectors = M), rows of C collected. Folding
``(PE, SIMD)`` picks the physical array; ``NF=MH/PE`` and ``SF=MW/SIMD`` are the
sequential fold counts, so one input vector costs ``SF*NF`` cycles (II=1).
"""

import math
import re


# ── Constants ────────────────────────────────────────────────────────────────

#: FINN's ``mvau_wwidth_max``: the per-PE weight-stream width cap that bounds the
#: SIMD search (weight_bitwidth * SIMD <= this). Keeps the weight beat routable.
WWIDTH_MAX = 36

#: Accumulator guard bits added on top of ceil(log2 K) + w + a.
ACCU_GUARD = 1


# ── Precision parsing (local; geometry stays import-free of the core) ─────────

def parse_width(precision, default=8):
    """Total bit width of an ``ac_fixed``-style string like ``fixed<8,4>``.

    Returns *default* for an unset/unparseable precision.
    """
    if not precision:
        return default
    m = re.search(r"u?fixed<\s*(\d+)", str(precision))
    return int(m.group(1)) if m else default


def parse_signed(precision):
    """True unless the precision is an unsigned (``ufixed``) type."""
    if not precision:
        return True
    return not str(precision).lstrip().startswith("ufixed")


def parse_int_bits(precision, default=None):
    """Integer-bit count (the second ``fixed<W,I>`` field). Defaults to *default*."""
    if not precision:
        return default
    m = re.search(r"u?fixed<\s*\d+\s*,\s*(-?\d+)", str(precision))
    return int(m.group(1)) if m else default


def parse_frac(precision, default=0):
    """Fractional-bit count (W - I) of a precision like ``fixed<8,4>``.

    Operands are fed as fixed-point integer *codes* (value·2^frac), so the raw
    dot product carries ``2^(frac_a+frac_b)`` which the drain rescales out.
    Returns 0 for an unset/unparseable precision (integer-coded, no rescale).
    """
    if not precision:
        return default
    m = re.search(r"u?fixed<\s*(\d+)\s*,\s*(-?\d+)", str(precision))
    return (int(m.group(1)) - int(m.group(2))) if m else default


# ── Divisors ──────────────────────────────────────────────────────────────────

def divisors(num):
    """Ascending divisors of *num* (>=1)."""
    return [x for x in range(1, int(num) + 1) if num % x == 0]


# ── Device -> DSP block -> compute core ───────────────────────────────────────

def dsp_block_for_part(part):
    """Map an FPGA part string to its DSP generation.

    Heuristic (mirrors the intent of FINN's ``get_dsp_block``): Versal -> DSP58,
    UltraScale/UltraScale+ -> DSP48E2, 7-series -> DSP48E1. Defaults to DSP48E2
    (the ATLAS default part is ``xcvu13p``) when unknown.
    """
    if not part:
        return "DSP48E2"
    p = str(part).lower()
    # Versal families: xcvc / xcvp / xcvm / xcve / xcvh (but NOT xcvu = US+ Virtex)
    if re.match(r"^xcv[cpmeh]", p):
        return "DSP58"
    # UltraScale / UltraScale+ : xcvu, xcku, xczu, xcau, xcu...
    if re.match(r"^xc(vu|ku|zu|au|u)", p):
        return "DSP48E2"
    # 7-series and older
    if re.match(r"^xc7", p):
        return "DSP48E1"
    return "DSP48E2"


def select_core(dsp_block, weight_width, act_width):
    """Pick the FINN COMPUTE_CORE string for a device + operand widths.

    Mirrors FINN's ``_resolve_impl_style``: DSP58 covers the whole <=9b/<=8b
    envelope with one core; on DSP48 the 4-bit path packs 4 MACs/DSP, else the
    8-bit path packs 2.
    """
    if dsp_block == "DSP58":
        return "mvu_vvu_8sx9_dsp58"
    if weight_width <= 4 and act_width <= 4:
        return "mvu_4sx4u_dsp48e1" if dsp_block == "DSP48E1" else "mvu_4sx4u_dsp48e2"
    return "mvu_8sx8u_dsp48"


# ── Envelope validation ───────────────────────────────────────────────────────

def check_envelope(dsp_block, weight_width, act_width, signed_act):
    """Reject operand widths outside the native MVU envelope (integer only).

    Raises ``ValueError`` for >8b weights, or >8b activations (>9b even when
    signed on DSP58). These cores have no path for wider or floating operands;
    bit-plane decomposition is a documented, out-of-scope extension.
    """
    if weight_width > 8:
        raise ValueError(
            f"weight width {weight_width}b exceeds the MVU envelope (<=8b); "
            "reject or bit-plane decompose (out of scope)")
    if act_width > 8:
        ok = signed_act and act_width == 9 and dsp_block == "DSP58"
        if not ok:
            raise ValueError(
                f"activation width {act_width}b exceeds the MVU envelope "
                "(<=8b, or <=9b signed on DSP58)")


# ── Accumulator / segment / narrow ────────────────────────────────────────────

def accu_width(k, weight_width, act_width, guard=ACCU_GUARD):
    """Accumulator bits to hold a length-K dot product without overflow.

    ``ceil(log2 K) + WEIGHT_WIDTH + ACTIVATION_WIDTH + guard`` -- the MVU core
    asserts on overflow in sim, so the generator must size this itself.
    """
    return int(math.ceil(math.log2(max(1, int(k))))) + int(weight_width) + int(act_width) + int(guard)


def dsp_estimate(core, pe, simd):
    """FINN cost-model DSP count for one MVU tile (doc 04.5): DSP58 packs 3
    MACs/DSP (PE·ceil(SIMD/3)), the 4-bit DSP48 core 4 (ceil(PE/4)·SIMD), the
    8-bit DSP48 core 2 (ceil(PE/2)·SIMD)."""
    if core == "mvu_vvu_8sx9_dsp58":
        return pe * math.ceil(simd / 3)
    if core in ("mvu_4sx4u_dsp48e1", "mvu_4sx4u_dsp48e2"):
        return math.ceil(pe / 4) * simd
    return math.ceil(pe / 2) * simd   # mvu_8sx8u_dsp48


def latency_cycles(core, sf, simd, segmentlen):
    """Deterministic fill latency (first input beat -> first output beat) of one
    MVU tile. RTL is a fixed pipeline, so this is exact, not an estimate --
    calibrated against standalone XSIM runs (temp_space/mvau-lat):

      DSP58: SF + ceil(CHAINLEN/SEGMENTLEN) + 2   (CHAINLEN = ceil(SIMD/3))
      DSP48 (8sx8u / 4sx4u): SF + 5               (5 fixed core stages)

    Verified: DSP58 SF=2/4 -> 5/7; 8sx8u SF=2/4 -> 7/9.
    """
    if core == "mvu_vvu_8sx9_dsp58":
        chainlen = math.ceil(simd / 3)
        max_pipe = math.ceil(chainlen / max(1, segmentlen))
        return sf + max_pipe + 2
    return sf + 5


def output_ii(sf):
    """Deterministic steady-state II (cycles between output beats): the core
    accumulates over SF synapse-fold beats per output at 1 beat/cycle, so one
    output every SF cycles. Verified: II == SF across all measured folds."""
    return sf


def segment_len(simd, clock_ns, dsp_block):
    """DSP58 cascade segment length from the target clock (FINN's model).

    ``critical_path_dsps = floor((clk - 0.741)/0.605 + 1)`` then clamp to the
    chain length ``ceil(SIMD/3)``. Returns 0 (unused) for non-DSP58 cores.
    """
    if dsp_block != "DSP58":
        return 0
    if clock_ns is None or clock_ns <= 0.741:
        # Can't meet the first-DSP delay; fall back to full chain length.
        return int(math.ceil(simd / 3))
    cp = int(math.floor((clock_ns - 0.741) / 0.605 + 1))
    return max(1, min(cp, int(math.ceil(simd / 3))))


def narrow_weights(weights, weight_width, signed=True):
    """FINN ``NARROW_WEIGHTS``: 1 iff the weights never use the most-negative code.

    With weights unknown (None) returns 0. Mandatory 1 for the 4-bit core on
    DSP48E1 -- callers targeting that core with unknown weights must constrain
    the range or target DSP48E2/DSP58 (validated in :func:`fold_plan`).
    """
    if weights is None:
        return 0
    wmin = -(1 << (weight_width - 1)) if signed else 0
    try:
        actual_min = min(int(w) for w in _flatten(weights))
    except (TypeError, ValueError):
        return 0
    return 0 if actual_min == wmin else 1


def _flatten(x):
    if hasattr(x, "flatten"):
        return list(x.flatten())
    out = []
    for e in x:
        if isinstance(e, (list, tuple)) or hasattr(e, "__iter__"):
            out.extend(_flatten(e))
        else:
            out.append(e)
    return out


# ── Folding search (FINN SetFolding, MVAU branch) ─────────────────────────────

def exp_cycles(k, n, pe, simd):
    """Per-input-vector cycle cost = SF*NF = (K/SIMD)*(N/PE)."""
    return (k // simd) * (n // pe)


def fold(k, n, weight_width, target_cycles, wwidth_max=WWIDTH_MAX):
    """Choose ``(PE, SIMD)`` for one MVU instance over ``(K=MW, N=MH)``.

    Mirrors FINN ``SetFolding`` (``set_folding.py:129-152``): reset PE=SIMD=1;
    ramp SIMD over divisors of K until the per-vector cycle target is met or the
    weight-stream width cap is hit; then ramp PE over divisors of N until the
    target is met. Divisor-only, so ``K%SIMD==0`` / ``N%PE==0`` by construction.
    """
    target = max(1, int(target_cycles))
    simd = 1
    for s in divisors(k):
        prev = simd
        simd = s
        if exp_cycles(k, n, 1, simd) < target:
            break
        if weight_width * simd > wwidth_max:
            simd = prev
            break
    pe = 1
    for p in divisors(n):
        pe = p
        if exp_cycles(k, n, pe, simd) < target:
            break
    return pe, simd


def target_from_knobs(m, reuse_factor=1, strategy="latency", target_cycles=None):
    """Map hls4ml knobs to the per-vector cycle target the fold search stops at.

    - ``ReuseFactor`` is the per-vector II == NF*SF, so target = RF.
    - ``TargetCycles`` (whole-frame budget) -> per-vector = ceil(target/M).
    - ``Strategy: Latency`` leaves RF at its default (1) => full unroll.

    Returns ``(target, source)`` where source names which knob drove it.
    """
    if target_cycles is not None:
        return max(1, int(math.ceil(int(target_cycles) / max(1, int(m))))), "target_cycles"
    return max(1, int(reuse_factor or 1)), "reuse_factor"


# ── Byte-aligned stream widths (match mvu_vvu_axi) ────────────────────────────

def _ba(bits):
    """Round *bits* up to a byte multiple (AXIS byte alignment)."""
    return (int(bits) + 7) // 8 * 8


def stream_widths(pe, simd, weight_width, act_width, accu):
    """Byte-aligned (weight, input, output) AXIS beat widths for one instance
    in MVU mode (IS_MVU=1)."""
    return {
        "weight_ba": _ba(pe * simd * weight_width),
        "input_ba": _ba(simd * act_width),
        "output_ba": _ba(pe * accu),
    }


# ── Top-level plan ────────────────────────────────────────────────────────────

def fold_plan(m, k, n, *, weight_precision=None, input_precision=None,
              output_precision=None, part=None, clock_period_ns=5.0,
              reuse_factor=1, strategy="latency", target_cycles=None,
              parallelization_factor=1, n_tiles=1, weights=None):
    """Resolve the full folding/geometry plan for a GEMM ``(m, k, n)``.

    N-tiling splits N into ``n_tiles`` equal column blocks, each an independent
    MVU instance (shared A, own B^T slice); every tile is folded identically.
    Returns a dict with the per-tile MVU parameters and the aggregate view.
    """
    m, k, n = int(m), int(k), int(n)
    weight_width = parse_width(weight_precision, 8)
    act_width = parse_width(input_precision, 8)
    out_width = parse_width(output_precision, 8)
    signed_act = parse_signed(input_precision)
    # Fixed-point fractional bits for the drain: raw product carries
    # 2^(input_frac + weight_frac); the output stores 2^output_frac.
    input_frac = parse_frac(input_precision)
    weight_frac = parse_frac(weight_precision)
    output_frac = parse_frac(output_precision)
    output_int = parse_int_bits(output_precision, default=out_width)

    if n_tiles < 1 or n % n_tiles != 0:
        raise ValueError(f"n_tiles={n_tiles} must be >=1 and divide N={n}")
    n_tile = n // n_tiles

    dsp_block = dsp_block_for_part(part)
    check_envelope(dsp_block, weight_width, act_width, signed_act)
    core = select_core(dsp_block, weight_width, act_width)

    target, target_src = target_from_knobs(
        m, reuse_factor=reuse_factor, strategy=strategy, target_cycles=target_cycles)

    pe, simd = fold(k, n_tile, weight_width, target)

    accu = accu_width(k, weight_width, act_width)
    seg = segment_len(simd, clock_period_ns, dsp_block)
    narrow = narrow_weights(weights, weight_width, signed_act)

    if core == "mvu_4sx4u_dsp48e1" and narrow == 0:
        raise ValueError(
            "mvu_4sx4u on DSP48E1 requires NARROW_WEIGHTS=1 (weights must exclude "
            "the most-negative code); constrain the range or target DSP48E2/DSP58")

    sf, nf = k // simd, n_tile // pe
    widths = stream_widths(pe, simd, weight_width, act_width, accu)

    # per-vector II and the RF it corresponds to (after divisor snapping)
    ii_per_vector = exp_cycles(k, n_tile, pe, simd)
    achieved_rf = ii_per_vector  # RF == NF*SF per vector
    requested_rf = int(reuse_factor or 1)

    tile = {
        "is_mvu": 1,
        "compute_core": core,
        "mw": k, "mh": n_tile,
        "pe": pe, "simd": simd, "sf": sf, "nf": nf,
        "activation_width": act_width,
        "weight_width": weight_width,
        "accu_width": accu,
        "signed_activations": 1 if signed_act else 0,
        "narrow_weights": narrow,
        "segmentlen": seg,
        "weight_stream_width_ba": widths["weight_ba"],
        "input_stream_width_ba": widths["input_ba"],
        "output_stream_width_ba": widths["output_ba"],
        "wmem": (k * n_tile) // (pe * simd),  # weight beats per input vector
        "dsp_estimate": dsp_estimate(core, pe, simd),   # FINN cost model, per tile
        "latency_cycles": latency_cycles(core, sf, simd, seg),  # deterministic fill latency
        "ii": output_ii(sf),                            # deterministic output II (= SF)
    }

    return {
        "m": m, "k": k, "n": n,
        "dsp_block": dsp_block,
        "output_width": out_width,
        "output_int": output_int,
        "output_frac": output_frac,
        "input_frac": input_frac,
        "weight_frac": weight_frac,
        "product_frac": input_frac + weight_frac,   # raw dot-product carries 2^this
        "n_tiles": n_tiles,
        "n_tile": n_tile,
        "num_input_vectors": m,
        "tile": tile,             # tiles are identical; instantiate n_tiles of these
        "target_cycles": target,
        "target_source": target_src,
        "achieved_reuse_factor": achieved_rf,
        "requested_reuse_factor": requested_rf,
        "reuse_factor_snapped": achieved_rf != requested_rf and target_src == "reuse_factor",
        "parallelization_factor": int(parallelization_factor or 1),
    }
