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
import warnings


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


def fold(k, n, weight_width, target_cycles, wwidth_max=WWIDTH_MAX, simd_cap=None):
    """Choose ``(PE, SIMD)`` for one MVU instance over ``(K=MW, N=MH)``.

    Mirrors FINN ``SetFolding`` (``set_folding.py:129-152``): reset PE=SIMD=1;
    ramp SIMD over divisors of K until the per-vector cycle target is met or the
    weight-stream width cap is hit; then ramp PE over divisors of N until the
    target is met. Divisor-only, so ``K%SIMD==0`` / ``N%PE==0`` by construction.

    ``simd_cap`` (optional) additionally bounds SIMD — used to hold SIMD at the
    DSP-packing-optimal value (e.g. a multiple of 3 on DSP58, where each DSP packs 3
    K-lanes; SIMD=4 would spill into a 2-DSP cascade at 33% waste).
    """
    target = max(1, int(target_cycles))
    simd = 1
    for s in divisors(k):
        prev = simd
        simd = s
        if exp_cycles(k, n, 1, simd) < target:
            break
        if weight_width * simd > wwidth_max or (simd_cap and simd > simd_cap):
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

    Knob contract (mvau): **ReuseFactor only**. Strategy answers "*how* is the MAC
    computed?" (which soft kernel) — but the MVAU is one hardened FINN DSP MVU, there is
    no kernel to select, so **Strategy is read-but-inert** for mvau (as is
    ParallelizationFactor). RF answers "*how many* / what II?", which is exactly the fold
    the MVAU exposes. So:
    - ``target = RF`` (per-vector II = NF*SF*... ). RF=1 (the default) => maximum
      parallelism / minimum II; reaching II=1 on K when SIMD is capped needs K-tiling
      (see fold_plan's ``k_tiles_full_spatial``).
    - ``TargetCycles`` (whole-frame budget) -> per-vector = ceil(target/M), overrides RF.
    ``strategy`` is accepted for signature compatibility but ignored.

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
              parallelization_factor=1, n_tiles=1, k_tiles=1, weights=None,
              pe=None, simd=None):
    """Resolve the full folding/geometry plan for a GEMM ``(m, k, n)``.

    N-tiling splits N into ``n_tiles`` equal column blocks, each an independent
    MVU instance (shared A, own B^T slice; outputs concatenated).

    K-tiling splits K into ``k_tiles`` equal row blocks (each MVU reduces its K-slice
    for ALL outputs; the partial results are SUMMED). ``k_tiles="auto"`` chooses the
    smallest divisor of SF that meets the RF/II target (RF=1 -> full spatial, SF_tile=1).
    ``k_tiles=1`` (default) is the single-K-core path. The summed accumulator is widened
    by ceil(log2 k_tiles).

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

    # K-padding so SIMD can reach the DSP-packing-optimal value. SIMD is bounded by
    # weight_width*SIMD <= WWIDTH_MAX (routability), but the *packing* granularity is the
    # DSP's K-lanes-per-DSP: DSP58 packs 3 (mvu_vvu_8sx9_dsp58), so SIMD should be a
    # multiple of 3 (SIMD=3 -> 1 DSP, no waste; SIMD=4 -> 2 DSPs at 33% waste). DSP48
    # cores pack along PE, not SIMD, so no SIMD granularity preference there.
    # simd_target = largest DSP-optimal SIMD within the cap; pad K to a multiple of it
    # (zero rows -> bit-exact) so the divisor-only fold can actually use it. A tiny K
    # (< simd_target) keeps SIMD=K, no pad.
    simd_cap = max(1, WWIDTH_MAX // weight_width)
    pack = 3 if core == "mvu_vvu_8sx9_dsp58" else 1        # DSP58 K-lanes per DSP
    simd_target = (simd_cap // pack) * pack if simd_cap >= pack else simd_cap
    simd_target = max(1, simd_target)
    if pe is not None and simd is not None:
        # ── user-directed fold: use (pe, simd) verbatim, no search ──
        pe, simd = int(pe), int(simd)
        if weight_width * simd > WWIDTH_MAX:
            raise ValueError(f"weight_width*simd = {weight_width * simd} exceeds the weight-stream "
                             f"cap WWIDTH_MAX={WWIDTH_MAX}; reduce simd")
        if pack and simd % pack:
            warnings.warn(f"mvau: simd={simd} is not a multiple of the {core} DSP-packing factor "
                          f"{pack}; the DSP cascade is under-utilized.", stacklevel=2)
        # K-pad to a multiple of simd*k_tiles so SF_full/k_tiles is integral.
        kt_int = 1 if k_tiles in (None, "auto") else max(1, int(k_tiles))
        unit = simd * kt_int
        k_pad = math.ceil(k / unit) * unit if k > 0 else unit
        if n_tile % pe:
            raise ValueError(f"pe={pe} must divide n_tile (= N/n_tiles = {n_tile})")
        target, target_src = (k_pad // simd) * (n_tile // pe), "manual"
    else:
        k_pad = k if k < simd_target else math.ceil(k / simd_target) * simd_target
        target, target_src = target_from_knobs(
            m, reuse_factor=reuse_factor, strategy=strategy, target_cycles=target_cycles)
        pe, simd = fold(k_pad, n_tile, weight_width, target, simd_cap=simd_target)

    # DSP48 cores pack MACs along PE (8sx8u: 2/DSP, 4sx4u: 4/DSP). When PE is not a
    # multiple of that factor the DSP estimate's ceil() rounds up -> wasted DSP. This
    # bites N-tiling with a small PE_tile and single cores on prime-ish N. DSP58 packs
    # along SIMD, so PE alignment is irrelevant there. Warn only -- no guard/snap.
    _pack_pe = {"mvu_8sx8u_dsp48": 2,
                "mvu_4sx4u_dsp48e1": 4, "mvu_4sx4u_dsp48e2": 4}.get(core)
    if _pack_pe and pe % _pack_pe:
        warnings.warn(
            f"mvau: PE={pe} is not a multiple of the {core} DSP-packing factor "
            f"{_pack_pe}; DSP packing is under-utilized (rounds up as if "
            f"PE={math.ceil(pe / _pack_pe) * _pack_pe}). Consider an aligned "
            f"reuse_factor or n_tiles for better DSP efficiency.",
            stacklevel=2)

    accu = accu_width(k_pad, weight_width, act_width)
    seg = segment_len(simd, clock_period_ns, dsp_block)
    narrow = narrow_weights(weights, weight_width, signed_act)

    if core == "mvu_4sx4u_dsp48e1" and narrow == 0:
        raise ValueError(
            "mvu_4sx4u on DSP48E1 requires NARROW_WEIGHTS=1 (weights must exclude "
            "the most-negative code); constrain the range or target DSP48E2/DSP58")

    sf_full, nf = k_pad // simd, n_tile // pe

    # K-tiling: split the SF_full K-folds across k_tiles MVU cores, each reducing a
    # K_pad/k_tiles slice for ALL outputs; the partials are summed downstream. Each tile
    # is folded identically (MW = K_pad/k_tiles), so SF drops to SF_full/k_tiles.
    if k_tiles == "auto":
        want = max(1, int(math.ceil((sf_full * nf) / max(1, target))))
        gk = next((d for d in divisors(sf_full) if d >= want), sf_full)
    else:
        gk = max(1, int(k_tiles))
        if sf_full % gk != 0:
            raise ValueError(f"k_tiles={gk} must divide SF={sf_full} (K_pad/SIMD)")
    k_per_tile = k_pad // gk
    sf = sf_full // gk                     # per-tile SF
    accu_sum = accu + (math.ceil(math.log2(gk)) if gk > 1 else 0)  # summed-partials width

    widths = stream_widths(pe, simd, weight_width, act_width, accu)

    # per-vector II and the RF it corresponds to (after divisor snapping + K-tiling)
    ii_per_vector = sf * nf
    achieved_rf = ii_per_vector  # RF == NF*SF per vector
    requested_rf = int(reuse_factor or 1)

    tile = {
        "is_mvu": 1,
        "compute_core": core,
        "mw": k_per_tile, "mh": n_tile,
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
        "wmem": (k_per_tile * n_tile) // (pe * simd),  # weight beats per input vector
        "dsp_estimate": dsp_estimate(core, pe, simd),   # FINN cost model, per tile
        "latency_cycles": latency_cycles(core, sf, simd, seg),  # deterministic fill latency
        "ii": output_ii(sf),                            # deterministic output II (= SF)
    }

    return {
        "m": m, "k": k, "k_pad": k_pad, "n": n,
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
        # K-tiling: k_tiles MVU cores each reduce K_pad/k_tiles for all outputs; partials
        # summed (accumulator widened to accu_sum). k_tiles=1 => single K-core (default).
        "simd_pack": pack,                 # DSP K-lanes/DSP (3 on DSP58) -> SIMD granularity
        "k_tiles": gk,                     # number of K-tiles (MVU cores summed along K)
        "k_per_tile": k_per_tile,          # each tile's MW (K slice), a multiple of SIMD
        "sf_full": sf_full,                # pre-tiling SF; k_tiles=SF_full => SF_tile=1
        "accu_sum": accu_sum,              # width of the summed-partials accumulator
    }


#: config keys fold_plan accepts (resolve_plan filters the config down to these).
_FOLD_PLAN_KEYS = (
    "weight_precision", "input_precision", "output_precision", "part",
    "clock_period_ns", "reuse_factor", "strategy", "target_cycles",
    "parallelization_factor", "n_tiles", "k_tiles", "weights", "pe", "simd",
)


def resolve_plan(shape, **cfg):
    """Resolve the MVU plan for *shape* ``(m, k, n)``.

    On this branch there is no shared mapper: the fold is user-directed (the tiling
    knobs ``n_tiles``/``k_tiles`` and ReuseFactor come straight from the config) and
    :func:`fold_plan` derives everything. This is a thin compatibility entry so the
    emitters (``rtl``/``golden``/``package``) can call one plan resolver; it filters
    the (possibly package-level) config down to the folding params ``fold_plan`` takes.
    """
    m, k, n = int(shape[0]), int(shape[1]), int(shape[2])
    kw = {key: cfg[key] for key in _FOLD_PLAN_KEYS if cfg.get(key) is not None}
    return fold_plan(m, k, n, **kw)
