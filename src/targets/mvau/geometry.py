"""mvau target geometry: ReuseFactor -> (PE, SIMD) resolution, and the geometry
derived from it (accumulator width, stream widths, DSP/latency estimates).

Self-contained (no ``gemm_ip`` imports) so ``rtl.py`` / ``golden.py`` / ``package.py``
can import it with only the target directory on ``sys.path`` -- mirrors the
convention of ``targets/tensor_slice/geometry.py``.

The compute is FINN's RTL MVU (``mvu_vvu_axi`` + ``mvu_vvu_8sx9_dsp58``, the only
core this target supports -- see ``finn_space/MVU_space/03_compute_cores.md`` for
the core's internal structure). A GEMM ``C[M,N] = A[M,K] . B[K,N]`` maps to one MVU
as ``W := B^T`` (MH=N, MW=K), rows of A streamed as activations (numInputVectors =
M), rows of C collected.

ReuseFactor is hls4ml's *MACs per multiplier per input vector*:
``RF = K*N / (PE*SIMD)``. It is a resource knob, not a cycle knob -- see
:func:`resolve_fold` for how RF picks (PE, SIMD).

Fold-N (the default) and fold-K cost the same DSPs on this core: fold-N spends
``PE*ceil(K/3)`` DSPs (``PE = N/RF``), fold-K spends ``N*ceil(K/(3*RF))``, both
``~= K*N/(3*RF)`` when divisible. Fold-N wins on two counts: the whole K reduction
runs inside one DSP58 PCOUT cascade (no fabric adders, no SF-beat accumulation),
and the ``ceil`` waste of a partially filled cascade is paid once per PE, of which
fold-N has RF times fewer than fold-K.
"""

import math
import re


# ── Constants ────────────────────────────────────────────────────────────────

#: Accumulator guard bits added on top of ceil(log2 K) + w + a.
ACCU_GUARD = 1

#: The only compute core this target supports.
CORE = "mvu_vvu_8sx9_dsp58"
DSP_BLOCK = "DSP58"

#: Vitis HLS C-simulation's default max width for a single ``ap_int``/``ap_uint``.
#: This is a property of the hls4ml-side translation unit that includes the
#: generated ``_gemm_ip.h`` header, not of gemm-ip-gen's own sources -- that TU
#: cannot have ``AP_INT_MAX_W`` raised (see ``package.py``'s ``_with_ap_int_max_w``,
#: which only patches gemm-ip-gen's own C++/header TUs). Any beat this target
#: declares as one ``ap_uint`` in that header must fit under this cap.
AP_UINT_MAX_BITS = 1024


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


# ── Part gate ──────────────────────────────────────────────────────────────────

def require_versal(part):
    """Raise unless *part* is a Versal device -- the only family this target
    supports (single compute core ``mvu_vvu_8sx9_dsp58``, DSP58 only)."""
    if not part or not re.match(r"^xcv[cpmeh]", str(part), re.IGNORECASE):
        raise ValueError(
            f"mvau requires a Versal part (xcv[cpmeh]...), got {part!r}; "
            "pass a Versal part via --part")


# ── Envelope validation ───────────────────────────────────────────────────────

def check_envelope(weight_width, act_width, signed_act):
    """Reject operand widths outside the native MVU envelope (integer only).

    Raises ``ValueError`` for >8b weights, or >8b activations (>9b even when
    signed). These cores have no path for wider or floating operands;
    bit-plane decomposition is a documented, out-of-scope extension.
    """
    if weight_width > 8:
        raise ValueError(
            f"weight width {weight_width}b exceeds the MVU envelope (<=8b); "
            "reject or bit-plane decompose (out of scope)")
    if act_width > 8:
        ok = signed_act and act_width == 9
        if not ok:
            raise ValueError(
                f"activation width {act_width}b exceeds the MVU envelope "
                "(<=8b, or <=9b signed)")


# ── Accumulator / segment / narrow ────────────────────────────────────────────

def accu_width(k, weight_width, act_width, guard=ACCU_GUARD):
    """Accumulator bits to hold a length-K dot product without overflow.

    ``ceil(log2 K) + WEIGHT_WIDTH + ACTIVATION_WIDTH + guard`` -- the MVU core
    asserts on overflow in sim, so the generator must size this itself.
    """
    return int(math.ceil(math.log2(max(1, int(k))))) + int(weight_width) + int(act_width) + int(guard)


def dsp_estimate(pe, simd):
    """DSP58 packs 3 K-lanes per DSP (``mvu_vvu_8sx9_dsp58``): ``PE*ceil(SIMD/3)``."""
    return pe * math.ceil(simd / 3)


def latency_cycles(sf, simd, segmentlen):
    """Deterministic fill latency (first input beat -> first output beat) of one
    MVU tile. RTL is a fixed pipeline, so this is exact, not an estimate --
    calibrated against standalone XSIM runs (temp_space/mvau-lat):

      SF + ceil(CHAINLEN/SEGMENTLEN) + 2   (CHAINLEN = ceil(SIMD/3))

    Verified: SF=2/4 -> 5/7.
    """
    chainlen = math.ceil(simd / 3)
    max_pipe = math.ceil(chainlen / max(1, segmentlen))
    return sf + max_pipe + 2


def output_ii(sf):
    """Deterministic steady-state II (cycles between output beats): the core
    accumulates over SF synapse-fold beats per output at 1 beat/cycle, so one
    output every SF cycles. Verified: II == SF across all measured folds."""
    return sf


def segment_len(simd, clock_ns):
    """DSP58 cascade segment length from the target clock (FINN's model).

    ``critical_path_dsps = floor((clk - 0.741)/0.605 + 1)`` then clamp to the
    chain length ``ceil(SIMD/3)``.
    """
    if clock_ns is None or clock_ns <= 0.741:
        # Can't meet the first-DSP delay; fall back to full chain length.
        return int(math.ceil(simd / 3))
    cp = int(math.floor((clock_ns - 0.741) / 0.605 + 1))
    return max(1, min(cp, int(math.ceil(simd / 3))))


def narrow_weights(weights, weight_width, signed=True):
    """FINN ``NARROW_WEIGHTS``: 1 iff the weights never use the most-negative code.

    With weights unknown (None) returns 0.
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


# ── ReuseFactor -> (PE, SIMD) ──────────────────────────────────────────────────

def _warn(req, legal, hi_desc, name=None):
    where = f' in layer "{name}"' if name else ""
    return (f"WARNING: Invalid ReuseFactor={req}{where}. Using ReuseFactor={legal} "
            f"instead. Valid ReuseFactor(s): {hi_desc}.")


def resolve_fold(k, n, reuse_factor, fold_axis="n", name=None):
    """Resolve ``ReuseFactor`` to ``(PE, SIMD)`` for a GEMM ``(K, N)``.

    ReuseFactor is *how many times each MAC is used per input vector*:
    ``RF = K*N / (PE*SIMD)``. The fold axis decides which dimension pads to
    honor the request exactly (rather than snapping RF itself):

    - ``"n"`` (default): SIMD=K (the whole K reduction stays inside one DSP58
      cascade per PE), N pads up to a multiple of RF, PE = n_pad/RF. RF > N
      cannot be honored (PE would be < 1); legalizes to RF=N (PE=1).
    - ``"k"``: PE=N, K pads up to a multiple of RF, SIMD = k_pad/RF. SIMD is
      never folded below 3 (the DSP58 packs 3 K-lanes/DSP) unless K itself is
      below 3, in which case SIMD=K and K never folds. RF beyond ``K/3``
      legalizes to the SIMD=3 floor.
    - ``"kn"``: both rules applied independently, each with its own padding
      and bound/warning. The multiplier count is then ``K*N/RF**2`` -- each
      MAC is reused ``RF**2`` times per input vector, not RF; callers should
      report the effective reuse (``k_pad*n_pad/(pe*simd)``) alongside the
      requested knob so the two are never confused.

    Returns a dict: ``pe``, ``simd``, ``k_pad``, ``n_pad``, ``reuse_factor``
    (legalized -- for "kn" this is the common per-axis RF, not the effective
    reuse), ``warnings`` (list of str, empty if the request needed no
    legalization).
    """
    k, n = int(k), int(n)
    rf = max(1, int(reuse_factor or 1))
    warns = []

    fold_n = fold_axis in ("n", "kn")
    fold_k = fold_axis in ("k", "kn")

    if fold_axis not in ("n", "k", "kn"):
        raise ValueError(f"fold_axis={fold_axis!r} must be one of 'n', 'k', 'kn'")

    # N side (fold-N and fold-KN): SIMD=K, PE=n_pad/RF.
    if fold_n:
        rf_n = rf
        if rf_n > n:
            warns.append(_warn(rf_n, n, f"1..{n}", name))
            rf_n = n
        n_pad = math.ceil(n / rf_n) * rf_n
        pe = n_pad // rf_n
        simd_n = k
    else:
        rf_n = rf
        n_pad = n
        pe = n
        simd_n = None

    # K side (fold-K and fold-KN): PE=N, SIMD=k_pad/RF, SIMD floor 3.
    if fold_k:
        rf_k = rf
        if k < 3:
            # K never folds below 3; SIMD=K is the only legal point.
            if rf_k != 1:
                warns.append(_warn(rf_k, 1, "1", name))
            rf_k = 1
            k_pad = k
            simd_k = k
        else:
            simd_floor_rf = math.ceil(k / 3)
            if rf_k > simd_floor_rf:
                warns.append(_warn(rf_k, simd_floor_rf, f"1..{simd_floor_rf}", name))
                rf_k = simd_floor_rf
            k_pad = math.ceil(k / rf_k) * rf_k
            simd_k = k_pad // rf_k
        if not fold_n:
            pe = n
    else:
        rf_k = rf
        k_pad = k
        simd_k = k

    if fold_axis == "n":
        simd = simd_n
        legal_rf = rf_n
    elif fold_axis == "k":
        simd = simd_k
        legal_rf = rf_k
        n_pad = n
    else:  # kn
        simd = simd_k
        pe = pe  # already n_pad // rf_n from the N-side branch
        legal_rf = rf_n if rf_n == rf_k else rf_n  # report the N-side RF; see effective reuse

    return {
        "pe": pe,
        "simd": simd,
        "k_pad": k_pad,
        "n_pad": n_pad,
        "reuse_factor": legal_rf,
        "warnings": warns,
    }


# ── Byte-aligned stream widths (match mvu_vvu_axi) ────────────────────────────

def _ba(bits):
    """Round *bits* up to a byte multiple (AXIS byte alignment)."""
    return (int(bits) + 7) // 8 * 8


def stream_widths(pe, simd, weight_width, act_width, accu, out_width=None):
    """Byte-aligned (weight, input, output) AXIS beat widths for one instance
    in MVU mode (IS_MVU=1). ``output_ba`` is the shim's post-requant output
    beat (``PE*out_width``); when *out_width* is omitted it falls back to the
    pre-requant raw accumulator beat (``PE*accu``) for callers that only need
    the internal core width."""
    ow = accu if out_width is None else out_width
    return {
        "weight_ba": _ba(pe * simd * weight_width),
        "input_ba": _ba(simd * act_width),
        "output_ba": _ba(pe * ow),
    }


def rv_width(accu, has_bias):
    """Just-wide-enough intermediate width for the requant stage's biased
    accumulator value (shared by the C twin and the RTL requant stage so the
    two never drift). With a bias add, headroom is kept wide (>= 32b) plus a
    couple guard bits; without one, one guard bit above the accumulator
    suffices."""
    return (max(int(accu), 32) + 2) if has_bias else (int(accu) + 1)


def check_beat_limits(pe, simd, weight_width, act_width, output_ba, n_tiles=1,
                       k_tiles=1, weights_in_core=True, name=None):
    """Reject a fold whose hls4ml-facing stream beats would need an ``ap_uint``
    wider than :data:`AP_UINT_MAX_BITS`.

    Checks every beat ``_gemm_ip.h`` (see ``package.py``) declares as a single
    ``ap_uint``: the weight beat (``PE*SIMD*weight_width`` -- only when
    ``weights_in_core`` is False, i.e. the weight matrix is a C++ stream
    rather than baked into the RTL memstream), the input beat
    (``SIMD*act_width``), the per-tile output beat (post-requant,
    ``PE*out_width``, byte-aligned -- *output_ba* must already be computed
    with ``out_width``, not the raw accumulator width), and the concatenated
    output beat that N-tiling and K-tiling glue side by side
    (``n_tiles*k_tiles*`` the per-tile output beat). This is independent of
    gemm-ip-gen's own ``_with_ap_int_max_w`` machinery, which only raises
    ``AP_INT_MAX_W`` for gemm-ip-gen's own C++/header translation units -- it
    cannot help the hls4ml-side TU that includes the generated header.
    """
    where = f' in layer "{name}"' if name else ""
    beats = []
    if not weights_in_core:
        beats.append(("weight beat (PE*SIMD*weight_width)",
                       pe * simd * weight_width))
    beats += [
        ("input beat (SIMD*act_width)", simd * act_width),
        ("output beat (PE*out_width)", output_ba),
        ("concatenated output beat (n_tiles*k_tiles*PE*out_width)",
         n_tiles * k_tiles * output_ba),
    ]
    for label, width in beats:
        if width > AP_UINT_MAX_BITS:
            raise ValueError(
                f"{label} is {width} bits{where}, exceeding the "
                f"{AP_UINT_MAX_BITS}-bit ap_uint limit of Vitis HLS C "
                "simulation. Reduce the parallelism on that beat: a larger "
                "ReuseFactor or FoldAxis kn shrinks PE and SIMD, KTiles splits "
                "the input beat across cores, NTiles splits the output beat")


# ── Top-level plan ────────────────────────────────────────────────────────────

def fold_plan(m, k, n, *, weight_precision=None, input_precision=None,
              output_precision=None, part=None, clock_period_ns=5.0,
              reuse_factor=1, fold_axis="n", n_tiles=1, k_tiles=1, weights=None,
              pe=None, simd=None, weights_in_core=True, name=None, **_ignored):
    """Resolve the full folding/geometry plan for a GEMM ``(m, k, n)``.

    N-tiling splits N into ``n_tiles`` equal column blocks, each an independent
    MVU instance (shared A, own B^T slice; outputs concatenated).

    K-tiling splits K into ``k_tiles`` equal row blocks (each MVU reduces its K-slice
    for ALL outputs; the partial results are SUMMED). ``k_tiles="auto"`` is treated as
    1 (K-tiling is a physical/BRAM-shape choice, not something ``resolve_fold`` derives).
    ``k_tiles=1`` (default) is the single-K-core path. The summed accumulator is widened
    by ceil(log2 k_tiles).

    ``**_ignored`` swallows legacy config keys (``strategy``, ``target_cycles``,
    ``parallelization_factor``, ``simd_cap``) so older callers/manifests don't break;
    they are not consulted.

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

    require_versal(part)
    check_envelope(weight_width, act_width, signed_act)
    core = CORE
    dsp_block = DSP_BLOCK

    requested_rf = int(reuse_factor or 1)
    fold_warnings = []

    if pe is not None and simd is not None:
        # ── user-directed fold: use (pe, simd) verbatim, no resolution ──
        pe, simd = int(pe), int(simd)
        # K-pad to a multiple of simd*k_tiles so SF_full/k_tiles is integral.
        kt_int = 1 if k_tiles in (None, "auto") else max(1, int(k_tiles))
        unit = simd * kt_int
        k_pad = math.ceil(k / unit) * unit if k > 0 else unit
        if n_tile % pe:
            raise ValueError(f"pe={pe} must divide n_tile (= N/n_tiles = {n_tile})")
        n_pad = n_tile
        legal_rf = requested_rf
    else:
        resolved = resolve_fold(k, n_tile, requested_rf, fold_axis=fold_axis, name=name)
        pe, simd = resolved["pe"], resolved["simd"]
        k_pad, n_pad = resolved["k_pad"], resolved["n_pad"]
        legal_rf = resolved["reuse_factor"]
        fold_warnings = resolved["warnings"]

    accu = accu_width(k_pad, weight_width, act_width)
    seg = segment_len(simd, clock_period_ns)
    narrow = narrow_weights(weights, weight_width, signed_act)

    sf_full, nf = k_pad // simd, n_pad // pe

    # K-tiling: split the SF_full K-folds across k_tiles MVU cores, each reducing a
    # K_pad/k_tiles slice for ALL outputs; the partials are summed downstream. Each tile
    # is folded identically (MW = K_pad/k_tiles), so SF drops to SF_full/k_tiles.
    gk = 1 if k_tiles in (None, "auto") else max(1, int(k_tiles))
    if sf_full % gk != 0:
        raise ValueError(f"k_tiles={gk} must divide SF={sf_full} (K_pad/SIMD)")
    k_per_tile = k_pad // gk
    sf = sf_full // gk                     # per-tile SF
    accu_sum = accu + (math.ceil(math.log2(gk)) if gk > 1 else 0)  # summed-partials width

    # Post-requant per-tile output beat (PE*out_width): the beat the shim now emits
    # after the requantize stage, not the raw pre-drain accumulator beat.
    widths = stream_widths(pe, simd, weight_width, act_width, accu, out_width=out_width)
    check_beat_limits(pe, simd, weight_width, act_width, widths["output_ba"],
                       n_tiles=n_tiles, k_tiles=gk,
                       weights_in_core=weights_in_core, name=name)

    effective_reuse = (k_pad * n_pad) // (pe * simd)

    tile = {
        "is_mvu": 1,
        "compute_core": core,
        "mw": k_per_tile, "mh": n_pad,
        "pe": pe, "simd": simd, "sf": sf, "nf": nf,
        "activation_width": act_width,
        "weight_width": weight_width,
        "accu_width": accu,
        "signed_activations": 1 if signed_act else 0,
        "narrow_weights": narrow,
        "segmentlen": seg,
        # requant scale/width, duplicated onto the tile dict so rtl.py (which is
        # only ever handed the tile, not the full plan) can size its requant
        # stage without a second geometry call.
        "output_width": out_width,
        "output_int": output_int,
        "output_frac": output_frac,
        "product_frac": input_frac + weight_frac,
        "accu_sum": accu_sum,
        "weight_stream_width_ba": widths["weight_ba"],
        "input_stream_width_ba": widths["input_ba"],
        "output_stream_width_ba": widths["output_ba"],
        "wmem": (k_per_tile * n_pad) // (pe * simd),  # weight beats per input vector
        "dsp_estimate": dsp_estimate(pe, simd),         # FINN cost model, per tile
        "latency_cycles": latency_cycles(sf, simd, seg),  # deterministic fill latency
        "ii": output_ii(sf),                            # deterministic output II (= SF)
    }

    return {
        "m": m, "k": k, "k_pad": k_pad, "n": n, "n_pad": n_pad,
        "dsp_block": dsp_block,
        "output_width": out_width,
        "output_int": output_int,
        "output_frac": output_frac,
        "input_frac": input_frac,
        "weight_frac": weight_frac,
        "product_frac": input_frac + weight_frac,   # raw dot-product carries 2^this
        "n_tiles": n_tiles,
        "n_tile": n_pad,
        "num_input_vectors": m,
        "tile": tile,             # tiles are identical; instantiate n_tiles of these
        "fold_axis": fold_axis,
        "requested_reuse_factor": requested_rf,
        "reuse_factor": legal_rf,
        "effective_reuse": effective_reuse,
        "fold_warnings": fold_warnings,
        # K-tiling: k_tiles MVU cores each reduce K_pad/k_tiles for all outputs; partials
        # summed (accumulator widened to accu_sum). k_tiles=1 => single K-core (default).
        "k_tiles": gk,                     # number of K-tiles (MVU cores summed along K)
        "k_per_tile": k_per_tile,          # each tile's MW (K slice), a multiple of SIMD
        "sf_full": sf_full,                # pre-tiling SF; k_tiles=SF_full => SF_tile=1
        "accu_sum": accu_sum,              # width of the summed-partials accumulator
    }


#: config keys fold_plan accepts (resolve_plan filters the config down to these).
_FOLD_PLAN_KEYS = (
    "weight_precision", "input_precision", "output_precision", "part",
    "clock_period_ns", "reuse_factor", "fold_axis", "n_tiles", "k_tiles",
    "weights", "pe", "simd", "weights_in_core", "name",
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
