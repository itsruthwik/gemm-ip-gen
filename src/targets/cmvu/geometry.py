"""cmvu geometry: fold legalization, resident-slot map, widths, latency, tails.

Self-contained by design (no ``gemm_ip`` imports): the RTL, golden, and package
generators import this module with only ``src/targets/cmvu`` on ``sys.path``.
Normative sources: ``docs/architecture.md`` (sections 1, 4, 5, 8, 11) and
``docs/mode_1_user_guide.md``.

The CMVU K x N problem is folded onto a grid of ``cmvu_mode1`` blocks:

* ``k_spatial`` cascade columns each own one 4-row K tile per pass; the
  remaining K tiles are visited over ``k_passes`` temporal passes.
* ``n_spatial`` broadcast rows each own one 8-lane N tile per group; the
  remaining N tiles are visited over ``n_passes`` output groups.
* A block therefore needs ``k_passes * n_passes`` resident 4x8 weight tiles
  and ``cmvu_mode1`` has only ``MEM_TILES = 8`` slots (load-all-upfront
  residency; mid-stream preload is out of v1 scope).
"""

from pathlib import Path

# ── Physical constants (V1 hardened block; do not re-parameterize) ────────────

K_PHYS = 4          # grid rows: contraction lanes per pass (architecture.md §1)
N_PHYS = 8          # grid cols: output lanes (§1)
L = 6               # all-present pipeline latency (§5)
MEM_TILES = 8       # resident 4x8 weight tiles per block (§4)
IN_WIDTH = 8        # int8 activation codes (§2)
COEF_WIDTH = 8      # int8 weight codes (§2)
ACC_WIDTH = 32      # internal/cascade width (§2)
BIAS_WIDTH = 32     # per-lane signed bias code width, accumulator scale (§2, §8)
RESULT_WIDTH = 16   # physical y_out lane width (int16, MVU@678a60a); the runtime
                    # effective width W (<= RESULT_WIDTH) is the layer's logical
                    # output width, selected via the block's out_w port (= W-1).
SHIFT_WIDTH = 5
MAX_SHIFT = (1 << SHIFT_WIDTH) - 1  # runtime shift_amt range 0..31 (§2)

#: Vendored CMVU block RTL instantiated by the generated wrapper (one copy per
#: package root). ``cmvu_array.sv`` is reference/oracle only, never shipped.
VENDORED_SV = ("cmvu_mode1.sv", "cmvu_w_mem.sv", "cmvu_regbank.sv")

#: VTR-facing hard-block model for cmvu_mode1 (package.gen_cmvu_mode1_vtr_model):
#: a generated ``(* blackbox *)`` port-list-only stub, no body -- VTR's front
#: end (parmys/yosys) maps instances of this module name to the arch's
#: cmvu_mode1 pb_type; simulators never see it (they use the real vendored
#: ``cmvu_mode1.sv``). Named distinctly from ``VENDORED_SV`` so both can sit
#: in the same package root without colliding.
CMVU_MODE1_VTR_MODEL = "cmvu_mode1_vtr_blackbox.sv"

#: Directory holding the vendored CMVU block RTL, vendored next to this module
#: (see rtl_static/MVU_COMMIT.txt for provenance).
CMVU_RTL_DIR = Path(__file__).resolve().parent / "rtl_static"


def vendored_rtl_dir():
    """Return the directory holding the vendored CMVU block RTL.

    Returns ``CMVU_RTL_DIR`` if all of ``VENDORED_SV`` are present there,
    else ``None``; callers that need it raise with a clear message.
    """
    if all((CMVU_RTL_DIR / f).is_file() for f in VENDORED_SV):
        return CMVU_RTL_DIR
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ceil_div(a, b):
    return (a + b - 1) // b


def _check_pos_int(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} must be a positive int, got {value!r}")


# ── Per-axis fold legalization ────────────────────────────────────────────────


def _resolve_axis_fold(dim, fold, axis, granule, name=None):
    """Legalize one axis's requested temporal pass count.

    ``chunks = ceil(dim/granule)`` tiles are covered by ``spatial =
    ceil(chunks/fold)`` parallel tiles, visited in ``passes =
    ceil(chunks/spatial)`` temporal passes (the tensor_slice tile-group-fold
    law, with the cmvu granules: 4 on K, 8 on N). ``fold = 1`` means fully
    spatial (one pass); the legal range is ``1..chunks``. The knob is required
    (``None`` is an error) and out-of-range values are errors, not silent
    clamp -- but a non-divisible request may still legalize down (e.g.
    chunks=5, fold=4 -> spatial=2, passes=3) with a warning.
    """
    who = f" for layer {name}" if name is not None else ""
    if fold is None:
        raise ValueError(
            f"{axis}Fold{who} is required: cmvu supports the KFold+NFold knobs "
            f"only (both must be set; no legacy ReuseFactor/FoldAxis).")
    if isinstance(fold, bool) or not isinstance(fold, int):
        raise ValueError(f"{axis}Fold{who} must be an int, got {fold!r}")
    chunks = _ceil_div(dim, granule)
    if fold < 1 or fold > chunks:
        raise ValueError(
            f"{axis}Fold{who} is invalid: legal range 1..{chunks} for "
            f"{axis}={dim} (granule {granule}), got {fold}.")
    spatial = _ceil_div(chunks, fold)
    passes = _ceil_div(chunks, spatial)
    warnings = []
    if passes != fold:
        warnings.append(
            f"WARNING: {axis}Fold={fold}{who} legalized to {passes} temporal "
            f"pass{'es' if passes != 1 else ''} (spatial={spatial} of {chunks} "
            f"{axis} tiles).")
    return {
        "dim": int(dim),
        "granule": int(granule),
        "chunks": chunks,
        "chunks_pad": passes * spatial,
        "spatial": spatial,
        "passes": passes,
        "fold_requested": fold,
        "fold_effective": passes,
        "warnings": warnings,
    }


def resolve_geometry(m, k, n, kfold, nfold, name=None, result_width=None):
    """Resolve the full cmvu tiling plan for shape ``(m, k, n)``.

    ``m`` is the stream length (never folded; any number of streams), ``k``/``n``
    the logical contraction/output sizes. ``kfold``/``nfold`` are the temporal
    pass counts (required, legal ``1..ceil(K/4)`` / ``1..ceil(N/8)``).

    Returns a dict with the derived per-axis splits, the block grid, the
    per-block resident-tile budget, the spec's N-outer/K-inner slot order,
    stream widths, tail-lane counts, and the wrapper first-output latency
    ``L + (n_spatial-1) + (k_spatial-1)`` (architecture.md §7).
    Raises ValueError for missing/invalid knobs and for per-block
    oversubscription (> ``MEM_TILES``).
    """
    _check_pos_int(m, "m")
    _check_pos_int(k, "k")
    _check_pos_int(n, "n")

    fk = _resolve_axis_fold(k, kfold, "K", K_PHYS, name)
    fn = _resolve_axis_fold(n, nfold, "N", N_PHYS, name)

    k_passes = fk["passes"]
    n_passes = fn["passes"]
    slots = k_passes * n_passes
    if slots > MEM_TILES:
        who = f"layer '{name}': " if name is not None else ""
        raise ValueError(
            f"{who}KFold={kfold}/NFold={nfold} need {slots} resident weight "
            f"tiles per block ({k_passes} K passes x {n_passes} N passes) but "
            f"cmvu_mode1 has only {MEM_TILES}; lower KFold/NFold (more spatial "
            f"blocks) or reduce K/N.")

    blocks = fk["spatial"] * fn["spatial"]
    latency = L + (fn["spatial"] - 1) + (fk["spatial"] - 1)

    # Effective (logical) result width W: the block emits RESULT_WIDTH-bit lanes
    # with the W-bit value sign-extended; the wrapper slices back to W. Default
    # is the full physical lane (W = RESULT_WIDTH).
    rw = RESULT_WIDTH if result_width is None else int(result_width)
    if not (1 <= rw <= RESULT_WIDTH):
        raise ValueError(
            f"{'layer ' + str(name) + ': ' if name else ''}result_width must be "
            f"1..{RESULT_WIDTH}, got {rw}")

    return {
        "m": int(m),
        "k": int(k),
        "n": int(n),

        "kfold_requested": fk["fold_requested"],
        "kfold_effective": fk["fold_effective"],
        "k_spatial": fk["spatial"],
        "k_passes": k_passes,
        "k_chunks": fk["chunks"],
        "k_chunks_pad": fk["chunks_pad"],

        "nfold_requested": fn["fold_requested"],
        "nfold_effective": fn["fold_effective"],
        "n_spatial": fn["spatial"],
        "n_passes": n_passes,
        "n_chunks": fn["chunks"],
        "n_chunks_pad": fn["chunks_pad"],

        # grid: rows = broadcast (N), cols = cascade (K)
        "block_rows": fn["spatial"],
        "block_cols": fk["spatial"],
        "blocks": blocks,
        "multipliers": K_PHYS * N_PHYS * blocks,

        "slots_per_block": slots,
        "slot_budget": MEM_TILES,

        # Public stream row widths (packed LSB-first, architecture.md §11.1):
        # one whole K row per M beat in, one whole N row per M beat out.
        # a_port_bits (K*8, unpadded) is the single source for the wrapper's
        # a_row port width -- element k at bits [8k +: 8], no K-tail padding
        # exposed at the interface. a_row_bits (K_PHYS/chunks_pad-padded) is
        # kept only as rtl.py's internal a_row_buf width, which the wrapper
        # zero-extends the unpadded port into.
        "a_port_bits": int(k) * IN_WIDTH,
        "a_row_bits": fk["chunks_pad"] * K_PHYS * IN_WIDTH,
        # res_port_bits (N*W, unpadded) is the single source for the
        # wrapper's res_row port width -- lane n at bits [W*n +: W], no
        # N-tail padding exposed at the interface. res_row_bits
        # (chunks_pad/N_PHYS-padded) is kept only as rtl.py's internal
        # padded result-buffer width, which the wrapper slices the first N
        # (real) lanes out of.
        "res_port_bits": int(n) * rw,
        "res_row_bits": fn["chunks_pad"] * N_PHYS * rw,
        "result_width": rw,
        "out_w": rw - 1,

        # Baked weight data: one canonical 4x8 tile per (K,N) chunk.
        "weight_tiles": fk["chunks"] * fn["chunks"],
        "weight_tile_bits": K_PHYS * N_PHYS * COEF_WIDTH,

        # Valid lanes per tail chunk (zero-pad the rest).
        "k_tail_lanes": tuple(k_tail_lanes(k, i) for i in range(fk["chunks"])),
        "n_tail_lanes": tuple(n_tail_lanes(n, i) for i in range(fn["chunks"])),

        # Spec's K-first schedule: output groups outermost, K passes innermost.
        "slot_sequence": slot_sequence(n_passes, k_passes),

        "first_out_latency": latency,
        "warnings": fk["warnings"] + fn["warnings"],

        # Row-major runtime-B load (rtl.generate_core(b_row_major=True)): a
        # block's write port holds its tile address across a whole 4-beat
        # transaction, so N_PASSES>1 resident slots can't be interleaved
        # beat-by-beat. Slot 0 is written live off every incoming row; slots
        # 1..N_PASSES-1 are buffered per row-band (4 rows) and drained after
        # the band's 4th row, stalling new input for this many cycles. 0 when
        # N_PASSES==1 (no buffer, no stall -- every row writes live).
        "row_major_drain_cycles": 4 * (n_passes - 1),
    }


# ── Slot map and schedule ─────────────────────────────────────────────────────


def slot_of(k_pass, n_group, k_passes):
    """Slot index for one block's ``(k_pass, n_group)`` tile.

    Slots are assigned in schedule order (N-outer/K-inner), so a block's
    ``tile_sel`` simply walks 0..slots_per_block-1 over a row's passes.
    """
    return n_group * k_passes + k_pass


def slot_sequence(n_passes, k_passes):
    """Spec schedule order: ``[(k_pass, n_group)]`` with N outer, K inner."""
    return tuple((kp, np) for np in range(n_passes) for kp in range(k_passes))


def ktile_of(k_pass, col, k_spatial):
    """Logical K-tile index of cascade column *col* on temporal pass *k_pass*."""
    return k_pass * k_spatial + col


def ntile_of(n_group, row, n_spatial):
    """Logical N-tile index of broadcast row *row* in group *n_group*."""
    return n_group * n_spatial + row


# ── Tail masks ────────────────────────────────────────────────────────────────


def k_tail_lanes(k, k_tile):
    """Valid contraction lanes in logical K tile *k_tile* (0 = empty)."""
    remain = int(k) - int(k_tile) * K_PHYS
    if remain <= 0:
        return 0
    return min(K_PHYS, remain)


def n_tail_lanes(n, n_tile):
    """Valid output lanes in logical N tile *n_tile* (0 = empty)."""
    remain = int(n) - int(n_tile) * N_PHYS
    if remain <= 0:
        return 0
    return min(N_PHYS, remain)


# ── Runtime-B load schedule ───────────────────────────────────────────────────


def runtime_b_load_schedule(k, n, geo, b_row_major):
    """Per-cycle load-window pattern of the runtime-B wrapper's load FSM:
    True where it consumes an external B beat, False where it runs on its own
    (padding columns/rows, drains). The wrapper has no B ready, so the C++
    model and the Icarus TBs must present beats exactly on the True cycles.

    Column-major: every n-group (8 virtual columns, block rows inner) costs
    one cycle per column; real N columns consume a beat, padding columns are
    self-clocked. With K_PASSES > 2 each group is then followed by its drain,
    4 cycles per staged tile (K_PASSES-2 staged when the group's base slot
    n_group*K_PASSES is even, K_PASSES-1 when odd).

    Row-major: every K tile costs 4 fill cycles (real rows consume a beat)
    plus, when N_PASSES > 1, 4*(N_PASSES-1) drain cycles -- for every tile,
    since the fill/drain split is unconditional on tile index.
    """
    if not b_row_major:
        kp = int(geo["k_passes"])
        ns = int(geo["n_spatial"])
        sched = []
        for g in range(int(geo["n_chunks_pad"])):
            sched.extend((g * N_PHYS + c) < int(n) for c in range(N_PHYS))
            if kp > 2:
                n_group = g // ns
                staged = kp - 2 if (n_group * kp) % 2 == 0 else kp - 1
                sched.extend([False] * (K_PHYS * staged))
        return sched
    np_n = int(geo["n_passes"])
    sched = []
    for kt in range(int(geo["k_chunks_pad"])):
        sched.extend((kt * K_PHYS + i) < int(k) for i in range(K_PHYS))
        if np_n > 1:
            sched.extend([False] * (K_PHYS * (np_n - 1)))
    return sched


def runtime_b_beat_waits(k, n, geo, b_row_major):
    """Idle load cycles after each real B beat before the next one is taken
    (0 after the last beat): what a beat-driving TB must wait so it never
    presents a beat on a cycle the load FSM would not consume."""
    sched = runtime_b_load_schedule(k, n, geo, b_row_major)
    real = [i for i, v in enumerate(sched) if v]
    return [real[j + 1] - real[j] - 1 for j in range(len(real) - 1)] + [0]
