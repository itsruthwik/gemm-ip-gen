"""Weight-stationary (const-weight) ROM construction.

The weight-stationary GEMM IP variant bakes the constant weights into an internal
ROM in the RTL wrapper (no external weight port) and streams one weight beat per
cycle into the tensor slice — exactly as the external ``b_cols`` port does today.

hls4ml emits the weights as a raw fixed-point integer ``.dat``
(``hls4ml/writer/gemm_ip_weights.py``), column-major ``[gemm_n][gemm_k]`` by default
or row-major ``[gemm_k][gemm_n]`` under ``SecondOperandRowMajor``; the manifest's
``weight_layout`` says which. The file is read as written; a target that does not
consume that layout (``Target.weight_layouts``) is refused by the CLI, not adapted.

The ROM contents are produced by reusing the SAME per-beat packer the verified
testbench uses (``pack_b_chunk``), so the baked ROM is byte-identical to the beat
sequence the external port would have received — cosim stays bit-exact.
"""

import sys as _sys
from pathlib import Path

import numpy as np


def _ts_dir_on_path():
    ts_dir = str(Path(__file__).resolve().parent.parent / "targets" / "tensor_slice")
    if ts_dir not in _sys.path:
        _sys.path.insert(0, ts_dir)


def load_weight_dat(path, n, k, layout="column_major"):
    """Load an hls4ml raw-int weight ``.dat`` as ``B`` shaped ``[K, N]``.

    ``layout`` is the manifest's ``weight_layout``, i.e. how hls4ml wrote the file
    (``SecondOperandRowMajor``); it is read as written, never re-ordered:
      - ``column_major``: one output column per line, ``k`` integers (``[n][k]``),
        which is B transposed;
      - ``row_major``: one contraction row per line, ``n`` integers (``[k][n]``),
        which is B itself.
    Whether a target can consume a layout is decided by the CLI against
    ``Target.weight_layouts`` before this is called.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([int(x) for x in line.split()])
    dat = np.asarray(rows, dtype=np.int64)
    layout = (layout or "column_major").lower()
    if layout == "row_major":
        if dat.shape != (k, n):
            raise ValueError(f"weight .dat {path} (row_major): shape {dat.shape} != expected ({k}, {n})")
        return dat.copy()  # [k][n] == B
    if layout != "column_major":
        raise ValueError(f"weight .dat {path}: unknown weight_layout {layout!r}")
    if dat.shape != (n, k):
        raise ValueError(f"weight .dat {path} (column_major): shape {dat.shape} != expected ({n}, {k})")
    return dat.T.copy()  # [n][k] -> [k][n]


def build_weight_rom(B, m, n, k):
    """Per-beat ROM values matching the TB/RTL ``b_cols`` feed order.

    Returns a list of ints, one per beat, each ``grid_cols*64`` bits wide, in the
    exact ``for chunk: for t`` order the RUN loop consumes (see ``_gen_all_stimulus``
    in ``golden.py``). ``B`` is ``[K, N]``.
    """
    _ts_dir_on_path()
    from golden import pack_b_chunk  # noqa: F401

    grid_cols = (n + 7) // 8
    k_chunks = (k + 7) // 8
    rom = []
    for chunk in range(k_chunks):
        for t in range(n):
            rom.append(int(pack_b_chunk(B, t, chunk, grid_cols, n, k)))
    return rom


def build_weight_rom_k_spatial(B, m, n, k, k_spatial):
    """Per-beat ROM values for the general K-spatial feed. ``B`` is ``[K, N]``.

    ``k_spatial`` parallel K partitions cover ``k_chunks = ceil(k/8)`` chunks in
    ``passes = ceil(k_chunks/k_spatial)`` passes; each pass feeds ONE
    ``max(M, N)``-beat sweep, but only the first ``N`` beats of each pass carry a
    real column (beat ``t >= N`` is a wasted repeat), so the ROM holds only
    ``passes*N`` entries of ``64*k_spatial`` bits; the wrapper zero-fills beats
    ``t >= N`` at read time. ``k_spatial == 1`` is today's chunked endpoint
    (``build_weight_rom``); ``k_spatial == k_chunks`` (one pass) is today's
    full-K endpoint (``build_weight_rom_full_k``). Widening the word rather
    than adding beats per chunk is deliberate: serialising the baked weights
    would cost the very latency K-spatial folding exists to avoid.

    Uses the NARROW packer (partition p at bits ``[p*64, p*64+64)``, no grid
    tile offset) because the K-spatial wrapper RTL re-inserts the tile offset
    by beat index — see the ``a_bits``/``b_bits`` derivation in
    ``catapult.gen_public_header``. Same packer the verified testbench
    stimulus uses, so the baked ROM matches the beats the external ``b_cols``
    port would have received.

    K need not be a multiple of 8: the packer walks only real K bytes, leaving
    tail lanes (and fully-padded chunks beyond ``k_chunks``) zero, which is
    the masking the RTL contract requires.
    """
    if k_spatial == 1:
        return build_weight_rom(B, m, n, k)
    _ts_dir_on_path()
    from golden import pack_b_k_spatial_narrow  # noqa: F401

    k_chunks = (k + 7) // 8
    passes = -(-k_chunks // k_spatial)
    rom = []
    for pass_idx in range(passes):
        for t in range(n):
            rom.append(int(pack_b_k_spatial_narrow(B, t, pass_idx, n, k, k_spatial)))
    return rom


def build_weight_rom_fold_n(B_full, m, core_n, k, k_spatial, n_passes):
    """Fold-N weight ROM: ``n_passes`` column groups of ``core_n`` columns each,
    group ``g``'s beats at base ``g * core_n``. ``B_full`` is ``[K, n_passes*core_n]``
    (real columns first, zero-padded tail).

    Built per group so every word is sized for ``core_n`` columns -- the width the
    fold-N core's ``b_cols`` register and csim ``B_ROM`` actually have. Building it
    in one call with ``n = n_passes*core_n`` would, on the chunked layout
    (``k_spatial == 1``), place group ``g``'s column in tile ``g`` of a
    ``grid_cols(n_full)``-wide word, which the ``core_n``-wide consumers truncate.
    The narrow K-spatial layout is width-independent of ``n`` so it was unaffected.
    """
    rom = []
    for g in range(n_passes):
        Bg = np.asarray(B_full)[:, g * core_n:(g + 1) * core_n]
        rom.extend(build_weight_rom_k_spatial(Bg, m, core_n, k, k_spatial))
    return rom


def build_weight_rom_full_k(B, m, n, k):
    """Today's full-K endpoint of :func:`build_weight_rom_k_spatial` (one pass)."""
    k_chunks = (k + 7) // 8
    return build_weight_rom_k_spatial(B, m, n, k, k_chunks)


def weight_rom_from_dat(path, m, n, k, k_spatial=1, layout="column_major"):
    """Convenience: load a ``.dat`` and build the ROM beats in one call."""
    B = load_weight_dat(path, n, k, layout)
    return build_weight_rom_k_spatial(B, m, n, k, k_spatial)
