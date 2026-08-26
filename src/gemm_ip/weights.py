"""Weight-stationary (const-weight) ROM construction.

The weight-stationary GEMM IP variant bakes the constant weights into an internal
ROM in the RTL wrapper (no external weight port) and streams one weight beat per
cycle into the tensor slice — exactly as the external ``b_cols`` port does today.

hls4ml emits the weights as a column-major ``[gemm_n][gemm_k]`` raw fixed-point
integer ``.dat`` (``hls4ml/writer/gemm_ip_weights.py``). The tensor slice's B feed
expects ``B`` shaped ``[K, N]``, so we load the ``.dat`` and transpose.

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


def load_weight_dat(path, n, k):
    """Load an hls4ml column-major ``[n][k]`` raw-int ``.dat`` as ``B`` shaped ``[K, N]``.

    Each line is one output column (n) with ``k`` space-separated signed integers.
    Returns an ``int64`` array of shape ``(k, n)`` (the [K, N] the B feed expects).
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([int(x) for x in line.split()])
    dat = np.asarray(rows, dtype=np.int64)  # [n][k]
    if dat.shape != (n, k):
        raise ValueError(f"weight .dat {path}: shape {dat.shape} != expected ({n}, {k})")
    return dat.T.copy()  # -> [k][n]


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
    input_beats = max(m, n)
    rom = []
    for chunk in range(k_chunks):
        for t in range(input_beats):
            rom.append(int(pack_b_chunk(B, t, chunk, grid_cols, n, k)))
    return rom


def build_weight_rom_full_k(B, m, n, k):
    """Per-beat ROM values for the full-K-spatial feed. ``B`` is ``[K, N]``.

    Full-K feeds every K chunk in ONE ``max(M, N)``-beat pass, so the ROM holds
    ``input_beats`` entries of ``64*k_chunks`` bits — the transpose of the chunked
    layout's ``k_chunks*input_beats`` entries of ``grid_cols*64`` bits. Widening the
    word rather than adding beats is deliberate: serialising the baked weights would
    cost the very latency full-K exists to avoid.

    Uses the NARROW packer (one tile at position 0, K chunk ``c`` at bits
    ``[c*64, c*64+64)``, no grid tile offset) because the full-K wrapper RTL
    re-inserts the tile offset by beat index — see the ``a_bits``/``b_bits``
    derivation in ``catapult.gen_public_header``. Same packer the verified
    testbench stimulus uses, so the baked ROM matches the beats the external
    ``b_cols`` port would have received.

    K need not be a multiple of 8: the packer walks only real K bytes, leaving
    tail lanes zero, which is the masking the RTL contract requires.
    """
    _ts_dir_on_path()
    from golden import pack_b_full_k_spatial_narrow  # noqa: F401

    input_beats = max(m, n)
    return [int(pack_b_full_k_spatial_narrow(B, t, n, k)) for t in range(input_beats)]


def weight_rom_from_dat(path, m, n, k, full_k_spatial=False):
    """Convenience: load a ``.dat`` and build the ROM beats in one call."""
    B = load_weight_dat(path, n, k)
    if full_k_spatial:
        return build_weight_rom_full_k(B, m, n, k)
    return build_weight_rom(B, m, n, k)
