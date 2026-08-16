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
    ts_dir = str(Path(__file__).resolve().parent.parent / "tensor-slice")
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
    in ``generate_verilog_tb.py``). ``B`` is ``[K, N]``.
    """
    _ts_dir_on_path()
    from generate_verilog_tb import pack_b_chunk  # noqa: F401

    grid_cols = (n + 7) // 8
    k_chunks = (k + 7) // 8
    input_beats = max(m, n)
    rom = []
    for chunk in range(k_chunks):
        for t in range(input_beats):
            rom.append(int(pack_b_chunk(B, t, chunk, grid_cols, n, k)))
    return rom


def weight_rom_from_dat(path, m, n, k):
    """Convenience: load a ``.dat`` and build the ROM beats in one call."""
    B = load_weight_dat(path, n, k)
    return build_weight_rom(B, m, n, k)
