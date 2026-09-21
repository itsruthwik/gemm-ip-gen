"""The combined-fold weight ROM must use the narrow word (64*k_spatial bits) at
every k_spatial. At k_spatial == 1 the single-axis builder falls back to the
chunked wide word, which puts column t in tile t // 8: the combined-fold core and
its C twin read only the low 64 bits, so every column past the first tile used to
come out as zero weights."""
import numpy as np
import pytest

from test_tensor_slice_operand_guard import _pkg  # noqa: F401  (puts src/ on sys.path)
from gemm_ip.weights import build_weight_rom_combined_fold
from targets.tensor_slice.golden import pack_b_k_spatial_narrow


@pytest.mark.parametrize("k,core_n,k_spatial,n_passes", [
    (16, 32, 1, 1),   # transformer ffn1 shape: one K partition, four column tiles
    (16, 16, 1, 1),
    (16, 16, 1, 2),
    (32, 16, 2, 1),
])
def test_combined_fold_rom_words_are_narrow(k, core_n, k_spatial, n_passes):
    rng = np.random.default_rng(0)
    b_full = rng.integers(-8, 9, size=(k, n_passes * core_n))
    b_full[b_full == 0] = 1   # no zero weights, so a dropped column cannot hide
    passes = -(-((k + 7) // 8) // k_spatial)
    rom = build_weight_rom_combined_fold(b_full, 8, core_n, k, k_spatial, n_passes)
    assert len(rom) == passes * n_passes * core_n
    for chunk in range(passes):
        for ng in range(n_passes):
            bg = b_full[:, ng * core_n:(ng + 1) * core_n]
            for t in range(core_n):
                word = rom[(chunk * n_passes + ng) * core_n + t]
                assert 0 < word < (1 << (64 * k_spatial))
                assert word == pack_b_k_spatial_narrow(bg, t, chunk, core_n, k, k_spatial)
