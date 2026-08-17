"""C++ ccore execution tests.

Compiles the generated public header's ``{name}_ccore`` with the MGC AC
datatypes and replays the wrapper FEED protocol call-by-call (one preload
call, then total_beats data calls, then idle calls), checking every emitted
row against an independent numpy golden model and the emission timing against
latency_cycles — for both the sequential cadence (the generated wrapper's
FEED -> DRAIN drive) and true back-to-back frames (frame II = total_beats+1).

Skipped when the dcs-gcc toolchain or the AC headers are unavailable.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_ts_dir = str(Path(__file__).resolve().parent.parent / "src" / "tensor-slice")
if _ts_dir not in sys.path:
    sys.path.insert(0, _ts_dir)

from gemm_ip.catapult import gen_public_header, latency_cycles  # noqa: E402
from generate_verilog_tb import (  # noqa: E402
    _random_matrices,
    pack_a_chunk,
    pack_a_full_k_spatial_narrow,
    pack_b_chunk,
    pack_b_full_k_spatial_narrow,
    pack_bias,
)

MGC = Path(os.environ.get("MGC_HOME", "/home/tools/siemens/catapult/Mgc_home"))
GXX = MGC / "pkgs" / "dcs_gcc" / "gcc-13.4.0" / "bin" / "g++"
ACDIR = MGC / "shared" / "include"
HAVE_CXX = GXX.exists() and (ACDIR / "ac_int.h").exists()

MAIN_TPL = r"""
#include <cstdio>
#include <cstring>
#include <ac_fixed.h>
#include "gemm_hdr.h"

template <int W>
ac_int<W, false> parse_hex(const char *s) {
    ac_int<W, false> v = 0;
    int n = (int) strlen(s);
    for (int i = 0; i < n && 4 * i < W; i++) {
        char c = s[n - 1 - i];
        int d = (c >= '0' && c <= '9') ? c - '0'
              : (c >= 'a' && c <= 'f') ? c - 'a' + 10
              : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0;
        v.set_slc(4 * i, ac_int<4, false>(d));
    }
    return v;
}

int main() {
    nnet::NAME_ccore core;
    char kind[4], ah[1024], bh[1024], biash[1024];
    int call = 0;
    while (scanf("%3s %1023s %1023s %1023s", kind, ah, bh, biash) == 4) {
        ac_int<ABITS, false> a = parse_hex<ABITS>(ah);
        ac_int<BBITS, false> b = parse_hex<BBITS>(bh);
        ac_int<BIASBITS, false> bias = parse_hex<BIASBITS>(biash);
        ac_int<1, false> pre = (kind[0] == 'P');
        ac_int<1, false> iv = (kind[0] == 'D');
        ac_int<CBITS, false> c_row;
        ac_int<1, false> v, l;
        core.run(a, b, bias, pre, iv, c_row, v, l);
        if (v) {
            printf("OUT %d %d ", call, (int) l);
            for (int nib = CBITS - 4; nib >= 0; nib -= 4)
                printf("%x", (int) c_row.template slc<4>(nib).to_uint());
            printf("\n");
        }
        call++;
    }
    printf("END %d\n", call);
    return 0;
}
"""


def _word_hex(val):
    return format(val, "x") if val else "0"


def _frame_calls(A, B, biases, m, k, n, full_k):
    """One frame's calls in wrapper FEED order: preload + total_beats data."""
    grid_rows = (m + 7) // 8
    grid_cols = (n + 7) // 8
    input_beats = max(m, n)
    k_chunks = (k + 7) // 8
    bias_word = pack_bias(biases, grid_cols, n)
    calls = [("P", "0", "0", _word_hex(bias_word))]
    if full_k:
        for t in range(input_beats):
            calls.append(("D",
                          _word_hex(pack_a_full_k_spatial_narrow(A, t, m, k)),
                          _word_hex(pack_b_full_k_spatial_narrow(B, t, n, k)),
                          _word_hex(bias_word)))
    else:
        for chunk in range(k_chunks):
            for t in range(input_beats):
                calls.append(("D",
                              _word_hex(pack_a_chunk(A, t, chunk, grid_rows, m, k)),
                              _word_hex(pack_b_chunk(B, t, chunk, grid_cols, n, k)),
                              _word_hex(bias_word)))
    return calls


def _golden_rows(A, B, biases, m, n, grid_cols):
    """Expected c_row hex per output row: int16 lanes, raw accumulator."""
    C = A.astype(np.int64) @ B.astype(np.int64) + biases.astype(np.int64)
    C = np.clip(C, -32768, 32767)
    rows = []
    for r in range(m):
        val = 0
        for ct in range(grid_cols):
            for cl in range(8):
                col = ct * 8 + cl
                lane = int(C[r, col]) & 0xFFFF if col < n else 0
                val |= lane << (ct * 128 + cl * 16)
        rows.append(val)
    return rows


@pytest.mark.skipif(not HAVE_CXX, reason="dcs-gcc / AC headers not available")
class TestCcoreCsim:
    @pytest.mark.parametrize("m,k,n,gemm_k_spatial", [
        pytest.param(8, 8, 8, 1, id="8x8x8"),
        pytest.param(10, 10, 10, 2, id="10x10x10-fullk"),
        pytest.param(16, 16, 16, 2, id="16x16x16-fullk"),
        pytest.param(16, 72, 8, 9, id="16x72x8-fullk"),
        pytest.param(16, 72, 8, 1, id="16x72x8-chunked"),
        pytest.param(10, 27, 10, 4, id="10x27x10-fullk"),
    ])
    def test_ccore_frames_sequential_and_back_to_back(self, m, k, n, gemm_k_spatial, tmp_path):
        grid_rows = (m + 7) // 8
        grid_cols = (n + 7) // 8
        k_chunks = (k + 7) // 8
        full_k = k_chunks > 1 and gemm_k_spatial == k_chunks
        input_beats = max(m, n)
        total_beats = input_beats if full_k else k_chunks * input_beats
        first_out = latency_cycles(m, k, n, grid_rows, grid_cols, full_k_spatial=full_k)
        a_bits = 64 * k_chunks if full_k else grid_rows * 64
        b_bits = 64 * k_chunks if full_k else grid_cols * 64
        bias_bits = grid_cols * 64
        c_bits = grid_cols * 128

        hdr = gen_public_header("uut", m, k, n, grid_rows, grid_cols,
                                gemm_k_spatial=gemm_k_spatial)
        (tmp_path / "gemm_hdr.h").write_text(hdr)
        main = (MAIN_TPL
                .replace("NAME_", "uut_")
                .replace("ABITS", str(a_bits))
                .replace("BBITS", str(b_bits))
                .replace("BIASBITS", str(bias_bits))
                .replace("CBITS", str(c_bits)))
        (tmp_path / "main.cpp").write_text(main)

        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (
            str(MGC / "pkgs" / "dcs_gcc" / "gcc-13.4.0" / "lib64")
            + ":" + env.get("LD_LIBRARY_PATH", ""))
        comp = subprocess.run(
            [str(GXX), "-std=c++17", "-O1", f"-I{ACDIR}",
             str(tmp_path / "main.cpp"), "-o", str(tmp_path / "harness")],
            capture_output=True, text=True, timeout=300, env=env)
        assert comp.returncode == 0, comp.stderr[:800]

        # Stimulus: 3 sequential frames (idle gap covers the full drain, like
        # the wrapper's FEED -> DRAIN drive), then 4 true back-to-back frames.
        frames, calls, starts = [], [], []
        drain_idle = first_out + 1 + m + 2
        for v in range(7):
            A, B, biases, _ = _random_matrices(m, k, n, seed=100 + v)
            frames.append((A, B, biases))
        for v in range(3):
            fc = _frame_calls(*frames[v], m, k, n, full_k)
            starts.append(len(calls) + 1)            # first data call index
            calls += fc + [("I", "0", "0", "0")] * drain_idle
        for v in range(3, 7):
            fc = _frame_calls(*frames[v], m, k, n, full_k)
            starts.append(len(calls) + 1)
            calls += fc
        calls += [("I", "0", "0", "0")] * drain_idle  # final flush

        stim = "\n".join(" ".join(c) for c in calls) + "\n"
        run = subprocess.run([str(tmp_path / "harness")], input=stim,
                             capture_output=True, text=True, timeout=120, env=env)
        assert run.returncode == 0, run.stderr[:500]
        outs = re.findall(r"OUT (\d+) (\d) ([0-9a-f]+)", run.stdout)
        assert len(outs) == 7 * m, f"expected {7 * m} rows, got {len(outs)}"

        for v in range(7):
            A, B, biases = frames[v]
            golden = _golden_rows(A, B, biases, m, n, grid_cols)
            rows = outs[v * m:(v + 1) * m]
            # Data: every row bit-exact vs the numpy golden.
            for r, (_, last, hexval) in enumerate(rows):
                assert int(hexval, 16) == golden[r], (v, r)
                assert int(last) == (1 if r == m - 1 else 0)
            # Timing: first row of frame v at its first data call + first_out + 1.
            assert int(rows[0][0]) == starts[v] + first_out + 1, (v, rows[0][0])

        # Back-to-back frame II: first-output spacing == total_beats + 1.
        period = total_beats + 1
        b2b_firsts = [int(outs[v * m][0]) for v in range(3, 7)]
        deltas = [b - a for a, b in zip(b2b_firsts, b2b_firsts[1:])]
        assert all(d == period for d in deltas), (deltas, period)


# Compiling the generated header for the WEIGHTLESS (weight-stationary) variant.
# The RTL-side tests drive iverilog on the generated Verilog and never touch the
# generated C++, so a broken weightless wrapper compiled fine in CI and only failed
# in the real HLS flow: the full-K feed loop emitted the external B beat
# unconditionally, giving "'b_beat_T' was not declared in this scope" and a run()
# arity mismatch. This forces a real template instantiation of the weightless entry
# so both classes of error surface here.
WEIGHTLESS_TU = r"""
#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>

STUB_HERE

#include "gemm_hdr.h"

struct cfg {
    static const unsigned gemm_m = MVAL;
    static const unsigned gemm_k = KVAL;
    static const unsigned gemm_n = NVAL;
    static const unsigned n_in = KVAL;
    static const unsigned n_out = NVAL;
    typedef ac_fixed<32, 16, true> accum_t;
    typedef ac_fixed<16, 8, true> weight_t;
    typedef ac_fixed<16, 8, true> bias_t;
};

typedef nnet_array_stub<KVAL> a_beat_t;
typedef nnet_array_stub<NVAL> res_beat_t;

int main() {
    ac_channel<a_beat_t> a_stream;
    ac_channel<res_beat_t> res_stream;
    int biases[NVAL];
    for (int i = 0; i < NVAL; i++) biases[i] = 0;
    // Force instantiation of the weightless entry point.
    nnet::NAME_gemm_ip_stream_weightless<a_beat_t, int, res_beat_t, cfg>(
        a_stream, biases, res_stream);
    return 0;
}
"""


@pytest.mark.skipif(not HAVE_CXX, reason="dcs-gcc / AC headers not available")
class TestWeightlessHeaderCompiles:
    @pytest.mark.parametrize("m,k,n,gemm_k_spatial", [
        pytest.param( 8,  8,  8, 1, id="8x8x8-chunked"),
        pytest.param( 8, 16,  8, 1, id="8x16x8-chunked"),
        pytest.param( 8, 16,  8, 2, id="8x16x8-fullk"),
        pytest.param(24, 16, 16, 2, id="24x16x16-fullk"),
        pytest.param(15, 24, 14, 3, id="15x24x14-fullk"),
        pytest.param( 8, 12,  8, 2, id="8x12x8-fullk-tail"),
    ])
    def test_weightless_header_compiles(self, m, k, n, gemm_k_spatial, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "gemm_ip"))
        import weights as _weights

        k_chunks = (k + 7) // 8
        full_k = k_chunks > 1 and gemm_k_spatial == k_chunks
        rng = np.random.default_rng(3)
        B = rng.integers(-4, 5, size=(k, n)).astype(np.int8)
        rom = (_weights.build_weight_rom_full_k(B, m, n, k) if full_k
               else _weights.build_weight_rom(B, m, n, k))

        hdr = gen_public_header("uut", m, k, n, (m + 7) // 8, (n + 7) // 8,
                                gemm_k_spatial=gemm_k_spatial, weight_rom=rom)
        # Weight-stationary drops the external B operand everywhere; a leftover
        # reference is the exact failure this test exists to catch.
        assert "b_beat" not in hdr, "weightless header still references the external B beat"

        # Minimal nnet::array stand-in. A beats carry gemm_k lanes, result rows
        # gemm_n, so the width must be a template parameter — a single fixed size
        # only happens to satisfy both when k == n.
        stub = (
            "template <unsigned SZ>\n"
            "struct nnet_array_stub {\n"
            "    static const unsigned size = SZ;\n"
            "    typedef ac_fixed<16, 8, true> value_type;\n"
            "    value_type d[SZ];\n"
            "    value_type &operator[](int i) { return d[i]; }\n"
            "    const value_type &operator[](int i) const { return d[i]; }\n"
            "};\n"
        )
        (tmp_path / "gemm_hdr.h").write_text(hdr)
        tu = (WEIGHTLESS_TU
              .replace("STUB_HERE", stub)
              .replace("NAME_", "uut_")
              .replace("MVAL", str(m)).replace("KVAL", str(k)).replace("NVAL", str(n)))
        (tmp_path / "tu.cpp").write_text(tu)

        comp = subprocess.run(
            [str(GXX), "-std=c++17", "-fsyntax-only", f"-I{ACDIR}", str(tmp_path / "tu.cpp")],
            capture_output=True, text=True, cwd=str(tmp_path))
        assert comp.returncode == 0, (
            f"weightless header failed to compile (m={m} k={k} n={n} "
            f"k_spatial={gemm_k_spatial}):\n{comp.stderr[-3000:]}")
