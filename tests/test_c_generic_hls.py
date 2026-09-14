"""Unit tests for the generic/catapult kernel emitter (step 3 of
jojo-track/open/catapult-generic-target/plan.md), `src/targets/c_generic/hls.py`.

All tool-free except the final g++ syntax-check smoke test (no Catapult/Vitis
invocation).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TARGET_DIR = _REPO_ROOT / "gemm-ip-gen" / "src" / "targets" / "c_generic"
_TEMP = _REPO_ROOT / "temp_space" / "c_generic"

_SRC = _REPO_ROOT / "gemm-ip-gen" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from targets.c_generic import hls  # noqa: E402

FORBIDDEN_PATTERNS = [r"#pragma HLS", r"hls::stream", r"\bap_int\b", r"\bap_uint\b", r"\bap_fixed\b"]


def _assert_no_forbidden_idioms(text):
    for pat in FORBIDDEN_PATTERNS:
        matches = re.findall(pat, text)
        assert matches == [], f"forbidden idiom {pat!r} found: {matches[:3]}"


# ── ac_type / mode mapping ────────────────────────────────────────────────────

@pytest.mark.parametrize("src,expected", [
    ("fixed<16,6>", "ac_fixed<16,6,true,AC_TRN,AC_WRAP>"),
    ("ufixed<16,6>", "ac_fixed<16,6,false,AC_TRN,AC_WRAP>"),
    ("fixed<16,6,RND,SAT>", "ac_fixed<16,6,true,AC_RND,AC_SAT>"),
    ("fixed<16,6,RND_CONV,SAT_SYM>", "ac_fixed<16,6,true,AC_RND_CONV,AC_SAT_SYM>"),
    ("fixed<16,6,TRN,WRAP>", "ac_fixed<16,6,true,AC_TRN,AC_WRAP>"),
    ("fixed<16,6,RND_ZERO,SAT>", "ac_fixed<16,6,true,AC_RND_ZERO,AC_SAT>"),
    ("ap_fixed<16,6>", "ac_fixed<16,6,true,AC_TRN,AC_WRAP>"),
    ("ap_ufixed<10,4>", "ac_fixed<10,4,false,AC_TRN,AC_WRAP>"),
    ("ap_int<8>", "ac_int<8,true>"),
    ("ap_uint<8>", "ac_int<8,false>"),
    ("int<8>", "ac_int<8,true>"),
    ("uint<8>", "ac_int<8,false>"),
    ("ac_fixed<18,8,true>", "ac_fixed<18,8,true,AC_TRN,AC_WRAP>"),
    ("ac_int<4,false>", "ac_int<4,false>"),
])
def test_ac_type_mapping(src, expected):
    assert hls.ac_type(src) == expected


def test_ac_type_default_when_unset():
    assert hls.ac_type(None, "ac_fixed<16,6,true>") == "ac_fixed<16,6,true>"
    assert hls.ac_type("", "ac_fixed<16,6,true>") == "ac_fixed<16,6,true>"


# ── flatten / weight layouts ─────────────────────────────────────────────────

def test_flatten_weights_row_major_false():
    # weight_matrix given as [K][N]; dense_resource's OWN index arithmetic
    # (traced from the copied RF regime loops -- see hls.py's module
    # docstring) is out-major: index = nn*k+kk, i.e. [n_out][n_in] flattened.
    k, n = 2, 3
    wm = [[1, 2, 3], [4, 5, 6]]  # wm[kk][nn] == W[kk][nn]
    flat = hls._flatten_weights(wm, k, n, weights_row_major=False)
    assert flat == [1, 4, 2, 5, 3, 6]


def test_flatten_weights_row_major_true():
    # weight_matrix given as [N][K] (transposed); same logical W.
    k, n = 2, 3
    wm_t = [[1, 4], [2, 5], [3, 6]]  # wm_t[nn][kk] == W[kk][nn]
    flat = hls._flatten_weights(wm_t, k, n, weights_row_major=True)
    assert flat == [1, 4, 2, 5, 3, 6]


# ── combined_header() shape ───────────────────────────────────────────────────

def _mk_item(name="l0", m=2, k=4, n=3, rf=1, weight_matrix=None, bias_values=None,
            has_bias=True, weights_row_major=False, idx=0):
    return dict(
        name=name, gemm_m=m, gemm_k=k, gemm_n=n, reuse_factor=rf,
        gemm_ip_index=idx, weight_matrix=weight_matrix, bias_values=bias_values,
        has_bias=has_bias, weights_row_major=weights_row_major,
        weights_in_core=weight_matrix is not None,
    )


def test_combined_header_has_four_entry_points():
    item = _mk_item(weight_matrix=[[1] * 3 for _ in range(4)], bias_values=[0, 0, 0])
    text = hls.combined_header([item])
    assert re.search(r"void\s+gemm_array\s*\(", text)
    assert re.search(r"void\s+gemm_array_const_weights\s*\(", text)
    assert re.search(r"void\s+gemm_stream\s*\(", text)
    assert re.search(r"void\s+gemm_stream_const_weights\s*\(", text)
    assert "ac_channel" in text
    assert "l0_config" in text


def test_combined_header_forbidden_idioms():
    item = _mk_item(weight_matrix=[[1] * 3 for _ in range(4)])
    text = hls.combined_header([item])
    _assert_no_forbidden_idioms(text)


def test_gemm_ip_header_forbidden_idioms_and_guard():
    text = hls.gemm_ip_header("mylayer", 2, 4, 3, weight_matrix=[[1] * 3 for _ in range(4)])
    _assert_no_forbidden_idioms(text)
    assert "MYLAYER_GEMM_IP_H_" in text


# ── CONFIG_T fields ────────────────────────────────────────────────────────

def test_config_fields_present():
    item = _mk_item(m=5, k=8, n=6, rf=2)
    text = hls.combined_header([item])
    assert "static const unsigned gemm_m = 5;" in text
    assert "static const unsigned gemm_k = 8;" in text
    assert "static const unsigned gemm_n = 6;" in text
    assert "static const unsigned reuse_factor = 2;" in text
    assert "weights_row_major" in text


# ── RF snapping reflected downstream ─────────────────────────────────────────

def test_reuse_factor_reflected_in_emitted_config():
    item = _mk_item(k=8, n=16, rf=1)
    from gemm_ip.behavioral import snap_reuse_factor  # noqa
    item["reuse_factor"] = 3  # invalid for k=8,n=16
    snap_reuse_factor(item)
    text = hls.combined_header([item])
    assert f"static const unsigned reuse_factor = {item['reuse_factor']};" in text
    assert item["reuse_factor"] != 3


# ── bias / no-bias ────────────────────────────────────────────────────────

def test_no_bias_bakes_zero_rom():
    item = _mk_item(n=3, bias_values=[1, 2, 3], has_bias=False)
    text = hls.combined_header([item])
    m = re.search(r"l0_bias_rom\[3\] = \{ ([^}]*) \}", text)
    assert m
    vals = [v.strip() for v in m.group(1).split(",")]
    assert vals == ["0", "0", "0"]


def test_has_bias_keeps_values():
    item = _mk_item(n=3, bias_values=[1, 2, 3], has_bias=True)
    text = hls.combined_header([item])
    m = re.search(r"l0_bias_rom\[3\] = \{ ([^}]*) \}", text)
    assert m
    vals = [v.strip() for v in m.group(1).split(",")]
    assert vals == ["1", "2", "3"]


def test_has_bias_trait_specializations():
    item = _mk_item(has_bias=False, idx=0)
    text = hls.combined_header([item])
    assert "gemm_ip_has_bias<0> { static const bool value = false; }" in text


# ── g++ syntax-check smoke test ──────────────────────────────────────────────

def _find_ac_types_include_dir():
    candidates = [
        Path("/home/tools/siemens/catapult/Mgc_home/shared/include"),
        _REPO_ROOT / "hls4ml-gemm" / "hls4ml" / "templates" / "catapult" / "ac_types" / "include",
    ]
    for c in candidates:
        if (c / "ac_fixed.h").exists() and (c / "ac_int.h").exists() and (c / "ac_channel.h").exists():
            return c
    return None


_AC_INCLUDE_DIR = _find_ac_types_include_dir()
_GXX = shutil.which("g++")


@pytest.mark.skipif(_AC_INCLUDE_DIR is None, reason="no ac_types include dir with ac_fixed/ac_int/ac_channel found")
@pytest.mark.skipif(_GXX is None, reason="no g++ on PATH")
@pytest.mark.parametrize("shape", [(2, 4, 3), (4, 16, 8)])
def test_gxx_syntax_check(shape):
    m, k, n = shape
    _TEMP.mkdir(parents=True, exist_ok=True)
    weight_matrix = [[(i + j) % 5 for j in range(n)] for i in range(k)]
    bias_values = [i % 3 for i in range(n)]
    item = _mk_item(name=f"shape_{m}_{k}_{n}", m=m, k=k, n=n, rf=1,
                    weight_matrix=weight_matrix, bias_values=bias_values)
    header_text = hls.combined_header([item])
    header_path = _TEMP / f"gemm_ip_combined_{m}_{k}_{n}.h"
    header_path.write_text(header_text)

    a_t = f"shape_{m}_{k}_{n}_config"
    src_text = f"""#include "{header_path.name}"

typedef {a_t}::input_t a_row_T_elem;

struct a_row_t {{
    static const unsigned size = {k};
    typedef a_row_T_elem value_type;
    a_row_T_elem data[{k}];
    a_row_T_elem &operator[](unsigned i) {{ return data[i]; }}
}};

typedef {a_t}::result_t res_row_T_elem;
struct res_row_t {{
    static const unsigned size = {n};
    typedef res_row_T_elem value_type;
    res_row_T_elem data[{n}];
    res_row_T_elem &operator[](unsigned i) {{ return data[i]; }}
}};

void instantiate() {{
    a_row_t a_rows[{m}];
    res_row_t results[{m}];
    nnet::gemm_array_const_weights<a_row_t, res_row_t, {a_t}>(a_rows, results);
}}
"""
    src_path = _TEMP / f"gxx_check_{m}_{k}_{n}.cpp"
    src_path.write_text(src_text)

    result = subprocess.run(
        [_GXX, "-fsyntax-only", "-std=c++17", f"-I{_AC_INCLUDE_DIR}", f"-I{_TEMP}", str(src_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
