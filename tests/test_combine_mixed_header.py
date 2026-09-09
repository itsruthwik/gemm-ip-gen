"""Codegen tests for the mixed per-layer-target combined header (gemm_ip.combine).

Tool-free (no Vitis). These lock in the contract the merged gemm_ip_combined.h must
satisfy so a mixed mvau+generic design compiles/simulates like a single-target one:
the copied generic soft kernels reference gemm_rf<CONFIG_T> and
gemm_ip_has_bias<CONFIG_T::gemm_ip_id>, so the merge must declare both traits, and every
gemm_stream_const_weights signature is 2-arg (bias is never a call argument -- soft reads
CONFIG_T::gemm_bias() gated by the trait; an IP bakes it into the core). csim/cosim of the
mixed build is validated separately (temp_space; jojo-track status).
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from gemm_ip.combine import unified_combined_header  # noqa: E402

# Two mvau IP layers + a generic (behavioral) head, the mlp_tiny_mvau_vitis mixed case.
_LAYERS = [
    {"name": "gemm_fc1", "id": 14, "target": "mvau", "has_bias": True,
     "reuse_factor": 1, "gemm_k": 49, "gemm_n": 32, "func": "stream_const_weights"},
    {"name": "gemm_fc2", "id": 15, "target": "mvau", "has_bias": True,
     "reuse_factor": 1, "gemm_k": 32, "gemm_n": 16, "func": "stream_const_weights"},
    {"name": "gemm_head", "id": 16, "target": "generic", "has_bias": False,
     "reuse_factor": 2, "gemm_k": 16, "gemm_n": 10, "func": "stream_const_weights"},
]


def test_declares_traits_the_soft_body_references():
    h = unified_combined_header(_LAYERS)
    # gemm_rf: the base template and the per-layer override the soft kernels' pragmas use.
    assert "template <typename CONFIG_T> struct gemm_rf {" in h
    assert "template <unsigned id> struct gemm_rf_override" in h
    assert "gemm_rf_override<16>" in h  # generic head's snapped reuse factor
    # gemm_ip_has_bias: base template plus the head's has_bias=False specialization.
    assert "template <unsigned id> struct gemm_ip_has_bias" in h
    assert "gemm_ip_has_bias<16>" in h
    # Both traits must be declared BEFORE the soft body that references them.
    assert h.index("struct gemm_rf {") < h.index("gemm_soft_stream_const_weights")
    assert h.index("struct gemm_ip_has_bias") < h.index("gemm_soft_stream_const_weights")


def test_gemm_stream_const_weights_is_two_arg():
    h = unified_combined_header(_LAYERS)
    # No stale 3-arg bias-array convention anywhere.
    assert "bias_t b[" not in h
    assert "biases[CONFIG_T::n_out]" not in h
    # The single public entry hls4ml calls is 2-arg.
    assert ("void gemm_stream_const_weights(hls::stream<data_T> &a_stream, "
            "hls::stream<res_T> &res_stream) {") in h
    # Each mvau IP specialization forwards 2-arg to its per-IP kernel.
    assert "gemm_fc1_gemm_stream_const_weights<data_T, res_T, CONFIG_T>(a, r);" in h
    assert "gemm_fc2_gemm_stream_const_weights<data_T, res_T, CONFIG_T>(a, r);" in h
    # The soft primary (behavioral/generic fallback) forwards 2-arg too.
    assert "gemm_soft_stream_const_weights<data_T, res_T, CONFIG_T>(a, r);" in h


def test_generic_head_rides_soft_primary_no_include():
    h = unified_combined_header(_LAYERS)
    # Behavioral (generic) layers get no per-IP include and no dispatch specialization;
    # they fall through to the soft primary.
    assert "gemm_head/gemm_head_gemm_ip.h" not in h
    assert "struct gemm_ip_dispatch<16>" not in h
    # The mvau IP layers do get includes + specializations.
    assert '#include "mvau/gemm_fc1/gemm_fc1_gemm_ip.h"' in h
    assert "struct gemm_ip_dispatch<14>" in h
