"""Mixed per-layer-target designs: the whole-model header is the union of the per-target
combined headers. Every io_stream GEMM node is called by its own name
(``gemm_stream_<layer>``) and each target defines only its own layers' functions, so
no dispatch is needed and the per-target headers just need distinct include guards."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemm_ip.combine import unified_combined_header  # noqa: E402
from targets.v_generic import hls as ghls  # noqa: E402
from targets.mvau import package as mpkg  # noqa: E402

_LAYERS = [
    {"name": "gemm_fc1", "id": 14, "target": "mvau"},
    {"name": "gemm_fc2", "id": 15, "target": "mvau"},
    {"name": "gemm_head", "id": 16, "target": "generic"},
]


def test_unified_header_is_the_union_of_per_target_headers():
    h = unified_combined_header(_LAYERS)
    assert h.count('#include "mvau/gemm_ip_combined.h"') == 1
    assert h.count('#include "generic/gemm_ip_combined.h"') == 1
    # no per-id dispatch survives: the call sites use the node names directly
    assert "gemm_ip_dispatch" not in h and "gemm_ip_id" not in h


def test_per_target_headers_have_distinct_guards():
    generic = ghls.combined_header([])
    mvau = mpkg.gen_combined_header([])
    assert "#ifndef GEMM_IP_COMBINED_GENERIC_H_" in generic
    assert "#ifndef GEMM_IP_COMBINED_MVAU_H_" in mvau
    assert "GEMM_IP_COMBINED_H_" not in generic.replace("GEMM_IP_COMBINED_GENERIC_H_", "")
    assert "GEMM_IP_COMBINED_H_" not in mvau.replace("GEMM_IP_COMBINED_MVAU_H_", "")


def test_generic_defines_packed_entry_per_io_stream_layer():
    items = [
        {"name": "gemm_head", "gemm_ip_index": 16, "interface": "stream", "k": 16, "n": 10,
         "input_precision": "fixed<8,2>", "output_precision": "fixed<16,6>",
         "weights_in_core": True, "reuse_factor": 2, "has_bias": False},
        {"name": "gemm_qk", "gemm_ip_index": 17, "interface": "stream", "k": 8, "n": 4,
         "input_precision": "fixed<8,2>", "weight_precision": "fixed<8,2>",
         "output_precision": "fixed<16,8>", "weights_in_core": False,
         "second_operand_row_major": False, "reuse_factor": 1, "has_bias": False},
    ]
    h = ghls.combined_header(items)
    assert '#include "nnet_utils/nnet_gemm_pack.h"' in h
    assert ("inline void gemm_stream_gemm_head(hls::stream<ap_uint<nnet::gemm_packed_bits<"
            "nnet::array<ap_fixed<8,2>, 16>>::value> > &a," in h)
    assert "nnet::gemm_stream_packed_const_weights<nnet::array<ap_fixed<8,2>, 16>, nnet::array<ap_fixed<16,6>, 10>, config16>(a, p);" in h
    # two-operand, col-major B: one K-wide column per beat
    assert "nnet::gemm_stream_packed<nnet::array<ap_fixed<8,2>, 8>, nnet::array<ap_fixed<8,2>, 8>, nnet::array<ap_fixed<16,8>, 4>, config17>(a, b, p);" in h
