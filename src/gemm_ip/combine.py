"""Unified combined header for ATLAS mixed per-layer-target designs.

When a single model routes different GEMM layers to different targets (e.g. mvau on
some layers, generic on others), each target still emits its own self-contained
``gemm_ip_combined.h`` for the single-target case. But a mixed design needs ONE header
that the firmware includes, routing every ``nnet::gemm_stream_const_weights<..,configN>``
call to the layer's owning IP by ``CONFIG_T::gemm_ip_id`` (== node index == the
manifest's ``gemm_ip_index``).

This module builds that header: the behavioral/soft blanket (reused verbatim from the
generic target, with the const_weights entry renamed) is the dispatch *primary* — so any
behavioral/generic layer falls through to soft logic — and each non-behavioral IP
``#include``s its per-IP header and specializes the dispatch to its own
``<name>_gemm_stream_const_weights`` entry.

Historically the merge just ``#include``d each target's ``gemm_ip_combined.h``; they
share the guard ``GEMM_IP_COMBINED_H_`` (so the second was skipped) and both defined
``gemm_stream_const_weights`` (an ODR clash) — the blackbox IP was never actually called.
"""

# Reuse the Vitis generic target's soft compute templates verbatim (single source
# of truth) -- the only tool a mixed per-layer design can combine with an IP target
# today (mvau is Vitis-only too).
from targets.v_generic import hls as _ghls

# Target names whose layers are behavioral (no per-IP blackbox header) and so ride
# the soft dispatch primary rather than a specialization. There is now exactly one
# user-facing behavioral target name ("generic") regardless of tool (vitis/catapult);
# kept as a set (rather than a bare string) so a future second behavioral name is a
# one-line add, not a rework.
_BEHAVIORAL_TARGETS = {"generic"}


def unified_combined_header(layers):
    """Build the one combined header for a mixed-target design.

    layers: ordered list of dicts, each ``{"name": str, "id": int, "target": str,
    "has_bias": bool, "func": str}`` (id is the manifest ``gemm_ip_index``; has_bias
    keys the soft body's ``gemm_ip_has_bias`` trait). Behavioral-target layers route to the soft
    primary; every other layer gets an ``#include`` of its per-IP header and a
    ``gemm_ip_dispatch<id>`` specialization forwarding to ``<name>_gemm_stream_const_weights``.
    """
    # Soft blanket, const_weights entry renamed so the single public entry is the dispatcher.
    soft = _ghls._GEMM_IP_COMBINED_FUNCS.replace(
        "gemm_stream_const_weights", "gemm_soft_stream_const_weights")

    ip_layers = [l for l in layers if l["target"] not in _BEHAVIORAL_TARGETS]
    includes = "".join(
        f'#include "{l["target"]}/{l["name"]}/{l["name"]}_gemm_ip.h"\n' for l in ip_layers)

    # const_weights (one-operand) layers specialize the const_weights dispatcher; two-operand
    # (func == "stream") layers specialize the two-operand dispatcher (see below). A layer's
    # func defaults to const_weights so existing single-operand manifests are unchanged.
    wl_layers = [l for l in ip_layers if l.get("func", "stream_const_weights") != "stream"]
    two_op_layers = [l for l in ip_layers if l.get("func") == "stream"]

    # Bias, when a const-weights layer has one, is baked as a compile-time constant inside
    # its per-IP header (mvau) or read from CONFIG_T::gemm_bias() and gated by the
    # gemm_ip_has_bias trait (generic soft) -- never a call argument. So every
    # <name>_gemm_stream_const_weights takes exactly (a, r), matching hls4ml's 2-arg call.
    specs = "\n".join(
        f"template <> struct gemm_ip_dispatch<{l['id']}> {{\n"
        f"    template <class data_T, class res_T, typename CONFIG_T>\n"
        f"    static void stream_const_weights(hls::stream<data_T> &a, hls::stream<res_T> &r) {{\n"
        f"        {l['name']}_gemm_stream_const_weights<data_T, res_T, CONFIG_T>(a, r);\n"
        f"    }}\n"
        f"}};"
        for l in wl_layers)

    # Two-operand (runtime-B) GEMM never has a real bias, so the call site never passes one.
    two_op_specs = "\n".join(
        f"template <> struct gemm_ip_stream_dispatch<{l['id']}> {{\n"
        f"    template <class data0_T, class data1_T, class res_T, typename CONFIG_T>\n"
        f"    static void stream(hls::stream<data0_T> &a, hls::stream<data1_T> &b, hls::stream<res_T> &r) {{\n"
        f"        {l['name']}_gemm_stream<data0_T, data1_T, res_T, CONFIG_T>(a, b, r);\n"
        f"    }}\n"
        f"}};"
        for l in two_op_layers)

    # The two-operand dispatcher + public entry are only emitted when a two-operand IP layer
    # exists (no soft two-operand primary today, so an unrouted id would be a compile error --
    # which is the correct signal that a gemm_stream layer wasn't given an IP).
    two_op_block = "" if not two_op_layers else (
        "\ntemplate <int ID> struct gemm_ip_stream_dispatch;\n"
        f"{two_op_specs}\n\n"
        "template <class data0_T, class data1_T, class res_T, typename CONFIG_T>\n"
        "void gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,\n"
        "                 hls::stream<res_T> &res_stream) {\n"
        "    gemm_ip_stream_dispatch<CONFIG_T::gemm_ip_id>::template stream<data0_T, data1_T, res_T, CONFIG_T>(\n"
        "        a_stream, b_stream, res_stream);\n"
        "}\n")

    # The soft primary (generic fallback) is built from _GEMM_IP_COMBINED_FUNCS (the
    # generic target's resource core), whose kernels reference gemm_rf<CONFIG_T> and
    # gemm_ip_has_bias<CONFIG_T::gemm_ip_id>. The single-target generic combined_header
    # prepends both traits before that body; the mixed merge must do the same, keyed on
    # each layer's manifest fields.
    trait_items = [{"gemm_ip_index": l["id"], "has_bias": l.get("has_bias"),
                    "reuse_factor": l.get("reuse_factor"),
                    "gemm_k": l.get("gemm_k"), "gemm_n": l.get("gemm_n")} for l in layers]
    rf_trait = _ghls._gemm_rf_trait(trait_items)
    has_bias_trait = _ghls._gemm_has_bias_trait(trait_items)

    return (
        "#ifndef GEMM_IP_COMBINED_H_\n"
        "#define GEMM_IP_COMBINED_H_\n\n"
        "// Unified per-config-id GEMM IP dispatch (ATLAS mixed per-layer targets).\n"
        "// Routes nnet::gemm_stream_const_weights<..,CONFIG_T> to each layer's IP by\n"
        "// CONFIG_T::gemm_ip_id; behavioral/generic layers use the soft primary.\n"
        "#include <hls_stream.h>\n"
        f"{rf_trait}\n"
        f"{has_bias_trait}\n"
        f"{soft}\n"
        f"{includes}\n"
        "namespace nnet {\n\n"
        "template <int ID> struct gemm_ip_dispatch {  // primary: behavioral/soft fallback\n"
        "    template <class data_T, class res_T, typename CONFIG_T>\n"
        "    static void stream_const_weights(hls::stream<data_T> &a, hls::stream<res_T> &r) {\n"
        "        gemm_soft_stream_const_weights<data_T, res_T, CONFIG_T>(a, r);\n"
        "    }\n"
        "};\n"
        f"{specs}\n\n"
        "// The single io_stream const_weights entry hls4ml calls; routes by config id.\n"
        "// Bias is never a call argument (soft: CONFIG_T::gemm_bias() gated by the has_bias\n"
        "// trait; IP: baked into the core), matching hls4ml's 2-arg gemm call.\n"
        "template <class data_T, class res_T, typename CONFIG_T>\n"
        "void gemm_stream_const_weights(hls::stream<data_T> &a_stream, hls::stream<res_T> &res_stream) {\n"
        "    gemm_ip_dispatch<CONFIG_T::gemm_ip_id>::template stream_const_weights<data_T, res_T, CONFIG_T>(\n"
        "        a_stream, res_stream);\n"
        "}\n"
        f"{two_op_block}\n"
        "} // namespace nnet\n"
        "#endif // GEMM_IP_COMBINED_H_\n"
    )
