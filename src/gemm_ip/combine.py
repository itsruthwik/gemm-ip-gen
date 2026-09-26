"""Unified combined header for ATLAS mixed per-layer-target designs.

When a single model routes different GEMM layers to different targets (e.g. mvau on
some layers, generic on others), each target emits its own ``gemm_ip_combined.h``
under ``pkg/<target>/``. Since every io_stream GEMM node is called by its own name,
``gemm_stream_<layer>``, and each target defines only its own layers' functions,
the whole-model header is simply the union: one include per target. The per-target
headers carry target-specific include guards so both survive the union.
"""


def unified_combined_header(layers):
    """``layers``: ordered ``[{"name": str, "id": int, "target": str, ...}]``; only the
    distinct targets matter here (each target's header already defines that target's
    layers' ``gemm_stream_<name>`` functions)."""
    targets = []
    for l in layers:
        if l["target"] not in targets:
            targets.append(l["target"])
    incs = "".join(f'#include "{t}/gemm_ip_combined.h"\n' for t in targets)
    return (
        "#ifndef GEMM_IP_COMBINED_H_\n"
        "#define GEMM_IP_COMBINED_H_\n\n"
        "// Mixed per-layer targets: each io_stream GEMM node is called by its own name\n"
        "// (gemm_stream_<layer>), so the whole-model header is the union of the per-target\n"
        "// headers, each defining only its own layers' functions.\n"
        f"{incs}"
        "#endif // GEMM_IP_COMBINED_H_\n"
    )
