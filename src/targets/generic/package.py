"""Assemble a `generic` behavioral-HLS GEMM package for Vitis HLS.

No RTL blackbox: the package is plain synthesizable C++ that Vitis HLS turns into
RTL. Emits, per ``<name>``: nnet_types.h, <name>_gemm_ip.h (the four funcs),
<name>_config.h, <name>_top.cpp (set_top), <name>_tb.cpp (csim), <name>_weights.h
(weight-stationary ROM), and run_vitis.tcl (csim + csynth, no -blackbox).
"""

import sys
from pathlib import Path

_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

import hls as _hls  # noqa: E402
import golden as _golden  # noqa: E402

DEFAULT_PART = "xcvu13p-flga2577-2-e"


def _default_weight_matrix(k, n):
    """Deterministic B[K][N] matching the TB's b_val formula (const_weights standalone)."""
    return [[((row + 2 * col) % 3) - 1 for col in range(n)] for row in range(k)]


def _default_bias(n):
    """Deterministic bias matching the TB's bias_val formula (standalone csim)."""
    return [(col % 3) - 1 for col in range(n)]


def _run_vitis_tcl(name, part, clock_ns):
    return f"""open_project {name}_proj
set_top {name}
add_files {name}_top.cpp -cflags "-I."
add_files -tb {name}_tb.cpp -cflags "-I."
open_solution "sol1"
set_part {{{part}}}
create_clock -period {clock_ns} -name default
csim_design
csynth_design
exit
"""


def generate_generic_pkg(m, k, n, name, output_dir, interface="array",
                         weights_in_core=False, weight_matrix=None,
                         input_precision=None, weight_precision=None,
                         output_precision=None, bias_precision=None,
                         accum_precision=None, part=DEFAULT_PART, clock_period_ns=5,
                         strategy="latency", reuse_factor=1,
                         has_bias=None, bias=None,
                         **_ignored):
    if interface not in ("stream", "array"):
        raise ValueError(f"generic target: unsupported interface '{interface}'")
    pkg_dir = Path(output_dir) / name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Bias is always baked as a compile-time ROM (never a top-level port): the
    # manifest's real per-column values when has_bias is True, else a zero array
    # (standalone/unit generation with no manifest falls back to the TB's own
    # deterministic bias_val() formula, so the self-check stays meaningful).
    if has_bias is True and not bias:
        raise ValueError(
            "has_bias is True but the manifest has no bias values to bake "
            "(bias is missing/empty)")
    if bias:
        bias_values = bias
    elif has_bias is None:
        bias_values = _default_bias(n)
    else:
        bias_values = None
    # gemm_ip_header()'s gemm_ip_has_bias<0> trait: false only when has_bias is
    # explicitly False (the const-weight kernel then folds its bias add away
    # entirely, matching what the manifest's combined-header path does for the
    # same layer) -- has_bias=None (no manifest; falls back to _default_bias
    # above) and has_bias=True both keep the trait true.
    kernel_has_bias = has_bias is not False

    (pkg_dir / "nnet_types.h").write_text(_hls.nnet_types_header())
    (pkg_dir / f"{name}_gemm_ip.h").write_text(
        _hls.gemm_ip_header(name, strategy=strategy, has_bias=kernel_has_bias))
    if weights_in_core:
        B = weight_matrix if weight_matrix is not None else _default_weight_matrix(k, n)
        (pkg_dir / f"{name}_weights.h").write_text(_hls.weights_header(name, m, k, n, B))
    bias_t = _hls._ap_type(bias_precision, _hls._ap_type(output_precision, "ap_fixed<16,6>"))
    (pkg_dir / f"{name}_bias.h").write_text(_hls.bias_header(name, n, bias_t, bias_values))
    # config.h itself #includes <name>_bias.h (and, when weights_in_core,
    # <name>_weights.h) right after the typedefs they need, so its CONFIG_T can
    # expose gemm_bias() / gemm_weight_beats() -- write it after those two files'
    # *content* is decided above, though on-disk write order doesn't matter to the
    # preprocessor (see config_header()'s docstring).
    (pkg_dir / f"{name}_config.h").write_text(_hls.config_header(
        name, m, k, n,
        input_precision=input_precision, weight_precision=weight_precision,
        output_precision=output_precision, bias_precision=bias_precision,
        accum_precision=accum_precision, weights_in_core=weights_in_core,
        strategy=strategy, reuse_factor=reuse_factor))
    (pkg_dir / f"{name}_top.cpp").write_text(
        _hls.top_cpp(name, m, k, n, interface, weights_in_core))
    (pkg_dir / f"{name}_tb.cpp").write_text(
        _golden.tb_cpp(name, m, k, n, interface, weights_in_core, has_bias=kernel_has_bias))
    (pkg_dir / "run_vitis.tcl").write_text(_run_vitis_tcl(name, part, clock_period_ns))

    print(f"Generated {pkg_dir}  (generic vitis: M={m}, K={k}, N={n}, "
          f"interface={interface}, weights_in_core={weights_in_core}, "
          f"strategy={strategy}, reuse_factor={reuse_factor})")
    return pkg_dir
