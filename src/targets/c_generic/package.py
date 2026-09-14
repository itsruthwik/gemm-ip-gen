"""Assemble a `c-generic` behavioral-HLS GEMM package for Catapult.

The Catapult twin of ``v-generic/package.py``: same file *shape* (per-IP
``<name>_gemm_ip.h`` / ``_config.h`` / ``_bias.h`` [/ ``_weights.h``],
``<name>_top.cpp``, ``<name>_tb.cpp``, ``nnet_types.h``) plus
``gemm_ip_combined.h`` (the whole-model integration header hls4ml-gemm's
Catapult writer consumes -- a single-item combined header here since a
standalone package has exactly one layer), and ``run_catapult.tcl`` (solution
setup + ``go analyze`` .. ``go extract`` + SCVerify, modeled on
``tensor_slice/package.py``'s ``gen_tcl`` idiom -- matches the Xilinx library
and 5 ns clock hls4ml's Catapult build_prj.tcl uses for the real design flow,
so results are comparable).

Interface is always ``stream`` (``ac_channel``) -- the point of a Catapult
package is exercising the streaming beat idiom under SCVerify; the array
(io_parallel) entry points exist and are grepped/synthesis-checked by the
tool-free tests, but are not separately packaged here.
"""

import re
import subprocess
from pathlib import Path

from . import hls as _hls

DEFAULT_CLOCK_NS = 5.0


def _default_weight_matrix(k, n):
    """Deterministic B[K][N], matching the TB's b_val formula (two-operand)."""
    return [[((row + 2 * col) % 3) - 1 for col in range(n)] for row in range(k)]


def _default_bias(n):
    return [(col % 3) - 1 for col in range(n)]


def emit_behavioral(shape, name="gemm_c_generic", **kwargs):
    """The synthesizable kernel body only (shape-independent)."""
    return _hls.kernel_header(name)


def golden(shape, name="gemm_c_generic", weights_in_core=False, **kwargs):
    m, k, n = shape
    return _tb_cpp(name, m, k, n, weights_in_core=weights_in_core,
                    has_bias=kwargs.get("has_bias", True),
                    weights_row_major=kwargs.get("weights_row_major", False))


def _tb_cpp(name, m, k, n, weights_in_core, has_bias=True, weights_row_major=False):
    """ac_channel-driven self-checking TB: the same deterministic stimulus and
    0.5-tolerance golden compare golden_gemm.py's Vitis TB uses, adapted to
    Catapult's ac_channel blocking read/write and this target's beat types
    (``{name}_a_row_t`` / ``{name}_b_col_t`` / ``{name}_res_row_t``, defined
    identically to ``top_cpp``'s).
    """
    a_row_t, b_col_t, res_row_t = f"{name}_a_row_t", f"{name}_b_col_t", f"{name}_res_row_t"
    beat_typedefs = (
        f"typedef nnet::array<{name}_input_t, {k}> {a_row_t};\n"
        f"typedef nnet::array<{name}_result_t, {n}> {res_row_t};\n"
    )
    if not weights_in_core:
        beat_typedefs += f"typedef nnet::array<{name}_weight_t, {k}> {b_col_t};\n"

    if weights_in_core:
        proto = (f"void {name}(ac_channel<{a_row_t}> &a_stream, "
                  f"ac_channel<{res_row_t}> &res_stream);")
        b_ref = (f"{name}_config::gemm_weight_beats()[kk][nn].to_double()" if weights_row_major
                 else f"{name}_config::gemm_weight_beats()[nn][kk].to_double()")
        stim = f"""    ac_channel<{a_row_t}> a_stream;
    ac_channel<{res_row_t}> res_stream;
    for (int mm = 0; mm < {m}; mm++) {{
        {a_row_t} a_row;
        for (int kk = 0; kk < {k}; kk++) a_row[kk] = a_val(mm, kk);
        a_stream.write(a_row);
    }}
    CCS_DESIGN({name})(a_stream, res_stream);
"""
        bias_expr = "bias_val(nn)" if has_bias else "0.0"
        wl_label = " const_weights"
    else:
        proto = (f"void {name}(ac_channel<{a_row_t}> &a_stream, "
                  f"ac_channel<{b_col_t}> &b_stream, ac_channel<{res_row_t}> &res_stream);")
        b_ref = "b_val(kk, nn)"
        stim = f"""    ac_channel<{a_row_t}> a_stream;
    ac_channel<{b_col_t}> b_stream;
    ac_channel<{res_row_t}> res_stream;
    for (int mm = 0; mm < {m}; mm++) {{
        {a_row_t} a_row;
        for (int kk = 0; kk < {k}; kk++) a_row[kk] = a_val(mm, kk);
        a_stream.write(a_row);
    }}
    for (int nn = 0; nn < {n}; nn++) {{
        {b_col_t} b_col;
        for (int kk = 0; kk < {k}; kk++) b_col[kk] = b_val(kk, nn);
        b_stream.write(b_col);
    }}
    CCS_DESIGN({name})(a_stream, b_stream, res_stream);
"""
        # A two-operand GEMM never owns a bias (matches golden_gemm.py's rule).
        bias_expr = "0.0"
        wl_label = ""

    return f"""#include <cstdio>
#include <cmath>
#include <ac_channel.h>
#include <mc_scverify.h>
#include "nnet_types.h"
#include "{name}_gemm_ip.h"
#include "{name}_config.h"

{beat_typedefs}
{proto}

static double a_val(int mm, int kk) {{ return (double)(((mm + kk) % 3) - 1); }}
static double b_val(int kk, int nn) {{ return (double)(((kk + 2 * nn) % 3) - 1); }}
static double bias_val(int nn) {{ return {name}_config::gemm_bias()[nn].to_double(); }}

CCS_MAIN(int argc, char **argv) {{
{stim}
    {res_row_t} results[{m}];
    for (int mm = 0; mm < {m}; mm++) results[mm] = res_stream.read();

    int errors = 0;
    for (int mm = 0; mm < {m}; mm++) {{
        for (int nn = 0; nn < {n}; nn++) {{
            double golden = 0.0;
            for (int kk = 0; kk < {k}; kk++) golden += a_val(mm, kk) * ({b_ref});
            golden += {bias_expr};
            double got = results[mm][nn].to_double();
            if (std::fabs(got - golden) > 0.5) {{
                if (errors < 20)
                    std::printf("MISMATCH [%d][%d] got=%f exp=%f\\n", mm, nn, got, golden);
                errors++;
            }}
        }}
    }}
    if (errors == 0)
        std::printf("GENERIC CSIM PASS ({name} {m}x{k}x{n} stream{wl_label})\\n");
    else
        std::printf("GENERIC CSIM FAIL errors=%d\\n", errors);
    return errors ? 1 : 0;
}}
"""


def _run_catapult_tcl(name, clock_ns):
    """Solution/tech setup mirroring tensor_slice's ``gen_tcl`` (matches the
    Xilinx library and 5 ns clock hls4ml's Catapult build_prj.tcl uses for
    the real design flow, so results are comparable), but for a
    behavioral (non-blackbox) top: ``go analyze`` straight through
    ``go extract`` on ``{name}_top.cpp`` itself, then SCVerify.
    """
    return f"""\
set project_name "{name}_proj"
set solution_name "{name}_sol"

project new -name $project_name
solution new $solution_name
solution options defaults
solution options set /Output/OutputVerilog true
solution options set /Output/GenerateCycleNetlist false

# Turn on SCVerify (RTL cosim against the C++ testbench) before go analyze,
# same as Catapult's own SCVerify-flow examples.
flow package require /SCVerify

solution file add ./{name}_top.cpp -type C++
solution file add ./{name}_tb.cpp -type C++ -exclude true

directive set -DESIGN_GOAL area
directive set -SPECULATE true
directive set -MERGEABLE true
directive set -REGISTER_THRESHOLD 4096
directive set -MEM_MAP_THRESHOLD 4096
directive set -LOGIC_OPT false
directive set -FSM_ENCODING none
directive set -UNROLL no
directive set -IO_MODE super
directive set -CHAN_IO_PROTOCOL use_library
directive set -TIMING_CHECKS true

go new
solution design set {name} -top
go analyze
go compile

solution library add mgc_Xilinx-KINTEX-u-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family KINTEX-u -speed -2 -part xcku115-flvb2104-2-i
solution library add Xilinx_RAMS
solution library add Xilinx_ROMS
go libraries

directive set -CLOCKS {{clk {{-CLOCK_PERIOD {clock_ns} -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME {clock_ns / 2} -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND both -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low}}}}

go assembly
go architect
go allocate
go schedule
go extract

flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {{}} SIMTOOL=msim sim

project save
puts "{name} Catapult run complete."
"""


REQUIRED_FILES = [
    "gemm_ip_combined.h",
    "{name}_gemm_ip.h",
    "{name}_config.h",
    "{name}_bias.h",
    "{name}_top.cpp",
    "{name}_tb.cpp",
    "nnet_types.h",
    "run_catapult.tcl",
]


def generate_c_generic_pkg(shape, cfg):
    cfg = dict(cfg)
    name = cfg.pop("name")
    output_dir = cfg.pop("output_dir")
    m, k, n = shape

    interface = cfg.pop("interface", "stream")
    if interface != "stream":
        raise ValueError("generic (catapult) target: only the 'stream' package interface is supported")

    weight_matrix = cfg.pop("weight_matrix", None)
    weights_in_core = bool(cfg.pop("weights_in_core", weight_matrix is not None))
    weights_row_major = bool(cfg.pop("weights_row_major", False))
    if weights_in_core and weight_matrix is None:
        weight_matrix = _default_weight_matrix(k, n)

    reuse_factor = cfg.pop("reuse_factor", 1)
    input_precision = cfg.pop("input_precision", None)
    weight_precision = cfg.pop("weight_precision", None)
    output_precision = cfg.pop("output_precision", None)
    bias_precision = cfg.pop("bias_precision", None)
    accum_precision = cfg.pop("accum_precision", None)
    clock_period_ns = cfg.pop("clock_period_ns", DEFAULT_CLOCK_NS)
    has_bias = cfg.pop("has_bias", None)
    bias = cfg.pop("bias", None)
    cfg.pop("second_operand_row_major", None)
    cfg.pop("part", None)

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
    kernel_has_bias = has_bias is not False

    pkg_dir = Path(output_dir) / name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    input_t = _hls.ac_type(input_precision, "ac_fixed<16,6,true>")
    weight_t = _hls.ac_type(weight_precision, input_t)
    result_t = _hls.ac_type(output_precision, "ac_fixed<16,6,true>")
    bias_t = _hls.ac_type(bias_precision, result_t)

    (pkg_dir / "nnet_types.h").write_text(_hls.nnet_types_header())
    (pkg_dir / f"{name}_gemm_ip.h").write_text(_hls.kernel_header(name))
    if weights_in_core:
        (pkg_dir / f"{name}_weights.h").write_text(
            _hls.weights_header(name, m, k, n, weight_matrix, weights_row_major))
    bias_rom = bias_values if kernel_has_bias else [0] * n
    (pkg_dir / f"{name}_bias.h").write_text(_hls.bias_header(name, n, bias_t, bias_rom))
    (pkg_dir / f"{name}_config.h").write_text(_hls.config_header(
        name, m, k, n, input_precision=input_precision, weight_precision=weight_precision,
        output_precision=output_precision, bias_precision=bias_precision,
        accum_precision=accum_precision, weights_in_core=weights_in_core,
        reuse_factor=reuse_factor, weights_row_major=weights_row_major))
    (pkg_dir / f"{name}_top.cpp").write_text(
        _hls.top_cpp(name, m, k, n, interface="stream", weights_in_core=weights_in_core))
    (pkg_dir / f"{name}_tb.cpp").write_text(
        _tb_cpp(name, m, k, n, weights_in_core, has_bias=kernel_has_bias,
                weights_row_major=weights_row_major))

    # gemm_ip_combined.h: the same single layer, through the whole-model path
    # (this is what hls4ml-gemm's Catapult writer actually #includes).
    item = dict(
        name=name, gemm_m=m, gemm_k=k, gemm_n=n, reuse_factor=reuse_factor,
        gemm_ip_index=0, input_precision=input_precision, weight_precision=weight_precision,
        output_precision=output_precision, bias_precision=bias_precision,
        accum_precision=accum_precision, weights_row_major=weights_row_major,
        weight_matrix=weight_matrix if weights_in_core else None,
        weights_in_core=weights_in_core, bias_values=bias_values, has_bias=has_bias,
    )
    (pkg_dir / "gemm_ip_combined.h").write_text(_hls.combined_header([item]))

    (pkg_dir / "run_catapult.tcl").write_text(_run_catapult_tcl(name, clock_period_ns))

    print(f"Generated {pkg_dir}  (generic/catapult: M={m}, K={k}, N={n}, "
          f"weights_in_core={weights_in_core}, reuse_factor={reuse_factor})")
    return pkg_dir


def verify(package):
    """Run Catapult csim (via ``go compile``) + SCVerify RTL cosim on a
    generated package, then parse the schedule report for the actual
    multiplier count. Requires ``catapult`` on PATH.

    Returns a dict: ``{"ok": bool, "log": str, "multiplier_count": int|None}``.
    """
    import shutil
    pkg = Path(package)
    name = pkg.name
    missing = [f for f in REQUIRED_FILES
               if not (pkg / f.format(name=name)).is_file()
               or (pkg / f.format(name=name)).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"{name}: package incomplete, missing/empty: {missing}")
    if shutil.which("catapult") is None:
        print("ERROR: catapult not found on PATH", file=sys.stderr)
        return {"ok": False, "log": "", "multiplier_count": None, "skipped": True}

    r = subprocess.run(["catapult", "-shell", "-product", "hls", "-file", "run_catapult.tcl"],
                       cwd=str(pkg), text=True, capture_output=True, timeout=1800)
    log = r.stdout + r.stderr
    (pkg / "catapult_run.log").write_text(log)

    csim_ok = ("Error" not in log.split("go compile")[-1].split("go libraries")[0]
               if "go compile" in log else False)
    scverify_ok = "Simulation PASSED" in log
    mult_count = _parse_multiplier_count(pkg, name)
    ok = scverify_ok and (r.returncode == 0)
    if not ok:
        print(log[-6000:], file=sys.stderr)
    return {"ok": ok, "log": log, "multiplier_count": mult_count,
            "scverify_ok": scverify_ok, "csim_ok": csim_ok}


def _parse_multiplier_count(pkg, name):
    """Sum the multiplier instance counts out of Catapult's ``rtl.rpt`` Bill
    Of Materials (the last numeric column on each multiplier BOM row is its
    instance count in the extracted netlist).

    Matches both ``mgc_mul(...)`` (the plain-multiplier component the
    nangate ASIC library maps to) and ``mgc_muladd1(...)`` (the fused
    multiply-add component the Xilinx library maps the same scheduled
    product terms to) -- same underlying multiplier count, different
    component name per target library.

    Returns ``0`` (not ``None``) when the report has no such row at all --
    a real, expected outcome for const-weight layers: the weight ROM is
    baked as compile-time constants, so Catapult's constant propagation
    folds "multiply by a known constant" into adder/shift networks (and, for
    larger ROMs, small lookup-table components) instead of instantiating a
    generic multiplier, even though the *emitter's* scheduled product-term
    count (``multiplier_limit``) is unchanged by that fold. Only genuinely
    unknown-at-synthesis operands (the two-operand ``gemm_stream``/
    ``gemm_array`` entries, B arriving at run time) leave behind real
    multiplier instances to count here.
    """
    reports = sorted(pkg.glob(f"{name}_proj/{name}_sol.v*/rtl.rpt"))
    if not reports:
        return None
    text = reports[-1].read_text()
    total = 0
    found = False
    for line in text.splitlines():
        m = re.match(r"\s*mgc_mul\w*\([^)]*\)\s+.*?(\d+)\s+(\d+)\s*$", line)
        if m:
            found = True
            total += int(m.group(2))
    return total if found else 0


def rtl_test(package=None, **kwargs):
    """SCVerify RTL cosim is the RTL test here (no separate blackbox); same as
    ``verify()``."""
    if package is None:
        raise ValueError("rtl_test(package=...) needs a generated package dir")
    result = verify(package)
    return 0 if result.get("ok") else (2 if result.get("skipped") else 1)
