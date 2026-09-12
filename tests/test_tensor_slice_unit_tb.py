"""Compile-and-run tests for the standalone unit testbench tensor_slice emits
alongside every generated package (`<name>_tb.cpp` / `<name>_inst.cpp`, see
package.py's gen_tb / gen_inst_cpp).

These call the header's real entry points with g++ (no Catapult), so a
package can be checked without the hls4ml/Catapult flow. Skips cleanly if
g++ or the AC datatype headers aren't available in this environment.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from test_tensor_slice_operand_guard import _load_tensor_slice_package
from test_tensor_slice_rf import _with_tensor_slice_on_path

_generate_catapult_pkg = _load_tensor_slice_package().generate_catapult_pkg

AC_INCLUDE = "/home/tools/siemens/catapult/Mgc_home/shared/include"
LD_LIBRARY_PATH_EXTRA = (
    "/home/tools/siemens/catapult/Mgc_home/pkgs/dcs_gcc/gcc-13.4.0/lib64"
)


def _gxx():
    return shutil.which("g++")


def _skip_if_no_toolchain():
    if _gxx() is None:
        pytest.skip("g++ not found on PATH")
    if not Path(AC_INCLUDE, "ac_int.h").is_file():
        pytest.skip(f"AC datatype headers not found under {AC_INCLUDE}")


def generate_catapult_pkg(*args, **kwargs):
    return _with_tensor_slice_on_path(_generate_catapult_pkg, *args, **kwargs)


def _compile_and_run(pkg_dir, name):
    """Compile <name>_tb.cpp + <name>_inst.cpp with g++ and run the binary.

    Returns (returncode, stdout+stderr).
    """
    tb = pkg_dir / f"{name}_tb.cpp"
    inst = pkg_dir / f"{name}_inst.cpp"
    exe = pkg_dir / "tb"
    cmd = [
        _gxx(), "-std=c++17",
        f"-I{pkg_dir}", f"-I{AC_INCLUDE}",
        str(tb), str(inst), "-o", str(exe),
    ]
    compile_res = subprocess.run(cmd, capture_output=True, text=True)
    assert compile_res.returncode == 0, (
        f"g++ failed for {name}:\ncmd: {' '.join(cmd)}\n"
        f"stdout:\n{compile_res.stdout}\nstderr:\n{compile_res.stderr}"
    )
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH_EXTRA + ":" + env.get("LD_LIBRARY_PATH", "")
    run_res = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
    return run_res.returncode, run_res.stdout + run_res.stderr


def _random_weight_matrix(k, n, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(-64, 64, size=(k, n)).astype(int)


def _nonuniform_bias(n, scale=0.05):
    """A real-valued, non-zero, non-uniform bias vector, small enough that
    its baked 16-bit stage-2 intermediate code always fits (see
    generate_catapult_pkg's own bias-code-fits check)."""
    return list((np.arange(n) - (n - 1) / 2.0) * scale)


# Each case: (case_id, gen kwargs). m/k/n and reuse_factor chosen so the
# accum_precision default keeps S1 == 0 (see generate_catapult_pkg: S1 is
# only forced above 0 when accum_precision asks for more than 16 bits of
# gemm-scale range; none of these cases set accum_precision, so S1 == 0 and
# the tb's own single-round golden -- which explicitly assumes S1 == 0 --
# matches the two-stage DUT exactly).
_TWO_OPERAND_CASES = [
    pytest.param("two_op_stream_k", dict(m=8, k=8, n=8, interface="stream", fold_axis="k"), id="two_op_stream_k"),
    pytest.param("two_op_array_k", dict(m=8, k=8, n=8, interface="array", fold_axis="k"), id="two_op_array_k"),
    pytest.param("two_op_stream_m", dict(m=16, k=8, n=8, interface="stream", fold_axis="m", reuse_factor=2), id="two_op_stream_m"),
    pytest.param("two_op_array_m", dict(m=16, k=8, n=8, interface="array", fold_axis="m", reuse_factor=2), id="two_op_array_m"),
    pytest.param("two_op_stream_n", dict(m=8, k=8, n=16, interface="stream", fold_axis="n", reuse_factor=2), id="two_op_stream_n"),
    pytest.param("two_op_array_n", dict(m=8, k=8, n=16, interface="array", fold_axis="n", reuse_factor=2), id="two_op_array_n"),
]

# Baked-weights (weight_matrix given) cases, each with a real non-uniform,
# non-zero bias vector baked into the core.
_BAKED_CASES = [
    pytest.param("baked_stream_k", dict(m=8, k=8, n=8, interface="stream", fold_axis="k"), id="baked_stream_k"),
    pytest.param("baked_array_k", dict(m=8, k=8, n=8, interface="array", fold_axis="k"), id="baked_array_k"),
    pytest.param("baked_stream_m", dict(m=16, k=8, n=16, interface="stream", fold_axis="m", reuse_factor=2), id="baked_stream_m"),
    pytest.param("baked_array_m", dict(m=16, k=8, n=16, interface="array", fold_axis="m", reuse_factor=2), id="baked_array_m"),
    pytest.param("baked_stream_n", dict(m=8, k=8, n=32, interface="stream", fold_axis="n", reuse_factor=2), id="baked_stream_n"),
    pytest.param("baked_stream_n_2rows", dict(m=16, k=8, n=32, interface="stream", fold_axis="n", reuse_factor=2), id="baked_stream_n_2rows"),
    pytest.param("baked_array_n", dict(m=8, k=8, n=32, interface="array", fold_axis="n", reuse_factor=2), id="baked_array_n"),
]


@pytest.mark.parametrize("name,kwargs", _TWO_OPERAND_CASES)
def test_two_operand_unit_tb(tmp_path, name, kwargs):
    _skip_if_no_toolchain()
    generate_catapult_pkg(
        name=name, output_dir=str(tmp_path),
        output_precision="fixed<10,3>",
        input_precision="fixed<8,2>", weight_precision="fixed<8,2>",
        **kwargs,
    )
    pkg_dir = tmp_path / name
    rc, out = _compile_and_run(pkg_dir, name)
    assert rc == 0, f"{name} testbench exited {rc}:\n{out}"
    assert "Test passed" in out, f"{name}: PASS line missing:\n{out}"


@pytest.mark.parametrize("name,kwargs", _BAKED_CASES)
def test_baked_weights_unit_tb(tmp_path, name, kwargs):
    _skip_if_no_toolchain()
    n = kwargs["n"]
    k = kwargs["k"]
    weight_matrix = _random_weight_matrix(k, n, seed=1)
    bias = _nonuniform_bias(n)
    generate_catapult_pkg(
        name=name, output_dir=str(tmp_path),
        output_precision="fixed<10,3>",
        input_precision="fixed<8,2>", weight_precision="fixed<8,2>",
        weight_matrix=weight_matrix, has_bias=True, bias=bias,
        bias_precision="fixed<16,8>",
        **kwargs,
    )
    pkg_dir = tmp_path / name
    rc, out = _compile_and_run(pkg_dir, name)
    assert rc == 0, f"{name} testbench exited {rc}:\n{out}"
    assert "Test passed" in out, f"{name}: PASS line missing:\n{out}"
