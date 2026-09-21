"""The combined-fold core's drain-side frame index must wrap to group 0 after the
last group. It picks the emitted frame's N-group bias, so holding the last group
gave every inference after the first the last group's bias on all of its groups."""
import re
from pathlib import Path

import numpy as np

from test_tensor_slice_operand_guard import _pkg as _tensor_slice_pkg
from test_tensor_slice_rf import _with_tensor_slice_on_path


def test_drain_frame_wraps_to_the_first_group(tmp_path):
    name = "t"
    rng = np.random.default_rng(0)
    _with_tensor_slice_on_path(
        _tensor_slice_pkg.generate_catapult_pkg,
        m=1, k=64, n=32, name=name, output_dir=str(tmp_path), interface="stream",
        input_precision="fixed<8,4>", weight_precision="fixed<8,1>",
        output_precision="fixed<15,6,RND,WRAP,0>", bias_precision="fixed<8,1>",
        has_bias=True, bias=list(rng.integers(-30, 30, 32) / 128.0),
        weight_matrix=rng.integers(-8, 9, size=(64, 32)),
        k_reuse_factor=2, n_reuse_factor=2,
    )
    core = (Path(tmp_path) / name / f"{name}_core.v").read_text()
    update = re.search(r"drain_frame <= \(drain_frame \+ 16'd1 == 16'd(\d+)\) \?\s*(\S+) :", core)
    assert update, "drain_frame update not found in the combined-fold core"
    assert update.group(1) == "2"          # two N groups per inference
    assert update.group(2) == "16'd0"      # wraps, does not hold the last group
