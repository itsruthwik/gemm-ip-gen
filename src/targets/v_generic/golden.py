"""Thin re-export shim: the real module now lives at ``gemm_ip.golden_gemm``
(tool-neutral, shared with the Catapult ``c-generic`` target). Kept here,
under the bare name ``golden`` that ``v-generic``'s own ``flow.py`` /
``package.py`` import as a sibling module, so neither needed to change.
"""

from gemm_ip.golden_gemm import *  # noqa: F401,F403
from gemm_ip.golden_gemm import tb_cpp  # noqa: F401
