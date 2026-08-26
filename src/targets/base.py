"""The Target contract for gemm-ip-gen.

A *target* is one hardblock welded to exactly one HLS tool — one tool per
hardblock, so there is no hardblock×tool matrix and no shared protocol/shim
layer. Each target lives in its own directory under ``src/targets/`` and exposes
this interface so the thin framework core (and any tooling) can drive it without
knowing hardblock specifics. Adding a hardblock is a new ``src/targets/<name>/``
directory whose ``flow.py`` provides a ``Target`` and a name entry in the
registry — the core and other targets are untouched.

``shape`` is the GEMM shape ``(m, k, n)`` throughout.
"""

from abc import ABC, abstractmethod


class Target(ABC):
    #: short target name, e.g. "tensor_slice"
    name = None
    #: the single HLS tool this hardblock is welded to, e.g. "catapult"
    tool = None

    @abstractmethod
    def geometry(self, shape):
        """Tile geometry for *shape* (grid rows/cols, k chunks, stream widths, latency)."""

    @abstractmethod
    def emit_rtl(self, shape, **kwargs):
        """Return the synthesizable (structural) Verilog for *shape*."""

    @abstractmethod
    def emit_behavioral(self, shape, **kwargs):
        """Return the behavioral (simulation) Verilog for *shape*."""

    @abstractmethod
    def golden(self, shape, **kwargs):
        """Return a self-checking Verilog testbench for *shape*."""

    @abstractmethod
    def package(self, shape, cfg):
        """Generate a full blackbox package on disk.

        *cfg* is a mapping carrying ``name`` and ``output_dir`` plus any
        target-specific options (interface, precisions, weight matrix, …).
        """

    @abstractmethod
    def verify(self, package):
        """Structurally verify a generated package (a directory path). Raise on failure."""

    @abstractmethod
    def rtl_test(self, **kwargs):
        """Run the target's RTL-level regression; return a process-style code (0 = pass)."""

    def sources_tcl(self, items):
        """Return the tcl the HLS tool sources to bring this target's IP into the
        project. This is the seam that lets the frontend stay target-blind: it emits a
        single ``source $tcldir/gemm_pkg/gemm_ip_sources.tcl`` line, and the target
        decides what goes in it — ``add_files -blackbox <wrapper.json>`` for an
        RTL-blackbox target, or nothing (header-only) for a behavioral-HLS one. The
        default is a no-op suitable for header-only targets."""
        return "# no IP sources to add (header-only target)\n"

    def __repr__(self):
        return f"<Target {self.name!r} tool={self.tool!r}>"
