"""cmvu target: binds the CMVU hardblock to the Target contract.

Delegates each contract method to the sibling modules -- ``geometry`` (fold
legalization + slot map), ``rtl`` (the generated wrapper), ``golden`` (oracle +
self-checking TB), ``package`` (the full Catapult blackbox package), and
``run_rtl_tests`` (the Icarus/VCS pre-filter regression). Heavy modules are
imported lazily so importing this module stays cheap.

Design locks that live here rather than in the generators:

* KFold + NFold only, both required (no tensor_slice ``fold_axis``/per-axis
  fold knobs). The raw config is checked before ``gemm_ip.config`` normalizes
  it, so a tensor_slice knob can never be silently ignored. ``reuse_factor`` is
  accepted and ignored: the hls4ml manifest carries it on every GEMM layer as
  hls4ml's own ReuseFactor, which is independent of the IP fold knobs.
* Stream interface only (``interface="array"`` -> ``ValueError``).
* Requant shift/signedness derived from the layer precisions via
  ``gemm_ip.quant``; cmvu's requant is a plain truncating (floor) shift, and
  RND output is reproduced by folding the rounding constant into the 32-bit
  bias (``golden.bias_codes``). int8 operand codes only. SAT/SAT_SYM output
  overflow and any round mode other than RND/TRN are rejected at package
  time (see ``_output_rounds``); so is a bias not exactly representable at
  the product's fractional precision, and an ``accum_precision`` narrower
  than input-frac + weight-frac.
* Effective result width ``W`` (1..16) derived from ``output_precision``,
  defaulting to the physical ``RESULT_WIDTH`` (16) when unset; lanes are always
  sign-extended, even for unsigned ``output_precision`` -- there is no
  zero-extend path.
* Column-major runtime-B needs K_PASSES<=2 (the load FSM pairs at most two
  resident k-passes per column beat); checked here, up front, so a config
  that needs a third resident k-pass fails at package time with a
  ``ValueError`` naming the layer, not as a ``NotImplementedError`` deep in
  RTL generation.
* Every rejection above (and the ones in ``geometry.resolve_geometry`` --
  missing/illegal KFold/NFold, per-block slot-budget oversubscription) is a
  ``ValueError`` naming the layer and the reason, raised before any RTL is
  generated -- never a ``NotImplementedError``, and never buried in
  ``rtl.generate_core``.
"""

import re
import sys
import zlib
from pathlib import Path

from ..base import Target
from . import geometry as _geom
from . import rtl as _rtl
from . import golden as _golden


# tensor_slice fold knobs. Their presence in the raw config means the caller
# is using the wrong target contract, not that they are unset.
_LEGACY_KEYS = ("fold_axis", "m_reuse_factor", "k_reuse_factor",
                "n_reuse_factor")


def _package():
    from . import package
    return package


def _run_rtl_tests():
    from . import run_rtl_tests
    return run_rtl_tests


def _raw_items(cfg):
    """Yield the raw config item dicts in the order ``_normalize_config_items``
    emits them (list, single-item shorthand, or named dict)."""
    if isinstance(cfg, list):
        return list(cfg)
    if isinstance(cfg, dict):
        if "m" in cfg and "k" in cfg and "n" in cfg and "name" in cfg:
            return [cfg]
        return list(cfg.values())
    raise TypeError(f"Unsupported config format: {type(cfg)}")


def _synth_weights(k, n, name):
    """Deterministic int8 weights for the standalone CLI (no ``weight_file``).

    The hls4ml integration always supplies a real weight matrix; this only keeps
    ``--target cmvu`` usable without one, with a name-stable seed so repeated
    runs produce identical packages."""
    import numpy as np
    seed = zlib.crc32(str(name or "cmvu").encode()) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, size=(int(k), int(n)), dtype=np.int64)


_OK_ROUND_MODES = ("RND", "TRN")
_BAD_OVERFLOW_MODES = ("SAT_SYM", "SAT")  # order matters: SAT_SYM contains SAT


def _precision_mode_fields(precision):
    """(round_mode, overflow_mode) parsed from 'fixed<W,I,round,overflow,...>'.

    Either field is ``None`` when absent/unparseable (an unset precision, or a
    precision with no mode fields) -- callers apply today's defaults (RND
    round, WRAP overflow) in that case. cmvu has no helper for this in
    ``gemm_ip.quant``, so it is parsed locally.
    """
    if not precision:
        return (None, None)
    m = re.search(r"u?fixed<\s*\d+\s*,\s*-?\d+\s*(?:,\s*(\w+))?\s*(?:,\s*(\w+))?", str(precision))
    if not m:
        return (None, None)
    return (m.group(1), m.group(2))


def _output_rounds(output_precision, name):
    """True (RND) / False (TRN) requant mode for the layer's output precision.

    Rejects any other explicit round mode (``RND_CONV``, ``RND_ZERO``, etc.)
    and any explicit saturating overflow mode (``SAT``/``SAT_SYM``) -- the
    block only wraps. An unset mode field defaults to RND (today's behavior);
    an unset overflow field defaults to WRAP (today's behavior).
    """
    rmode, omode = _precision_mode_fields(output_precision)
    if rmode is not None and rmode.upper().lstrip("AC_") not in _OK_ROUND_MODES:
        raise ValueError(
            f"layer '{name}': cmvu output rounding mode must be RND or TRN "
            f"(got {rmode!r} in output_precision {output_precision!r}); the "
            f"block only truncates or rounds via a folded bias.")
    if omode is not None:
        upper = omode.upper()
        if any(bad in upper for bad in _BAD_OVERFLOW_MODES):
            raise ValueError(
                f"layer '{name}': cmvu output overflow mode must not be "
                f"SAT/SAT_SYM (got {omode!r} in output_precision "
                f"{output_precision!r}); the block only wraps.")
    return rmode is None or rmode.upper().lstrip("AC_") == "RND"


def _derive_shift(input_precision, weight_precision, output_precision,
                  accum_precision=None, name=None):
    from gemm_ip.quant import _frac_bits
    in_frac = _frac_bits(input_precision)
    w_frac = _frac_bits(weight_precision)
    out_frac = _frac_bits(output_precision)
    product_frac = in_frac + w_frac
    if accum_precision is not None:
        accum_frac = _frac_bits(accum_precision)
        # cmvu accumulates exact products in int32, so accum_precision does not
        # change what the block computes. A narrower hls4ml accumulator rounds
        # every product first, so its reference can differ by about an output LSB.
        if accum_frac < product_frac:
            print(f"WARNING: layer '{name}': accum_precision {accum_precision!r} "
                  f"(frac {accum_frac}) is narrower than the product frac "
                  f"({product_frac}); cmvu accumulates exact products in 32 bits, "
                  f"so results can differ from hls4ml's rounded accumulator.",
                  file=sys.stderr)
    shift = product_frac - out_frac
    if not (0 <= shift <= _geom.MAX_SHIFT):
        raise ValueError(
            f"layer '{name}': cmvu requant shift {shift} (input frac "
            f"{in_frac} + weight frac {w_frac} - output frac "
            f"{out_frac}) is outside 0..{_geom.MAX_SHIFT}.")
    return shift, product_frac


def _check_bias_exact(bias, product_frac, name):
    """Reject a bias that is not exactly representable at ``product_frac``.

    hls4ml adds the bias into ``accum_t`` at the product's fractional
    precision; a bias with bits below that LSB cannot be baked exactly into
    the int32 accumulator-scale code cmvu uses.
    """
    import numpy as np
    scale = 1 << int(product_frac)
    for lane, value in enumerate(np.asarray(bias, dtype=np.float64).reshape(-1)):
        scaled = float(value) * scale
        if abs(scaled - round(scaled)) > 1e-6:
            raise ValueError(
                f"layer '{name}': bias lane {lane} = {value} is not exactly "
                f"representable at frac {product_frac} (input frac + weight "
                f"frac); cmvu bias must have no bits below the accumulator LSB.")


def _output_width(precision, name=None):
    """Effective result-lane width W from ``output_precision``.

    Defaults to the physical ``RESULT_WIDTH`` (16) when the precision is
    unset/unparseable; cmvu lanes are at most ``RESULT_WIDTH``, so a wider
    precision is an error rather than a silent truncation.
    """
    if precision:
        m = re.search(r"u?(?:ac_)?(?:fixed|int)<\s*(\d+)", str(precision))
        if m:
            width = int(m.group(1))
            if not (1 <= width <= _geom.RESULT_WIDTH):
                raise ValueError(
                    f"layer '{name}': cmvu result width must be "
                    f"1..{_geom.RESULT_WIDTH}; output_precision {precision!r} "
                    f"is {width}-bit.")
            return width
    return _geom.RESULT_WIDTH


def _operand_signed(precision, what):
    from gemm_ip.quant import _operand_bits
    bits = _operand_bits(precision)
    if bits is None:
        return True
    width, signed = bits
    if width > 8:
        raise ValueError(
            f"cmvu {what} codes must fit int8; precision {precision!r} is "
            f"{width}-bit (narrower codes sign-extend into the 8-bit lane).")
    return signed


class CmvuTarget(Target):
    name = "cmvu"
    tool = "catapult"

    # Both canonical GEMM weight layouts are accepted and converted to the
    # canonical row-major [K][N] tiles the init block / C++ baking use.
    weight_layouts = ("column_major", "row_major")

    knobs = [
        {"name": "KFold", "key": "kfold", "type": "int", "default": None,
         "description": "K-axis temporal pass count (required). Legal "
                         "1..ceil(K/4); 1 means K is fully spatial. cmvu has no "
                         "FoldAxis, and ignores ReuseFactor."},
        {"name": "NFold", "key": "nfold", "type": "int", "default": None,
         "description": "N-axis temporal group count (required). Legal "
                         "1..ceil(N/8); 1 means N is fully spatial."},
    ]

    # ── geometry / RTL / golden (thin delegation) ────────────────────────────

    def geometry(self, shape, kfold=None, nfold=None, **kwargs):
        geo = _geom.resolve_geometry(shape[0], shape[1], shape[2], kfold, nfold,
                                     kwargs.get("name"),
                                     result_width=kwargs.get("result_width"))
        for w in geo["warnings"]:
            print(w, file=sys.stderr)
        return geo

    def emit_rtl(self, shape, **kwargs):
        return _rtl.generate_core(*shape, **kwargs)

    def emit_behavioral(self, shape, **kwargs):
        # Single-branch RTL: behavioral IS structural (no sim-only model).
        return _rtl.generate_core(*shape, **kwargs)

    def golden(self, shape, seed=42, **kwargs):
        return _golden.generate_tb(*shape, seed=seed, **kwargs)

    # ── packaging ────────────────────────────────────────────────────────────

    def package(self, shape, cfg):
        cfg = dict(cfg)
        name = cfg.pop("name")
        self.validate_knobs(cfg, name)
        output_dir = cfg.pop("output_dir")

        kfold = cfg.pop("kfold", None)
        nfold = cfg.pop("nfold", None)
        if kfold is None or nfold is None:
            raise ValueError(
                f"layer '{name}': cmvu requires both KFold and NFold "
                f"(got KFold={kfold!r}, NFold={nfold!r}); the legacy "
                f"FoldAxis/ReuseFactor knobs are not supported.")

        interface = str(cfg.get("interface", "stream")).lower()
        if interface != "stream":
            raise ValueError(
                f"layer '{name}': cmvu supports interface='stream' only "
                f"(got {interface!r}); the array (io_parallel) interface is "
                f"not implemented.")

        shift, product_frac = _derive_shift(
            cfg.get("input_precision"), cfg.get("weight_precision"),
            cfg.get("output_precision"), cfg.get("accum_precision"), name)
        rnd = _output_rounds(cfg.get("output_precision"), name)
        a_signed = _operand_signed(cfg.get("input_precision"), "activation")
        b_signed = _operand_signed(cfg.get("weight_precision"), "weight")
        result_width = _output_width(cfg.get("output_precision"), name)

        # bias_codes is always built (int32, accumulator scale) so the RND
        # rounding constant has somewhere to live even on a bias-less layer;
        # a TRN layer with no bias collapses to an all-zero bias word.
        has_bias = bool(cfg.get("has_bias")) and cfg.get("bias") is not None
        if has_bias:
            _check_bias_exact(cfg["bias"], product_frac, name)
            bias_values = cfg["bias"]
        else:
            import numpy as np
            bias_values = np.zeros(shape[2])
        if has_bias or rnd:
            bias_codes, warns = _golden.bias_codes(
                bias_values, product_frac, rnd=rnd, requant_shift=shift)
            for w in warns:
                print(w, file=sys.stderr)
        else:
            bias_codes = None

        # weights_in_core=False selects the runtime-B (two-operand) path: B is
        # streamed and loaded into the blocks before the A rows, no baked ROM.
        runtime_b = not bool(cfg.get("weights_in_core", True))
        weight_matrix = cfg.get("weight_matrix")
        if weight_matrix is None and not runtime_b:
            weight_matrix = _synth_weights(shape[1], shape[2], name)

        # The manifest's weight_layout (validated against weight_layouts by
        # the CLI) only mattered to the baked-ROM path before runtime_b: the
        # streamed b_beat format has to match it too, or the RTL wrapper's
        # load FSM mis-decodes hls4ml's beat order.
        weight_layout = str(cfg.get("weight_layout") or "column_major").lower()
        b_row_major = runtime_b and weight_layout == "row_major"

        # Column-major runtime-B's load FSM pairs at most two resident
        # k-passes per column beat (rtl.generate_core's column-major branch);
        # a third resident k-pass needs a load-time staging buffer that does
        # not exist yet. Checked here, up front, so this fails at package
        # time naming the layer, not as a NotImplementedError deep in RTL
        # generation.
        if runtime_b and not b_row_major:
            geo = _geom.resolve_geometry(shape[0], shape[1], shape[2], kfold,
                                         nfold, name, result_width=result_width)
            if geo["k_passes"] > 2:
                raise ValueError(
                    f"layer '{name}': column-major runtime-B needs "
                    f"K_PASSES<=2 (single or paired tile writes per column "
                    f"beat); KFold={kfold}/K={shape[1]} legalizes to "
                    f"K_PASSES={geo['k_passes']}. Lower KFold (more spatial "
                    f"blocks) or use weight_layout='row_major'.")

        return _package().generate_catapult_pkg(
            shape[0], shape[1], shape[2], name, output_dir, kfold, nfold,
            weight_matrix, bias_codes=bias_codes, shift=shift,
            a_signed=a_signed, b_signed=b_signed,
            clock_period_ns=cfg.get("clock_period_ns") or 5.0,
            runtime_b=runtime_b, b_row_major=b_row_major, interface=interface,
            result_width=result_width)

    def verify(self, package):
        """Structural check that packaging produced a well-formed package.

        The functional bar (Catapult SCVerify cosim) is the tool-level gate and
        is run separately.
        """
        pkg = Path(package)
        name = pkg.name
        required = [f"{name}_core.sv", "nnet_types.h", f"{name}_gemm_ip.h",
                    f"{name}_inst.cpp", f"{name}_tb.cpp", "run_catapult.tcl"]
        missing = [f for f in required
                   if not (pkg / f).is_file() or (pkg / f).stat().st_size == 0]
        # the vendored block RTL ships once per package root, not per layer
        for sv in _geom.VENDORED_SV:
            if not (pkg.parent / sv).is_file():
                missing.append(f"../{sv}")
        if missing:
            raise RuntimeError(
                f"{name}: package incomplete, missing/empty: {missing}")
        return True

    def rtl_test(self, cases=None, seeds=None, keep=False):
        """Cheap Icarus/VCS pre-filter (not the acceptance gate)."""
        return _run_rtl_tests().run(cases=cases, seeds=seeds, keep=keep)

    def catapult_test(self, cases=None, workdir=None, resume=True, keep=False):
        """The acceptance gate: Catapult SCVerify over the tiered matrix."""
        from . import run_catapult_tests
        return run_catapult_tests.run(cases=cases, workdir=workdir,
                                      resume=resume, keep=keep)

    # ── batch orchestration (multi-package; used by the CLI) ─────────────────

    def normalize_config(self, cfg):
        # Check the RAW config: after normalization "unset" and "explicitly
        # set" are indistinguishable.
        for raw in _raw_items(cfg):
            layer = raw.get("name") or raw.get("layer_name")
            legacy = [k for k in _LEGACY_KEYS if raw.get(k) is not None]
            if legacy:
                raise ValueError(
                    f"layer '{layer}': cmvu uses the KFold/NFold knobs only; "
                    f"tensor_slice knob(s) {legacy} present. Remove them (or use "
                    f"the tensor_slice target).")
            rf = raw.get("reuse_factor")
            if rf is not None and rf != 1:
                print(f"WARNING: layer '{layer}': cmvu ignores reuse_factor={rf}; "
                      f"its folding is set by KFold/NFold only.", file=sys.stderr)

        from gemm_ip.config import _normalize_config_items
        items = _normalize_config_items(cfg)
        raw_items = _raw_items(cfg)
        for raw, item in zip(raw_items, items):
            name = item.get("name")
            if raw.get("cmvu_blocked"):
                raise ValueError(
                    f"layer '{name}': not representable by the cmvu datapath "
                    f"({raw.get('cmvu_block_reason', 'unsupported')}); remove it "
                    f"or use a representable subset.")
            # kfold/nfold survive in the list/shorthand branches but are dropped
            # by the named-dict branch; re-attach from the raw item either way.
            kfold = item.get("kfold", raw.get("kfold"))
            nfold = item.get("nfold", raw.get("nfold"))
            item["kfold"] = kfold
            item["nfold"] = nfold
            self.validate_knobs(item, name)

            interface = str(item.get("interface", "stream")).lower()
            if interface != "stream":
                raise ValueError(
                    f"layer '{name}': cmvu supports interface='stream' only "
                    f"(got {interface!r}); the array (io_parallel) interface is "
                    f"not implemented.")
            if kfold is None or nfold is None:
                raise ValueError(
                    f"layer '{name}': cmvu requires both KFold and NFold "
                    f"(got KFold={kfold!r}, NFold={nfold!r}).")

            geo = _geom.resolve_geometry(item["m"], item["k"], item["n"],
                                         kfold, nfold, name)
            for w in geo["warnings"]:
                print(w, file=sys.stderr)
            item["kfold"] = int(kfold)
            item["nfold"] = int(nfold)
            item["k_spatial"] = geo["k_spatial"]
            item["n_spatial"] = geo["n_spatial"]
            item["k_passes"] = geo["k_passes"]
            item["n_passes"] = geo["n_passes"]
            item["slots_per_block"] = geo["slots_per_block"]
        return items

    def combined_header(self, items):
        return _package().gen_combined_header(items)

    def integration_manifest(self, items):
        return _package().gen_integration_manifest(items)

    def blackbox_tcl(self, items):
        return _package().gen_blackbox_tcl(items)

    def sources_tcl(self, items):
        return _package().gen_sources_tcl()

    def finalize(self, items, output_dir):
        src = _geom.vendored_rtl_dir()
        if src is None:
            raise FileNotFoundError(
                f"cmvu vendored block RTL not found; expected "
                f"{', '.join(_geom.VENDORED_SV)} under {_geom.CMVU_RTL_DIR}.")
        dst = Path(output_dir)
        for sv in _geom.VENDORED_SV:
            (dst / sv).write_text((src / sv).read_text())
        # VTR-facing cmvu_mode1 hard-block model (blackbox stub, no body);
        # see package.gen_cmvu_mode1_vtr_model.
        (dst / _geom.CMVU_MODE1_VTR_MODEL).write_text(
            _package().gen_cmvu_mode1_vtr_model())


TARGET = CmvuTarget()
