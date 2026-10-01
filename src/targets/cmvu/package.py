"""cmvu package: Catapult blackbox package emission.

Per package (``<output_dir>/<name>/``) this emits the tensor_slice-style file
set adapted to cmvu:

* ``<name>_core.sv``  - the generated wrapper (real `cmvu_mode1` blocks)
* ``nnet_types.h``    - the ``nnet::array`` pack type
* ``<name>_gemm_ip.h``- public stream entry + ``ccore`` class:
    - ``__SYNTHESIS__ && BLACKBOX_FLOW``: ``ac_blackbox`` binding to the core
    - otherwise: a behavioral C++ twin (exact int math with the same baked
      weights/bias) whose result timing mirrors the wrapper's out_valid so
      Catapult csim and SCVerify see the same transaction schedule
* ``<name>_inst.cpp`` - Catapult top (stream const-weights entry)
* ``<name>_tb.cpp``   - self-checking csim TB (same golden arithmetic)
* ``run_catapult.tcl``- project setup + SCVerify RTL cosim (msim)

The vendored ``cmvu_mode1.sv``/``cmvu_w_mem.sv``/``cmvu_regbank.sv`` are added
``-exclude true`` (sim models); ``finalize`` copies them next to the output.
"""
import json
import zlib
from pathlib import Path
from string import Template

import numpy as np

try:
    from . import geometry as _geometry
    from . import golden as _golden
    from . import rtl as _rtl
except ImportError:  # standalone/script import
    import geometry as _geometry
    import golden as _golden
    import rtl as _rtl


VENDORED_SV = _geometry.VENDORED_SV


def _codes(weight_matrix, k, n):
    W = np.asarray(weight_matrix, dtype=np.int64)
    if W.shape != (int(k), int(n)):
        raise ValueError(f"weight_matrix shape {W.shape} != (k={k}, n={n})")
    if W.min() < -128 or W.max() > 127:
        raise ValueError("weight codes must fit int8")
    return W


def _cpp_int_array(name, rows, cols, table, signed=True):
    typ = "signed char" if signed else "int"
    lines = [f"static const {typ} {name}[{rows}][{cols}] = {{"]
    for i in range(rows):
        lines.append("    {" + ", ".join(str(int(v)) for v in table[i]) + "},")
    lines.append("};")
    return "\n".join(lines)


def _cpp_1d(name, values):
    return (f"static const int {name}[{len(values)}] = {{" +
            ", ".join(str(int(v)) for v in values) + "};")


# ── nnet types ────────────────────────────────────────────────────────────────


def gen_nnet_types_header():
    return """\
#ifndef NNET_TYPES_H_
#define NNET_TYPES_H_

#include <cstddef>

namespace nnet {

template <typename T, unsigned N> struct array {
    typedef T value_type;
    static const unsigned size = N;
    T data[N];

    T &operator[](size_t pos) { return data[pos]; }
    const T &operator[](size_t pos) const { return data[pos]; }
};

} // namespace nnet

#endif
"""


# ── header building blocks shared by compile-time-weight and runtime-B ────────
#
# _HEADER (compile-time weights) and _HEADER_RB (runtime B) are two-thirds
# identical: same guard/includes, same pack/unpack/requant helpers, same
# result-slot polling loop inside ccore::run(), same KP/NP/D/QN state and
# reset_state() shape. The pieces below render those shared spans once each;
# gen_public_header/gen_runtime_b_header substitute them into ${...}
# placeholders in _HEADER/_HEADER_RB, so each template's literal text is only
# the part that's genuinely different (blackbox result value, compute_row's
# weight source, extra runtime-B state, the stream-loop bodies).

_HEADER_GUARD_OPEN = Template("""\
#ifndef ${NAME}_GEMM_IP_H
#define ${NAME}_GEMM_IP_H

#include <cassert>
#include "ac_int.h"
#include "ac_channel.h"

#if defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW)
#include "ac_blackbox.h"
#endif

namespace nnet {""")

_HEADER_GUARD_CLOSE = Template("""\
} // namespace nnet

#endif // ${NAME}_GEMM_IP_H
""")

# Pack/unpack loops are unrolled on purpose: a rolled loop inside the II=1
# feed loop is an unschedulable feedback path (SCHD-3), the same trap
# tensor_slice documents for its capture body.
_PACK_A_ROW_FN = Template("""\
template <class data_T>
ac_int<${A_BITS}, false> ${name}_pack_a_row(const data_T &row) {
    ac_int<${A_BITS}, false> packed = 0;
    #pragma hls_unroll
    for (int i = 0; i < ${k}; i++)
        packed.set_slc(i * 8, ac_int<8, false>(row[i].template slc<8>(0)));
    return packed;
}""")

_UNPACK_RES_ROW_FN = Template("""\
// Store each lane's W-bit code as the element's raw bits (the same bit-level
// convention as pack): assigning the ac_int by value would read the code as an
// integer value and wrap it into the fixed-point range.
template <class res_T>
res_T ${name}_unpack_res_row(ac_int<${RES_BITS}, false> packed) {
    typedef typename res_T::value_type res_elem_T;
    static_assert(res_T::size == ${n}, "cmvu expects one N-wide result row per beat");
    static_assert(res_elem_T::width == ${W}, "cmvu result lanes are W bits wide");
    res_T out;
    #pragma hls_unroll
    for (int j = 0; j < ${n}; j++) {
        res_elem_T v;
        v.set_slc(0, packed.template slc<${W}>(j * ${W}));
        out[j] = v;
    }
    return out;
}""")

_REQUANT_FN = Template("""\
static inline int ${name}_requant_cpp(int total) {
    // 64-bit so the W = 32 mask and sign test stay defined.
    long long q = (long long)total >> ${shift};
    q &= (1LL << ${W}) - 1;
    return (int)((q >= (1LL << (${W} - 1))) ? (q - (1LL << ${W})) : q);
}""")

_BLACKBOX_BIND = Template("""\
        ac_blackbox()
            .entity("${name}_core")
            .verilog_files("${name}_core.sv")
            .outputs("out_valid res_row")
            .area(2048.0)
            .delay(${bb_delay})
            .latency(1)
            .init_delay(1)
            .clock_name("clk")
            .posedge_clock(true)
            .sync_reset_name("rst")
            .active_high_sync_reset(true)
            .has_state(true)
            .start_name("en")
            .end();""")

# Common to both ccore::run() behavioral bodies: drain any result slot whose
# countdown just hit zero, then arm a fresh slot for a newly-accepted row.
_RESULT_SLOT_POLL_CAPTURE = Template("""\
        for (int q = 0; q < QN; q++) {
            if (rem[q] > 0) {
                rem[q]--;
                if (rem[q] == 0) {
                    res_row   = pending[q];
                    out_valid = 1;
                }
            }
        }
        if (in_valid) {
            // QN is sized so a free slot always exists at the fixed cadence;
            // a geometry that breaks that must fail here, not walk off rem[].
            int q = 0;
            while (q < QN && rem[q] > 0) q++;
            assert(q < QN && "cmvu C model: no free result slot (QN too small)");
            rem[q]     = KP * NP + D + 1;
            pending[q] = ${name}_compute_row(a_row);
        }""")

# extra_members/extra_reset let the runtime-B header append its per-instance
# B_model buffer without duplicating the KP/NP/D/QN block or the rem/pending
# reset loop.
_STATE_MEMBERS_BLOCK = Template("""\
  private:
    static const int KP = ${KP};
    static const int NP = ${NP};
    static const int D  = ${D};
    static const int QN = ${QN};
#if !(defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW))
    ac_int<${RES_BITS}, false> pending[QN];
    int rem[QN];${extra_members}
#endif""")

_RESET_STATE_FN = Template("""\
    void ${name}_reset_state() {
#if !(defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW))
        for (int q = 0; q < QN; q++) { rem[q] = 0; pending[q] = 0; }${extra_reset}
#endif
    }""")


# ── public header (blackbox binding + behavioral twin) ────────────────────────

_HEADER = Template("""\
${guard_open}

${pack_a_row_comment}
${pack_a_row_fn}

${unpack_res_row_fn}

// Behavioral requant (the golden/TB/model all share this): truncating
// (floor) arithmetic shift then two's-complement wrap-to-W, matching
// cmvu_mode1 exactly. The block never rounds; RND output is reproduced by
// folding the rounding constant into the bias (int32, accumulator scale)
// before it reaches here -- see golden.bias_codes.
${requant_fn}

#if !(defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW))
${weights_decl}
${bias_decl}
#endif

class ${name}_ccore {
  public:
    ${name}_ccore() {}

#pragma hls_design interface ccore blackbox
    void run(ac_int<${A_BITS}, false>  a_row,
             ac_int<1, false>          in_valid,
             ac_int<${RES_BITS}, false>       &res_row,
             ac_int<1, false>         &out_valid) {
#if defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW)
${blackbox_bind}
        out_valid = 0;
        res_row   = (ac_int<${RES_BITS}, false>)a_row;
#else
        // Result-timing mirror of the generated wrapper: the entry feeds on a
        // fixed cadence, so each accepted row's result must land exactly
        // KP*NP + D + 1 calls after its accept call (D = L + cascade skew +
        // row de-skew), like the RTL's aligned `done`. Up to QN rows are
        // in flight. `in_ready` is intentionally NOT a model output: the HLS
        // schedule must not depend on any DUT signal.
        out_valid = 0;
${result_slot_poll_capture}
#endif
    }

#if !(defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW))
    static ac_int<${RES_BITS}, false> ${name}_compute_row(ac_int<${A_BITS}, false> a_row) {
        ac_int<${RES_BITS}, false> packed = 0;
        for (int j = 0; j < ${n}; j++) {
            int total = ${bias_expr};
            for (int i = 0; i < ${k}; i++) {
                int a_code = (int)a_row.template slc<8>(i * 8);
                if (a_code >= 128) a_code -= 256;
                total += a_code * (int)${W_NAME}[i][j];
            }
            long long code = (long long)${name}_requant_cpp(total) & ((1LL << ${W}) - 1);
            packed.set_slc(j * ${W}, ac_int<${W}, false>(code));
        }
        return packed;
    }
#endif

${state_members}

  public:
    // In Catapult the object is persistent across run() calls; csim initializes
    // the in-flight slots on construction (the RTL resets from `rst`).
${reset_state_fn}
};

// The entry is its own free-running block: its main loop is pipelined at
// II=1, so every call is one clock of the wrapper and frames overlap (the next
// frame's row enters while the previous frame's result is still draining).
// hls4ml's per-layer stage only forwards its channels to it.
#pragma hls_design block
#pragma hls_pipeline_init_interval 1
template <class data_T, class res_T, typename CONFIG_T>
void ${name}_gemm_stream_const_weights(ac_channel<data_T> &a_stream,
                                       ac_channel<res_T> &res_stream) {
    // One K-wide A row per beat; a narrower beat would under-read each row.
    static_assert(data_T::size == ${k}, "cmvu expects one K-wide A row per beat");
    static ${name}_ccore ccore;
#if defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW)
    // One wrapper clock per call. State persists across calls, so there is no
    // per-frame drain: a result is written whenever the wrapper emits one.
    // `gap` counts calls until the wrapper takes its next row (one every
    // KP*NP = ${PERIOD} cycles); it depends only on this side's own history,
    // never on a DUT output. The non-blocking read lets in-flight rows keep
    // draining when no new row is waiting; a full output channel stalls the
    // whole block, which gates `en` and freezes the wrapper with it.
    static ac_int<${A_BITS}, false> a_row = 0;
    static ac_int<4, false> gap = 0;
    ac_int<1, false> in_valid = 0;
    if (gap == 0) {
        data_T beat;
        if (a_stream.nb_read(beat)) {
            a_row = ${name}_pack_a_row<data_T>(beat);
            in_valid = 1;
            gap = ${PERIOD} - 1;
        }
    } else {
        gap--;
    }
    ac_int<${RES_BITS}, false> res_row;
    ac_int<1, false> out_valid;
    ccore.run(a_row, in_valid, res_row, out_valid);
    if (out_valid) {
        res_stream.write(${name}_unpack_res_row<res_T>(res_row));
    }
#else
    // C model: one whole frame per call (hls4ml csim and the SCVerify golden
    // call the layer once per frame). The output stream is identical to the
    // free-running RTL's; only the timing differs, and SCVerify compares
    // streams, not cycles.
    ccore.${name}_reset_state();
    // Feed-forward schedule: the blackbox contract has no
    // in_ready, so the HLS schedule cannot depend on any DUT output. One A row
    // is presented every PERIOD calls (compile-time constant) and results are
    // captured by polling out_valid. The trip count is fixed, so Catapult keeps
    // `en` asserted every call and never inserts handshake wait states -- the
    // old closed-loop entry let Catapult derive en/core_wen from sampled DUT
    // signals, which drifted the software and RTL timelines for multi-pass
    // schedules.
    ac_int<${A_BITS}, false> a_row = 0;
    int rows_in = 0;
    #pragma hls_pipeline_init_interval 1
    RUN: for (int step = 0; step < ${TOTAL}; step++) {
        ac_int<1, false> in_valid = 0;
        if ((step % ${PERIOD}) == 0 && rows_in < ${m}) {
            data_T beat = a_stream.read();
            a_row = ${name}_pack_a_row<data_T>(beat);
            in_valid = 1;
            rows_in++;
        }
        ac_int<${RES_BITS}, false> res_row;
        ac_int<1, false> out_valid;
        ccore.run(a_row, in_valid, res_row, out_valid);
        if (out_valid) {
            res_stream.write(${name}_unpack_res_row<res_T>(res_row));
        }
    }
#endif
}

${guard_close}""")


def gen_public_header(name, m, k, n, weight_codes, bias_codes, shift,
                      geo, bb_delay=3.5):
    W = _codes(weight_codes, k, n)
    W_NAME = f"_{name}_w"
    weights_decl = _cpp_int_array(W_NAME, k, n, W.tolist())
    bias_list = [int(v) for v in (bias_codes if bias_codes is not None
                                  else [0] * n)]
    bias_decl = _cpp_1d(f"_{name}_bias", bias_list)
    bias_expr = f"_{name}_bias[j]"
    D = _geometry.L + (geo["k_spatial"] - 1) + (geo["n_spatial"] - 1)
    KP, NP = geo["k_passes"], geo["n_passes"]
    # Feed-forward entry timing: the wrapper accepts a row every KP*NP
    # en-cycles (the next row is taken on the current row's last pass), and
    # the model emits each result KP*NP + D + 1 calls after its accept call.
    # TOTAL is the fixed trip count that covers every feed and every result;
    # QN bounds the rows in flight so the model's slot table can't overflow.
    PERIOD = KP * NP
    LAT = KP * NP + D + 1
    TOTAL = (int(m) - 1) * PERIOD + LAT + 1
    QN = (LAT + PERIOD - 1) // PERIOD + 2
    A_BITS, RES_BITS, W = geo["a_port_bits"], geo["res_port_bits"], geo["result_width"]
    pack_a_row_comment = (
        "// Lane packing (architecture.md 11.1): lane 0 in the LSBs.\n"
        "// Pack/unpack loops are unrolled on purpose: a rolled loop inside the II=1\n"
        "// feed loop is an unschedulable feedback path (SCHD-3), the same trap\n"
        "// tensor_slice documents for its capture body.")
    state_members = _STATE_MEMBERS_BLOCK.substitute(
        KP=KP, NP=NP, D=D, QN=QN, RES_BITS=RES_BITS, extra_members="")
    reset_state_fn = _RESET_STATE_FN.substitute(name=name, extra_reset="")
    return _HEADER.substitute(
        NAME=name.upper(), name=name, m=int(m), k=int(k), n=int(n),
        A_BITS=A_BITS, RES_BITS=RES_BITS,
        shift=int(shift), weights_decl=weights_decl,
        bias_decl=bias_decl, bias_expr=bias_expr, W_NAME=W_NAME,
        KP=KP, NP=NP, D=D, QN=QN, PERIOD=PERIOD, TOTAL=TOTAL,
        W=W, bb_delay=bb_delay,
        guard_open=_HEADER_GUARD_OPEN.substitute(NAME=name.upper()),
        guard_close=_HEADER_GUARD_CLOSE.substitute(NAME=name.upper()),
        pack_a_row_comment=pack_a_row_comment,
        pack_a_row_fn=_PACK_A_ROW_FN.substitute(name=name, k=int(k), A_BITS=A_BITS),
        unpack_res_row_fn=_UNPACK_RES_ROW_FN.substitute(
            name=name, n=int(n), W=W, RES_BITS=RES_BITS),
        requant_fn=_REQUANT_FN.substitute(name=name, shift=int(shift), W=W),
        blackbox_bind=_BLACKBOX_BIND.substitute(name=name, bb_delay=bb_delay),
        result_slot_poll_capture=_RESULT_SLOT_POLL_CAPTURE.substitute(name=name),
        state_members=state_members, reset_state_fn=reset_state_fn)


# ── runtime-B header (two-stream: load B, then stream A) ──────────────────────

_HEADER_RB = Template("""\
${guard_open}

${pack_a_row_comment}
${pack_a_row_fn}

${unpack_res_row_fn}

${requant_fn}

#if !(defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW))
${bias_decl}
#endif

class ${name}_ccore {
  public:
    ${name}_ccore() {}

#pragma hls_design interface ccore blackbox
    void run(ac_int<${A_BITS}, false>  a_row,
             ac_int<1, false>          in_valid,
             ac_int<${B_BEAT_BITS}, false>    b_beat,
             ac_int<1, false>          b_valid,
             ac_int<${RES_BITS}, false>       &res_row,
             ac_int<1, false>         &out_valid) {
#if defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW)
${blackbox_bind}
        out_valid = 0;
        res_row   = 0;
#else
        // Behavioral twin: capture the runtime B stream directly in hls4ml's
        // own beat format (same layout the RTL wrapper's load FSM consumes --
        // no native-tile reorder), then the same KP*NP + D + 1 result delay
        // as the const-weights model. B is a per-instance runtime operand.
        out_valid = 0;
        if (b_valid) {
            #pragma hls_unroll
            for (int lane = 0; lane < ${B_LANES}; lane++) {
                signed char v = (signed char)b_beat.template slc<8>(lane * 8).to_int();
${B_MODEL_STORE}
            }
${B_MODEL_ADVANCE}
        }
${result_slot_poll_capture}
#endif
    }

#if !(defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW))
    // Non-static: uses the per-instance runtime B_model.
    ac_int<${RES_BITS}, false> ${name}_compute_row(ac_int<${A_BITS}, false> a_row) {
        ac_int<${RES_BITS}, false> packed = 0;
        for (int j = 0; j < ${n}; j++) {
            int total = ${bias_expr};
            for (int i = 0; i < ${k}; i++) {
                int a_code = (int)a_row.template slc<8>(i * 8);
                if (a_code >= 128) a_code -= 256;
                total += a_code * (int)B_model[i][j];
            }
            long long code = (long long)${name}_requant_cpp(total) & ((1LL << ${W}) - 1);
            packed.set_slc(j * ${W}, ac_int<${W}, false>(code));
        }
        return packed;
    }
#endif

${state_members}

  public:
${reset_state_fn}
};

// hls4ml's two-operand contract (nnet_gemm_stream.h gemm_stream) streams B
// FIRST, in the layout selected by the manifest's weight_layout
// (rtl.generate_core's b_row_major): column-major (one gemm_k-high beat per
// real column, element k of column n holding B[k][n]) or row-major (one
// gemm_n-wide beat per real row, element n of row k holding B[k][n]). Each
// beat is passed to the core as one raw-bit word -- no C++ reorder/buffer --
// and the load schedule below (beat count, gap, and any row-major drain
// stalls) mirrors the RTL wrapper's load FSM exactly (see golden.py's
// generate_runtime_b_tb / b_load_beats_col_major / b_load_beats_row_major,
// the reference this cadence is derived from).
// Free-running block, like the weight-stationary entry: one wrapper clock per
// call, frames overlap, and with double-buffered slot sets the next frame's
// B loads while the current frame computes.
#pragma hls_design block
#pragma hls_pipeline_init_interval 1
template <class data_T, class b_T, class res_T, typename CONFIG_T>
void ${name}_gemm_stream_runtime_b(ac_channel<data_T> &a_stream,
                                   ac_channel<b_T> &b_stream,
                                   ac_channel<res_T> &res_stream) {
    // One K-wide A row per beat; a narrower beat would under-read each row.
    static_assert(data_T::size == ${k}, "cmvu expects one K-wide A row per beat");
    static_assert(b_T::size == ${B_ASSERT_SIZE}, "${B_ASSERT_MSG}");
    static ${name}_ccore ccore;
    // Per-cycle load pattern of the wrapper's load FSM: true where it takes a
    // real B beat (and waits for one), false where it steps on its own.
    static const bool ${name}_load_valid[${LOAD_LEN}] = {${LOAD_VALID}};
#if defined(__SYNTHESIS__) && defined(BLACKBOX_FLOW)
    // The blackbox has no ready outputs and the schedule may not depend on
    // DUT signals, so this keeps a cycle-exact copy of the wrapper's control
    // state (load FSM position, slot sets, row sequencer), advanced from the
    // same inputs presented to it. Inputs are read non-blocking and only
    // presented when the copy says the wrapper will take them; a missing B
    // beat holds the load FSM on that entry, exactly as the wrapper does.
    static bool loading = true;
    static bool loading_d = false;
    static bool busy = false;
    static bool rd_set = false;
    static bool wr_set = false;
    static bool set_full0 = false;
    static bool set_full1 = false;
    static ac_int<${LD_W}, false> ld_idx = 0;
    static ac_int<${PASS_W}, false> pass = 0;
    static ac_int<${ACC_W}, false> acc = 0;
    static ac_int<${A_BITS}, false> a_row = 0;
    static ac_int<${B_BEAT_BITS}, false> b_beat = 0;

    // Pre-edge view (what the wrapper sees on this clock).
    bool full_rd = rd_set ? set_full1 : set_full0;
    bool full_other = rd_set ? set_full0 : set_full1;
    bool full_wr = wr_set ? set_full1 : set_full0;
    bool row_last = busy && (pass == ${SLOTS} - 1);
    bool frame_end = row_last && (acc == 0);
    bool next_full = (busy && frame_end) ? (${DBUF} && full_other) : full_rd;
    bool in_ready = (!busy || row_last) && next_full;
    bool arm = !loading && !loading_d && !full_wr;

    ac_int<1, false> b_valid = 0;
    bool ld_step = false;
    if (loading) {
        if (${name}_load_valid[ld_idx]) {
            b_T beat;
            if (b_stream.nb_read(beat)) {
                ac_int<${B_BEAT_BITS}, false> raw = 0;
                #pragma hls_unroll
                for (int i = 0; i < ${B_BEAT_LANES}; i++)
                    raw.set_slc(i * 8, ac_int<8, false>(beat[i].template slc<8>(0)));
                b_beat = raw;
                b_valid = 1;
                ld_step = true;
            }
        } else {
            ld_step = true;
        }
    }
    ac_int<1, false> in_valid = 0;
    if (in_ready) {
        data_T beat;
        if (a_stream.nb_read(beat)) {
            a_row = ${name}_pack_a_row<data_T>(beat);
            in_valid = 1;
        }
    }
    ac_int<${RES_BITS}, false> res_row;
    ac_int<1, false> out_valid;
    ccore.run(a_row, in_valid, b_beat, b_valid, res_row, out_valid);
    if (out_valid) {
        res_stream.write(${name}_unpack_res_row<res_T>(res_row));
    }

    // Post-edge state, from the pre-edge values above.
    bool loading_next = loading;
    if (loading) {
        if (ld_step) {
            if (ld_idx == ${LOAD_LEN} - 1) {
                loading_next = false;
                ld_idx = 0;
            } else {
                ld_idx++;
            }
        }
    } else if (arm) {
        loading_next = true;
        ld_idx = 0;
    }
    if (loading_d && !loading) {             // the last write lands
        if (wr_set) set_full1 = true; else set_full0 = true;
        if (${DBUF}) wr_set = !wr_set;
    }
    if (busy && frame_end) {                  // the frame's last pass issues
        if (rd_set) set_full1 = false; else set_full0 = false;
        if (${DBUF}) rd_set = !rd_set;
    }
    loading_d = loading;
    loading = loading_next;
    if (in_valid) {
        busy = true;
        pass = 0;
        acc = (acc == ${m} - 1) ? ac_int<${ACC_W}, false>(0) : ac_int<${ACC_W}, false>(acc + 1);
    } else if (busy) {
        if (pass == ${SLOTS} - 1) {
            pass = 0;
            busy = false;
        } else {
            pass++;
        }
    }
#else
    // C model: one whole frame per call (load B, then the frame's M rows).
    ccore.${name}_reset_state();
    ac_int<${A_BITS}, false> a_row = 0;
    ac_int<${B_BEAT_BITS}, false> b_beat = 0;
    int rows_in = 0;
    #pragma hls_pipeline_init_interval 1
    RUN: for (int step = 0; step < ${TOTAL}; step++) {
        ac_int<1, false> in_valid = 0;
        ac_int<1, false> b_valid = 0;
        if (step < ${LOAD_LEN} && ${name}_load_valid[step]) {
            b_T beat = b_stream.read();
            ac_int<${B_BEAT_BITS}, false> raw = 0;
            #pragma hls_unroll
            for (int i = 0; i < ${B_BEAT_LANES}; i++)
                raw.set_slc(i * 8, ac_int<8, false>(beat[i].template slc<8>(0)));
            b_beat = raw;
            b_valid = 1;
        }
        if (step >= ${LOAD} && ((step - ${LOAD}) % ${PERIOD}) == 0
                && rows_in < ${m}) {
            data_T beat = a_stream.read();
            a_row = ${name}_pack_a_row<data_T>(beat);
            in_valid = 1;
            rows_in++;
        }
        ac_int<${RES_BITS}, false> res_row;
        ac_int<1, false> out_valid;
        ccore.run(a_row, in_valid, b_beat, b_valid, res_row, out_valid);
        if (out_valid) {
            res_stream.write(${name}_unpack_res_row<res_T>(res_row));
        }
    }
#endif
}

${guard_close}""")


def _runtime_b_load_schedule(k, n, geo, b_row_major):
    """The runtime-B load FSM's per-cycle beat pattern (see
    ``geometry.runtime_b_load_schedule``)."""
    return _geometry.runtime_b_load_schedule(k, n, geo, b_row_major)


def gen_runtime_b_header(name, m, k, n, bias_codes, shift, geo, bb_delay=3.5,
                         b_row_major=False):
    bias_list = [int(v) for v in (bias_codes if bias_codes is not None
                                  else [0] * n)]
    bias_decl = _cpp_1d(f"_{name}_bias", bias_list)
    bias_expr = f"_{name}_bias[j]"
    D = _geometry.L + (geo["k_spatial"] - 1) + (geo["n_spatial"] - 1)
    KP, NP = geo["k_passes"], geo["n_passes"]
    PERIOD = KP * NP

    load_sched = _runtime_b_load_schedule(k, n, geo, b_row_major)
    LOAD_LEN = len(load_sched)
    LOAD_VALID = ", ".join("true" if v else "false" for v in load_sched)
    LOAD = LOAD_LEN + 2

    LAT = KP * NP + D + 1
    TOTAL = LOAD + (int(m) - 1) * PERIOD + LAT + 1
    QN = (LAT + PERIOD - 1) // PERIOD + 2

    if b_row_major:
        B_BEAT_BITS = int(n) * 8
        B_BEAT_LANES = int(n)
        B_ASSERT_SIZE = int(n)
        B_ASSERT_MSG = ("cmvu expects one N-wide B row beat per real K row "
                        "(row-major runtime B)")
        B_LANES = int(n)
        B_MODEL_STORE = "                B_model[ld_idx][lane] = v;"
        B_MODEL_ADVANCE = "            ld_idx++;"
        B_MODEL_IDX = "ld_idx"
    else:
        B_BEAT_BITS = int(k) * 8
        B_BEAT_LANES = int(k)
        B_ASSERT_SIZE = int(k)
        B_ASSERT_MSG = ("cmvu expects one K-high B column beat per real N "
                        "column (column-major runtime B)")
        B_LANES = int(k)
        B_MODEL_STORE = "                B_model[lane][ld_idx] = v;"
        B_MODEL_ADVANCE = "            ld_idx++;"
        B_MODEL_IDX = "ld_idx"

    A_BITS, RES_BITS, W = geo["a_port_bits"], geo["res_port_bits"], geo["result_width"]
    # Slot sets, as in the wrapper: two when two copies fit (set 1 on an even
    # slot), one otherwise.
    slots = KP * NP
    dbuf = slots + slots % 2 + slots <= _geometry.MEM_TILES
    extra_members = (f"\n    signed char B_model[{int(k)}][{int(n)}];"
                     f"\n    int {B_MODEL_IDX};")
    extra_reset = (f"\n        for (int a = 0; a < {int(k)}; a++)"
                  f"\n            for (int b = 0; b < {int(n)}; b++) B_model[a][b] = 0;"
                  f"\n        {B_MODEL_IDX} = 0;")
    state_members = _STATE_MEMBERS_BLOCK.substitute(
        KP=KP, NP=NP, D=D, QN=QN, RES_BITS=RES_BITS, extra_members=extra_members)
    reset_state_fn = _RESET_STATE_FN.substitute(name=name, extra_reset=extra_reset)
    return _HEADER_RB.substitute(
        NAME=name.upper(), name=name, m=int(m), k=int(k), n=int(n),
        A_BITS=A_BITS, RES_BITS=RES_BITS,
        shift=int(shift), bias_decl=bias_decl, bias_expr=bias_expr,
        KP=KP, NP=NP, D=D, QN=QN, PERIOD=PERIOD, TOTAL=TOTAL, LOAD=LOAD,
        LOAD_LEN=LOAD_LEN, LOAD_VALID=LOAD_VALID,
        SLOTS=slots, DBUF="true" if dbuf else "false",
        LD_W=max(1, (LOAD_LEN - 1).bit_length()),
        PASS_W=max(1, (slots - 1).bit_length()),
        ACC_W=max(1, (int(m) - 1).bit_length()),
        B_BEAT_BITS=B_BEAT_BITS, B_BEAT_LANES=B_BEAT_LANES,
        B_ASSERT_SIZE=B_ASSERT_SIZE, B_ASSERT_MSG=B_ASSERT_MSG,
        B_LANES=B_LANES, B_MODEL_STORE=B_MODEL_STORE,
        B_MODEL_ADVANCE=B_MODEL_ADVANCE, B_MODEL_IDX=B_MODEL_IDX,
        W=W, bb_delay=bb_delay,
        guard_open=_HEADER_GUARD_OPEN.substitute(NAME=name.upper()),
        guard_close=_HEADER_GUARD_CLOSE.substitute(NAME=name.upper()),
        pack_a_row_comment="// Lane packing (architecture.md 11.1): lane 0 in the LSBs.",
        pack_a_row_fn=_PACK_A_ROW_FN.substitute(name=name, k=int(k), A_BITS=A_BITS),
        unpack_res_row_fn=_UNPACK_RES_ROW_FN.substitute(
            name=name, n=int(n), W=W, RES_BITS=RES_BITS),
        requant_fn=_REQUANT_FN.substitute(name=name, shift=int(shift), W=W),
        blackbox_bind=_BLACKBOX_BIND.substitute(name=name, bb_delay=bb_delay),
        result_slot_poll_capture=_RESULT_SLOT_POLL_CAPTURE.substitute(name=name),
        state_members=state_members, reset_state_fn=reset_state_fn)


# ── inst/TB building blocks shared across const-weights and runtime-B ─────────
#
# The Catapult top (_INST/_INST_RB) and csim TBs (_TB/_TB_RB/_TB_RB_MULTI) all
# open with the same "#include + CONFIG_T struct" (and the TBs all wrap main()
# in the same CCS_SCVERIFY/mc_testbench guard and end with the same
# PASS/FAIL-and-return footer). These pieces render those spans once; the
# per-variant differences (typedefs, weight source, single- vs multi-call
# body, the pass message) stay inline in each template.

_CONFIG_STRUCT = Template("""\
struct ${name}_config {
    static const unsigned gemm_m = ${m};
    static const unsigned gemm_k = ${k};
    static const unsigned gemm_n = ${n};
    static const unsigned n_in = ${k};
    static const unsigned n_out = ${n};
};""")

_SCVERIFY_OR_STDIO = """\
#ifdef CCS_SCVERIFY
#include "mc_testbench.h"
#include "mc_scverify.h"
#else
#include <stdio.h>
#endif"""

_MAIN_OPEN = """\
#ifdef CCS_SCVERIFY
CCS_MAIN(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif"""

_PASS_FAIL_FOOTER = Template("""\
    if (failed) {
        printf("CMVU CSIM: FAIL\\n");
#ifdef CCS_SCVERIFY
        CCS_RETURN(1);
#else
        return 1;
#endif
    }
    printf("CMVU CSIM: PASS (${pass_msg})\\n");
#ifdef CCS_SCVERIFY
    CCS_RETURN(0);
#else
    return 0;
#endif
}""")


# ── inst top + TB ─────────────────────────────────────────────────────────────

_INST = Template("""\
#include "nnet_types.h"
#include "${name}_gemm_ip.h"

${config_struct}

typedef nnet::array<ac_int<8, true>, ${k}> a_beat_t;
typedef nnet::array<ac_int<${W}, true>, ${n}> res_t;

#pragma hls_design top
void ${name}_inst(ac_channel<a_beat_t> &a_stream,
                  ac_channel<res_t> &res_stream) {
    nnet::${name}_gemm_stream_const_weights<a_beat_t, res_t, ${name}_config>(
        a_stream, res_stream);
}
""")


def gen_inst_cpp(name, m, k, n, result_width):
    return _INST.substitute(
        name=name, m=int(m), k=int(k), n=int(n), W=int(result_width),
        config_struct=_CONFIG_STRUCT.substitute(name=name, m=int(m), k=int(k), n=int(n)))


_TB = Template("""\
${scverify_or_stdio}

#include "nnet_types.h"
#include "${name}_gemm_ip.h"

${config_struct}

typedef nnet::array<ac_int<8, true>, ${k}> a_beat_t;
typedef nnet::array<ac_int<${W}, true>, ${n}> res_t;

static const signed char _W[${k}][${n}] = {
${w_rows}
};
static const int _BIAS[${n}] = {${bias_lit}};

static const int _A[${m}][${k}] = {
${a_rows}
};

static int check_row(const res_t &out, int row, int &failed) {
    for (int j = 0; j < ${n}; j++) {
        int total = _BIAS[j];
        for (int i = 0; i < ${k}; i++)
            total += _A[row][i] * (int)_W[i][j];
        int expected = nnet::${name}_requant_cpp(total);
        if (out[j].to_int() != expected) {
            printf("Mismatch row %d col %d: got %d expected %d\\n",
                   row, j, out[j].to_int(), expected);
            failed = 1;
        }
    }
    return failed;
}

${main_open}
    ac_channel<a_beat_t> a_stream;
    ac_channel<res_t> res_stream;
    int failed = 0;

    for (int i = 0; i < ${m}; i++) {
        a_beat_t beat;
        for (int kk = 0; kk < ${k}; kk++)
            beat[kk] = _A[i][kk];
        a_stream.write(beat);
    }

#ifdef CCS_SCVERIFY
    CCS_DESIGN(${name}_inst)(a_stream, res_stream);
#else
    nnet::${name}_gemm_stream_const_weights<a_beat_t, res_t, ${name}_config>(
        a_stream, res_stream);
#endif

    for (int i = 0; i < ${m}; i++) {
        res_t out = res_stream.read();
        check_row(out, i, failed);
    }

${pass_fail_footer}
""")


def gen_runtime_b_inst_cpp(name, m, k, n, result_width, b_row_major=False):
    b_size = int(n) if b_row_major else int(k)
    return _INST_RB.substitute(
        name=name, m=int(m), k=int(k), n=int(n), B_SIZE=b_size, W=int(result_width),
        config_struct=_CONFIG_STRUCT.substitute(name=name, m=int(m), k=int(k), n=int(n)))


_INST_RB = Template("""\
#include "nnet_types.h"
#include "${name}_gemm_ip.h"

${config_struct}

typedef nnet::array<ac_int<8, true>, ${k}> a_beat_t;
typedef nnet::array<ac_int<8, true>, ${B_SIZE}> b_beat_t;
typedef nnet::array<ac_int<${W}, true>, ${n}> res_t;

#pragma hls_design top
void ${name}_inst(ac_channel<a_beat_t> &a_stream,
                  ac_channel<b_beat_t> &b_stream,
                  ac_channel<res_t> &res_stream) {
    nnet::${name}_gemm_stream_runtime_b<a_beat_t, b_beat_t, res_t,
                                        ${name}_config>(
        a_stream, b_stream, res_stream);
}
""")


def gen_runtime_b_tb(name, m, k, n, weight_codes, bias_codes, shift, geo,
                     seed=42, b_row_major=False):
    """Standalone C++ TB: drives the hls4ml B format for the selected layout
    into the runtime-B entry -- column-major (gemm_n K-high column beats,
    element k of column j == W[k][j]) or row-major (gemm_k N-wide row beats,
    element n of row i == W[i][n]) -- exercising the entry's own layout
    handling end to end.
    """
    W = _codes(weight_codes, k, n)
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(int(m), int(k)), dtype=np.int64)
    w_rows = "\n".join("    {" + ", ".join(str(int(v)) for v in W[i]) + "},"
                       for i in range(int(k)))
    a_rows = "\n".join("    {" + ", ".join(str(int(v)) for v in A[i]) + "},"
                       for i in range(int(m)))
    bias_list = [int(v) for v in (bias_codes if bias_codes is not None
                                  else [0] * n)]

    if b_row_major:
        b_size = int(n)
        n_beats = int(k)
        b_beats = "\n".join(
            "    {" + ", ".join(str(int(W[i][j])) for j in range(int(n))) + "},"
            for i in range(int(k)))
        b_comment = ("// hls4ml B format (row-major): gemm_k N-wide row beats, "
                    "element j of row i == W[i][j].")
        b_write_loop = f"""\
    for (int i = 0; i < {int(k)}; i++) {{
        b_beat_t beat;
        for (int jj = 0; jj < {int(n)}; jj++) beat[jj] = _B_BEATS[i][jj];
        b_stream.write(beat);
    }}"""
    else:
        b_size = int(k)
        n_beats = int(n)
        b_beats = "\n".join(
            "    {" + ", ".join(str(int(W[i][j])) for i in range(int(k))) + "},"
            for j in range(int(n)))
        b_comment = ("// hls4ml B format (column-major): gemm_n K-high column "
                    "beats, element i of column j == W[i][j].")
        b_write_loop = f"""\
    for (int j = 0; j < {int(n)}; j++) {{
        b_beat_t beat;
        for (int kk = 0; kk < {int(k)}; kk++) beat[kk] = _B_BEATS[j][kk];
        b_stream.write(beat);
    }}"""

    return _TB_RB.substitute(
        name=name, m=int(m), k=int(k), n=int(n), b_size=b_size,
        n_beats=n_beats, b_beats=b_beats, b_comment=b_comment,
        b_write_loop=b_write_loop, w_rows=w_rows, a_rows=a_rows,
        bias_lit=", ".join(str(b) for b in bias_list),
        W=geo["result_width"],
        scverify_or_stdio=_SCVERIFY_OR_STDIO,
        config_struct=_CONFIG_STRUCT.substitute(name=name, m=int(m), k=int(k), n=int(n)),
        main_open=_MAIN_OPEN,
        pass_fail_footer=_PASS_FAIL_FOOTER.substitute(pass_msg=f"{int(m)} rows"))


def gen_runtime_b_multi_call_tb(name, m, k, n, bias_codes, shift, geo,
                                seed=42, b_row_major=False, n_calls=3):
    """Multi-call C++ TB: calls the generated runtime-B entry ``n_calls``
    times in a row (fresh channels each call, same persistent ``static
    ccore``), each call with an independently random B and A, checking every
    call's results.

    This is the C++/SCVerify counterpart of
    :func:`golden.generate_runtime_b_multi_call_tb`: hls4ml streams a NEW B
    ahead of every call's A rows (a fresh K for QK, a fresh V for aV), so a
    single-call TB (``gen_runtime_b_tb``) cannot catch an RTL wrapper whose
    load FSM only ever re-arms at reset -- it never exercises a second call.
    """
    W = geo["result_width"]
    rng = np.random.default_rng(seed)
    bias_list = [int(v) for v in (bias_codes if bias_codes is not None
                                  else [0] * n)]
    b_size = int(n) if b_row_major else int(k)
    n_beats = int(k) if b_row_major else int(n)

    def w_rows(Wc):
        return "\n".join(
            "    {" + ", ".join(str(int(v)) for v in Wc[i]) + "},"
            for i in range(int(k)))

    def a_rows(Ac):
        return "\n".join(
            "    {" + ", ".join(str(int(v)) for v in Ac[i]) + "},"
            for i in range(int(m)))

    def b_beats(Wc):
        if b_row_major:
            return "\n".join(
                "    {" + ", ".join(str(int(Wc[i][j])) for j in range(int(n)))
                + "}," for i in range(int(k)))
        return "\n".join(
            "    {" + ", ".join(str(int(Wc[i][j])) for i in range(int(k)))
            + "}," for j in range(int(n)))

    def b_write_loop(c):
        if b_row_major:
            return (f"        for (int i = 0; i < {int(k)}; i++) {{ "
                    f"b_beat_t beat; for (int jj = 0; jj < {int(n)}; jj++) "
                    f"beat[jj] = _B_BEATS{c}[i][jj]; b_stream.write(beat); }}")
        return (f"        for (int j = 0; j < {int(n)}; j++) {{ "
                f"b_beat_t beat; for (int kk = 0; kk < {int(k)}; kk++) "
                f"beat[kk] = _B_BEATS{c}[j][kk]; b_stream.write(beat); }}")

    data_decls, calls_code = [], []
    for c in range(int(n_calls)):
        Wc = rng.integers(-128, 128, size=(int(k), int(n)), dtype=np.int64)
        Ac = rng.integers(-128, 128, size=(int(m), int(k)), dtype=np.int64)
        data_decls.append(f"""\
static const signed char _W{c}[{k}][{n}] = {{
{w_rows(Wc)}
}};
static const signed char _B_BEATS{c}[{n_beats}][{b_size}] = {{
{b_beats(Wc)}
}};
static const int _A{c}[{m}][{k}] = {{
{a_rows(Ac)}
}};""")
        calls_code.append(f"""\
    {{
        ac_channel<a_beat_t> a_stream;
        ac_channel<b_beat_t> b_stream;
        ac_channel<res_t> res_stream;
{b_write_loop(c)}
        for (int i = 0; i < {m}; i++) {{
            a_beat_t beat;
            for (int kk = 0; kk < {k}; kk++) beat[kk] = _A{c}[i][kk];
            a_stream.write(beat);
        }}
#ifdef CCS_SCVERIFY
        CCS_DESIGN({name}_inst)(a_stream, b_stream, res_stream);
#else
        nnet::{name}_gemm_stream_runtime_b<a_beat_t, b_beat_t, res_t,
                                           {name}_config>(
            a_stream, b_stream, res_stream);
#endif
        for (int i = 0; i < {m}; i++) {{
            res_t out = res_stream.read();
            check_row(out, {c}, i, _W{c}, _A{c}, failed);
        }}
    }}""")

    return _TB_RB_MULTI.substitute(
        name=name, m=int(m), k=int(k), n=int(n), b_size=b_size,
        n_calls=int(n_calls), data_decls="\n".join(data_decls),
        calls_code="\n".join(calls_code),
        bias_lit=", ".join(str(b) for b in bias_list), W=W,
        scverify_or_stdio=_SCVERIFY_OR_STDIO,
        config_struct=_CONFIG_STRUCT.substitute(name=name, m=int(m), k=int(k), n=int(n)),
        main_open=_MAIN_OPEN,
        pass_fail_footer=_PASS_FAIL_FOOTER.substitute(
            pass_msg=f"{int(n_calls)} calls x {int(m)} rows"))


_TB_RB_MULTI = Template("""\
${scverify_or_stdio}

#include "nnet_types.h"
#include "${name}_gemm_ip.h"

${config_struct}

typedef nnet::array<ac_int<8, true>, ${k}> a_beat_t;
typedef nnet::array<ac_int<8, true>, ${b_size}> b_beat_t;
typedef nnet::array<ac_int<${W}, true>, ${n}> res_t;

static const int _BIAS[${n}] = {${bias_lit}};

${data_decls}

static void check_row(const res_t &out, int callc, int row,
                      const signed char W[][${n}], const int A[][${k}],
                      int &failed) {
    for (int j = 0; j < ${n}; j++) {
        int total = _BIAS[j];
        for (int i = 0; i < ${k}; i++)
            total += A[row][i] * (int)W[i][j];
        int expected = nnet::${name}_requant_cpp(total);
        if (out[j].to_int() != expected) {
            printf("Mismatch call %d row %d col %d: got %d expected %d\\n",
                   callc, row, j, out[j].to_int(), expected);
            failed = 1;
        }
    }
}

${main_open}
    int failed = 0;

${calls_code}

${pass_fail_footer}
""")


_TB_RB = Template("""\
${scverify_or_stdio}

#include "nnet_types.h"
#include "${name}_gemm_ip.h"

${config_struct}

typedef nnet::array<ac_int<8, true>, ${k}> a_beat_t;
typedef nnet::array<ac_int<8, true>, ${b_size}> b_beat_t;
typedef nnet::array<ac_int<${W}, true>, ${n}> res_t;

static const signed char _W[${k}][${n}] = {
${w_rows}
};
static const int _BIAS[${n}] = {${bias_lit}};
${b_comment}
static const signed char _B_BEATS[${n_beats}][${b_size}] = {
${b_beats}
};
static const int _A[${m}][${k}] = {
${a_rows}
};

static int check_row(const res_t &out, int row, int &failed) {
    for (int j = 0; j < ${n}; j++) {
        int total = _BIAS[j];
        for (int i = 0; i < ${k}; i++)
            total += _A[row][i] * (int)_W[i][j];
        int expected = nnet::${name}_requant_cpp(total);
        if (out[j].to_int() != expected) {
            printf("Mismatch row %d col %d: got %d expected %d\\n",
                   row, j, out[j].to_int(), expected);
            failed = 1;
        }
    }
    return failed;
}

${main_open}
    ac_channel<a_beat_t> a_stream;
    ac_channel<b_beat_t> b_stream;
    ac_channel<res_t> res_stream;
    int failed = 0;

${b_write_loop}
    for (int i = 0; i < ${m}; i++) {
        a_beat_t beat;
        for (int kk = 0; kk < ${k}; kk++) beat[kk] = _A[i][kk];
        a_stream.write(beat);
    }

#ifdef CCS_SCVERIFY
    CCS_DESIGN(${name}_inst)(a_stream, b_stream, res_stream);
#else
    nnet::${name}_gemm_stream_runtime_b<a_beat_t, b_beat_t, res_t,
                                        ${name}_config>(
        a_stream, b_stream, res_stream);
#endif

    for (int i = 0; i < ${m}; i++) {
        res_t out = res_stream.read();
        check_row(out, i, failed);
    }

${pass_fail_footer}
""")


def gen_tb(name, m, k, n, weight_codes, bias_codes, shift, result_width,
           seed=42):
    W = _codes(weight_codes, k, n)
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(int(m), int(k)), dtype=np.int64)
    w_rows = "\n".join("    {" + ", ".join(str(int(v)) for v in W[i]) + "},"
                       for i in range(int(k)))
    a_rows = "\n".join("    {" + ", ".join(str(int(v)) for v in A[i]) + "},"
                       for i in range(int(m)))
    bias_list = [int(v) for v in (bias_codes if bias_codes is not None
                                  else [0] * n)]
    return _TB.substitute(
        name=name, m=int(m), k=int(k), n=int(n),
        w_rows=w_rows, a_rows=a_rows,
        bias_lit=", ".join(str(b) for b in bias_list),
        W=int(result_width),
        scverify_or_stdio=_SCVERIFY_OR_STDIO,
        config_struct=_CONFIG_STRUCT.substitute(name=name, m=int(m), k=int(k), n=int(n)),
        main_open=_MAIN_OPEN,
        pass_fail_footer=_PASS_FAIL_FOOTER.substitute(pass_msg=f"{int(m)} rows"))


# ── Catapult tcl ──────────────────────────────────────────────────────────────

_TCL = Template("""\
set project_name "${name}_proj"
set solution_name "${name}_sol"

project new -name $$project_name
solution new $$solution_name
solution options defaults
solution options set /Output/OutputVerilog true
solution options set /Output/GenerateCycleNetlist false

# SCVerify (C++ TB vs RTL cosim); must be required before go analyze.
flow package require /SCVerify

options set Input/CompilerFlags {-DBLACKBOX_FLOW}

solution file add ./${name}_inst.cpp -type C++
solution file add ./${name}_tb.cpp -type C++
# Generated wrapper RTL (sim + blackbox binding), excluded from the C++ netlist.
solution file add ./${name}_core.sv -type SystemVerilog -exclude true
# Vendored CMVU block RTL: simulation models for the blackbox, one copy per
# package root (shipped by the target's finalize/sources_tcl).
solution file add ../cmvu_mode1.sv -type SystemVerilog -exclude true
solution file add ../cmvu_w_mem.sv -type SystemVerilog -exclude true
solution file add ../cmvu_regbank.sv -type SystemVerilog -exclude true

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
solution design set ${name}_inst -top
go analyze
go compile

solution library add mgc_Altera-Agilex-2_beh -- -rtlsyntool Quartus -manufacturer Altera -family Agilex -speed 2 -part AGFB014R24B2E2V
solution library add Altera_M20K
solution library add Altera_MLAB
solution library add Altera_DIST
solution library add Altera_ROMS
go libraries

directive set -CLOCKS {clk {-CLOCK_PERIOD ${clock_period} -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME ${clock_high} -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND sync -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low}}

# Stream ports map to ccs_ioport resources (const-weights top has no B/bias).
directive set /${name}_inst/a_stream:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_wait
directive set /${name}_inst/res_stream:rsc -MAP_TO_MODULE ccs_ioport.ccs_out_wait
${b_stream_map}

go assembly
go architect
go allocate
go schedule
go extract

# RTL co-simulation (QuestaSim/msim) of the generated wrapper + vendored
# cmvu_mode1 blocks.
flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim sim

project save
puts "${name} Catapult run complete."
""")


def gen_tcl(name, clock_period_ns=3.0, runtime_b=False):
    b_map = (f"directive set /{name}_inst/b_stream:rsc "
             f"-MAP_TO_MODULE ccs_ioport.ccs_in_wait"
             if runtime_b else "")
    return _TCL.substitute(name=name, clock_period=float(clock_period_ns),
                           clock_high=float(clock_period_ns) / 2.0,
                           b_stream_map=b_map)


# ── batch artifacts ───────────────────────────────────────────────────────────


def gen_blackbox_tcl(items):
    lines = ["# Generated by gemm-ip-gen cmvu target: wrapper cores (sim/blackbox)."]
    seen = set()
    for item in items:
        name = item["name"]
        if name in seen:
            continue
        seen.add(name)
        lines.append(f'solution file add [file join $script_dir {name} '
                     f'{name}_core.sv] -type SystemVerilog -exclude true')
    return "\n".join(lines) + "\n"


def gen_sources_tcl():
    lines = ["# cmvu target: vendored block RTL simulation models (one per design)."]
    for sv in VENDORED_SV:
        lines.append(f'solution file add [file join [file dirname [info script]] '
                     f'{sv}] -type SystemVerilog -exclude true')
    # The VTR-facing cmvu_mode1 stub ships in the package root for VTR only and
    # must NOT be added here: -exclude true only keeps a file out of synthesis,
    # SCVerify still compiles it, and its port-only `module cmvu_mode1` then
    # replaces the real block in the simulation library.
    return "\n".join(lines) + "\n"


def gen_cmvu_mode1_vtr_model():
    """VTR-facing hard-block model for ``cmvu_mode1``: the exact port list and
    the parameter values the generated wrapper (``rtl.generate_core``)
    instantiates it with (``geometry.py``'s physical constants -- K_PHYS,
    N_PHYS, MEM_TILES, and the IN/COEF/ACC/BIAS/RESULT/SHIFT widths -- are the
    single source, so this can never drift from what the wrapper actually
    instantiates), with ``(* blackbox *)`` and no body.

    VTR's front end (parmys/yosys) discards the body and maps instances of
    this module name to the arch's ``cmvu_mode1`` pb_type -- exactly
    tensor_slice's ``tensor_slice_int8_atlas.v`` pattern for its hard block.
    Unlike that file, this one carries no behavioral twin: Catapult
    csim/SCVerify already has the real vendored ``rtl_static/cmvu_mode1.sv``
    for simulation, so there is nothing this VTR-only file needs to model.
    The wrapper's per-instance parameter override (``CASCADE_EN`` on the
    cascade head) and per-instance port connections come from the
    instantiation, not this stub -- only the port list/widths and the
    defaults need to match.
    """
    g = _geometry
    return f"""\
// Generated by gemm-ip-gen cmvu target -- do not edit.
// VTR-facing hard-block model for cmvu_mode1: port list and parameter
// defaults from geometry.py's physical constants (the single source the
// wrapper itself instantiates against). (* blackbox *) tells VTR's front
// end (parmys/yosys) to discard the body and map instances to the arch
// model of the same name -- simulators (iverilog/Questa) never see this
// file; they use the real vendored rtl_static/cmvu_mode1.sv instead.
(* blackbox *)
module cmvu_mode1 #(
    parameter int unsigned IN_WIDTH        = {g.IN_WIDTH},
    parameter int unsigned COEF_WIDTH      = {g.COEF_WIDTH},
    parameter int unsigned ACC_WIDTH       = {g.ACC_WIDTH},
    parameter int unsigned BIAS_WIDTH      = {g.BIAS_WIDTH},
    parameter int unsigned RESULT_WIDTH    = {g.RESULT_WIDTH},
    parameter int unsigned SHIFT_WIDTH     = {g.SHIFT_WIDTH},
    parameter int unsigned K               = {g.K_PHYS},
    parameter int unsigned N               = {g.N_PHYS},
    parameter int unsigned M_MEM_TILES     = {g.MEM_TILES},

    parameter bit REG_MULT_PRESENT = 1'b1,
    parameter bit REG_RED_PRESENT  = 1'b1,
    parameter bit REG_OUT_PRESENT  = 1'b1,

    parameter bit CASCADE_EN       = 1'b1,
    parameter bit REQUANT_PRESENT  = 1'b1
) (
    input  wire                          clk,
    input  wire                          rst,

    input  wire                          valid,
    input  wire                          acc_first,
    input  wire                          acc_last,
    input  wire [K*IN_WIDTH-1:0]         a_in,
    input  wire [63:0]                   b_in,
    input  wire                          w_we,
    input  wire                          w_load_start,
    input  wire                          w_col_major,
    input  wire                          w_dual_tile,
    input  wire [$clog2(M_MEM_TILES < 2 ? 2 : M_MEM_TILES)-1:0] tile_sel,
    input  wire [$clog2(M_MEM_TILES < 2 ? 2 : M_MEM_TILES)-1:0] w_tile_sel,
    input  wire                          a_signed,
    input  wire                          b_signed,
    input  wire [SHIFT_WIDTH-1:0]        shift_amt,
    input  wire [$clog2(RESULT_WIDTH < 2 ? 2 : RESULT_WIDTH)-1:0] out_w,
    input  wire [N*ACC_WIDTH-1:0]        cascade_in,
    input  wire [N*BIAS_WIDTH-1:0]       bias_in,

    output wire [N*RESULT_WIDTH-1:0]     y_out,
    output wire [N*ACC_WIDTH-1:0]        cascade_out,
    output wire                          y_valid,
    output wire                          done
);
endmodule
"""


def _dispatch_condition(item):
    """Local copy of tensor_slice's dispatch condition -- cmvu is self-contained
    and must not import from tensor_slice. Prefers the manifest's gemm_ip_id when
    present; falls back to shape-only matching for synthetic/legacy configs that
    never got an index assigned."""
    shape_condition = (
        f"CONFIG_T::gemm_m == {item['m']} &&\n"
        f"                  CONFIG_T::gemm_k == {item['k']} &&\n"
        f"                  CONFIG_T::gemm_n == {item['n']}"
    )
    if item.get("gemm_ip_index") is not None:
        return (
            f"CONFIG_T::gemm_ip_id == {item['gemm_ip_index']} &&\n"
            f"                  {shape_condition}"
        )
    return shape_condition


def gen_combined_header(items):
    """Combined package header: #includes each per-layer header, then defines
    the four hls4ml Catapult call sites (nnet::gemm_stream,
    nnet::gemm_stream_const_weights, nnet::gemm_array,
    nnet::gemm_array_const_weights -- see nnet_gemm_ip.h / nnet_gemm_stream.h)
    by dispatching on CONFIG_T::gemm_ip_id/gemm_m/gemm_k/gemm_n to the matching
    per-layer function.

    cmvu is stream-only (no io_parallel core), so gemm_array and
    gemm_array_const_weights are static_assert-only stubs: any io_parallel
    layer routed at a cmvu package is a config-time error, not something this
    header can service.

    Per-layer functions already match the frontend's call signatures exactly
    (argument order and template-parameter order), so each branch just
    forwards the call -- no argument repacking needed, unlike tensor_slice's
    array-based buffered-B wrapper.
    """
    includes = "\n".join(
        f'#include "{item["name"]}/{item["name"]}_gemm_ip.h"' for item in items)

    const_weights_branches = []
    runtime_b_branches = []
    for item in items:
        if item.get("weights_in_core"):
            const_weights_branches.append(f"""\
    if constexpr ({_dispatch_condition(item)}) {{
        nnet::{item["name"]}_gemm_stream_const_weights<data_T, res_T, CONFIG_T>(
            data_stream, res_stream);
    }}""")
        else:
            runtime_b_branches.append(f"""\
    if constexpr ({_dispatch_condition(item)}) {{
        nnet::{item["name"]}_gemm_stream_runtime_b<data0_T, data1_T, res_T, CONFIG_T>(
            a_stream, b_stream, res_stream);
    }}""")

    const_weights_text = " else ".join(const_weights_branches)
    if not const_weights_text:
        const_weights_text = """\
    static_assert(CONFIG_T::gemm_m == 0,
                  "No generated cmvu weight-stationary GEMM IP implementation "
                  "is present in this package.");"""
    else:
        const_weights_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated cmvu weight-stationary GEMM IP "
                      "implementation matches this CONFIG_T.");
    }"""

    runtime_b_text = " else ".join(runtime_b_branches)
    if not runtime_b_text:
        runtime_b_text = """\
    static_assert(CONFIG_T::gemm_m == 0,
                  "No generated cmvu runtime-B GEMM IP implementation is "
                  "present in this package.");"""
    else:
        runtime_b_text += """ else {
        static_assert(CONFIG_T::gemm_m == 0,
                      "No generated cmvu runtime-B GEMM IP implementation "
                      "matches this CONFIG_T.");
    }"""

    return f"""\
// Generated by gemm-ip-gen cmvu target: combined per-layer headers plus the
// four hls4ml Catapult GEMM IP call sites (see nnet_gemm_ip.h / nnet_gemm_stream.h).
#ifndef GEMM_IP_COMBINED_H_
#define GEMM_IP_COMBINED_H_

#include "ac_channel.h"
{includes}

namespace nnet {{

// hls4ml call site: nnet::gemm_stream_const_weights<data_T, res_T, CONFIG_T>(a, res).
// Weight-stationary (weights_in_core=True) layers: baked ROM + bias live in
// the per-layer ccore; no bias argument here (matches the frontend signature).
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream) {{
{const_weights_text}
}}

// hls4ml call site: nnet::gemm_stream<a_T, b_T, res_T, CONFIG_T>(a, b, res).
// Runtime-B (weights_in_core=False) layers: B streams in ahead of A, no bias
// (a two-operand GEMM never owns one).
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(ac_channel<data0_T> &a_stream, ac_channel<data1_T> &b_stream,
                 ac_channel<res_T> &res_stream) {{
{runtime_b_text}
}}

// cmvu has no io_parallel interface (stream-only datapath); any layer routed
// here at a cmvu package is a config-time error, not something this header
// can service.
template <class a_row_T, class b_col_T, class res_row_T, typename CONFIG_T>
void gemm_array(a_row_T a_rows[CONFIG_T::gemm_m], b_col_T weight_cols[CONFIG_T::gemm_n],
                res_row_T results[CONFIG_T::gemm_m]) {{
    static_assert(CONFIG_T::gemm_m == 0,
                  "cmvu has no io_parallel interface; gemm_array is not "
                  "implemented by this package.");
}}

template <class a_row_T, class res_row_T, typename CONFIG_T>
void gemm_array_const_weights(a_row_T a_rows[CONFIG_T::gemm_m], res_row_T results[CONFIG_T::gemm_m]) {{
    static_assert(CONFIG_T::gemm_m == 0,
                  "cmvu has no io_parallel interface; gemm_array_const_weights "
                  "is not implemented by this package.");
}}

}} // namespace nnet

#endif // GEMM_IP_COMBINED_H_
"""


def gen_integration_manifest(items):
    cores = []
    for item in items:
        cores.append({
            "name": item["name"],
            "entity": f'{item["name"]}_core',
            "rtl": f'{item["name"]}/{item["name"]}_core.sv',
            "m": item.get("m"), "k": item.get("k"), "n": item.get("n"),
            "kfold": item.get("kfold"), "nfold": item.get("nfold"),
            "k_spatial": item.get("k_spatial"),
            "n_spatial": item.get("n_spatial"),
            "k_passes": item.get("k_passes"),
            "n_passes": item.get("n_passes"),
            "slots_per_block": item.get("slots_per_block"),
            "interface": "stream",
            "reset": {"name": "rst", "sync_active": "high"},
        })
    return json.dumps({
        "package_format": "single_top_catapult_blackboxes",
        "header": "gemm_ip_combined.h",
        "sources_tcl": "gemm_ip_sources.tcl",
        "cores": cores,
    }, indent=2)


# ── package generation ────────────────────────────────────────────────────────


def generate_catapult_pkg(m, k, n, name, output_dir, kfold, nfold,
                          weight_matrix, bias_codes=None, shift=0,
                          a_signed=True, b_signed=True, clock_period_ns=3.0,
                          runtime_b=False, result_width=None,
                          b_row_major=False, **kwargs):
    if str(kwargs.get("interface", "stream")).lower() != "stream":
        raise NotImplementedError(
            f"{name}: cmvu target supports interface='stream' only "
            f"(got {kwargs.get('interface')!r}); the array interface is not "
            f"implemented.")
    geo = _geometry.resolve_geometry(m, k, n, kfold, nfold, name,
                                     result_width=result_width)
    if not (0 <= int(shift) <= _geometry.MAX_SHIFT):
        raise ValueError(f"{name}: shift must be 0..{_geometry.MAX_SHIFT}, "
                         f"got {shift}")
    if runtime_b:
        # No baked weights: the csim/SCVerify testbench supplies a self-contained
        # test B (a real deployment streams B at runtime). Use the given matrix
        # when present (reproducible tests), else a name-stable synthetic one.
        if weight_matrix is not None:
            W = _codes(weight_matrix, k, n)
        else:
            seed = zlib.crc32(str(name).encode()) & 0xFFFFFFFF
            W = np.random.default_rng(seed).integers(-128, 128, size=(int(k), int(n)), dtype=np.int64)
    else:
        W = _codes(weight_matrix, k, n)

    pkg_dir = Path(output_dir) / name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    core = _rtl.generate_core(m, k, n, kfold, nfold, W, bias_codes=bias_codes,
                              shift=shift, a_signed=a_signed, b_signed=b_signed,
                              module_name=f"{name}_core", name=name,
                              runtime_b=runtime_b, b_row_major=b_row_major,
                              result_width=geo["result_width"])
    # Same blackbox budget as tensor_slice: 70% of the period, but always
    # leaving Catapult >= 1.5 ns for the glue around the block, below which
    # its scheduler rejects the component.
    period = float(clock_period_ns)
    bb_delay = round(min(0.7 * period, period - 1.5), 2)

    (pkg_dir / f"{name}_core.sv").write_text(core)
    (pkg_dir / "nnet_types.h").write_text(gen_nnet_types_header())
    if runtime_b:
        (pkg_dir / f"{name}_gemm_ip.h").write_text(
            gen_runtime_b_header(name, m, k, n, bias_codes, shift, geo,
                                 bb_delay=bb_delay, b_row_major=b_row_major))
        (pkg_dir / f"{name}_inst.cpp").write_text(
            gen_runtime_b_inst_cpp(name, m, k, n, geo["result_width"],
                                   b_row_major=b_row_major))
        (pkg_dir / f"{name}_tb.cpp").write_text(
            gen_runtime_b_tb(name, m, k, n, W, bias_codes, shift, geo,
                             b_row_major=b_row_major))
    else:
        (pkg_dir / f"{name}_gemm_ip.h").write_text(
            gen_public_header(name, m, k, n, W, bias_codes, shift, geo,
                              bb_delay=bb_delay))
        (pkg_dir / f"{name}_inst.cpp").write_text(
            gen_inst_cpp(name, m, k, n, geo["result_width"]))
        (pkg_dir / f"{name}_tb.cpp").write_text(
            gen_tb(name, m, k, n, W, bias_codes, shift, geo["result_width"]))
    (pkg_dir / "run_catapult.tcl").write_text(
        gen_tcl(name, clock_period_ns=clock_period_ns, runtime_b=runtime_b))
    return pkg_dir
