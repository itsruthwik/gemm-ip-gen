#!/usr/bin/env bash
# run_all_synth_scverify.sh
# Runs Catapult synthesis + SCVerify (RTL-v vs C++) for all 4 GEMM IP packages.
# GLIBCXX and QuestaSim path fixes per conductor/tech-stack.md.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output_synth"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH=/home/tools/siemens/catapult/Mgc_home/pkgs/dcs_gcc/gcc-13.4.0/lib64:${LD_LIBRARY_PATH}
export MGC_HOME=/home/tools/siemens/catapult/Mgc_home
export PATH=/opt/siemens/questasim/linux_x86_64:${PATH}

PACKAGES=(gemm_8x8x8 gemm_6x6x6 gemm_16x8x8 gemm_14x6x6)
SCV_MAKEFILE="Verify_rtl_v_msim.mk"

echo "================================================================"
echo "  GEMM IP Batch Synthesis + SCVerify"
echo "  $(date)"
echo "================================================================"

FAILED=()
PASSED=()

for PKG in "${PACKAGES[@]}"; do
    PKG_DIR="${OUTPUT_DIR}/${PKG}"
    LOG_SYNTH="${LOG_DIR}/${PKG}_synth.log"
    LOG_SCV="${LOG_DIR}/${PKG}_scverify.log"

    # ── Skip synthesis if project already exists ──────────────────────────────
    SOL_DIR="${PKG_DIR}/${PKG}_proj/${PKG}_sol.v1"
    if [ -d "${SOL_DIR}" ]; then
        echo ""
        echo "────────────────────────────────────────────────────────────────"
        echo "  [SYNTH] ${PKG}  (already synthesised, skipping)"
        echo "────────────────────────────────────────────────────────────────"
    else
        echo ""
        echo "────────────────────────────────────────────────────────────────"
        echo "  [SYNTH] ${PKG}"
        echo "────────────────────────────────────────────────────────────────"
        pushd "${PKG_DIR}" > /dev/null
        if catapult -shell -file run_catapult.tcl > "${LOG_SYNTH}" 2>&1; then
            echo "  [OK]   synthesis passed  →  ${LOG_SYNTH}"
        else
            echo "  [FAIL] synthesis failed  →  ${LOG_SYNTH}"
            FAILED+=("${PKG}:synth")
            popd > /dev/null
            continue
        fi
        popd > /dev/null
    fi

    # ── SCVerify ──────────────────────────────────────────────────────────────
    SCV_DIR="${SOL_DIR}/scverify"
    SCV_MK="${SCV_DIR}/${SCV_MAKEFILE}"
    # ccs_env.mk lives one level up; symlink it into scverify/ if missing
    if [ ! -f "${SCV_DIR}/ccs_env.mk" ]; then
        ln -sf "${SOL_DIR}/ccs_env.mk" "${SCV_DIR}/ccs_env.mk"
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  [SCVERIFY] ${PKG}"
    echo "────────────────────────────────────────────────────────────────"

    if [ ! -f "${SCV_MK}" ]; then
        echo "  [FAIL] SCVerify Makefile not found: ${SCV_MK}"
        FAILED+=("${PKG}:scverify")
        continue
    fi

    pushd "${SCV_DIR}" > /dev/null
    if make -f "${SCV_MAKEFILE}" > "${LOG_SCV}" 2>&1; then
        echo "  [OK]   scverify passed  →  ${LOG_SCV}"
        PASSED+=("${PKG}")
    else
        echo "  [FAIL] scverify failed  →  ${LOG_SCV}"
        echo "         last 10 lines:"
        tail -10 "${LOG_SCV}" | sed 's/^/         /'
        FAILED+=("${PKG}:scverify")
    fi
    popd > /dev/null
done

echo ""
echo "================================================================"
echo "  Summary  $(date)"
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  All packages: PASSED"
else
    echo "  Passed: ${PASSED[*]:-none}"
    echo "  FAILED:"
    for F in "${FAILED[@]}"; do echo "    - ${F}"; done
fi
echo ""
echo "  Logs: ${LOG_DIR}/"
