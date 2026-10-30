#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
VENDOR_NAME="cannsim_950"
SOC_VERSION="Ascend950"
VENDOR_INSTALL_SUFFIX="${VENDOR_NAME}_math"

usage() {
    echo "Usage: bash $0 <op_name> [options]"
    echo ""
    echo "Build, install, compile example, and run cannsim for an operator on Ascend950."
    echo ""
    echo "Arguments:"
    echo "    op_name                 Operator name, e.g. add_example"
    echo ""
    echo "Options:"
    echo "    --skip-build            Skip operator package build and install"
    echo "    --skip-cannsim          Stop after compiling example, do not run cannsim"
    echo "    --gen-report            Generate cannsim performance report (--gen-report)"
    echo "    --clean                 Clean previous vendor package before building"
    echo "    -j <num>                Parallel build jobs (default: 16)"
    echo "    -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "    bash $0 add_example"
    echo "    bash $0 add_example --skip-build"
    echo "    bash $0 add_example --gen-report"
    echo "    bash $0 add_example --clean -j8"
    echo ""
    echo "Output:"
    echo "    Vendor package:  build_out/cann-ops-math-${VENDOR_NAME}_linux-*.run"
    echo "    Install path:    \${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_INSTALL_SUFFIX}/"
    echo "    Executable:      build/test_aclnn_<op_name>"
    echo "    Cannsim result:  build/cannsim_*_<op_name>/"
}

OP_NAME=""
SKIP_BUILD=FALSE
SKIP_CANNSIM=FALSE
GEN_REPORT=FALSE
CLEAN=FALSE
JOBS=16

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --skip-build)
            SKIP_BUILD=TRUE
            shift
            ;;
        --skip-cannsim)
            SKIP_CANNSIM=TRUE
            shift
            ;;
        --gen-report)
            GEN_REPORT=TRUE
            shift
            ;;
        --clean)
            CLEAN=TRUE
            shift
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        -*)
            echo "[ERROR] Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            if [[ -z "$OP_NAME" ]]; then
                OP_NAME="$1"
            else
                echo "[ERROR] Unexpected argument: $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$OP_NAME" ]]; then
    echo "[ERROR] Operator name is required."
    usage
    exit 1
fi

if [[ -z "$ASCEND_HOME_PATH" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Please run: source <cann_path>/set_env.sh"
    exit 1
fi

cd "${PROJECT_ROOT}"

VENDOR_DIR="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_INSTALL_SUFFIX}"
VENDOR_LIB_DIR="${VENDOR_DIR}/op_api/lib"
RUN_PKG_PATTERN="build_out/cann-ops-math-${VENDOR_NAME}_linux-*.run"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

elapsed_str() {
    local seconds=$1
    local mins=$((seconds / 60))
    local secs=$((seconds % 60))
    if [[ $mins -gt 0 ]]; then
        printf "%dm%02ds" $mins $secs
    else
        printf "%ds" $secs
    fi
}

TOTAL_START=$(date +%s)

echo "============================================================"
    echo "[cannsim] Operator     : ${OP_NAME}"
    echo "[cannsim] Target SoC   : ${SOC_VERSION}"
    echo "[cannsim] Vendor       : ${VENDOR_NAME} (install: ${VENDOR_INSTALL_SUFFIX})"
    echo "[cannsim] Project root : ${PROJECT_ROOT}"
echo "============================================================"

# ==================== Stage 0: Clean (optional) ====================
if [[ "$CLEAN" == "TRUE" ]]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "[$(timestamp)] [Stage 0] Cleaning previous build and vendor"
    echo "------------------------------------------------------------"
    if [[ -d "$VENDOR_DIR" ]]; then
        echo "[cannsim] Removing installed vendor: ${VENDOR_DIR}"
        rm -rf "$VENDOR_DIR"
    fi
    RUN_PKG=$(ls ${RUN_PKG_PATTERN} 2>/dev/null || true)
    if [[ -n "$RUN_PKG" ]]; then
        echo "[cannsim] Removing run package: ${RUN_PKG}"
        rm -f ${RUN_PKG_PATTERN}
    fi
fi

# ==================== Stage 1: Build operator package ====================
if [[ "$SKIP_BUILD" == "FALSE" ]]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "[$(timestamp)] [Stage 1] Building operator package"
    echo "------------------------------------------------------------"
    STAGE1_START=$(date +%s)

    bash build.sh --pkg --soc=${SOC_VERSION} --vendor_name=${VENDOR_NAME} --ops=${OP_NAME} -j${JOBS}

    STAGE1_END=$(date +%s)
    STAGE1_ELAPSED=$((STAGE1_END - STAGE1_START))
    echo "[cannsim] Stage 1 (build package) finished in $(elapsed_str $STAGE1_ELAPSED)"

    # ==================== Stage 2: Install operator package ====================
    echo ""
    echo "------------------------------------------------------------"
    echo "[$(timestamp)] [Stage 2] Installing operator package"
    echo "------------------------------------------------------------"
    STAGE2_START=$(date +%s)

    RUN_PKG=$(ls ${PROJECT_ROOT}/${RUN_PKG_PATTERN} 2>/dev/null || true)
    if [[ -z "$RUN_PKG" ]]; then
        echo "[ERROR] Run package not found: ${PROJECT_ROOT}/${RUN_PKG_PATTERN}"
        exit 1
    fi
    echo "[cannsim] Installing: ${RUN_PKG}"
    ${RUN_PKG} --force

    STAGE2_END=$(date +%s)
    STAGE2_ELAPSED=$((STAGE2_END - STAGE2_START))
    echo "[cannsim] Stage 2 (install package) finished in $(elapsed_str $STAGE2_ELAPSED)"
else
    echo ""
    echo "[cannsim] Stage 1 & 2 skipped (--skip-build)"
    STAGE1_ELAPSED=0
    STAGE2_ELAPSED=0
fi

# ==================== Stage 3: Compile example ====================
echo ""
echo "------------------------------------------------------------"
echo "[$(timestamp)] [Stage 3] Compiling example (--noexec)"
echo "------------------------------------------------------------"
STAGE3_START=$(date +%s)

export LD_LIBRARY_PATH="${VENDOR_LIB_DIR}:${LD_LIBRARY_PATH}"
bash build.sh --run_example ${OP_NAME} eager cust --vendor_name=${VENDOR_NAME} --noexec

EXECUTABLE="${PROJECT_ROOT}/build/test_aclnn_${OP_NAME}"
if [[ ! -f "$EXECUTABLE" ]]; then
    echo "[ERROR] Executable not found: ${EXECUTABLE}"
    exit 1
fi
echo "[cannsim] Executable ready: ${EXECUTABLE}"

STAGE3_END=$(date +%s)
STAGE3_ELAPSED=$((STAGE3_END - STAGE3_START))
echo "[cannsim] Stage 3 (compile example) finished in $(elapsed_str $STAGE3_ELAPSED)"

# ==================== Stage 4: Run cannsim ====================
if [[ "$SKIP_CANNSIM" == "FALSE" ]]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "[$(timestamp)] [Stage 4] Running cannsim simulation"
    echo "------------------------------------------------------------"
    STAGE4_START=$(date +%s)

    CANNSIM_ARGS="record ${EXECUTABLE} -s ${SOC_VERSION}"
    if [[ "$GEN_REPORT" == "TRUE" ]]; then
        CANNSIM_ARGS="${CANNSIM_ARGS} --gen-report"
    fi

    echo "[cannsim] Command: cannsim ${CANNSIM_ARGS}"
    python3 -c "from cannsim import main; main.main()" ${CANNSIM_ARGS}

    CANNSIM_OUTPUT=$(ls -d ${PROJECT_ROOT}/build/cannsim_*_test_aclnn_${OP_NAME} 2>/dev/null | tail -1 || true)
    if [[ -n "$CANNSIM_OUTPUT" ]]; then
        echo "[cannsim] Result directory: ${CANNSIM_OUTPUT}"
        echo "[cannsim] Log file: ${CANNSIM_OUTPUT}/cannsim.log"
        if [[ "$GEN_REPORT" == "TRUE" ]]; then
            REPORT_DIR="${CANNSIM_OUTPUT}/report"
            if [[ -d "$REPORT_DIR" ]]; then
                echo "[cannsim] Report (trace_core0.json): ${REPORT_DIR}/trace_core0.json"
            fi
        fi
    fi

    STAGE4_END=$(date +%s)
    STAGE4_ELAPSED=$((STAGE4_END - STAGE4_START))
    echo "[cannsim] Stage 4 (cannsim simulation) finished in $(elapsed_str $STAGE4_ELAPSED)"
else
    echo ""
    echo "[cannsim] Stage 4 skipped (--skip-cannsim)"
    STAGE4_ELAPSED=0
fi

# ==================== Summary ====================
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================================"
echo "                    End-to-End Summary"
echo "============================================================"
echo "  Operator       : ${OP_NAME}"
echo "  Target SoC     : ${SOC_VERSION}"
echo "  Vendor         : ${VENDOR_NAME} -> ${VENDOR_INSTALL_SUFFIX}"
echo "  Executable     : ${EXECUTABLE}"
if [[ "$SKIP_BUILD" == "FALSE" ]]; then
echo "  Run Package    : ${RUN_PKG}"
fi
if [[ -n "${CANNSIM_OUTPUT:-}" ]]; then
echo "  Cannsim Output : ${CANNSIM_OUTPUT}"
fi
echo "------------------------------------------------------------"
echo "  Stage 1 (build package)   : $(elapsed_str $STAGE1_ELAPSED)"
echo "  Stage 2 (install package) : $(elapsed_str $STAGE2_ELAPSED)"
echo "  Stage 3 (compile example) : $(elapsed_str $STAGE3_ELAPSED)"
echo "  Stage 4 (cannsim sim)     : $(elapsed_str $STAGE4_ELAPSED)"
echo "------------------------------------------------------------"
echo "  TOTAL E2E Time            : $(elapsed_str $TOTAL_ELAPSED)"
echo "============================================================"
