#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
PROJECT_ROOT="${SCRIPT_DIR}"
PR_FILELIST="${PROJECT_ROOT}/.local_check_filelist.txt"

# ============================================================
# 参数变量定义
# ============================================================
DEFAULT_SOC="ascend910b"
DEFAULT_JOBS=16

SOC=""
JOBS=${DEFAULT_JOBS}
CANN_3RD_LIB_PATH="${PROJECT_ROOT}/third_party"
ASCEND_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}"
SKIP_BUILD=false
SKIP_LLT=false
SKIP_CANNSIM=false
OPS=""

LLT_A900_MODE=false
LLT_ST_MODE=false
LLT_EXP_UT_MODE=false
LLT_KERNEL_UT_MODE=false
LLT_STD_UT_MODE=false
CANNSIM_MODE=false

# compile.sh 功能覆盖参数
FILE_LIST=""
JIT_MODE=false
EXPERIMENTAL_MODE=false
SINGLE_MODE=false
A5_MODE=false
NORMAL_MODE=false

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================
# 公共函数
# ============================================================
print_header() {
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${CYAN}>>> [$1] $2${NC}"
    echo ""
}

print_ok() {
    echo -e "  ${GREEN}[PASS]${NC} $1"
}

print_warn() {
    echo -e "  ${YELLOW}[SKIP]${NC} $1"
}

print_fail() {
    echo -e "  ${RED}[FAIL]${NC} $1"
}

print_info() {
    echo -e "  ${CYAN}[INFO]${NC} $1"
}

soc_to_chip_short() {
    local soc="$1"
    case "${soc}" in
        ascend910b)   echo "910b" ;;
        ascend910_93) echo "910_93" ;;
        ascend950)    echo "950" ;;
        ascend310p)   echo "310p" ;;
        ascend310b)   echo "310b" ;;
        ascend910)    echo "910" ;;
        *)            echo "${soc#ascend}" ;;
    esac
}

op_supports_soc() {
    local op="$1"
    local soc="$2"
    local is_experimental="${3:-false}"
    local prefixes=("math" "conversion" "random")
    if [[ "$is_experimental" == "true" ]]; then
        prefixes=("experimental/math" "experimental/conversion" "experimental/random")
    fi
    for prefix in "${prefixes[@]}"; do
        local def_dir="${PROJECT_ROOT}/${prefix}/${op}/op_host"
        [[ -d "${def_dir}" ]] || continue
        for def_file in "${def_dir}"/*_def.cpp; do
            [[ -f "${def_file}" ]] || continue
            grep -q "AddConfig(\"${soc}\"" "${def_file}" && return 0
        done
    done
    return 1
}

filter_ops_by_soc() {
    local ops="$1"
    local soc="$2"
    local is_experimental="${3:-false}"
    local ops_arr=(${ops//,/ })
    local filtered=()
    for op in "${ops_arr[@]}"; do
        if op_supports_soc "$op" "$soc" "$is_experimental"; then
            filtered+=("${op}")
        else
            print_info "On [${soc}], [${op}] not supported, skipping" >&2
        fi
    done
    local result="${filtered[*]}"
    echo "${result// /,}"
}

make_filtered_pr_file() {
    local pr_file="$1"
    local filtered_ops="$2"
    local out_file="$3"
    > "${out_file}"
    local ops_arr=(${filtered_ops//,/ })
    while IFS= read -r line; do
        line=$(echo "$line" | xargs)
        [[ -z "$line" ]] && continue
        for op in "${ops_arr[@]}"; do
            if [[ "$line" == *"/$op/"* ]]; then
                echo "$line" >> "${out_file}"
                break
            fi
        done
    done < "${pr_file}"
}

usage() {
    cat <<EOF
Usage: bash local_check.sh [options]

Local verification script for ops-math development.
Fully covers compile.sh and LLT Flow functionality.

Change Detection Options:
  -f <file>                 Specify file list (default: auto-detect from git diff)

Build Mode Options (single task, default runs all):
  --jit                     JIT compilation only (compile.sh monitor/else mode)
  --experimental            Experimental compilation only (compile.sh experimentalX)
  --single                  Single operator packaging only (compile.sh singleX, check_pkg.sh)
  --a5                      A5/ascend950 arch35 only (compile.sh A5X, compile_a5_pkg.sh)
  --normal                  Normal compilation only (bash build.sh --pkg --soc, monitor package)

LLT/Cannsim Mode Options (single task, default runs all):
  --llt-a900                PreSmoke_A900 only (Upper + Lower)
  --llt-st                  PreSmoke_St only
  --llt-exp-ut              Experimental UT only
  --llt-kernel-ut           Kernel UT only
  --llt-std-ut              Standard UT only
  --cannsim                 Cannsim tests only
  Note: single-task mode auto-skips other main flows (compile/LLT/cannsim);
        combining params from different flows is not allowed

Build Parameters:
  --soc=<soc_version>       Target SoC
                            Supported: ascend910b ascend910_93 ascend950 ascend310p 
                                       ascend910 ascend310b
  -j <num>                  Parallel jobs (default: 16)
  --cann_3rd_lib_path=<path> CANN third party lib path

Skip Options:
  --skip-build              Skip compile flow
  --skip-llt                Skip LLT flow (PreSmoke_A900 + PreSmoke_St + UT)
  --skip-cannsim            Skip cannsim tests

Other Options:
  -h, --help                Show this help message

Check Items (Main Flows):
  1. Environment check      - Verify CANN toolkit, compilers, Python
  2. Compile Flow           - Default: all modes (normal/jit/experimental/single/a5)
                             - Single task: specify --jit/--experimental/--single/--a5/--normal
                             - Normal mode produces: cann-{chip}-ops-math_{ver}_linux-{arch}.run
                             - Normal mode also verifies: install/uninstall of the package
  3. LLT Flow               - PreSmoke_A900 + PreSmoke_St + UT tests
                              - Default: all tasks (a900/st/exp-ut/kernel-ut/std-ut)
                              - Single task: specify --llt-a900/--llt-st/--llt-exp-ut/--llt-kernel-ut/--llt-std-ut
                             - PreSmoke_A900 Upper: single.tar.gz + check_example.sh
                             - PreSmoke_A900 Lower: experimental package + --run_example
                             - PreSmoke_St: ops_st_test.sh (install built-in package first)
                             - UT: Experimental UT + Kernel UT + Standard UT (all by default)
  4. Cannsim Tests          - Cannsim simulation (requires ascend950, extra feature)
                              - Single task: specify --cannsim

Output:
  All logs saved to: logs/
  Build artifacts preserved: artifacts/ (normal/jit/experimental/single/a5)
  Build temp preserved: build/, build_out/

Examples:
  bash local_check.sh -f pr_filelist.txt           # default: run all tasks
  bash local_check.sh --jit -f pr_filelist.txt     # run only JIT compilation
  bash local_check.sh --experimental               # run only experimental compilation
  bash local_check.sh --single -f pr_filelist.txt  # run only single package build
  bash local_check.sh --a5                         # run only A5 compilation
  bash local_check.sh --normal                     # run only normal compilation
  bash local_check.sh --llt-a900                   # run only PreSmoke_A900
  bash local_check.sh --llt-st                     # run only PreSmoke_St
  bash local_check.sh --llt-exp-ut                 # run only Experimental UT
  bash local_check.sh --llt-kernel-ut              # run only Kernel UT
  bash local_check.sh --llt-std-ut                 # run only Standard UT
  bash local_check.sh --cannsim                    # run only cannsim tests
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f)
                if [[ -z "$2" ]]; then
                    echo "[ERROR] -f requires a file argument"
                    exit 1
                fi
                FILE_LIST="$2"
                shift 2
                ;;
            --jit)
                JIT_MODE=true
                shift
                ;;
            --experimental)
                EXPERIMENTAL_MODE=true
                shift
                ;;
            --single)
                SINGLE_MODE=true
                shift
                ;;
            --a5)
                A5_MODE=true
                shift
                ;;
            --normal)
                NORMAL_MODE=true
                shift
                ;;
            --soc=*)
                SOC="${1#*=}"
                shift
                ;;
            -j*)
                if [[ "${1}" == "-j" ]]; then
                    JOBS="$2"
                    shift 2
                else
                    JOBS="${1#-j}"
                    shift
                fi
                ;;
            --cann_3rd_lib_path=*)
                CANN_3RD_LIB_PATH="${1#*=}"
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-llt)
                SKIP_LLT=true
                shift
                ;;
            --skip-cannsim)
                SKIP_CANNSIM=true
                shift
                ;;
            --llt-a900)
                LLT_A900_MODE=true
                shift
                ;;
            --llt-st)
                LLT_ST_MODE=true
                shift
                ;;
            --llt-exp-ut)
                LLT_EXP_UT_MODE=true
                shift
                ;;
            --llt-kernel-ut)
                LLT_KERNEL_UT_MODE=true
                shift
                ;;
            --llt-std-ut)
                LLT_STD_UT_MODE=true
                shift
                ;;
            --cannsim)
                CANNSIM_MODE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "[ERROR] Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    }

detect_changed_ops() {
    print_step "1.1" "Detecting changed files"

    if [[ -n "${FILE_LIST}" ]]; then
        print_info "Using specified file list: ${FILE_LIST}"
        if [[ ! -f "${FILE_LIST}" ]]; then
            print_fail "File list not found: ${FILE_LIST}"
            return 1
        fi
        cp "${FILE_LIST}" "${PR_FILELIST}"
        print_ok "Loaded $(wc -l < "${PR_FILELIST}") files"
    else
        local base_ref="master"
        
        {
            git -C "${PROJECT_ROOT}" diff --name-only "${base_ref}...HEAD" 2>/dev/null || true
            git -C "${PROJECT_ROOT}" diff --name-only "HEAD~1...HEAD" 2>/dev/null || true
            git -C "${PROJECT_ROOT}" diff --cached --name-only 2>/dev/null || true
            git -C "${PROJECT_ROOT}" diff --name-only HEAD 2>/dev/null || true
        } | grep -v '^$' | sort -u > "${PR_FILELIST}"
        
        if [[ ! -s "${PR_FILELIST}" ]]; then
            print_warn "No changed files detected (committed, staged, or working)"
            print_info "Use -f/--file-list to manually specify which files to verify"
            return 1
        fi
        
        print_ok "Detected $(wc -l < "${PR_FILELIST}") changed files (committed + staged + working)"
    fi

    print_step "1.2" "Parsing affected operators"

    local normal_ops=""
    local exp_ops=""
    local merged_ops=""

    normal_ops=$(python3 "${PROJECT_ROOT}/scripts/ci/gen_ci_cmd.py" \
        -f "${PR_FILELIST}" --list_ops --experimental=FALSE) || {
        print_fail "gen_ci_cmd.py (normal) failed"
        return 1
    }
    exp_ops=$(python3 "${PROJECT_ROOT}/scripts/ci/gen_ci_cmd.py" \
        -f "${PR_FILELIST}" --list_ops --experimental=TRUE) || {
        print_fail "gen_ci_cmd.py (experimental) failed"
        return 1
    }

    if [[ -n "${normal_ops}" && -n "${exp_ops}" ]]; then
        merged_ops="${normal_ops},${exp_ops}"
    elif [[ -n "${normal_ops}" ]]; then
        merged_ops="${normal_ops}"
    elif [[ -n "${exp_ops}" ]]; then
        merged_ops="${exp_ops}"
    fi

    if [[ -z "${merged_ops}" ]]; then
        print_warn "No affected operators detected from changed files"
        print_info "Tasks requiring --ops will be skipped"
    fi

    OPS="${merged_ops}"
    print_ok "Affected operators: ${OPS}"
    return 0
}

check_environment() {
    print_header "Step 1: Environment Check"

    local env_ok=true

    print_step "1.1" "Checking CANN toolkit"
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        print_fail "ASCEND_HOME_PATH not set"
        print_info "Please source CANN set_env.sh first, e.g.:"
        print_info "  source /usr/local/Ascend/cann/set_env.sh"
        print_fail "Environment check failed"
        return 1
    fi
    print_ok "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

    print_step "1.2" "Checking compilers"
    command -v cmake &>/dev/null && print_ok "cmake: $(cmake --version | head -1)" || { print_fail "cmake not found"; env_ok=false; }
    command -v g++ &>/dev/null && print_ok "g++: $(g++ --version | head -1)" || { print_fail "g++ not found"; env_ok=false; }

    print_step "1.3" "Checking Python"
    command -v python3 &>/dev/null && print_ok "python3: $(python3 --version)" || { print_fail "python3 not found"; env_ok=false; }

    print_step "1.4" "Checking SoC"
    if [[ -z "${SOC}" ]]; then
        local chip_info=$(asys info -r=status 2>/dev/null || echo "")
        if echo "${chip_info}" | grep -q "Ascend 950"; then
            SOC="ascend950"
        elif echo "${chip_info}" | grep -q "Ascend 910"; then
            SOC="ascend910b"
        else
            SOC="${DEFAULT_SOC}"
            print_info "No NPU device detected, using default: ${SOC}"
        fi
    fi
    print_ok "Target SoC: ${SOC}"

    [[ "${env_ok}" != "true" ]] && { print_fail "Environment check failed"; return 1; }

    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        local original_ld="${LD_LIBRARY_PATH}"
        LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v "opp/vendors/.*op_api/lib" | grep -v "^$" | tr '\n' ':' | sed 's/:$//')
        export LD_LIBRARY_PATH
        if [[ "${LD_LIBRARY_PATH}" != "${original_ld}" ]]; then
            local removed_count=$(echo "${original_ld}" | tr ':' '\n' | grep "opp/vendors/.*op_api/lib" | wc -l)
            print_info "Cleaned LD_LIBRARY_PATH: removed ${removed_count} vendor op_api/lib paths (prevents libcust_opapi.so naming conflict)"
        fi
    fi

    print_ok "Environment check passed"
    return 0
}

CHECK_NAMES=()
CHECK_RESULTS=()
SUB_TASK_NAMES=()
SUB_TASK_RESULTS=()
SUB_TASK_ARTIFACTS=()
SUB_TASK_LOGS=()
SUB_TASK_COMMANDS=()

record_sub_task() {
    SUB_TASK_NAMES+=("$1")
    SUB_TASK_RESULTS+=("$2")
    SUB_TASK_ARTIFACTS+=("$3")
    SUB_TASK_LOGS+=("$4")
    SUB_TASK_COMMANDS+=("$5")
}

uninstall_vendor() {
    local vendor="$1"
    local log_file="$2"
    print_info "Uninstalling ${vendor} vendor to restore environment..."
    local uninstall_script="${ASCEND_HOME_PATH}/opp/vendors/${vendor}/scripts/uninstall.sh"
    if [[ -f "${uninstall_script}" ]]; then
        bash "${uninstall_script}" 2>&1 | tee "${log_file}" || print_warn "${vendor} vendor uninstall failed"
    else
        print_warn "Uninstall script not found: ${uninstall_script}"
    fi
}

run_check() {
    local name="$1"
    local func="$2"
    CHECK_NAMES+=("${name}")
    print_info "Starting: ${name}"
    set +e
    ${func}
    local ret=$?
    set -e
    case ${ret} in
        0) CHECK_RESULTS+=("PASS") ;;
        2) CHECK_RESULTS+=("SKIP") ;;
        *) CHECK_RESULTS+=("FAIL") ;;
    esac
}

# ============================================================
# 第一大类：编译流程 (compile.sh 功能覆盖)
# ============================================================
do_compile_flow() {
    print_header "Step 2: Compile Flow"

    if [[ "${SKIP_BUILD}" == "true" ]]; then
        print_warn "Compile flow skipped (--skip-build)"
        return 2
    fi

    # 检查是否指定了编译模式
    local any_mode=false
    [[ "${EXPERIMENTAL_MODE}" == "true" ]] && any_mode=true
    [[ "${JIT_MODE}" == "true" ]] && any_mode=true
    [[ "${SINGLE_MODE}" == "true" ]] && any_mode=true
    [[ "${A5_MODE}" == "true" ]] && any_mode=true
    [[ "${NORMAL_MODE}" == "true" ]] && any_mode=true
    
    if [[ "${any_mode}" == "false" ]]; then
        NORMAL_MODE=true
        EXPERIMENTAL_MODE=true
        SINGLE_MODE=true
        JIT_MODE=true
        A5_MODE=true
        print_info "Auto mode: all compile tasks enabled (default)"
    fi

    local start_time=$(date +%s)
    local compile_ret=0

    # 初始化产物目录（隔离 build_out，避免 build_package 清空）
    local ARTIFACTS="${PROJECT_ROOT}/artifacts"
    rm -rf "${ARTIFACTS}"
    mkdir -p "${ARTIFACTS}/normal" "${ARTIFACTS}/jit" "${ARTIFACTS}/experimental" "${ARTIFACTS}/single" "${ARTIFACTS}/a5"

    # Normal 模式
    if [[ "${NORMAL_MODE}" == "true" ]]; then
        print_step "2.1" "Normal compilation"
        local cmd_array=(bash build.sh --pkg --soc="${SOC}" -j"${JOBS}")
        [[ -n "${CANN_3RD_LIB_PATH}" ]] && cmd_array+=(--cann_3rd_lib_path="${CANN_3RD_LIB_PATH}")
        print_info "Command: ${cmd_array[*]}"
        
        if "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/build_normal.txt"; then
            print_ok "Normal compilation passed"
            record_sub_task "Normal compilation" "PASS" "artifacts/normal/" "logs/build_normal.txt" ""
            cp "${PROJECT_ROOT}"/build_out/cann-*-ops-math_*_linux-*.run "${ARTIFACTS}/normal/" 2>/dev/null || true
            local normal_pkgs=("${ARTIFACTS}/normal/"cann-*-ops-math_*.run)
            if [[ -f "${normal_pkgs[0]}" ]]; then
                print_info "Starting: Package Verification (Normal)"
                do_package_verify "${normal_pkgs[0]}"
                local pkg_verify_ret=$?
                case ${pkg_verify_ret} in
                    0) record_sub_task "Package Verification" "PASS" "" "logs/package_install.txt" "" ;;
                    2) record_sub_task "Package Verification" "SKIP" "" "" "" ;;
                    *) record_sub_task "Package Verification" "FAIL" "" "logs/package_install.txt" "${normal_pkgs[0]} --install --install-path=${PROJECT_ROOT}/tmp_verify" ;;
                esac
            else
                print_warn "No normal package found for verification"
            fi
        else
            print_fail "Normal compilation failed"
            record_sub_task "Normal compilation" "FAIL" "artifacts/normal/" "logs/build_normal.txt" "${cmd_array[*]}"
            compile_ret=1
        fi
    fi

    # JIT 模式
    if [[ "${JIT_MODE}" == "true" ]]; then
        print_step "2.2" "JIT compilation"
        local chip_short=$(soc_to_chip_short "${SOC}")
        local cmd_array=(bash build.sh --pkg --jit --soc="${SOC}" -j"${JOBS}")
        [[ -n "${CANN_3RD_LIB_PATH}" ]] && cmd_array+=(--cann_3rd_lib_path="${CANN_3RD_LIB_PATH}")
        print_info "Command: ${cmd_array[*]}"
        
        if "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/build_jit.txt"; then
            print_ok "JIT compilation passed"
            local jit_src_pattern="cann-${chip_short}-ops-math_*_linux-*.run"
            local jit_src_file="${PROJECT_ROOT}/build_out/${jit_src_pattern}"
            if compgen -G "${jit_src_file}" > /dev/null; then
                cp "${PROJECT_ROOT}"/build_out/${jit_src_pattern} "${ARTIFACTS}/jit/"
                local jit_pkg=$(ls "${ARTIFACTS}/jit/${jit_src_pattern}" 2>/dev/null | head -1)
                print_info "Copied package: $(basename "${jit_pkg}")"
            else
                print_warn "JIT package not found in build_out/ matching ${jit_src_pattern}"
            fi
            record_sub_task "JIT compilation" "PASS" "artifacts/jit/" "logs/build_jit.txt" ""
        else
            print_fail "JIT compilation failed"
            record_sub_task "JIT compilation" "FAIL" "artifacts/jit/" "logs/build_jit.txt" "${cmd_array[*]}"
            compile_ret=1
        fi
    fi

    # Experimental 模式
    if [[ "${EXPERIMENTAL_MODE}" == "true" ]]; then
        print_step "2.3" "Experimental compilation"
        local ops_comma=$(filter_ops_by_soc "${OPS}" "${SOC}" "true")

        if [[ -n "${OPS}" && -z "${ops_comma}" ]]; then
            print_warn "No experimental operators support ${SOC}, skipping Experimental compilation"
            record_sub_task "Experimental compilation" "SKIP" "artifacts/experimental/" "" ""
        else
            local cmd_array=(bash build.sh --experimental --pkg --soc="${SOC}" --vendor_name=experimental -j"${JOBS}")
            [[ -n "${ops_comma}" ]] && cmd_array+=(--ops="${ops_comma}")
            [[ -n "${CANN_3RD_LIB_PATH}" ]] && cmd_array+=(--cann_3rd_lib_path="${CANN_3RD_LIB_PATH}")
            print_info "Command: ${cmd_array[*]}"

            if "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/build_experimental.txt"; then
                print_ok "Experimental compilation passed"
                record_sub_task "Experimental compilation" "PASS" "artifacts/experimental/" "logs/build_experimental.txt" ""
                cp "${PROJECT_ROOT}"/build_out/cann-ops-math-experimental_linux-*.run "${ARTIFACTS}/experimental/" 2>/dev/null || true
            else
                print_fail "Experimental compilation failed"
                record_sub_task "Experimental compilation" "FAIL" "artifacts/experimental/" "logs/build_experimental.txt" "${cmd_array[*]}"
                compile_ret=1
            fi
        fi
    fi

    # Single 模式
    if [[ "${SINGLE_MODE}" == "true" ]]; then
        print_step "2.4" "Single package build"
        local cmd_array=(bash scripts/ci/check_pkg.sh "${PR_FILELIST}" --soc="${SOC}")
        print_info "Command: ${cmd_array[*]}"
        
        if "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/build_single.txt"; then
            print_ok "Single package build passed"
            record_sub_task "Single package build" "PASS" "artifacts/single/" "logs/build_single.txt" ""
            cp "${PROJECT_ROOT}"/single/cann-ops-math-*_linux*.run "${ARTIFACTS}/single/" 2>/dev/null || true
            cp "${PROJECT_ROOT}"/single.tar.gz "${ARTIFACTS}/single/" 2>/dev/null || true
        else
            print_fail "Single package build failed"
            record_sub_task "Single package build" "FAIL" "artifacts/single/" "logs/build_single.txt" "${cmd_array[*]}"
            compile_ret=1
        fi
    fi

    # A5 模式
    if [[ "${A5_MODE}" == "true" ]]; then
        print_step "2.5" "A5 compilation"
        local cmd_array=(bash scripts/ci/compile_a5_pkg.sh "${PR_FILELIST}")
        
        if "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/build_a5.txt"; then
            print_ok "A5 compilation passed"
            record_sub_task "A5 compilation" "PASS" "artifacts/a5/" "logs/build_a5.txt" ""
            cp "${PROJECT_ROOT}"/build_out/cann-ops-math-*.run "${ARTIFACTS}/a5/" 2>/dev/null || true
        else
            print_fail "A5 compilation failed"
            record_sub_task "A5 compilation" "FAIL" "artifacts/a5/" "logs/build_a5.txt" "${cmd_array[*]}"
            compile_ret=1
        fi
    fi

    local end_time=$(date +%s)
    if [[ ${compile_ret} -eq 0 ]]; then
        print_ok "Compile flow succeeded ($((end_time - start_time))s)"
        print_info "All build artifacts preserved in: artifacts/ (normal/jit/experimental/single/a5)"
    else
        print_fail "Compile flow failed"
        print_info "See logs/ for details"
        return 1
    fi

    return 0
}

# ============================================================
# 第二大类：LLT流程 (PreSmoke_A900 + PreSmoke_St + UT)
# ============================================================
do_presmoke_a900() {
    local task_ret=0
    
    print_step "3.1" "PreSmoke_A900 Upper: single.tar.gz + check_example.sh"
    
    local single_tgz="${PROJECT_ROOT}/artifacts/single/single.tar.gz"
    if [[ ! -f "${single_tgz}" ]]; then
        print_warn "single.tar.gz not found in artifacts/single/"
        record_sub_task "PreSmoke_A900 Upper" "SKIP" "artifacts/single/single.tar.gz" "" ""
        task_ret=2
    else
        print_info "Extracting single.tar.gz..."
        if ! tar -zxf "${single_tgz}" -C "${PROJECT_ROOT}" 2>&1 | tee "${PROJECT_ROOT}/logs/presmoke_a900_upper.txt"; then
            print_fail "Failed to extract single.tar.gz"
            record_sub_task "PreSmoke_A900 Upper" "FAIL" "artifacts/single/single.tar.gz" "logs/presmoke_a900_upper.txt" "tar -zxf ${single_tgz}"
            if [[ ${task_ret} -ne 1 ]]; then task_ret=1; fi
        fi
        
        local installed_vendors=()
        for single_run in "${PROJECT_ROOT}"/single/cann-ops-math-*_linux-*.run; do
            if [[ -f "${single_run}" ]]; then
                local op_name=$(basename "${single_run}" | sed -E 's/cann-ops-math-(.+)_linux-.*/\1/')
                installed_vendors+=("${op_name}_math")
            fi
        done
        
        local filtered_ops=$(filter_ops_by_soc "${OPS}" "${SOC}" "false")
        if [[ -n "${OPS}" && -z "${filtered_ops}" ]]; then
            print_warn "Specified operators do not support ${SOC}, skipping PreSmoke_A900 Upper"
            record_sub_task "PreSmoke_A900 Upper" "SKIP" "artifacts/single/single.tar.gz" "" "ops do not support ${SOC}"
            if [[ ${task_ret} -ne 1 ]]; then task_ret=2; fi
        elif [[ -z "${filtered_ops}" ]]; then
            print_warn "No operators detected, skipping PreSmoke_A900 Upper"
            record_sub_task "PreSmoke_A900 Upper" "SKIP" "artifacts/single/single.tar.gz" "" ""
            if [[ ${task_ret} -ne 1 ]]; then task_ret=2; fi
        else
            local filtered_pr="${PROJECT_ROOT}/logs/filtered_a900_upper.txt"
            make_filtered_pr_file "${PR_FILELIST}" "${filtered_ops}" "${filtered_pr}"
            local cmd_array=(bash scripts/ci/check_example.sh "${filtered_pr}" --soc="${SOC}")
            print_info "Command: ${cmd_array[*]}"
            
            "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/presmoke_a900_upper.txt"
            local ret=$?
            
            if [[ ${ret} -eq 0 ]]; then
                print_ok "PreSmoke_A900 Upper passed"
                record_sub_task "PreSmoke_A900 Upper" "PASS" "artifacts/single/single.tar.gz" "logs/presmoke_a900_upper.txt" ""
            else
                print_fail "PreSmoke_A900 Upper failed"
                record_sub_task "PreSmoke_A900 Upper" "FAIL" "artifacts/single/single.tar.gz" "logs/presmoke_a900_upper.txt" "${cmd_array[*]}"
                task_ret=1
            fi
        fi

        print_info "Uninstalling single package vendors to restore environment..."
        for vendor in "${installed_vendors[@]}"; do
            local uninstall_script="${ASCEND_HOME_PATH}/opp/vendors/${vendor}/scripts/uninstall.sh"
            if [[ -f "${uninstall_script}" ]]; then
                bash "${uninstall_script}" 2>&1 | tee "${PROJECT_ROOT}/logs/presmoke_a900_upper_uninstall_${vendor}.txt" || print_warn "Vendor ${vendor} uninstall failed"
            else
                print_warn "Uninstall script not found: ${uninstall_script}"
            fi
        done
    fi
    
    print_step "3.2" "PreSmoke_A900 Lower: experimental package + --run_example"
    
    local exp_pkgs=("${PROJECT_ROOT}"/artifacts/experimental/cann-ops-math-experimental_linux-*.run)
    if [[ ${#exp_pkgs[@]} -eq 0 ]] || [[ ! -f "${exp_pkgs[0]}" ]]; then
        print_warn "experimental package not found in artifacts/experimental/"
        record_sub_task "PreSmoke_A900 Lower" "SKIP" "artifacts/experimental/" "" ""
        if [[ ${task_ret} -ne 1 ]]; then task_ret=2; fi
    else
        local exp_pkg="${exp_pkgs[0]}"
        print_info "Found experimental package: $(basename ${exp_pkg})"
        
        print_info "Installing experimental package with --quiet..."
        chmod +x "${exp_pkg}"
        "${exp_pkg}" --quiet --force 2>&1 | tee "${PROJECT_ROOT}/logs/presmoke_a900_lower_install.txt"
        local install_ret=$?
        
        if [[ ${install_ret} -ne 0 ]]; then
            print_fail "Failed to install experimental package"
            record_sub_task "PreSmoke_A900 Lower" "FAIL" "artifacts/experimental/" "logs/presmoke_a900_lower_install.txt" "${exp_pkg} --quiet"
            task_ret=1
        else
            local filtered_exp_ops=$(filter_ops_by_soc "${OPS}" "${SOC}" "true")
            if [[ -n "${OPS}" && -z "${filtered_exp_ops}" ]]; then
                print_warn "Specified operators do not support ${SOC} in experimental mode, skipping PreSmoke_A900 Lower"
                record_sub_task "PreSmoke_A900 Lower" "SKIP" "artifacts/experimental/" "" "ops do not support ${SOC}"
                if [[ ${task_ret} -ne 1 ]]; then task_ret=2; fi
                uninstall_vendor "experimental_math" "${PROJECT_ROOT}/logs/presmoke_a900_lower_uninstall.txt"
            elif [[ -z "${filtered_exp_ops}" ]]; then
                print_warn "No operators detected, skipping PreSmoke_A900 Lower"
                record_sub_task "PreSmoke_A900 Lower" "SKIP" "artifacts/experimental/" "" ""
                if [[ ${task_ret} -ne 1 ]]; then task_ret=2; fi
                uninstall_vendor "experimental_math" "${PROJECT_ROOT}/logs/presmoke_a900_lower_uninstall.txt"
            else
                local lower_ret=0
                local last_fail_cmd=""
                local exp_arr=(${filtered_exp_ops//,/ })
                for op in "${exp_arr[@]}"; do
                    local cmd_array=(bash build.sh --experimental --run_example "${op}" eager cust --vendor_name=experimental --soc="${SOC}")
                    print_info "Command: ${cmd_array[*]}"

                    "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/presmoke_a900_lower_${op}_run.txt"
                    local run_ret=$?

                    if [[ ${run_ret} -eq 0 ]]; then
                        print_ok "PreSmoke_A900 Lower (${op}) passed"
                    else
                        print_fail "PreSmoke_A900 Lower (${op}) failed"
                        last_fail_cmd="${cmd_array[*]}"
                        lower_ret=1
                    fi
                done

                if [[ ${lower_ret} -eq 0 ]]; then
                    record_sub_task "PreSmoke_A900 Lower" "PASS" "artifacts/experimental/" "logs/presmoke_a900_lower_*_run.txt" ""
                else
                    record_sub_task "PreSmoke_A900 Lower" "FAIL" "artifacts/experimental/" "logs/presmoke_a900_lower_*_run.txt" "${last_fail_cmd}"
                    task_ret=1
                fi

                uninstall_vendor "experimental_math" "${PROJECT_ROOT}/logs/presmoke_a900_lower_uninstall.txt"
            fi
        fi
    fi
    
    return ${task_ret}
}

do_presmoke_st() {
    print_step "3.3" "PreSmoke_St: ops_st_test.sh"

    local chip_short=$(soc_to_chip_short "${SOC}")
    local monitor_pkgs=("${PROJECT_ROOT}"/artifacts/normal/cann-${chip_short}-ops-math_*_linux-*.run)
    if [[ ${#monitor_pkgs[@]} -gt 0 ]] && [[ -f "${monitor_pkgs[0]}" ]]; then
        local monitor_pkg="${monitor_pkgs[0]}"
        print_info "Found package: $(basename ${monitor_pkg})"
        chmod +x "${monitor_pkg}"
        "${monitor_pkg}" --full --install-path="${ASCEND_HOME_PATH}" 2>&1 | tee "${PROJECT_ROOT}/logs/presmoke_st_install.txt"
        local install_ret=$?
        
        if [[ ${install_ret} -eq 0 ]]; then
            print_ok "Package installed"
        else
            print_fail "Package install failed (exit code: ${install_ret})"
            return 1
        fi
    else
        print_warn "Package not found in artifacts/normal/: cann-${chip_short}-ops-math_*_linux-*.run"
    fi
    
    if [[ -z "${OPS}" ]]; then
        print_warn "No operators detected"
        if [[ -n "${monitor_pkg:-}" ]] && [[ -f "${monitor_pkg}" ]]; then
            print_info "Uninstalling normal package to restore environment..."
            "${monitor_pkg}" --uninstall --install-path="${ASCEND_HOME_PATH}" 2>&1 | tee "${PROJECT_ROOT}/logs/presmoke_st_uninstall.txt" || print_warn "Normal package uninstall failed"
        fi
        return 2
    fi
    
    local cmd_array=(bash scripts/ci/ops_st_test.sh --soc_version="${SOC}" --ops="${OPS}" --pr_filelist="${PR_FILELIST}")
    print_info "Command: ${cmd_array[*]}"
    
    "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/presmoke_st.txt"
    local ret=$?
    
    if [[ -n "${monitor_pkg:-}" ]] && [[ -f "${monitor_pkg}" ]]; then
        print_info "Uninstalling normal package to restore environment..."
        "${monitor_pkg}" --uninstall --install-path="${ASCEND_HOME_PATH}" 2>&1 | tee "${PROJECT_ROOT}/logs/presmoke_st_uninstall.txt" || print_warn "Normal package uninstall failed"
    fi
    
    if [[ ${ret} -eq 0 ]]; then
        print_ok "PreSmoke_St passed"
        record_sub_task "PreSmoke_St" "PASS" "" "logs/presmoke_st.txt" ""
        return 0
    else
        print_fail "PreSmoke_St failed"
        record_sub_task "PreSmoke_St" "FAIL" "" "logs/presmoke_st.txt" "${cmd_array[*]}"
        return 1
    fi
}

do_llt_flow() {
    print_header "Step 3: LLT Flow"
    
    if [[ "${SKIP_LLT}" == "true" ]]; then
        print_warn "LLT flow skipped (--skip-llt)"
        return 2
    fi
    
    local any_llt_mode=false
    [[ "${LLT_A900_MODE}" == "true" ]] && any_llt_mode=true
    [[ "${LLT_ST_MODE}" == "true" ]] && any_llt_mode=true
    [[ "${LLT_EXP_UT_MODE}" == "true" ]] && any_llt_mode=true
    [[ "${LLT_KERNEL_UT_MODE}" == "true" ]] && any_llt_mode=true
    [[ "${LLT_STD_UT_MODE}" == "true" ]] && any_llt_mode=true
    
    if [[ "${any_llt_mode}" == "false" ]]; then
        LLT_A900_MODE=true
        LLT_ST_MODE=true
        LLT_EXP_UT_MODE=true
        LLT_KERNEL_UT_MODE=true
        LLT_STD_UT_MODE=true
        print_info "Auto mode: all LLT tasks enabled (default)"
    fi
    
    local llt_ret=0
    
    # --- PreSmoke_A900 ---
    if [[ "${LLT_A900_MODE}" == "true" ]]; then
        do_presmoke_a900
        local a900_ret=$?
        if [[ ${a900_ret} -eq 1 ]]; then llt_ret=1; fi
    else
        print_warn "PreSmoke_A900 skipped (use --llt-a900 to enable)"
        record_sub_task "PreSmoke_A900" "SKIP" "" "" ""
    fi
    
    # --- PreSmoke_St ---
    if [[ "${LLT_ST_MODE}" == "true" ]]; then
        do_presmoke_st
        local st_ret=$?
        if [[ ${st_ret} -eq 1 ]]; then llt_ret=1; fi
    else
        print_warn "PreSmoke_St skipped (use --llt-st to enable)"
        record_sub_task "PreSmoke_St" "SKIP" "" "" ""
    fi
    
    # --- Experimental UT ---
    if [[ "${LLT_EXP_UT_MODE}" == "true" ]]; then
        print_step "3.4" "Experimental UT"
        local ops_comma=$(filter_ops_by_soc "${OPS}" "ascend910b" "true")

        if [[ -n "${OPS}" && -z "${ops_comma}" ]]; then
            print_warn "No experimental operators support ascend910b, skipping Experimental UT"
            record_sub_task "Experimental UT" "SKIP" "" "" ""
        else
            local cmd_array=(bash build.sh --experimental -u -j"${JOBS}")
            [[ -n "${ops_comma}" ]] && cmd_array+=(--ops="${ops_comma}")
            [[ -n "${CANN_3RD_LIB_PATH}" ]] && cmd_array+=(--cann_3rd_lib_path="${CANN_3RD_LIB_PATH}")
            print_info "Command: ${cmd_array[*]}"

            "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/ut_experimental.txt"
            local ret=$?

            if [[ ${ret} -eq 0 ]]; then
                print_ok "Experimental UT passed"
                record_sub_task "Experimental UT" "PASS" "" "logs/ut_experimental.txt" ""
            else
                print_fail "Experimental UT failed"
                record_sub_task "Experimental UT" "FAIL" "" "logs/ut_experimental.txt" "${cmd_array[*]}"
                llt_ret=1
            fi
        fi
    else
        print_warn "Experimental UT skipped (use --llt-exp-ut to enable)"
        record_sub_task "Experimental UT" "SKIP" "" "" ""
    fi
    
    # --- Kernel UT ---
    if [[ "${LLT_KERNEL_UT_MODE}" == "true" ]]; then
        print_step "3.5" "Kernel UT"
        local cmd_array=(bash scripts/ci/check_kernel_ut.sh "${PR_FILELIST}")
        print_info "Command: ${cmd_array[*]}"
        
        "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/ut_kernel.txt"
        local ret=$?
        
        if [[ ${ret} -eq 0 ]]; then
            print_ok "Kernel UT passed"
            record_sub_task "Kernel UT" "PASS" "" "logs/ut_kernel.txt" ""
        else
            print_fail "Kernel UT failed"
            record_sub_task "Kernel UT" "FAIL" "" "logs/ut_kernel.txt" "${cmd_array[*]}"
            llt_ret=1
        fi
    else
        print_warn "Kernel UT skipped (use --llt-kernel-ut to enable)"
        record_sub_task "Kernel UT" "SKIP" "" "" ""
    fi
    
    # --- Standard UT ---
    if [[ "${LLT_STD_UT_MODE}" == "true" ]]; then
        print_step "3.6" "Standard UT"
        local ops_comma=$(filter_ops_by_soc "${OPS}" "ascend910b" "false")

        if [[ -n "${OPS}" && -z "${ops_comma}" ]]; then
            print_warn "No operators support ascend910b, skipping Standard UT"
            record_sub_task "Standard UT" "SKIP" "" "" ""
        else
            local cmd_array=(bash build.sh -u -j"${JOBS}")
            [[ -n "${ops_comma}" ]] && cmd_array+=(--ops="${ops_comma}")
            [[ -n "${CANN_3RD_LIB_PATH}" ]] && cmd_array+=(--cann_3rd_lib_path="${CANN_3RD_LIB_PATH}")
            print_info "Command: ${cmd_array[*]}"

            "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/ut_standard.txt"
            local ret=$?

            if [[ ${ret} -eq 0 ]]; then
                print_ok "Standard UT passed"
                record_sub_task "Standard UT" "PASS" "" "logs/ut_standard.txt" ""
            else
                print_fail "Standard UT failed"
                record_sub_task "Standard UT" "FAIL" "" "logs/ut_standard.txt" "${cmd_array[*]}"
                llt_ret=1
            fi
        fi
    else
        print_warn "Standard UT skipped (use --llt-std-ut to enable)"
        record_sub_task "Standard UT" "SKIP" "" "" ""
    fi
    
    if [[ ${llt_ret} -eq 0 ]]; then
        print_ok "LLT Flow completed (all tasks passed)"
    else
        print_fail "LLT Flow completed with failures"
    fi
    
    return ${llt_ret}
}

do_cannsim_check() {
    print_header "Step 4: Cannsim Tests"

    if [[ "${SKIP_CANNSIM}" == "true" ]]; then
        print_warn "Cannsim skipped"
        return 2
    fi

    if [[ "${SOC}" != "ascend950" ]]; then
        print_warn "cannsim only supports ascend950"
        return 2
    fi

    if [[ -z "${OPS}" ]]; then
        print_warn "No operators detected"
        return 2
    fi

    local ops_arr=(${OPS//,/ })
    local cannsim_ops=()
    print_info "cannsim requires ascend950 support; operators without AddConfig(\"ascend950\") will be skipped"
    for op in "${ops_arr[@]}"; do
        if op_supports_soc "$op" "ascend950" "false" || \
           op_supports_soc "$op" "ascend950" "true"; then
            cannsim_ops+=("${op}")
        else
            print_warn "Skipping cannsim: ${op} (no ascend950 support)"
        fi
    done
    if [[ ${#cannsim_ops[@]} -eq 0 ]]; then
        print_warn "No ascend950-compatible operators for cannsim"
        return 2
    fi
    for op in "${cannsim_ops[@]}"; do
        print_step "4" "Running cannsim: ${op}"
        local cmd_array=(bash scripts/ci/run_cannsim_example.sh "${op}" -j "${JOBS}")
        print_info "Command: ${cmd_array[*]}"
        
        if "${cmd_array[@]}" 2>&1 | tee "${PROJECT_ROOT}/logs/cannsim_${op}.txt"; then
            print_ok "Cannsim passed: ${op}"
        else
            print_fail "Cannsim failed: ${op}"
            return 1
        fi
    done
}

do_package_verify() {
    local pkg_file="$1"
    if [[ -z "${pkg_file}" ]] || [[ ! -f "${pkg_file}" ]]; then
        print_warn "Package file not found: ${pkg_file}"
        return 2
    fi

    print_info "Package: $(basename "${pkg_file}")"
    local temp_dir="${PROJECT_ROOT}/tmp_verify"
    chmod +x "${pkg_file}"

    print_step "2.1.1" "Installing package"
    local install_log="${PROJECT_ROOT}/logs/package_install.txt"
    print_info "Command: ${pkg_file} --full --install-path=${temp_dir}"

    if "${pkg_file}" --full --install-path="${temp_dir}" 2>&1 | tee "${install_log}"; then
        print_ok "Install succeeded"
    else
        print_fail "Install failed"
        rm -rf "${temp_dir}"
        return 1
    fi

    print_step "2.1.2" "Uninstalling package"
    local uninstall_log="${PROJECT_ROOT}/logs/package_uninstall.txt"
    print_info "Command: ${pkg_file} --uninstall --install-path=${temp_dir}"

    if "${pkg_file}" --uninstall --install-path="${temp_dir}" 2>&1 | tee "${uninstall_log}"; then
        print_ok "Uninstall succeeded"
    else
        print_fail "Uninstall failed"
        rm -rf "${temp_dir}"
        return 1
    fi

    rm -rf "${temp_dir}"
    print_ok "Package verification completed"
}

# ============================================================
# 结果汇总
# ============================================================
print_summary() {
    local total_start=$1
    local total_end=$(date +%s)
    local elapsed=$((total_end - total_start))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))

    print_header "Summary"

    local idx=0
    for result in "${CHECK_RESULTS[@]}"; do
        idx=$((idx + 1))
        local name="${CHECK_NAMES[$((idx - 1))]}"
        case "${result}" in
            PASS) print_ok "${idx}. ${name}" ;;
            SKIP) print_warn "${idx}. ${name}" ;;
            FAIL) print_fail "${idx}. ${name}" ;;
        esac
    done

    print_header "Sub-task Details"

    local idx=0
    for result in "${SUB_TASK_RESULTS[@]}"; do
        idx=$((idx + 1))
        local name="${SUB_TASK_NAMES[$((idx - 1))]}"
        local artifact="${SUB_TASK_ARTIFACTS[$((idx - 1))]}"
        local log="${SUB_TASK_LOGS[$((idx - 1))]}"
        local cmd="${SUB_TASK_COMMANDS[$((idx - 1))]}"

        case "${result}" in
            PASS) echo -e "  ${GREEN}[PASS]${NC} ${name}" ;;
            SKIP) echo -e "  ${YELLOW}[SKIP]${NC} ${name}" ;;
            FAIL) echo -e "  ${RED}[FAIL]${NC} ${name}" ;;
        esac

        [[ -n "${artifact}" ]] && echo -e "    Artifact: ${artifact}"
        echo -e "    Log: ${log}"
        if [[ "${result}" == "FAIL" ]] && [[ -n "${cmd}" ]]; then
            echo -e "    Command: ${RED}${cmd}${NC}"
        fi
    done

    echo ""
    [[ ${mins} -gt 0 ]] && print_info "Total time: ${mins}m${secs}s" || print_info "Total time: ${secs}s"

    local has_fail=false
    for result in "${CHECK_RESULTS[@]}"; do
        [[ "${result}" == "FAIL" ]] && { has_fail=true; break; }
    done
    for result in "${SUB_TASK_RESULTS[@]}"; do
        [[ "${result}" == "FAIL" ]] && { has_fail=true; break; }
    done

    if [[ "${has_fail}" == "true" ]]; then
        print_fail "Some checks FAILED"
        return 1
    else
        print_ok "All checks PASSED"
        rm -f "${PR_FILELIST}"
        rm -rf "${PROJECT_ROOT}/tmp_verify"
        return 0
    fi
}

# ============================================================
# 主函数
# ============================================================
main() {
    parse_args "$@"
    
    # 设置 CI 脚本依赖的环境变量
    export REPOSITORY_NAME="ops-math"
    export WORKSPACE="${PROJECT_ROOT}"
    export CANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}"
    export ASCEND_3RD_LIB_PATH="${ASCEND_3RD_LIB_PATH}"
    
    # 全局单任务互斥检测
    local compile_task=false
    local llt_task=false
    local cannsim_task=false
    [[ "${JIT_MODE}" == "true" || "${EXPERIMENTAL_MODE}" == "true" || "${SINGLE_MODE}" == "true" || "${A5_MODE}" == "true" || "${NORMAL_MODE}" == "true" ]] && compile_task=true
    [[ "${LLT_A900_MODE}" == "true" || "${LLT_ST_MODE}" == "true" || "${LLT_EXP_UT_MODE}" == "true" || "${LLT_KERNEL_UT_MODE}" == "true" || "${LLT_STD_UT_MODE}" == "true" ]] && llt_task=true
    [[ "${CANNSIM_MODE}" == "true" ]] && cannsim_task=true

    local task_count=0
    [[ "${compile_task}" == "true" ]] && task_count=$((task_count + 1))
    [[ "${llt_task}" == "true" ]] && task_count=$((task_count + 1))
    [[ "${cannsim_task}" == "true" ]] && task_count=$((task_count + 1))
    if [[ ${task_count} -gt 1 ]]; then
        echo "[ERROR] Cannot specify single-task parameters from different flows simultaneously"
        echo "  Compile: --jit/--experimental/--single/--a5/--normal"
        echo "  LLT: --llt-a900/--llt-st/--llt-exp-ut/--llt-kernel-ut/--llt-std-ut"
        echo "  Cannsim: --cannsim"
        exit 1
    fi

    if [[ "${compile_task}" == "true" ]]; then
        SKIP_LLT=true
        SKIP_CANNSIM=true
        print_info "Single-task mode: compile only, LLT and cannsim auto-skipped"
    fi
    if [[ "${llt_task}" == "true" ]]; then
        SKIP_BUILD=true
        SKIP_CANNSIM=true
        print_info "Single-task mode: LLT only, compile and cannsim auto-skipped"
    fi
    if [[ "${cannsim_task}" == "true" ]]; then
        SKIP_BUILD=true
        SKIP_LLT=true
        print_info "Single-task mode: cannsim only, compile and LLT auto-skipped"
    fi
    
    cd "${PROJECT_ROOT}"
    mkdir -p "${PROJECT_ROOT}/logs"
    
    local TOTAL_START=$(date +%s)
    
    print_header "ops-math Local Check"
    echo -e "  Project:   ${PROJECT_ROOT}"
    echo -e "  SoC:       ${SOC:-auto-detect}"
    echo -e "  Jobs:      ${JOBS}"
    echo -e "  Ops:       ${OPS:-auto-detect}"
    echo -e "  Exp:       ${EXPERIMENTAL_MODE}"
    echo -e "  Mode:      JIT=${JIT_MODE} Exp=${EXPERIMENTAL_MODE} Single=${SINGLE_MODE} A5=${A5_MODE} Normal=${NORMAL_MODE}"
    echo -e "  LLT:       A900=${LLT_A900_MODE} St=${LLT_ST_MODE} ExpUT=${LLT_EXP_UT_MODE} KernelUT=${LLT_KERNEL_UT_MODE} StdUT=${LLT_STD_UT_MODE} Cannsim=${CANNSIM_MODE}"
    echo -e "  Logs:      ${PROJECT_ROOT}/logs/"
    
    check_environment || exit 1
    detect_changed_ops || { print_fail "Failed to detect changed files"; exit 1; }
    
    # 主流程
    run_check "Compile Flow"   do_compile_flow
    run_check "LLT Flow"       do_llt_flow
    
    # 仿真测试
    run_check "Cannsim Tests"  do_cannsim_check
    
    print_summary "${TOTAL_START}"
}

main "$@"
