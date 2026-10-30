#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# FusedMulAddN PyTorch ST 一键编译 + 运行脚本（真实 NPU + ACLNN 两段式）
#
# 用法:
#   bash run_torch.sh                 # 编译 torch_adapter 并跑 L0+L1 全量
#   bash run_torch.sh --golden-only   # 仅 CPU golden 自测（无需 NPU / 自定义包）
#   bash run_torch.sh --level L0       # 只跑 L0
#   bash run_torch.sh --case L1_023    # 只跑指定用例
#
# 自定义算子包（custom_math，含 aclnnFusedMulAddN + ascend910b kernel）查找优先级：
#   1) 环境变量 CUSTOM_OP_VENDOR_DIR 显式指定（指向 .../vendors/custom_math）
#   2) 仓库内已构建/解包的 build_out/installed_custom_math/packages/vendors/custom_math
#   3) 标准安装路径 $HOME/Ascend/opp/vendors/custom_math 或 $ASCEND_HOME_PATH/opp/vendors/custom_math
#
# 若 custom_math 尚未生成，请先构建并解包：
#   bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_mul_add_n
#   bash build_out/cann-ops-math-custom_linux-*.run --noexec --quiet \
#        --extract=$(pwd)/build_out/installed_custom_math
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 定位仓库根（worktree 根）：tests/st/torch -> 上溯到含 build.sh 的目录
REPO_ROOT="$SCRIPT_DIR"
while [[ "$REPO_ROOT" != "/" && ! -f "$REPO_ROOT/build.sh" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done

# --- golden-only 快速分支：无需 NPU / 自定义包 ---
for arg in "$@"; do
    if [[ "$arg" == "--golden-only" ]]; then
        python3 "$SCRIPT_DIR/test.py" --golden-only
        exit $?
    fi
done

# --- 解析自定义算子包 vendor 目录 ---
resolve_vendor() {
    local candidates=(
        "${CUSTOM_OP_VENDOR_DIR:-}"
        "$REPO_ROOT/build_out/installed_custom_math/packages/vendors/custom_math"
        "$HOME/Ascend/opp/vendors/custom_math"
        "${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/opp/vendors/custom_math"
    )
    for c in "${candidates[@]}"; do
        if [[ -n "$c" && -f "$c/op_api/include/aclnn_fused_mul_add_n.h" \
              && -f "$c/op_api/lib/libcust_opapi.so" ]]; then
            echo "$c"; return 0
        fi
    done
    return 1
}

VENDOR_DIR="$(resolve_vendor || true)"
if [[ -z "$VENDOR_DIR" ]]; then
    echo "[ERROR] 未找到 custom_math 自定义算子包（含 aclnn_fused_mul_add_n.h + libcust_opapi.so）。"
    echo "        请先构建并解包，或设置 CUSTOM_OP_VENDOR_DIR。详见本脚本头部注释。"
    exit 2
fi
echo "[INFO] custom_math vendor dir: $VENDOR_DIR"

# --- 编译 torch_adapter ---
BUILD_DIR="$SCRIPT_DIR/build"
TORCH_CMAKE_PREFIX="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
mkdir -p "$BUILD_DIR"
pushd "$BUILD_DIR" >/dev/null
cmake .. -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" -DCUSTOM_OP_VENDOR_DIR="$VENDOR_DIR" >/dev/null
make
popd >/dev/null

LIB="$BUILD_DIR/libtorch_adapter.so"
[[ -f "$LIB" ]] || { echo "[ERROR] 未生成 $LIB"; exit 3; }

# --- 运行时环境：让 ACLNN 找到自定义 kernel + op_api lib ---
export ASCEND_CUSTOM_OPP_PATH="$VENDOR_DIR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$VENDOR_DIR/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[INFO] ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
python3 "$SCRIPT_DIR/test.py" --lib "$LIB" "$@"
