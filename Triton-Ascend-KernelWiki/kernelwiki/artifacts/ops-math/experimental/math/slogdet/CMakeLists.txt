# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# NOTE: Portions of this code were AI-generated and have been
# technically reviewed for functional accuracy and security

# 设置算子定义时支持的芯片类型（Ascend 910B + 910C/A3，均复用 arch32 tiling）
set(SUPPORT_COMPUTE_UNIT "ascend910b" "ascend910_93")
# 设置每种芯片类型对应的 tiling 文件目录（op_host/arch32）。
# arch32 为实验树对 DAV_2201/910B/910C(A3) 家族沿用的目录标签（与 CLAUDE.md 中 arch22 同指 DAV_2201）。
set(SUPPORT_TILING_DIR "arch32" "arch32")
# ACLNNTYPE aclnn_exclude + DISABLE_IN_OPP TRUE：
#   - 不进 OPP 包（实验树构建隔离，避免与 math/slogdet 同时进同一包）
#   - aclnnSlogdet 符号仅在 experimental 构建时编译（与 math/slogdet 不同时构建）
add_all_modules_sources(OPTYPE slogdet ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)
