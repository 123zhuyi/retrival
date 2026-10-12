# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")

# 本算子只交付 ascend950（DAV_3510 / arch35），Kernel 入口固定为 arch35/square_sum_all.cpp。
# 注意：ops-math 的 add_kernel_sources 以算子根目录为基准推导 op_type，必须在此调用，
# 放进 op_kernel/CMakeLists.txt 会解析出空 op_type，导致 kernelSrc.value 不写入 ops-info。
add_kernel_sources(
    KERNEL_SRC arch35/square_sum_all.cpp
    COMPUTE_UNITS ${SUPPORT_COMPUTE_UNIT}
)

add_all_modules_sources(
    OPTYPE square_sum_all
    ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE)
