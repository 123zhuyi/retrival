# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

# AcosGradV2: AcosGrad 算子的 A2 (Atlas A2 / Ascend910B, DAV_2201) 适配版本
# 对应 math/acos_grad README: z = -dy / sqrt(1 - y^2)

# 设置算子定义时支持的芯片类型：A2 训练/推理系列产品 (ascend910b)
set(SUPPORT_COMPUTE_UNIT "ascend910b")
# 设置每种芯片类型对应的 tiling 文件目录：A2 -> arch22
set(SUPPORT_TILING_DIR "arch22")
add_all_modules_sources(OPTYPE acos_grad_v2 ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)

# 显式注册 kernel 入口（arch22 / Ascend910B）
add_kernel_sources(
    KERNEL_SRC arch22/acos_grad_v2.cpp
    COMPUTE_UNITS ascend910b
)
