# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the full text of the License in the software repository.
# ----------------------------------------------------------------------------

# 设置算子定义时支持的芯片类型
set(SUPPORT_COMPUTE_UNIT "ascend950")
# 设置每种芯片类型对应的tiling文件目录，即采用op_host目录下哪个文件夹下的tiling文件编译
set(SUPPORT_TILING_DIR "arch35")
add_all_modules_sources(OPTYPE histogram_fixed_width ACLNNTYPE aclnn_exclude COMPUTE_UNIT ascend950 TILING_DIR arch35)

add_kernel_sources(
    KERNEL_SRC arch35/histogram_fixed_width.cpp
    COMPUTE_UNITS ascend950
    AUTO_SYNC false
)
