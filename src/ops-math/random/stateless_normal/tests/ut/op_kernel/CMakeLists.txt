# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_KERNEL_UT)
    set(KERNEL_STAGING_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_dep_staging)
    file(MAKE_DIRECTORY ${KERNEL_STAGING_DIR}/stateless_normal/arch35)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${PROJECT_SOURCE_DIR}/random/random_common/op_kernel
        ${KERNEL_STAGING_DIR}/random_common)

    set(stateless_normal_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/stateless_normal_tiling_arch35.cpp)

    # 默认 float32 测试（锁步：DTYPE_Y/DTYPE_MEAN/DTYPE_STDEV 必须一致）
    # 测试代码已类型自适应，修改下方 -DDTYPE_Y/DTYPE_MEAN/DTYPE_STDEV 即可测试其他 dtype：
    #   float32: -DDTYPE_Y=float -DDTYPE_MEAN=float -DDTYPE_STDEV=float
    #   float16: -DDTYPE_Y=half -DDTYPE_MEAN=half -DDTYPE_STDEV=half
    #   bfloat16: -DDTYPE_Y=bfloat16_t -DDTYPE_MEAN=bfloat16_t -DDTYPE_STDEV=bfloat16_t
    # 注意：框架限制，同一 opName 同一平台仅支持一个 AddOpTestCase
    AddOpTestCase(
        stateless_normal
        "ascend950"
        "-DDTYPE_Y=float -DDTYPE_MEAN=float -DDTYPE_STDEV=float -DTestUtDefaultTilingStruct=RandomUnifiedSimtTilingDataStruct -I${KERNEL_STAGING_DIR}/stateless_normal/arch35"
        "${stateless_normal_tiling_files}")
endif()
