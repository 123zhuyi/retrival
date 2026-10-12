# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 需要将Tiling依赖的文件添加到CMakeLists.txt中
    # set(elewise_common_tiling_files
    #         ${CANN_ROOT}/ops/built-in/op_tiling/runtime/elewise_tiling.cc
    #         )
    # 算子自己的tiling文件路径
    set(is_inf_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/is_inf_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/is_inf_tiling_arch35.cpp
        )
    # 指定UT源文件，不指定则会编译全部以test_为前缀的源文件
    set(is_inf_ut_files
        ${CMAKE_CURRENT_SOURCE_DIR}/test_is_inf.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_is_inf_arch35.cpp
    )

    #AddOpsTestCase(
    #   OP_NAME is_inf
    #   SOC_VERSION "ascend910b"
    #   TILING_SRC_FILES ${is_inf_tiling_files}
    #   UT_SRC_FILES ${is_inf_ut_files}
    #)
endif()
