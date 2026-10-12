# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 说明：AddOpsTestCase 按 `${PROJECT_SOURCE_DIR}/*/${OP_NAME}/op_kernel/${OP_NAME}.cpp`
    # 约定查找 kernel 入口，仅匹配仓根下一层目录（如 math/<op>）；本算子位于
    # experimental/math/ 两层目录且仓内无同名一层算子，glob 取不到入口文件，
    # 需在各 soc 的 cases 目标上显式注入 kernel 入口源文件。

    # A5（ascend950）使用 arch35 kernel + arch35 tiling
    set(add_mat_mat_elements_plus_tiling_files_arch35
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/add_mat_mat_elements_plus_tiling_arch35.cpp
        )
    AddOpTestCase(add_mat_mat_elements_plus "ascend950" "-DDTYPE_C=float" "${add_mat_mat_elements_plus_tiling_files_arch35}")
    if (TARGET add_mat_mat_elements_plus_Ascend950PR_9599_cases_obj)
        target_sources(add_mat_mat_elements_plus_Ascend950PR_9599_cases_obj
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/add_mat_mat_elements_plus.cpp)
    endif()

    # A2（ascend910b）使用 arch22 kernel + arch22 tiling
    set(add_mat_mat_elements_plus_tiling_files_arch22
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch22/add_mat_mat_elements_plus_tiling_arch22.cpp
        )
    AddOpTestCase(add_mat_mat_elements_plus "ascend910b" "-DDTYPE_C=float" "${add_mat_mat_elements_plus_tiling_files_arch22}")
    if (TARGET add_mat_mat_elements_plus_Ascend910B1_cases_obj)
        target_sources(add_mat_mat_elements_plus_Ascend910B1_cases_obj
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/add_mat_mat_elements_plus_arch22.cpp)
    endif()

    # A3（ascend910_93）使用 arch22 kernel + arch22 tiling
    AddOpTestCase(add_mat_mat_elements_plus "ascend910_93" "-DDTYPE_C=float" "${add_mat_mat_elements_plus_tiling_files_arch22}")
    if (TARGET add_mat_mat_elements_plus_Ascend910_9391_cases_obj)
        target_sources(add_mat_mat_elements_plus_Ascend910_9391_cases_obj
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/add_mat_mat_elements_plus_arch22.cpp)
    endif()
endif()
