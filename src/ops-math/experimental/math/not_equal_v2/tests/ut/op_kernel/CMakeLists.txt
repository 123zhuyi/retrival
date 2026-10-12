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
    set(not_equal_tiling_files ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/not_equal_tiling.cpp)
    AddOpTestCase(not_equal_v2 "ascend910b" "-DDTYPE_X1=float -DDTYPE_X2=float -DDTYPE_Y=bool" "${not_equal_tiling_files}")
    target_compile_options(not_equal_v2_Ascend910B1_tiling_tmp PRIVATE -Wno-error)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/tests/ut/op_kernel/compare_data.py ${CMAKE_BINARY_DIR}/tests/ut/op_kernel/gen_data.py COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/not_equal_data/compare_data.py ${CMAKE_CURRENT_SOURCE_DIR}/not_equal_data/gen_data.py ${CMAKE_BINARY_DIR}/tests/ut/op_kernel DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/not_equal_data/compare_data.py ${CMAKE_CURRENT_SOURCE_DIR}/not_equal_data/gen_data.py)
    add_custom_target(not_equal_v2_Ascend910B1_sync_py DEPENDS ${CMAKE_BINARY_DIR}/tests/ut/op_kernel/compare_data.py ${CMAKE_BINARY_DIR}/tests/ut/op_kernel/gen_data.py)
    add_dependencies(not_equal_v2_Ascend910B1_cases_obj not_equal_v2_Ascend910B1_sync_py)
endif()
