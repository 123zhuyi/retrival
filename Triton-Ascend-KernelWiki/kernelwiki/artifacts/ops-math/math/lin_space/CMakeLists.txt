# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        if(DISABLE_AICPU)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu" AND IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu")
                MESSAGE(STATUS "disable aicpu kernel lin_space, skip it.")
                continue()
            endif()
        endif()
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

if(ENABLE_TEST AND (UT_TEST_ALL OR OP_KERNEL_AICPU_UT))
    list(FIND ASCEND_OP_NAME lin_space LIN_SPACE_INDEX)
    if("${ASCEND_OP_NAME}" STREQUAL "" OR NOT LIN_SPACE_INDEX EQUAL -1)
        add_aicpu_op_test_case(lin_space)
    endif()
endif()