# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

file(GLOB LIN_SPACE_AICPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu.cpp)
file(GLOB LIN_SPACE_AICPU_OP_DEF_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu_def.cpp)
if(LIN_SPACE_AICPU_OP_DEF_SRCS)
    set_property(GLOBAL APPEND PROPERTY AICPU_OPDEF_FILES ${LIN_SPACE_AICPU_OP_DEF_SRCS})
endif()
if(LIN_SPACE_AICPU_SRCS)
    if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
        add_aicpu_kernel_modules()
        target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${LIN_SPACE_AICPU_SRCS})
    else()
        get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
        get_filename_component(OP_NAME ${PARENT_DIR} NAME)
        add_aicpu_cust_kernel_modules(${OP_NAME} "${LIN_SPACE_AICPU_SRCS}")
    endif()
endif()