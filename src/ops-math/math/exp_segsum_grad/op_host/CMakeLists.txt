# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_modules_sources(OPTYPE exp_segsum_grad ACLNNTYPE aclnn_exclude)

add_tiling_modules()
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    set(EXP_SEGSUM_GRAD_TILING_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35/exp_segsum_grad_tiling_arch35.cpp)
else()
    set(EXP_SEGSUM_GRAD_TILING_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22/exp_segsum_grad_tiling.cpp)
endif()
target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${EXP_SEGSUM_GRAD_TILING_SRC})
