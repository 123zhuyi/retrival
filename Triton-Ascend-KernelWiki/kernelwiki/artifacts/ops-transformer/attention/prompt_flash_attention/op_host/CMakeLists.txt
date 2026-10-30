# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        prompt_flash_attention_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME PromptFlashAttention
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME PromptFlashAttention
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    if("${CONDITION_UNIT}" STREQUAL "ascend950")
        set(prompt_flash_attention_apt_depends attention/flash_attention_score attention/common CACHE INTERNAL "Dependencies for prompt_flash_attention")
    else()
        set(prompt_flash_attention_depends attention/flash_attention_score attention/common CACHE INTERNAL "Dependencies for prompt_flash_attention")
    endif()
else()
    set(prompt_flash_attention_depends attention/common CACHE INTERNAL "Dependencies for prompt_flash_attention")
    add_modules_sources_with_soc(OPTYPE prompt_flash_attention ACLNNTYPE aclnn_inner)
endif()

if(NOT BUILD_OPS_RTY_KERNEL) #
    if("${CONDITION_UNIT}" STREQUAL "ascend950")
        set(prompt_flash_attention_apt_depends attention/common CACHE INTERNAL "Dependencies for prompt_flash_attention")
    else()
        set(prompt_flash_attention_depends attention/common CACHE INTERNAL "Dependencies for prompt_flash_attention")
    endif()
    add_modules_sources_with_soc(OPTYPE prompt_flash_attention ACLNNTYPE aclnn_inner)
endif()
