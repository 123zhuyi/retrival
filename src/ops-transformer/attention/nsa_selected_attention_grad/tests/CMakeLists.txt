# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(_nsag_TilingSourcesExt
        ${OPS_ADV_DIR}/attention/nsa_selected_attention_grad/op_host/nsa_selected_attention_grad_tiling_bs1.cpp
        ${OPS_ADV_DIR}/attention/nsa_selected_attention_grad/op_host/nsa_selected_attention_grad_tiling_bs1_basic.cpp
        ${OPS_ADV_DIR}/attention/nsa_selected_attention_grad/op_host/nsa_selected_attention_grad_tiling_common.cpp
        ${OPS_ADV_DIR}/attention/nsa_selected_attention_grad/op_host/nsa_selected_attention_grad_tiling.cpp
)

set(_nsag_KernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_grad/op_host/nsa_selected_attention_grad_tiling.h
)

set(_nsag_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)
aux_source_directory(${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_grad/op_kernel _nsag_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer/attention
        BRIEF                                   nsaGrad
        SNAKE                                   nsa_selected_attention_grad
        KERNEL_TILING_DATA_DEF_H                ${_nsag_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_nsag_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_nsag_TilingSourcesExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_nsag_TargetPrivateLinkLibrariesExt}
)
