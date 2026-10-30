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
set(_NsaSelectedAttention_OpProtoSourcesExt
        ${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention/op_host/nsa_selected_attention_infershape.cpp
)

set(_NsaSelectedAttention_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/nsa_selected_attention/op_host/nsa_selected_attention_tiling.cpp
        ${OPS_TRANSFORMER_DIR}/attention/nsa_selected_attention/op_host/nsa_selected_attention_tiling_general.cpp
)

set(_NsaSelectedAttention_KernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention/op_host/nsa_selected_attention_tiling.h
)

set(_NsaSelectedAttention_TilingPrivateIncludesExt
        ${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention/op_host/
        ${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention/op_host/op_api/
)

set(_ncai_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
)

set(_NSA_COMPRESS_GRAD_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)

aux_source_directory(${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention/op_kernel _NsaSelectedAttention_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer/attention
        BRIEF                                   NsaSelectedAttention
        SNAKE                                   nsa_selected_attention
        PROTO_SOURCES_EXT                       ${_NsaSelectedAttention_OpProtoSourcesExt}
        TILING_SOURCES_EXT                      ${_NsaSelectedAttention_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_NsaSelectedAttention_TilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                      ${_NsaSelectedAttention_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_NsaSelectedAttention_KernelTilingDataDefH}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_NsaSelectedAttention_TargetPrivateLinkLibrariesExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_ncai_UTestCommonPrivateIncludeExt}
)
