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

set(_NSA_COMPRESS_GRAD_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/nsa_compress_grad/op_host/nsa_compress_grad_tiling.cpp
)

set(_NSA_COMPRESS_GRAD_KernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/nsa_compress_grad/op_host/nsa_compress_grad_tiling.h
)

set(_NSA_COMPRESS_GRAD_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/attention/nsa_compress_grad/op_host/
        ${CMAKE_SOURCE_DIR}/attention/nsa_compress_grad/op_host/op_api/
)

set(_ncai_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
)

set(_NSA_COMPRESS_GRAD_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)

set(_NSA_COMPRESS_GRAD_CompileDefintions
        DTYPE_X=float
)
aux_source_directory(${CMAKE_SOURCE_DIR}/attention/nsa_compress_grad/op_kernel _NSA_COMPRESS_GRAD_KernelSourcesExt)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   NsaCompressGrad
        SNAKE                                   nsa_compress_grad
        KERNEL_TILING_DATA_DEF_H                ${_NSA_COMPRESS_GRAD_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_NSA_COMPRESS_GRAD_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_NSA_COMPRESS_GRAD_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_NSA_COMPRESS_GRAD_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_NSA_COMPRESS_GRAD_CompileDefintions}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_NSA_COMPRESS_GRAD_TargetPrivateLinkLibrariesExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_ncai_UTestCommonPrivateIncludeExt}
)
