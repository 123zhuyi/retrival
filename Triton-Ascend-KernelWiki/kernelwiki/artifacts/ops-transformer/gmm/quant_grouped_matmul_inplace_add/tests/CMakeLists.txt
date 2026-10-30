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
set(_QGMMIA_TilingSourcesExt
        ${CMAKE_SOURCE_DIR}/src/transformer/gmm/quant_grouped_matmul_inplace_add/op_host/arch35/quant_grouped_matmul_inplace_add_tiling.cpp
        ${CMAKE_SOURCE_DIR}/src/transformer/gmm/grouped_matmul/op_host/arch35/grouped_quant_matmul_tiling.cpp
)

set(_QGMMIA_TilingPrivateIncludesExt
        ${CMAKE_SOURCE_DIR}/src/transformer/gmm/quant_grouped_matmul_inplace_add/op_host/arch35
)

set(_QGMMIA_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/src/transformer/gmm/quant_grouped_matmul_inplace_add/op_host/arch35/quant_grouped_matmul_inplace_add_tiling.h
)

set(_QGMMIA_OpApiSourcesExt
        ${CMAKE_SOURCE_DIR}/src/transformer/gmm/grouped_matmul/op_host/op_api/grouped_matmul_util.cpp
        ${CMAKE_SOURCE_DIR}/src/transformer/gmm/grouped_matmul/op_host/op_api/arch35/grouped_matmul_950_checker.cpp
        ${CMAKE_SOURCE_DIR}/src/transformer/gmm/quant_grouped_matmul_inplace_add/op_host/op_api/aclnn_quant_grouped_matmul_inplace_add.cpp
)

set(_QGMMIA_OpProtoSourcesExt
        ${CMAKE_SOURCE_DIR}/src/transformer/gmm/quant_grouped_matmul_inplace_add/op_host/arch35/quant_grouped_matmul_inplace_add_proto.cpp
)

set(_QGMMIA_TargetPrivateIncludeExt
        ${CMAKE_SOURCE_DIR}/src/transformer/gmm/quant_grouped_matmul_inplace_add/op_host/arch35
)

set(_QGMMIA_TillingPrivateLinkLibrariesExt
)

set(_QGMMIA_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/gmm/quant_grouped_matmul_inplace_add/ _QGMMIA_KernelSourcesExt)
set(_QGMMIA_KernelPrivateCompileDefinitionsExt
        KernelCtrlParam quant_grouped_matmul_inplace_add quant_int8 DTYPE_X1=int8_t DTYPE_X2=int8_t DTYPE_SCALE2=uint64_t DTYPE_Y=int8_t ORIG_DTYPE_X=DT_INT8 ORIG_DTYPE_X2=DT_INT8 ORIG_DTYPE_Y=DT_INT8)

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
        message(STATUS "The UTest of quant_grouped_matmul_inplace_add don't supported clang compiler yet.")
else ()
        # OpsTest_Level2_AddOp(
        #         SUB_SYSTEM                              transformer
        #         BRIEF                                   QGMMIA
        #         SNAKE                                   quant_grouped_matmul_inplace_add
        #         OPAPI_SOURCES_EXT                       ${_QGMMIA_OpApiSourcesExt}
        #         PROTO_SOURCES_EXT                       ${_QGMMIA_OpProtoSourcesExt}
        #         TILING_SOURCES_EXT                      ${_QGMMIA_TilingSourcesExt}
        #         TILING_PRIVATE_INCLUDES_EXT             ${_QGMMIA_TilingPrivateIncludesExt}
        #         TILING_PRIVATE_LINK_LIBRARIES_EXT       ${_QGMMIA_TillingPrivateLinkLibrariesExt}
        #         KERNEL_SOURCES_EXT                      ${_QGMMIA_KernelSourcesExt}
        #         KERNEL_TILING_DATA_DEF_H                ${_QGMMIA_KernelTilingDataDefH}
        #         KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_QGMMIA_KernelPrivateCompileDefinitionsExt}
        #         UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_QGMMIA_TargetPrivateIncludeExt}
        #         UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_QGMMIA_TargetPrivateLinkLibrariesExt}
        # )
endif()