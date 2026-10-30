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
set(_LIG_OpApiSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/lightning_indexer_grad/op_host/op_api/lightning_indexer_grad.cpp
)

set(_LIG_OpProtoSourcesExt
        ${CMAKE_SOURCE_DIR}/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_infershape.cpp
)

set(_LIG_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/lightning_indexer_grad/op_host/lightning_indexer_grad_tiling.cpp
)

set(_LIG_KernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/lightning_indexer_grad/op_kernel/lightning_indexer_grad_tiling.h
)

set(_LIG_TilingPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/attention/common/op_kernel/arch35
        ${CMAKE_SOURCE_DIR}/attention/lightning_indexer_grad/op_host/
        ${CMAKE_SOURCE_DIR}/attention/lightning_indexer_grad/op_host/op_api/
)

set(_LIG_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
        ${OPS_TRANSFORMER_DIR}/common/include/kernel
        ${OPS_TRANSFORMER_DIR}/attention/lightning_indexer_grad/op_host
        ${CMAKE_SOURCE_DIR}/attention/lightning_indexer_grad/op_host/op_api/
)

set(_LIG_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)
set(_LIG_OpKernelPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/common/include/kernel
)

set(_LIG_OpKernelPrivateCompileDefinitionsExt
        ORIG_DTYPE_QUERY=DT_FLOAT16
)

set(_LIG_UTestCompileDefinitions)
list(APPEND _LIG_UTestCompileDefinitions TESTS_UT_OPS_TEST_LIG)

file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
list(APPEND _LIG_UTestSourcesExtForce ${_Src1} ${_Src2})

aux_source_directory(${CMAKE_SOURCE_DIR}/attention/lightning_indexer_grad/op_kernel _LIG_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   lig
        SNAKE                                   lightning_indexer_grad
        OPAPI_SOURCES_EXT                       ${_LIG_OpApiSourcesExt}
        PROTO_SOURCES_EXT                       ${_LIG_OpProtoSourcesExt}
        TILING_SOURCES_EXT                      ${_LIG_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_LIG_TilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                      ${_LIG_KernelSourcesExt}
        KERNEL_PRIVATE_INCLUDES_EXT             ${_LIG_OpKernelPrivateIncludesExt}
        KERNEL_TILING_DATA_DEF_H                ${_LIG_KernelTilingDataDefH}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_LIG_OpKernelPrivateCompileDefinitionsExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT         ${_LIG_UTestCommonPrivateIncludeExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT   ${_LIG_TargetPrivateLinkLibrariesExt}
        UTEST_COMMON_PRIVATE_COMPILE_DEFINITIONS  ${_LIG_UTestCompileDefinitions}
        UTEST_SOURCES_EXT                         ${_LIG_UTestSourcesExtForce}
        UTEST_SOURCES_EXT_FORCE                   ON
        UTEST_PRIVATE_COMPILE_DEFINITIONS         ${_LIG_UTestCompileDefinitions}
)
