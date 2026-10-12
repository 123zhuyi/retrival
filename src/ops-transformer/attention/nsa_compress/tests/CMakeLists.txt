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

set(_NSA_OpApiSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/nsa_compress/op_host/op_api/nsa_compress.cpp
        ${OPS_TRANSFORMER_DIR}/attention/nsa_compress/op_host/op_api/aclnn_nsa_compress.cpp
)

set(_NSA_OpProtoSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/nsa_compress/op_host/nsa_compress_infershape.cpp
)

set(_NSA_OpTilingSourcesExt)
file(GLOB _Src1 "${OPS_TRANSFORMER_DIR}/attention/nsa_compress/op_host/nsa_compress_tiling*.cc")
file(GLOB _Src2 "${OPS_TRANSFORMER_DIR}/attention/nsa_compress/op_host/nsa_compress_tiling*.cpp")
list(APPEND _NSA_OpTilingSourcesExt ${_Src1} ${_Src2})

set(_NSA_OpTilingPrivateIncludesExt)

set(_NSA_OpKernelTilingDataDefH
        # 公共
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        # NSA
        ${OPS_TRANSFORMER_DIR}/attention/nsa_compress/op_host/nsa_compress_tiling.h
)

set(_NSA_OpKernelSourcesExt)

set(_NSA_OpKernelPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/common/include/kernel
)

set(_NSA_OpKernelPrivateCompileDefinitionsExt
        KernelCtrlParam nsa_compress fp16 ORIG_DTYPE_QUERY=DT_FLOAT16 DTYPE_DQ=half       KFC_L1_RESERVER_SIZE=0
        KernelCtrlParam nsa_compress bf16 ORIG_DTYPE_QUERY=DT_BF16    DTYPE_DQ=bfloat16_t KFC_L1_RESERVER_SIZE=0
)
set(_NSA_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
        ${OPS_TRANSFORMER_DIR}/attention/nsa_compress/op_host
        ${OPS_TRANSFORMER_DIR}/attention/nsa_compress/op_host/op_api
)
set(_NSA_UTestCommonPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling    # 用于 FAG 指定模板优先级相关实现
        error_manager                   # 用于 FAG 指定模板优先级相关实现
)

set(_NSA_UTestSourcesExtForce)
file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
list(APPEND _NSA_UTestSourcesExtForce ${_Src1} ${_Src2})

set(_NSA_UTestAclnnSourcesExtForce)
file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cpp")
list(APPEND _NSA_UTestAclnnSourcesExtForce ${_Src1} ${_Src2})

set(_NSA_UTestCompileDefinitions)

list(APPEND _NSA_OpKernelSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/nsa_compress/op_kernel/nsa_compress.cpp
)

file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
list(APPEND _NSA_UTestSourcesExtForce ${_Src1} ${_Src2})

file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cpp")
list(APPEND _NSA_UTestAclnnSourcesExtForce ${_Src1} ${_Src2})

list(APPEND _NSA_UTestCompileDefinitions)


OpsTest_Level2_AddOp(
        SUB_SYSTEM                                transformer
        BRIEF                                     NSA
        SNAKE                                     nsa_compress
        OPAPI_SOURCES_EXT                         ${_NSA_OpApiSourcesExt}
        PROTO_SOURCES_EXT                         ${_NSA_OpProtoSourcesExt}
        TILING_SOURCES_EXT                        ${_NSA_OpTilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT               ${_NSA_OpTilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                        ${_NSA_OpKernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                  ${_NSA_OpKernelTilingDataDefH}
        KERNEL_PRIVATE_INCLUDES_EXT               ${_NSA_OpKernelPrivateIncludesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT    ${_NSA_OpKernelPrivateCompileDefinitionsExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT         ${_NSA_UTestCommonPrivateIncludeExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT   ${_NSA_UTestCommonPrivateLinkLibrariesExt}
        UTEST_COMMON_PRIVATE_COMPILE_DEFINITIONS  ${_NSA_UTestCompileDefinitions}
        UTEST_SOURCES_EXT                         ${_NSA_UTestSourcesExtForce}
        UTEST_SOURCES_EXT_FORCE                   ON
        UTEST_PRIVATE_COMPILE_DEFINITIONS         ${_NSA_UTestCompileDefinitions}
        UTEST_ACLNN_SOURCES_EXT                   ${_NSA_UTestAclnnSourcesExtForce}
        UTEST_ACLNN_SOURCES_EXT_FORCE             ON
        UTEST_ACLNN_PRIVATE_COMPILE_DEFINITIONS   ${_NSA_UTestCompileDefinitions}
)
