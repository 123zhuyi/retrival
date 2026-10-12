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

set(_AttentionUpdate_OpApiSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/attention_update/op_host/op_api/aclnn_attention_update.cpp
        ${OPS_TRANSFORMER_DIR}/attention/attention_update/op_host/op_api/attention_update.cpp
)

#set(_AttentionUpdate_OpProtoSourcesExt
#        ${OPS_TRANSFORMER_DIR}/attention/nsa_compress_with_cache/op_host/nsa_compress_with_cache_infershape.cpp
#)

set(_AttentionUpdate_OpTilingSourcesExt)
file(GLOB _Src1 "${OPS_TRANSFORMER_DIR}/attention/attention_update/op_host/decode_update_tiling*.cc")
file(GLOB _Src2 "${OPS_TRANSFORMER_DIR}/attention/attention_update/op_host/decode_update_tiling*.cpp")
list(APPEND _AttentionUpdate_OpTilingSourcesExt ${_Src1} ${_Src2})

set(_AttentionUpdate_OpTilingPrivateIncludesExt)

set(_AttentionUpdate_OpKernelTilingDataDefH
        # 公共
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        # AttentionUpdate
        ${OPS_TRANSFORMER_DIR}/attention/attention_update/op_host/decode_update_tiling.h
)

set(_AttentionUpdate_OpKernelSourcesExt)

set(_AttentionUpdate_OpKernelPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/common/include/kernel
)

#set(_AttentionUpdate_OpKernelPrivateCompileDefinitionsExt
#        KernelCtrlParam attention_update fp32 ORIG_DTYPE_QUERY=DT_FLOAT32 DTYPE_DQ=half       KFC_L1_RESERVER_SIZE=0
#        KernelCtrlParam attention_update bf16 ORIG_DTYPE_QUERY=DT_BF16    DTYPE_DQ=bfloat16_t KFC_L1_RESERVER_SIZE=0
#)
set(_AttentionUpdate_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
        ${OPS_TRANSFORMER_DIR}/attention/attention_update/op_host
        ${OPS_TRANSFORMER_DIR}/attention/attention_update/op_host/op_api
)
set(_AttentionUpdate_UTestCommonPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)


list(APPEND _AttentionUpdate_OpKernelSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/attention_update/op_kernel/attention_update.cpp
)

set(_AttentionUpdate_UTestSourcesExtForce)
if (UT_TEST_ALL)
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
elseif(OP_HOST_UT)
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*tiling.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*tiling.cpp")
elseif(OP_KERNEL_UT)
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*kernel.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*kernel.cpp")
endif()
list(APPEND _AttentionUpdate_UTestSourcesExtForce ${_Src1} ${_Src2})

set(_AttentionUpdate_UTestAclnnSourcesExtForce)
if (OP_API_UT)
    file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cc")
    file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cpp")
list(APPEND _AttentionUpdate_UTestAclnnSourcesExtForce ${_Src1} ${_Src2})
else()
    list(APPEND _AttentionUpdate_UTestAclnnSourcesExtForce)
endif()

set(_AttentionUpdate_UTestCompileDefinitions)
list(APPEND _AttentionUpdate_UTestCompileDefinitions)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                                transformer
        BRIEF                                     AttentionUpdate
        SNAKE                                     attention_update
        OPAPI_SOURCES_EXT                         ${_AttentionUpdate_OpApiSourcesExt}
        #PROTO_SOURCES_EXT                         ${_AttentionUpdate_OpProtoSourcesExt}
        TILING_SOURCES_EXT                        ${_AttentionUpdate_OpTilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT               ${_AttentionUpdate_OpTilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                        ${_AttentionUpdate_OpKernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                  ${_AttentionUpdate_OpKernelTilingDataDefH}
        KERNEL_PRIVATE_INCLUDES_EXT               ${_AttentionUpdate_OpKernelPrivateIncludesExt}
        #KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT    ${_AttentionUpdate_OpKernelPrivateCompileDefinitionsExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT         ${_AttentionUpdate_UTestCommonPrivateIncludeExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT   ${_AttentionUpdate_UTestCommonPrivateLinkLibrariesExt}
        UTEST_COMMON_PRIVATE_COMPILE_DEFINITIONS  ${_AttentionUpdate_UTestCompileDefinitions}
        UTEST_SOURCES_EXT                         ${_AttentionUpdate_UTestSourcesExtForce}
        UTEST_SOURCES_EXT_FORCE                   ON
        UTEST_PRIVATE_COMPILE_DEFINITIONS         ${_AttentionUpdate_UTestCompileDefinitions}
        UTEST_ACLNN_SOURCES_EXT                   ${_AttentionUpdate_UTestAclnnSourcesExtForce}
        UTEST_ACLNN_SOURCES_EXT_FORCE             ON
        UTEST_ACLNN_PRIVATE_COMPILE_DEFINITIONS   ${_AttentionUpdate_UTestCompileDefinitions}
)
