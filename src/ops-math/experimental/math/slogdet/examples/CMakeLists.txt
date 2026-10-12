# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# slogdet 算子 aclnn 调用示例构建脚本（experimental/math/slogdet，原生 AscendC，fp32）。
#   图模式 (GE IR) 示例见 CMakeLists_geir.txt（由 run.sh --graph 临时切换）。

cmake_minimum_required(VERSION 3.14)
project(ACLNN_SLOGDET_EXAMPLE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# ASCEND_PATH（CANN 软件包目录）
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()
message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

# CANN 架构相关 include（aarch64-linux / x86_64-linux）
if(EXISTS "${ASCEND_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
    set(CANN_ARCH_INCLUDE ${ASCEND_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include)
elseif(EXISTS "${ASCEND_PATH}/aarch64-linux/include")
    set(CANN_ARCH_INCLUDE ${ASCEND_PATH}/aarch64-linux/include)
elseif(EXISTS "${ASCEND_PATH}/x86_64-linux/include")
    set(CANN_ARCH_INCLUDE ${ASCEND_PATH}/x86_64-linux/include)
else()
    set(CANN_ARCH_INCLUDE ${ASCEND_PATH}/include)
endif()

# 查找自定义算子头文件 aclnn_slogdet_native.h（优先 SLOGDET_CUSTOM_OPP 指向的 vendor 目录）
find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_slogdet_native.h
    PATHS
        $ENV{SLOGDET_CUSTOM_OPP}/op_api/include
        ${ASCEND_PATH}/opp/vendors/custom_math/op_api/include
        ${ASCEND_PATH}/opp/vendors/slogdet_custom/op_api/include
        ${ASCEND_PATH}/opp/vendors/customize/op_api/include
)
if(NOT CUSTOM_OP_INCLUDE_DIR)
    message(FATAL_ERROR "未找到自定义算子头文件 aclnn_slogdet_native.h，请先安装 slogdet 自定义算子包并设置 SLOGDET_CUSTOM_OPP")
endif()
message(STATUS "自定义算子头文件目录: ${CUSTOM_OP_INCLUDE_DIR}")

# 查找自定义算子库 libcust_opapi.so
find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        $ENV{SLOGDET_CUSTOM_OPP}/op_api/lib
        ${ASCEND_PATH}/opp/vendors/custom_math/op_api/lib
        ${ASCEND_PATH}/opp/vendors/slogdet_custom/op_api/lib
        ${ASCEND_PATH}/opp/vendors/customize/op_api/lib
)
if(NOT CUSTOM_OP_LIBRARY)
    message(FATAL_ERROR "未找到自定义算子库 libcust_opapi.so，请先安装 slogdet 自定义算子包")
endif()
message(STATUS "自定义算子库: ${CUSTOM_OP_LIBRARY}")

add_executable(test_aclnn_slogdet test_aclnn_slogdet.cpp)

target_include_directories(test_aclnn_slogdet PRIVATE
    ${ASCEND_PATH}/include
    ${CANN_ARCH_INCLUDE}
    ${CANN_ARCH_INCLUDE}/aclnnop
    ${CUSTOM_OP_INCLUDE_DIR}
)

# 关键：自定义 libcust_opapi.so 也导出 aclnnSlogdet/aclnnSlogdetGetWorkspaceSize，
#   上游 libopapi.so 同样导出同名符号（math/slogdet 转发壳 -> LogMatrixDeterminant）。
# 用 `-Wl,--no-as-needed ${CUSTOM_OP_LIBRARY}` 置最前，保证 DT_NEEDED 中自定义库
#   先于 libopapi，运行时按 BFS 优先绑定到自定义 AscendC kernel 实现。
target_link_libraries(test_aclnn_slogdet
    -Wl,--no-as-needed
    ${CUSTOM_OP_LIBRARY}
    -Wl,--as-needed
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libopapi.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
)
get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
target_link_options(test_aclnn_slogdet PRIVATE
    "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
    "-Wl,-rpath,${ASCEND_PATH}/lib64"
)

install(TARGETS test_aclnn_slogdet DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
