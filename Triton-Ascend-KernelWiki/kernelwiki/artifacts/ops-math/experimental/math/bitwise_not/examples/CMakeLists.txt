# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# BitwiseNot (experimental/math/bitwise_not) 调用示例独立编译工程。
#
# 两个示例:
#   - test_aclnn_bitwise_not : eager 两段式 aclnn 调用 (aclnnBitwiseNotGetWorkspaceSize / aclnnBitwiseNot)。
#   - test_geir_bitwise_not  : graph (GE IR) 模式 (Session AddGraph/RunGraph，OpType BitwiseNot)。
#
# 两段式 aclnnBitwiseNot 入口由自定义算子包提供（libcust_opapi.so）。
# eager 示例链接已安装自定义算子包的 op_api/lib/libcust_opapi.so，并 include op_api/include/aclnnop。
# 推荐用配套 run.sh 一键编译运行。
#
# 也可走仓库统一入口（在 ops-math 根目录）:
#   bash build.sh --run_example bitwise_not eager cust --experimental --vendor_name=custom
#   bash build.sh --run_example bitwise_not graph --experimental
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(bitwise_not_examples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 定位 CANN（与 build.sh 一致：ASCEND_HOME_PATH > ASCEND_TOOLKIT_HOME > 默认 latest） ----
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME $ENV{ASCEND_TOOLKIT_HOME})
else()
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()
message(STATUS "ASCEND_HOME = ${ASCEND_HOME}")

# OPP 路径（GE op 工厂头 array_ops.h/all_ops.h 等位于 ${OPP}/built-in/op_proto/inc）。
if(DEFINED ENV{ASCEND_OPP_PATH})
    set(ASCEND_OPP $ENV{ASCEND_OPP_PATH})
else()
    set(ASCEND_OPP "${ASCEND_HOME}/opp")
endif()

# 自定义算子包路径（含自动生成的 aclnn_bitwise_not.h 与 libcust_opapi.so）。
# run.sh 安装自定义包后 source set_env.bash 导出 ASCEND_CUSTOM_OPP_PATH 指向 vendors/<vendor>_math。
set(CUST_OPP_ROOT "")
if(DEFINED ENV{ASCEND_CUSTOM_OPP_PATH})
    # 取第一个路径段（冒号分隔）。
    string(REPLACE ":" ";" CUST_OPP_LIST "$ENV{ASCEND_CUSTOM_OPP_PATH}")
    list(GET CUST_OPP_LIST 0 CUST_OPP_ROOT)
endif()

option(BUILD_EAGER "Build eager (two-stage aclnnBitwiseNot) example" ON)
option(BUILD_GEIR  "Build graph (GE IR) example" ON)

# ============================================================================
# eager 示例：两段式 aclnnBitwiseNot（自动生成入口，链接 libcust_opapi）。
#   依赖 libascendcl（aclrt*）+ libcust_opapi（aclnnBitwiseNotGetWorkspaceSize / aclnnBitwiseNot）
#       + libnnopbase。
# ============================================================================
if(BUILD_EAGER)
    add_executable(test_aclnn_bitwise_not test_aclnn_bitwise_not.cpp)
    target_include_directories(test_aclnn_bitwise_not PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_HOME}/include
        ${ASCEND_HOME}/include/aclnnop
    )
    target_link_directories(test_aclnn_bitwise_not PRIVATE ${ASCEND_HOME}/lib64)
    target_link_libraries(test_aclnn_bitwise_not PRIVATE ascendcl)
    if(NOT CUST_OPP_ROOT STREQUAL "")
        message(STATUS "Custom opp (aclnnBitwiseNot) = ${CUST_OPP_ROOT}")
        target_include_directories(test_aclnn_bitwise_not PRIVATE
            ${CUST_OPP_ROOT}/op_api/include
            ${CUST_OPP_ROOT}/op_api/include/aclnnop
        )
        target_link_directories(test_aclnn_bitwise_not PRIVATE ${CUST_OPP_ROOT}/op_api/lib)
        target_link_libraries(test_aclnn_bitwise_not PRIVATE cust_opapi nnopbase)
        set_target_properties(test_aclnn_bitwise_not PROPERTIES
            BUILD_RPATH "${CUST_OPP_ROOT}/op_api/lib")
    else()
        message(WARNING "ASCEND_CUSTOM_OPP_PATH not set; eager example需先安装自定义算子包并 source set_env.bash 才能链接 aclnnBitwiseNot")
    endif()
endif()

# ============================================================================
# graph (GE IR) 示例：Session AddGraph/RunGraph。
#   include/link 对齐 build.sh compile_graph_example。
# ============================================================================
if(BUILD_GEIR)
    add_executable(test_geir_bitwise_not test_geir_bitwise_not.cpp)
    target_include_directories(test_geir_bitwise_not PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_HOME}/include
        ${ASCEND_HOME}/include/external
        ${ASCEND_HOME}/include/graph
        ${ASCEND_HOME}/include/ge
        ${ASCEND_OPP}/built-in/op_proto/inc
    )
    target_link_directories(test_geir_bitwise_not PRIVATE ${ASCEND_HOME}/lib64)
    target_link_libraries(test_geir_bitwise_not PRIVATE graph ge_runner graph_base ge_compiler)
endif()
