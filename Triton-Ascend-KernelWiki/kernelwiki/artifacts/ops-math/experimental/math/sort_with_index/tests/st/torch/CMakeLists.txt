# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for sort_with_index ST test (PyTorch adapter, ascend910b)
#
#   本 CMake 把 torch_adapter.cpp 编译为 libtorch_adapter.so（含 ACLNN 两段式封装）。
#   NPU 实跑需 torch_npu + 安装 sort_with_index 自定义算子包。
#
# 构建：
#   mkdir -p build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
#   make
#   python3 ../test.py --lib ./libtorch_adapter.so   # 仅 torch_npu+NPU 可用时实跑，否则 mock 闭环
cmake_minimum_required(VERSION 3.10)
project(aclnn_sort_with_index_torch_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Find PyTorch
# ============================================================================
find_package(Torch REQUIRED)
message(STATUS "PyTorch version: ${TORCH_VERSION}")
message(STATUS "PyTorch path: ${TORCH_INSTALL_PREFIX}")

# ============================================================================
# Find CANN
# ============================================================================
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set! source set_env.sh first.")
endif()
if(NOT EXISTS "$ENV{ASCEND_HOME_PATH}")
    message(FATAL_ERROR "ASCEND_HOME_PATH directory does not exist!")
endif()

set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
set(ASCENDCL_INCLUDE_DIRS ${CANN_ROOT}/include)
set(ASCENDCL_LIBRARIES ${CANN_ROOT}/lib64/libascendcl.so)
set(CANN_INCLUDE_DIRS ${CANN_ROOT}/include)
set(CANN_LIBRARIES ${CANN_ROOT}/lib64/libopapi.so)
message(STATUS "CANN_ROOT: ${CANN_ROOT}")

# ============================================================================
# Find torch_npu (NPU 后端)
# ============================================================================
execute_process(
    COMMAND python3 -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_NPU_ROOT)
    set(TORCH_NPU_INCLUDE_DIR "${TORCH_NPU_ROOT}/include")
endif()

if(TORCH_NPU_INCLUDE_DIR)
    message(STATUS "torch_npu include dir: ${TORCH_NPU_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "torch_npu include directory not found.")
endif()

find_library(TORCH_NPU_LIBRARY torch_npu
    PATHS ${TORCH_NPU_ROOT}/lib
    PATH_SUFFIXES lib
)
if(TORCH_NPU_LIBRARY)
    message(STATUS "torch_npu library: ${TORCH_NPU_LIBRARY}")
else()
    message(FATAL_ERROR "torch_npu library not found.")
endif()

# ============================================================================
# Find custom operator package (vendor "custom_math"; ships aclnn_sort_with_index.h + libcust_opapi.so)
#   优先 ASCEND_CUSTOM_OPP_PATH（当前激活包的唯一真值），再回退常见安装路径，与 C++ ST CMakeLists 对齐。
# ============================================================================
set(CUSTOM_OPP_VENDOR_ROOTS "")
if(DEFINED ENV{ASCEND_CUSTOM_OPP_PATH} AND NOT "$ENV{ASCEND_CUSTOM_OPP_PATH}" STREQUAL "")
    string(REPLACE ":" ";" CUSTOM_OPP_VENDOR_ROOTS "$ENV{ASCEND_CUSTOM_OPP_PATH}")
    list(REMOVE_ITEM CUSTOM_OPP_VENDOR_ROOTS "")
endif()

set(CUSTOM_OP_INCLUDE_CANDIDATES "")
set(CUSTOM_OP_LIB_CANDIDATES "")
foreach(vroot ${CUSTOM_OPP_VENDOR_ROOTS})
    list(APPEND CUSTOM_OP_INCLUDE_CANDIDATES "${vroot}/op_api/include")
    list(APPEND CUSTOM_OP_LIB_CANDIDATES "${vroot}/op_api/lib")
endforeach()
list(APPEND CUSTOM_OP_INCLUDE_CANDIDATES
    $ENV{HOME}/sort_with_index_custom_opp/vendors/custom_math/op_api/include
    ${CANN_ROOT}/opp/vendors/custom_math/op_api/include
    ${CANN_ROOT}/opp/vendors/sort_with_index_custom/op_api/include
    ${CANN_ROOT}/opp/vendors/customize/op_api/include
    /usr/local/Ascend/opp/vendors/custom_math/op_api/include
    $ENV{HOME}/Ascend/opp/vendors/custom_math/op_api/include)
list(APPEND CUSTOM_OP_LIB_CANDIDATES
    $ENV{HOME}/sort_with_index_custom_opp/vendors/custom_math/op_api/lib
    ${CANN_ROOT}/opp/vendors/custom_math/op_api/lib
    ${CANN_ROOT}/opp/vendors/sort_with_index_custom/op_api/lib
    ${CANN_ROOT}/opp/vendors/customize/op_api/lib
    /usr/local/Ascend/opp/vendors/custom_math/op_api/lib
    $ENV{HOME}/Ascend/opp/vendors/custom_math/op_api/lib)

find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_sort_with_index.h
    PATHS ${CUSTOM_OP_INCLUDE_CANDIDATES}
    NO_DEFAULT_PATH
)
if(CUSTOM_OP_INCLUDE_DIR)
    message(STATUS "Found custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Custom operator include directory not found (aclnn_sort_with_index.h). "
                        "Install the sort_with_index operator (op_api) first and source its set_env.bash.")
endif()

find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS ${CUSTOM_OP_LIB_CANDIDATES}
    NO_DEFAULT_PATH
)
if(NOT CUSTOM_OP_LIBRARY)
    message(FATAL_ERROR "cust_opapi library (libcust_opapi.so) not found. Install the operator (op_api) first.")
endif()
message(STATUS "Found custom operator library: ${CUSTOM_OP_LIBRARY}")

# ============================================================================
# Build shared library
# ============================================================================
add_library(torch_adapter SHARED torch_adapter.cpp)

target_include_directories(torch_adapter PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${ASCENDCL_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${CUSTOM_OP_INCLUDE_DIR}
)

target_link_libraries(torch_adapter PRIVATE
    ${TORCH_LIBRARIES}
    ${ASCENDCL_LIBRARIES}
    ${CANN_LIBRARIES}
    ${CANN_ROOT}/lib64/libnnopbase.so
    ${CUSTOM_OP_LIBRARY}
    ${TORCH_NPU_LIBRARY}
)

# Set rpath
get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
get_filename_component(TORCH_NPU_LIB_DIR ${TORCH_NPU_LIBRARY} DIRECTORY)
target_link_options(torch_adapter PRIVATE
    "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
    "-Wl,-rpath,${CANN_ROOT}/lib64"
    "-Wl,-rpath,${TORCH_NPU_LIB_DIR}"
)

install(TARGETS torch_adapter DESTINATION lib)

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Build config (SortWithIndex PyTorch ST adapter):")
message(STATUS "  Output: libtorch_adapter.so")
message(STATUS "  Test:   python3 ../test.py --lib ./libtorch_adapter.so")
message(STATUS "  torch_npu 不可用时 test.py 自动回退 mock 闭环")
message(STATUS "========================================")
message(STATUS "")
