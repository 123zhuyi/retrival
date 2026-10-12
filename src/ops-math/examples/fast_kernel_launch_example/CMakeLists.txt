# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
set(PKG_NAME AscendOps)

include(cmake/ascend.cmake)
find_package(ASC REQUIRED)
project(${PKG_NAME} VERSION 1.0.0 LANGUAGES C CXX ASC)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# set NPU architecture, default to dav-2201
set(SUPPORTED_NPU_ARCHS dav-2201 dav-3510)
if(DEFINED ENV{NPU_ARCH} AND NOT "$ENV{NPU_ARCH}" STREQUAL "")
    set(NPU_ARCH "$ENV{NPU_ARCH}" CACHE STRING "NPU architecture" FORCE)
elseif(NOT DEFINED NPU_ARCH OR "${NPU_ARCH}" STREQUAL "")
    set(NPU_ARCH "dav-2201" CACHE STRING "NPU architecture" FORCE)
endif()

if(NOT "${NPU_ARCH}" IN_LIST SUPPORTED_NPU_ARCHS)
    message(FATAL_ERROR "Unsupported NPU_ARCH: ${NPU_ARCH}. Supported values: ${SUPPORTED_NPU_ARCHS}")
endif()
message(STATUS "Using NPU_ARCH: ${NPU_ARCH}")

include(cmake/python.cmake)
include(cmake/torch.cmake)
include(cmake/torch_npu.cmake)
include(cmake/func.cmake)

# set EXTENSION_MODULE_NAME to `ascend_ops`
set(EXTENSION_MODULE_NAME "ascend_ops" CACHE STRING "Extension module name")

# include directories
set(INCLUDE_DIRECTORIES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
)

# link directories
set(LINK_DIRECTORIES
    ${TORCH_NPU_LIB_PATH}
)

# link libraries
set(LINK_LIBRARIES
    ${TORCH_LIBRARIES}
    torch_npu
)

# compile options
set(COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O3
    -fdiagnostics-color=always
    -w
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
)

add_subdirectory(csrc)
get_property(OPERATOR_TARGETS GLOBAL PROPERTY ASCEND_OPS_OPERATOR_TARGETS)
if(OPERATOR_TARGETS)
    message(STATUS "Operator targets: ${OPERATOR_TARGETS}")
else()
    message(WARNING "No operator targets found for NPU_ARCH=${NPU_ARCH}; building dummy extension module.")
endif()

set(OPERATOR_OBJECTS)
foreach(OPERATOR_TARGET IN LISTS OPERATOR_TARGETS)
    list(APPEND OPERATOR_OBJECTS $<TARGET_OBJECTS:${OPERATOR_TARGET}>)
endforeach()

# create shared library
set("EXTENSION_CPP" ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extension.cpp)
add_library(_C SHARED
    ${EXTENSION_CPP}
    ${OPERATOR_OBJECTS}
)
set_target_properties(_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".abi3.so"
    OUTPUT_NAME "_C"
)
target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x03080000)
target_compile_options(_C PRIVATE ${COMPILE_OPTIONS})
target_include_directories(_C PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(_C PRIVATE ${LINK_DIRECTORIES})
target_link_libraries(_C PRIVATE ${LINK_LIBRARIES})
if(OPERATOR_TARGETS)
    target_link_libraries(_C PRIVATE ${OPERATOR_TARGETS})
endif()

add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:_C>
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_MODULE_NAME}/$<TARGET_FILE_NAME:_C>
    COMMENT "Copying compiled extension $<TARGET_FILE_NAME:_C> to ${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_MODULE_NAME}/"
)
