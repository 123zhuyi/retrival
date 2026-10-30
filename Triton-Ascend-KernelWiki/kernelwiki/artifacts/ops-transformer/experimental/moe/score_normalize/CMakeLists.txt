# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
message(STATUS "BUILD_TORCH_OPS ON in score_normalize")
# SCORE_NORMALIZE operation sources
file(GLOB SCORE_NORMALIZE_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(SCORE_NORMALIZE_SOURCES ${SCORE_NORMALIZE_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${SCORE_NORMALIZE_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(score_normalize_objects OBJECT ${SCORE_NORMALIZE_SOURCES})

target_compile_options(score_normalize_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(score_normalize_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()