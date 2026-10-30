# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    # HistogramV2 has two tiling templates selected at runtime by SoC (IsRegbaseSocVersion):
    #   RegBase (ascend950)               -> arch35 (HistogramV2SimtTiling)
    #   non-RegBase (ascend310p/910_93/910b) -> arch32 (HistogramV2MembaseTiling)
    # Both arch UTs are compiled unconditionally: each test case fakes the SoC it needs via
    # TilingContextPara::socVersion_ (arch35 -> "ascend950", arch32 -> "ascend910b"), so a single
    # build (any BUILD_SOC_VERSION) exercises BOTH templates and covers both tiling sources.
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} TILING_DIR arch35)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} TILING_DIR arch32)
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    # histogram_v2_tiling_ut_util.h fetches/runs tiling via the op-impl space registry (to fake the
    # SoC per case). That header lives under ${ASCEND_DIR}/pkg_inc, which the shared tiling common
    # obj already includes but the cases obj does not — add it here just for this op's cases obj.
    # The registry symbols themselves come from libs the UT already links.
    if(TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
        target_include_directories(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE ${ASCEND_DIR}/pkg_inc)
    endif()
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
