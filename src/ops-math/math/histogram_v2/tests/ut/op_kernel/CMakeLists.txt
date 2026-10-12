# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 需要将Tiling依赖的文件添加到CMakeLists.txt中
    # set(elewise_common_tiling_files
    #         ${CANN_ROOT}/ops/built-in/op_tiling/runtime/elewise_tiling.cc
    #         )
    # 算子自己的tiling文件路径
    set(histogram_v2_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/histogram_v2_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/histogram_v2_simt_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/histogram_v2_op_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/histogram_v2_base_tiling.cpp
        )
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本，多个以分号分隔，例如："ascend950;ascend910b"
    # param3：自定义编译选项，一般填写测试的一种典型数据类型组合，不需要则传入空字符串，例如："-DDTYPE_X=float"，多个使用空格分隔，例如："-DDTYPE_X=float -DDTYPE_Y=float"
    # param4：该算子依赖的所有tiling源码文件
    AddOpTestCase(histogram_v2 "ascend910b" "-DDTYPE_X=float" "${histogram_v2_tiling_files}")
endif()

if (UT_TEST_ALL OR OP_HOST_UT)
    # HistogramV2 has two tiling templates selected at runtime by SoC (IsRegbaseSocVersion):
    #   RegBase (ascend950)                  -> arch35 (HistogramV2SimtTiling)
    #   non-RegBase (ascend310p/910_93/910b) -> arch22 (HistogramV2MembaseTiling)
    # The single-value TILING_DIR of add_all_modules_sources cannot express both arch dirs, so
    # register them explicitly here. Each case fakes the SoC it needs via
    # TilingContextPara::socVersion_, so a single build (any BUILD_SOC_VERSION) exercises
    # BOTH templates and covers both tiling sources.
    get_filename_component(histogram_v2_op_host_ut_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_host REALPATH)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE
                           DIR ${histogram_v2_op_host_ut_dir} TILING_DIR arch22)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE
                           DIR ${histogram_v2_op_host_ut_dir} TILING_DIR arch35)
    # histogram_v2_tiling_ut_util.h fetches/runs tiling via the op-impl space registry (to fake
    # the SoC per case). That header lives under ${ASCEND_DIR}/pkg_inc, which the shared tiling
    # common obj already includes but the cases obj does not — add it here just for this op.
    if (TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
        target_include_directories(${OP_TILING_MODULE_NAME}_cases_obj
                                   PRIVATE ${ASCEND_DIR}/pkg_inc)
    endif()
endif()
