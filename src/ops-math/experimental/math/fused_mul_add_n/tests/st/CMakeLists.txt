# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for fused_mul_add_n C++ ST test（全量用例）
#
# 本 C++ ST 以 Mock 模式（CPU golden 自测）验证测试框架与公式实现，覆盖全量用例：
# 全 dtype、多 shape、边界不变量 (x3=0/1/负)、极端输入 (NaN/Inf/全零/fp16 上界)、
# 整数回绕、广播等价、确定性。设备侧上板精度验收由 PyTorch（torch_npu）测试承担
# （见 tests/st/torch/），aclnn 两段式精度可另见 ATK 系统测试（见 tests/st/aclnnFusedMulAddN/）。
cmake_minimum_required(VERSION 3.10)
project(fused_mul_add_n_st_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Mock 模式：CPU golden 验证，无需 NPU（默认验证路径）
option(USE_MOCK "Use mock (CPU golden) for testing without real operator/NPU" OFF)

if(USE_MOCK)
    add_definitions(-DUSE_MOCK_ACLNN)
    message(STATUS "Building with MOCK (CPU golden) - full framework validation without NPU")
else()
    message(STATUS "Building in REAL mode - 本算子无 aclnn 接口，C++ ST 仅做 CPU golden 自测；")
    message(STATUS "             设备侧上板精度验收请使用 PyTorch (torch_npu) 独立任务。")
endif()

add_executable(test_aclnn_fused_mul_add_n test_aclnn_fused_mul_add_n.cpp)

# 安装目标
install(TARGETS test_aclnn_fused_mul_add_n DESTINATION bin)
