/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace aicpu;

class TEST_CEIL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "Ceil", "Ceil")                   \
        .Input({"x", data_types[0], shapes[0], datas[0]})            \
        .Output({"y", data_types[1], shapes[1], datas[1]})

#define ADD_CASE(base_type, aicpu_type)                                \
    TEST_F(TEST_CEIL_UT, TestCeil_##aicpu_type)                        \
    {                                                                  \
        std::vector<DataType> data_types = {aicpu_type, aicpu_type};   \
        std::vector<std::vector<int64_t>> shapes = {{24}, {24}};       \
        base_type input[24];                                           \
        base_type output[24] = {static_cast<base_type>(0)};            \
        base_type expected[24];                                        \
        for (size_t i = 0; i < 24; ++i) {                              \
            input[i] = static_cast<base_type>((i - 12.0) / 3.2);       \
            expected[i] = static_cast<base_type>(std::ceil(input[i])); \
        }                                                              \
        std::vector<void*> datas = {input, output};                    \
        CREATE_NODEDEF(shapes, data_types, datas);                     \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                  \
        EXPECT_TRUE(CompareResult(output, expected, 24));              \
    }

ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)

ADD_CASE(double, DT_DOUBLE)
