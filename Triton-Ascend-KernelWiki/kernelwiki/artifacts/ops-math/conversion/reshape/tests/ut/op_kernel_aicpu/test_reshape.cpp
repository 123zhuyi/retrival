/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

class TEST_RESHAPE_AICPU_UT : public testing::Test {};

template <typename T>
void BuildAndRunReshapeNode(
    DataType x_type, const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape, const void* x_data,
    const void* y_data, uint32_t expect_status)
{
    int64_t shape_data[8] = {0};
    const size_t shape_rank = y_shape.size();
    for (size_t idx = 0U; idx < shape_rank; ++idx) {
        shape_data[idx] = y_shape[idx];
    }

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "Reshape", "Reshape")
        .Input({"x", x_type, x_shape, const_cast<void*>(x_data)})
        .Input({"shape", DT_INT64, {static_cast<int64_t>(shape_rank)}, shape_data})
        .Output({"y", x_type, y_shape, const_cast<void*>(y_data)});
    RUN_KERNEL(node_def, HOST, expect_status);
}

template <typename T>
void RunSuccessCase(DataType x_type)
{
    std::vector<int64_t> x_shape = {2, 11};
    std::vector<int64_t> y_shape = {2, 11};
    std::vector<T> input(22);
    SetRandomValue<T>(input.data(), input.size());
    std::vector<T> output(22);

    BuildAndRunReshapeNode<T>(x_type, x_shape, y_shape, input.data(), output.data(), KERNEL_STATUS_OK);
    EXPECT_EQ(output, input);
}

TEST_F(TEST_RESHAPE_AICPU_UT, output_size_not_match)
{
    std::vector<int64_t> x_shape = {2, 2, 4};
    std::vector<int64_t> y_shape = {2, 2, 3};
    std::vector<int32_t> input(16);
    std::vector<int32_t> output(12);

    BuildAndRunReshapeNode<int32_t>(
        DT_INT32, x_shape, y_shape, input.data(), output.data(), KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RESHAPE_AICPU_UT, input_null_failed)
{
    std::vector<int64_t> x_shape = {2, 11};
    std::vector<int64_t> y_shape = {2, 11};
    std::vector<int32_t> output(22);

    BuildAndRunReshapeNode<int32_t>(DT_INT32, x_shape, y_shape, nullptr, output.data(), KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RESHAPE_AICPU_UT, output_null_failed)
{
    std::vector<int64_t> x_shape = {2, 11};
    std::vector<int64_t> y_shape = {2, 11};
    std::vector<int32_t> input(22);

    BuildAndRunReshapeNode<int32_t>(DT_INT32, x_shape, y_shape, input.data(), nullptr, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RESHAPE_AICPU_UT, float16_success)
{
    RunSuccessCase<Eigen::half>(DT_FLOAT16);
}
TEST_F(TEST_RESHAPE_AICPU_UT, float32_success)
{
    RunSuccessCase<float>(DT_FLOAT);
}
TEST_F(TEST_RESHAPE_AICPU_UT, int8_success)
{
    RunSuccessCase<int8_t>(DT_INT8);
}
TEST_F(TEST_RESHAPE_AICPU_UT, int16_success)
{
    RunSuccessCase<int16_t>(DT_INT16);
}
TEST_F(TEST_RESHAPE_AICPU_UT, int32_success)
{
    RunSuccessCase<int32_t>(DT_INT32);
}
TEST_F(TEST_RESHAPE_AICPU_UT, int64_success)
{
    RunSuccessCase<int64_t>(DT_INT64);
}
TEST_F(TEST_RESHAPE_AICPU_UT, uint8_success)
{
    RunSuccessCase<uint8_t>(DT_UINT8);
}
TEST_F(TEST_RESHAPE_AICPU_UT, uint16_success)
{
    RunSuccessCase<uint16_t>(DT_UINT16);
}
TEST_F(TEST_RESHAPE_AICPU_UT, uint32_success)
{
    RunSuccessCase<uint32_t>(DT_UINT32);
}
TEST_F(TEST_RESHAPE_AICPU_UT, uint64_success)
{
    RunSuccessCase<uint64_t>(DT_UINT64);
}