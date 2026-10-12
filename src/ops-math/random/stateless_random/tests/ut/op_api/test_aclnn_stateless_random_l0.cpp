/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_stateless_random_l0.cpp
 * \brief l0op::StatelessRandom L0 UT
 *
 * 该算子为 aclnn_exclude（无 L2 入口），被测对象是 op_api/stateless_random.cpp 的四个 L0 重载：
 *   A) 标量接口（带from/to）：StatelessRandom(self, int64_t seed, int64_t offset, from, to, executor)
 *   B) Tensor接口（带from/to）：StatelessRandom(self, seedTensor, offsetTensor, from, to, executor)
 *   C) 标量接口（无from/to）：StatelessRandomWithoutFromTo(self, int64_t seed, int64_t offset, executor)
 *   D) Tensor接口（无from/to）：StatelessRandomWithoutFromTo(self, seedTensor, offsetTensor, executor)
 *
 * 文件名必须以 test_aclnn_ 开头 —— cmake/ut.cmake 中 OP_API_MODULE_NAME 的 glob 是
 * `${MODULE_DIR}/test_aclnn_*.cpp`，命名不符则不会被编入 math_op_api_ut。
 *
 * 覆盖目标（四个重载各自内部有多条 dtype 分支）：
 *   - 支持的输出类型：FLOAT, FLOAT16, BF16, INT32, INT64, INT16, INT8, UINT8, BOOL
 *   - 不支持的类型回退到 INT32
 *
 * 附加覆盖：
 *   - 输出 dtype 与输入一致（非支持类型回退到 INT32）
 *   - 输出 shape 与输入一致
 *   - 多维输入（ToShapeVector / AllocIntArray 多维路径）
 *   - from/to 非默认区间（负值区间）
 *   - 非零 seed/offset
 */

#include <gtest/gtest.h>
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "random/stateless_random/op_api/stateless_random.h"

using namespace op;
using namespace std;

namespace {
constexpr int64_t DATA_SIZE = 256;
constexpr int64_t DEFAULT_SEED = 0;
constexpr int64_t DEFAULT_OFFSET = 0;
constexpr int64_t DEFAULT_FROM = 0;
constexpr int64_t DEFAULT_TO = 100;
} // namespace

class StatelessRandomL0Test : public ::testing::Test {
public:
    StatelessRandomL0Test() : exe(nullptr) {}

    aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dtype)
    {
        return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                               data);
    }

    void SetUp() override
    {
        SetPlatformNpuArch(NpuArch::DAV_3510);
        auto executor = &exe;
        auto unique_executor = CREATE_EXECUTOR();
        unique_executor.ReleaseTo(executor);
    }

    void TearDown() override { delete exe; }

public:
    aclOpExecutor* exe;
    float data[DATA_SIZE] = {1.0f};
};

// ===== 重载 A：标量 seed/offset（带 from/to）=====

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_float32)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_float16)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT16);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT16);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_bf16)
{
    auto self = CreateAclTensor({256}, ACL_BF16);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_BF16);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_int32)
{
    auto self = CreateAclTensor({256}, ACL_INT32);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_INT32);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_int64)
{
    auto self = CreateAclTensor({256}, ACL_INT64);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_INT64);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_int16)
{
    auto self = CreateAclTensor({256}, ACL_INT16);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_INT16);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_int8)
{
    auto self = CreateAclTensor({256}, ACL_INT8);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_INT8);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_uint8)
{
    auto self = CreateAclTensor({256}, ACL_UINT8);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_UINT8);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_bool)
{
    auto self = CreateAclTensor({256}, ACL_BOOL);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_BOOL);
}

// 不支持的类型回退到 INT32
TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_unsupported_dtype_fallback)
{
    auto self = CreateAclTensor({256}, ACL_DOUBLE);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_INT32);
}

// ===== 重载 B：Tensor 形式 seed/offset（带 from/to）=====

TEST_F(StatelessRandomL0Test, stateless_random_l0_tensor_float32)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessRandom(self, seedTensor, offsetTensor, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_tensor_float16)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT16);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessRandom(self, seedTensor, offsetTensor, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT16);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_tensor_bf16)
{
    auto self = CreateAclTensor({256}, ACL_BF16);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessRandom(self, seedTensor, offsetTensor, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_BF16);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_tensor_int32)
{
    auto self = CreateAclTensor({256}, ACL_INT32);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessRandom(self, seedTensor, offsetTensor, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_INT32);
}

// ===== 重载 C：标量 seed/offset（无 from/to）=====

TEST_F(StatelessRandomL0Test, stateless_random_l0_without_fromto_scalar_float32)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT);
    auto out = l0op::StatelessRandomWithoutFromTo(self, DEFAULT_SEED, DEFAULT_OFFSET, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_without_fromto_scalar_float16)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT16);
    auto out = l0op::StatelessRandomWithoutFromTo(self, DEFAULT_SEED, DEFAULT_OFFSET, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT16);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_without_fromto_scalar_bf16)
{
    auto self = CreateAclTensor({256}, ACL_BF16);
    auto out = l0op::StatelessRandomWithoutFromTo(self, DEFAULT_SEED, DEFAULT_OFFSET, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_BF16);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_without_fromto_scalar_int32)
{
    auto self = CreateAclTensor({256}, ACL_INT32);
    auto out = l0op::StatelessRandomWithoutFromTo(self, DEFAULT_SEED, DEFAULT_OFFSET, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_INT32);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_without_fromto_scalar_bool)
{
    auto self = CreateAclTensor({256}, ACL_BOOL);
    auto out = l0op::StatelessRandomWithoutFromTo(self, DEFAULT_SEED, DEFAULT_OFFSET, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_BOOL);
}

// ===== 重载 D：Tensor 形式 seed/offset（无 from/to）=====

TEST_F(StatelessRandomL0Test, stateless_random_l0_without_fromto_tensor_float32)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessRandomWithoutFromTo(self, seedTensor, offsetTensor, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_without_fromto_tensor_float16)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT16);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessRandomWithoutFromTo(self, seedTensor, offsetTensor, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT16);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_without_fromto_tensor_int64)
{
    auto self = CreateAclTensor({256}, ACL_INT64);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessRandomWithoutFromTo(self, seedTensor, offsetTensor, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_INT64);
}

// ===== 形状 / 参数覆盖 =====

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_multi_dim)
{
    auto self = CreateAclTensor({2, 8, 16}, ACL_FLOAT);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape().GetDimNum(), 3U);
    EXPECT_EQ(out->GetViewShape().GetDim(0), 2);
    EXPECT_EQ(out->GetViewShape().GetDim(1), 8);
    EXPECT_EQ(out->GetViewShape().GetDim(2), 16);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_shape_preserved)
{
    auto self = CreateAclTensor({16, 16}, ACL_FLOAT);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape(), self->GetViewShape());
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_negative_range)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, -50, 50, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_nonzero_seed)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT);
    auto out = l0op::StatelessRandom(self, 12345, 678, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_scalar_dtype_preserved)
{
    auto self = CreateAclTensor({4, 64}, ACL_FLOAT16);
    auto out = l0op::StatelessRandom(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT16);
    EXPECT_EQ(out->GetViewShape(), self->GetViewShape());
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_without_fromto_multi_dim)
{
    auto self = CreateAclTensor({4, 8, 16}, ACL_INT32);
    auto out = l0op::StatelessRandomWithoutFromTo(self, DEFAULT_SEED, DEFAULT_OFFSET, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape().GetDimNum(), 3U);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_INT32);
}

TEST_F(StatelessRandomL0Test, stateless_random_l0_tensor_multi_dim)
{
    auto self = CreateAclTensor({8, 32}, ACL_BF16);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessRandom(self, seedTensor, offsetTensor, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape().GetDimNum(), 2U);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_BF16);
}
