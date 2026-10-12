/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_stateless_uniform_l0.cpp
 * \brief l0op::StatelessUniform L0 UT
 *
 * 该算子为 aclnn_exclude（无 L2 入口），被测对象是 op_api/stateless_uniform.cpp 的两个 L0 重载：
 *   A) 标量接口：StatelessUniform(self, uint64_t seed, uint64_t offset, from, to, executor)
 *   B) Tensor接口：StatelessUniform(self, seedTensor, offsetTensor, from, to, executor)
 *
 * 文件名必须以 test_aclnn_ 开头 —— cmake/ut.cmake 中 OP_API_MODULE_NAME 的 glob 是
 * `${MODULE_DIR}/test_aclnn_*.cpp`，命名不符则不会被编入 math_op_api_ut。
 *
 * 覆盖目标（两个重载各自内部有 fp16 / bf16 / else 三条 AllocTensor 分支，共 6 条路径）：
 *   A-float32  : scalar_float32      A-float16 : scalar_float16      A-bfloat16 : scalar_bf16
 *   B-float32  : tensor_float32      B-float16 : tensor_float16      B-bfloat16 : tensor_bf16
 *
 * 附加覆盖：
 *   输出 dtype 与输入一致（非 float32 不被静默提升）: scalar_dtype_preserved
 *   输出 shape 与输入一致                            : scalar_shape_preserved
 *   多维输入（ToShapeVector / AllocIntArray 多维路径）: scalar_multi_dim
 *   from/to 非默认区间（负值区间）                    : scalar_negative_range
 *   非零 seed/offset                                 : scalar_nonzero_seed
 */

#include <gtest/gtest.h>
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "random/stateless_uniform/op_api/stateless_uniform.h"

using namespace op;
using namespace std;

namespace {
constexpr int64_t DATA_SIZE = 256;
constexpr uint64_t DEFAULT_SEED = 0;
constexpr uint64_t DEFAULT_OFFSET = 0;
constexpr double DEFAULT_FROM = 0.0;
constexpr double DEFAULT_TO = 1.0;
} // namespace

class StatelessUniformL0Test : public ::testing::Test {
public:
    StatelessUniformL0Test() : exe(nullptr) {}

    aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dtype)
    {
        return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                               data);
    }

    void SetUp() override
    {
        // 该算子仅注册 ascend950，统一在 950 平台下构图
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

// ===== 重载 A：标量 seed/offset =====

// case 1: float32 → else 分支，输出 DT_FLOAT
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_scalar_float32)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT);
    auto out = l0op::StatelessUniform(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
}

// case 2: float16 → DT_FLOAT16 分支
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_scalar_float16)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT16);
    auto out = l0op::StatelessUniform(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT16);
}

// case 3: bfloat16 → DT_BF16 分支
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_scalar_bf16)
{
    auto self = CreateAclTensor({256}, ACL_BF16);
    auto out = l0op::StatelessUniform(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_BF16);
}

// ===== 重载 B：Tensor 形式 seed/offset（图捕获模式）=====

// case 4: float32
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_tensor_float32)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessUniform(self, seedTensor, offsetTensor, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
}

// case 5: float16
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_tensor_float16)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT16);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessUniform(self, seedTensor, offsetTensor, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT16);
}

// case 6: bfloat16
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_tensor_bf16)
{
    auto self = CreateAclTensor({256}, ACL_BF16);
    auto seedTensor = CreateAclTensor({1}, ACL_INT64);
    auto offsetTensor = CreateAclTensor({1}, ACL_INT64);
    auto out = l0op::StatelessUniform(self, seedTensor, offsetTensor, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_BF16);
}

// ===== 形状 / 参数覆盖 =====

// case 7: 多维输入，ToShapeVector + AllocIntArray 走多维路径，输出 shape 应与输入一致
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_scalar_multi_dim)
{
    auto self = CreateAclTensor({2, 8, 16}, ACL_FLOAT);
    auto out = l0op::StatelessUniform(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape().GetDimNum(), 3U);
    EXPECT_EQ(out->GetViewShape().GetDim(0), 2);
    EXPECT_EQ(out->GetViewShape().GetDim(1), 8);
    EXPECT_EQ(out->GetViewShape().GetDim(2), 16);
}

// case 8: 输出 shape 与输入严格一致（2D）
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_scalar_shape_preserved)
{
    auto self = CreateAclTensor({16, 16}, ACL_FLOAT);
    auto out = l0op::StatelessUniform(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape(), self->GetViewShape());
}

// case 9: 负值区间 from/to，确认 double 参数被正常接收
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_scalar_negative_range)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT);
    auto out = l0op::StatelessUniform(self, DEFAULT_SEED, DEFAULT_OFFSET, -3.5, 2.5, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
}

// case 10: 非零 seed/offset
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_scalar_nonzero_seed)
{
    auto self = CreateAclTensor({256}, ACL_FLOAT);
    auto out = l0op::StatelessUniform(self, 12345UL, 678UL, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
}

// case 11: 非 float32 输入不应被静默提升为 float32（fp16 入 → fp16 出）
TEST_F(StatelessUniformL0Test, stateless_uniform_l0_scalar_dtype_preserved)
{
    auto self = CreateAclTensor({4, 64}, ACL_FLOAT16);
    auto out = l0op::StatelessUniform(self, DEFAULT_SEED, DEFAULT_OFFSET, DEFAULT_FROM, DEFAULT_TO, exe);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT16);
    EXPECT_EQ(out->GetViewShape(), self->GetViewShape());
}
