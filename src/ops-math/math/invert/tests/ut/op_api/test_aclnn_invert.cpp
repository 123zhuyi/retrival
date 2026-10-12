/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "opdev/make_op_executor.h"
#include "../../../op_api/invert.h"

using namespace op;
using namespace std;

const int64_t DATA_SIZE = 1024 * 1024;

class InvertTest : public ::testing::Test {
public:
    InvertTest() : exe(nullptr) {}

    aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dtype)
    {
        return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                               data);
    }

    void Clear() {}

    void SetUp() override
    {
        auto executor = &exe;
        auto unique_executor = CREATE_EXECUTOR();
        unique_executor.ReleaseTo(executor);
    }

    void TearDown() override { delete exe; }

public:
    aclOpExecutor* exe;
    int64_t data[DATA_SIZE] = {0};
};

// int16类型 (AICore支持)
TEST_F(InvertTest, Invert_int16)
{
    auto self = CreateAclTensor({2, 3}, ACL_INT16);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({2, 3});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// uint16类型 (AICore支持)
TEST_F(InvertTest, Invert_uint16)
{
    auto self = CreateAclTensor({2, 3}, ACL_UINT16);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({2, 3});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// int8类型
TEST_F(InvertTest, Invert_int8)
{
    auto self = CreateAclTensor({4, 5}, ACL_INT8);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({4, 5});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// int32类型
TEST_F(InvertTest, Invert_int32)
{
    auto self = CreateAclTensor({4, 5}, ACL_INT32);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({4, 5});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// int64类型
TEST_F(InvertTest, Invert_int64)
{
    auto self = CreateAclTensor({4, 5}, ACL_INT64);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({4, 5});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// uint8类型
TEST_F(InvertTest, Invert_uint8)
{
    auto self = CreateAclTensor({4, 5}, ACL_UINT8);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({4, 5});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// uint32类型
TEST_F(InvertTest, Invert_uint32)
{
    auto self = CreateAclTensor({4, 5}, ACL_UINT32);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({4, 5});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// uint64类型
TEST_F(InvertTest, Invert_uint64)
{
    auto self = CreateAclTensor({4, 5}, ACL_UINT64);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({4, 5});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// 不同shape
TEST_F(InvertTest, Invert_different_shape)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_INT16);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({3, 4, 5});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// 1维tensor
TEST_F(InvertTest, Invert_1d)
{
    auto self = CreateAclTensor({5}, ACL_INT16);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({5});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// 空tensor
TEST_F(InvertTest, Invert_empty)
{
    auto self = CreateAclTensor({0, 3}, ACL_INT16);
    auto result = l0op::Invert(self, exe);
    ASSERT_NE(result, nullptr);

    op::ShapeVector expectShape({0, 3});
    EXPECT_EQ(op::ToShapeVector(result->GetViewShape()), expectShape);
}

// float类型 (不支持, 验证不崩溃)
TEST_F(InvertTest, Invert_float)
{
    auto self = CreateAclTensor({2, 3}, ACL_FLOAT);
    auto result = l0op::Invert(self, exe);
    // float不是支持的类型, 结果可能为nullptr或仍然返回
    // l0op::Invert不做dtype检查, 它直接分配tensor并走AiCpu
    // 验证不崩溃即可
    (void)result;
}
