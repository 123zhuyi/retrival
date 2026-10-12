/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_zero.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2ZeroTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2ZeroTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2ZeroTest TearDown" << std::endl;
    }
};

// self的数据类型不在支持范围内
TEST_F(l2ZeroTest, l2_zero_test_unsupported_type)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2ZeroTest, l2_zero_test_null)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicore, int8
TEST_F(l2ZeroTest, l2_zero_test_int8)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径, aicore, int32
TEST_F(l2ZeroTest, l2_zero_test_int32)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径, aicore, int64
TEST_F(l2ZeroTest, l2_zero_test_int64)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径, aicore, uint8
TEST_F(l2ZeroTest, l2_zero_test_uint8)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径， aicore, float16
TEST_F(l2ZeroTest, l2_zero_test_float16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径， aicore, float32
TEST_F(l2ZeroTest, l2_zero_test_float32)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径， aicore, bool
TEST_F(l2ZeroTest, l2_zero_test_bool)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径, aicpu, double
TEST_F(l2ZeroTest, l2_zero_test_double)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径, aicpu, INT16
TEST_F(l2ZeroTest, l2_zero_test_int16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径, aicpu, UINT16, 原pytorch不支持uint16, 所以此处未设置精度测试
TEST_F(l2ZeroTest, l2_zero_test_uint16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicpu, COMPLEX64
TEST_F(l2ZeroTest, l2_zero_test_complex64)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径, aicpu, COMPLEX128
TEST_F(l2ZeroTest, l2_zero_test_complex128)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径, 测试 ACL_BF16 类型（仅在910B上支持aicore，其他型号走aicpu）
TEST_F(l2ZeroTest, l2_zero_test_bf16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, 触发 CheckDim 函数（维度为8时）
TEST_F(l2ZeroTest, l2_zero_test_max_dim)
{
    auto selfDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试多维度 aicpu 类型
TEST_F(l2ZeroTest, l2_zero_test_multi_dim_double)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}