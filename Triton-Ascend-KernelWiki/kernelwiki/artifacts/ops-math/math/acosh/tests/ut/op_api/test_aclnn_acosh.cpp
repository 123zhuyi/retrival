/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "math/acosh/op_api/aclnn_acosh.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_acosh_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_acosh_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_acosh_test TearDown" << endl;
    }
};

// input nullptr
TEST_F(l2_acosh_test, acosh_self_nullptr)
{
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(nullptr), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_acosh_test, acosh_out_nullptr)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// empty tensor
TEST_F(l2_acosh_test, acosh_empty_tensor)
{
    auto selfDesc = TensorDesc({2, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype float16
TEST_F(l2_acosh_test, acosh_dtype_float16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype float
TEST_F(l2_acosh_test, acosh_dtype_float)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype int8
TEST_F(l2_acosh_test, acosh_dtype_int8)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype int16
TEST_F(l2_acosh_test, acosh_dtype_int16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype int32
TEST_F(l2_acosh_test, acosh_dtype_int32)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype int64
TEST_F(l2_acosh_test, acosh_dtype_int64)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype uint8
TEST_F(l2_acosh_test, acosh_dtype_uint8)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype bool
TEST_F(l2_acosh_test, acosh_dtype_bool)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype bfloat16
TEST_F(l2_acosh_test, ascend910B2_acosh_dtype_bfloat16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// different shape
TEST_F(l2_acosh_test, acosh_different_shape)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test precision
TEST_F(l2_acosh_test, acosh_precision)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 6);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// not contiguous
TEST_F(l2_acosh_test, acosh_not_contiguous)
{
    auto selfDesc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {3, 5}).ValueRange(1, 2);
    auto outDesc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {3, 5}).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// shape 1d
TEST_F(l2_acosh_test, acosh_shape_1d)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// shape 9d
TEST_F(l2_acosh_test, acosh_shape_9d)
{
    auto selfDesc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype float_int8
TEST_F(l2_acosh_test, acosh_dtype_float_int8)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype complex64_float
TEST_F(l2_acosh_test, acosh_dtype_complex64_float)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosh, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test acosh_ float
TEST_F(l2_acosh_test, acosh_inplace_float)
{
    auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto ut = OP_API_UT(aclnnInplaceAcosh, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// test acosh_ int8
TEST_F(l2_acosh_test, acosh_inplace_int8)
{
    auto selfDesc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 2);
    auto ut = OP_API_UT(aclnnInplaceAcosh, INPUT(selfDesc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
