/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_eye.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_eye_test : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

// 浮点类型支持
TEST_F(l2_eye_test, eye_float)
{
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_eye_test, eye_float16)
{
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)5), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 整数类型支持
TEST_F(l2_eye_test, eye_int32)
{
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_eye_test, eye_int64)
{
    auto out_tensor_desc = TensorDesc({4, 6}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)4, (int64_t)6), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_eye_test, eye_int8)
{
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_eye_test, eye_int16)
{
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_eye_test, eye_uint8)
{
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_eye_test, eye_bool)
{
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 非方阵
TEST_F(l2_eye_test, eye_non_square)
{
    auto out_tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)5, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空指针校验
TEST_F(l2_eye_test, eye_out_nullptr)
{
    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持的数据类型
TEST_F(l2_eye_test, eye_unsupported_dtype)
{
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out维度不为2
TEST_F(l2_eye_test, eye_wrong_dim_1d)
{
    auto out_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_eye_test, eye_wrong_dim_3d)
{
    auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out shape与n,m不匹配
TEST_F(l2_eye_test, eye_shape_mismatch)
{
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负数n
TEST_F(l2_eye_test, eye_negative_n)
{
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)(-1), (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负数m
TEST_F(l2_eye_test, eye_negative_m)
{
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)(-1)), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor: n=0
TEST_F(l2_eye_test, eye_n_zero)
{
    auto out_tensor_desc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)0, (int64_t)3), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor: m=0
TEST_F(l2_eye_test, eye_m_zero)
{
    auto out_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)0), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 非连续tensor
TEST_F(l2_eye_test, eye_non_contiguous)
{
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 3}, 0, {5, 3});

    auto ut = OP_API_UT(aclnnEye, INPUT((int64_t)3, (int64_t)5), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
