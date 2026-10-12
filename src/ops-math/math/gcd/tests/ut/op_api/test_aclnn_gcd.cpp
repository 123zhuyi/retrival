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

#include "../../../op_api/aclnn_gcd.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_gcd_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Gcd Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "Gcd Test TearDown" << std::endl; }
};

// INT32 normal
TEST_F(l2_gcd_test, case_int32)
{
    auto self = TensorDesc({1, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT32 same shape
TEST_F(l2_gcd_test, case_int32_same_shape)
{
    auto self = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT16 (Ascend910B)
TEST_F(l2_gcd_test, ascend910B2_case_int16)
{
    auto self = TensorDesc({2, 3, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 3, 2}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT32 + INT16 mixed
TEST_F(l2_gcd_test, case_int32_int16_mixed)
{
    auto self = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT64 (Ascend910B)
TEST_F(l2_gcd_test, ascend910B2_case_int64)
{
    auto self = TensorDesc({1, 2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 2, 1, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT64 + INT16 mixed (Ascend910B)
TEST_F(l2_gcd_test, ascend910B2_case_int64_int16_mixed)
{
    auto self = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT64 + INT32 mixed (Ascend910B)
TEST_F(l2_gcd_test, ascend910B2_case_int64_int32_mixed)
{
    auto self = TensorDesc({2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 1D shape
TEST_F(l2_gcd_test, case_dim_1)
{
    auto self = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Empty tensor
TEST_F(l2_gcd_test, case_empty)
{
    auto self = TensorDesc({1, 0, 1, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto other = TensorDesc({1, 0, 1, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 0, 1, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Non-contiguous tensor
TEST_F(l2_gcd_test, case_non_contiguous)
{
    auto self = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto other = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto out = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull
TEST_F(l2_gcd_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND);

    auto ut_l = OP_API_UT(aclnnGcd, INPUT((aclTensor*)nullptr, tensor_desc), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_r = OP_API_UT(aclnnGcd, INPUT(tensor_desc, (aclTensor*)nullptr), OUTPUT(tensor_desc));
    aclRet = ut_r.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_o = OP_API_UT(aclnnGcd, INPUT(tensor_desc, tensor_desc), OUTPUT((aclTensor*)nullptr));
    aclRet = ut_o.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// Mismatched shapes (not broadcastable)
TEST_F(l2_gcd_test, case_mismatched_shape)
{
    auto self = TensorDesc({123, 11, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({123, 8, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({123, 11, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Error output shape
TEST_F(l2_gcd_test, case_error_output_shape)
{
    auto self = TensorDesc({123, 11, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({123, 11, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({123, 8, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Rank > 8
TEST_F(l2_gcd_test, case_rank_too_large)
{
    auto self = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({7, 8, 9, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({7, 8, 9, 10}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Unsupported dtype: FLOAT16
TEST_F(l2_gcd_test, case_unsupported_float16)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Unsupported dtype: FLOAT
TEST_F(l2_gcd_test, case_unsupported_float)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Unsupported dtype: COMPLEX64
TEST_F(l2_gcd_test, case_unsupported_complex64)
{
    auto self = TensorDesc({2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Error output dtype: BOOL
TEST_F(l2_gcd_test, case_error_output_dtype_bool)
{
    auto self = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({6, 2, 1, 2}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnGcd, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
