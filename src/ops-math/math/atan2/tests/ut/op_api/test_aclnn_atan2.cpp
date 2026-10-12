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

#include "../../../op_api/aclnn_atan2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_atan2_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Atan2 Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "Atan2 Test TearDown" << std::endl; }
};

// FLOAT
TEST_F(l2_atan2_test, case_float)
{
    auto self = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// FLOAT16
TEST_F(l2_atan2_test, case_float16)
{
    auto self = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// BF16
TEST_F(l2_atan2_test, ascend910B2_case_bf16)
{
    auto self = TensorDesc({5, 6, 7}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 7}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 7}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// DOUBLE
TEST_F(l2_atan2_test, case_double)
{
    auto self = TensorDesc({5, 6, 7}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 7}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 7}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT8 -> FLOAT
TEST_F(l2_atan2_test, case_int8)
{
    auto self = TensorDesc({5, 6, 7}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 7}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT16 -> FLOAT
TEST_F(l2_atan2_test, case_int16)
{
    auto self = TensorDesc({5, 6, 7}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 7}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT32 -> FLOAT
TEST_F(l2_atan2_test, case_int32)
{
    auto self = TensorDesc({5, 6, 7}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 7}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT64 -> FLOAT
TEST_F(l2_atan2_test, case_int64)
{
    auto self = TensorDesc({5, 6, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// UINT8 -> FLOAT
TEST_F(l2_atan2_test, case_uint8)
{
    auto self = TensorDesc({5, 6, 7}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 7}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// BOOL -> FLOAT
TEST_F(l2_atan2_test, case_bool)
{
    auto self = TensorDesc({5, 6, 7}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 7}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 1D shape
TEST_F(l2_atan2_test, case_dim_1)
{
    auto self = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 2D shape
TEST_F(l2_atan2_test, case_dim_2)
{
    auto self = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 4D shape
TEST_F(l2_atan2_test, case_dim_4)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto other = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 8D shape (max supported)
TEST_F(l2_atan2_test, case_dim_8)
{
    auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 9D shape (exceeds max, should fail)
TEST_F(l2_atan2_test, case_dim_9_error)
{
    auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Empty tensor
TEST_F(l2_atan2_test, case_empty)
{
    auto self = TensorDesc({5, 6, 0, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 0, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 0, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Broadcast
TEST_F(l2_atan2_test, case_broadcast)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Non-contiguous tensor
TEST_F(l2_atan2_test, case_non_contiguous)
{
    auto self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto other = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto out = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull
TEST_F(l2_atan2_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut_1 = OP_API_UT(aclnnAtan2, INPUT((aclTensor*)nullptr, tensor_desc), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnAtan2, INPUT(tensor_desc, (aclTensor*)nullptr), OUTPUT(tensor_desc));
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_3 = OP_API_UT(aclnnAtan2, INPUT(tensor_desc, tensor_desc), OUTPUT((aclTensor*)nullptr));
    aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// Mismatched shapes (not broadcastable)
TEST_F(l2_atan2_test, case_mismatched_shape)
{
    auto self = TensorDesc({10, 5, 2, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto other = TensorDesc({10, 5, 5, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto out = TensorDesc({10, 5, 2, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Mismatched output shape
TEST_F(l2_atan2_test, case_error_output_shape)
{
    auto self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Complex64 input (unsupported)
TEST_F(l2_atan2_test, case_complex64_error)
{
    auto self = TensorDesc({5, 6, 7}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({5, 6, 7}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({5, 6, 7}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAtan2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
