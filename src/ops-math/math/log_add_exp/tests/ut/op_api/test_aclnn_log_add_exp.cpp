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

#include "../../../op_api/aclnn_logaddexp.h"
#include "../../../op_api/aclnn_logaddexp2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_log_add_exp_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LogAddExp Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "LogAddExp Test TearDown" << std::endl; }
};

// ===================== aclnnLogAddExp tests =====================

// FP32
TEST_F(l2_log_add_exp_test, logaddexp_case_fp32)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(self).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// FP16
TEST_F(l2_log_add_exp_test, logaddexp_case_fp16)
{
    auto self = TensorDesc({1, 1, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({1, 1, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(self).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// DOUBLE (unsupported on current platform)
TEST_F(l2_log_add_exp_test, logaddexp_case_double)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto out = TensorDesc(self);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// INT32 input (unsupported)
TEST_F(l2_log_add_exp_test, logaddexp_case_int32_unsupported)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto out = TensorDesc(self);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// INT input -> FLOAT output (type promotion)
TEST_F(l2_log_add_exp_test, logaddexp_case_int_out_fp)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// BOOL input -> FLOAT16 output (type promotion)
TEST_F(l2_log_add_exp_test, logaddexp_case_bool_out_fp16)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Promote type ok: FP16 + FP32 -> FP32
TEST_F(l2_log_add_exp_test, logaddexp_case_promote_type_ok)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Promote type not ok: FP16 + FP32 -> INT32
TEST_F(l2_log_add_exp_test, logaddexp_case_promote_type_nok)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 1D shape
TEST_F(l2_log_add_exp_test, logaddexp_case_dim_1)
{
    auto self = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc(self).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 4D shape
TEST_F(l2_log_add_exp_test, logaddexp_case_dim_4)
{
    auto self = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto other = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto out = TensorDesc(self).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 8D shape (max supported)
TEST_F(l2_log_add_exp_test, logaddexp_case_dim_8)
{
    auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc(self).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 9D shape (exceeds max)
TEST_F(l2_log_add_exp_test, logaddexp_case_dim_9_error)
{
    auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc(self).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Broadcast ok
TEST_F(l2_log_add_exp_test, logaddexp_case_broadcast_ok)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Broadcast not ok
TEST_F(l2_log_add_exp_test, logaddexp_case_broadcast_nok)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({3, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Empty tensor (all empty, broadcastable)
TEST_F(l2_log_add_exp_test, logaddexp_case_empty_empty_empty)
{
    auto self = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({0, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({0, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull: input nullptr
TEST_F(l2_log_add_exp_test, logaddexp_case_nullptr_input)
{
    auto tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull: output nullptr
TEST_F(l2_log_add_exp_test, logaddexp_case_nullptr_output)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnLogAddExp, INPUT(self, other), OUTPUT((aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ===================== aclnnLogAddExp2 tests =====================

// FP32
TEST_F(l2_log_add_exp_test, logaddexp2_case_fp32)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(self).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// FP16
TEST_F(l2_log_add_exp_test, logaddexp2_case_fp16)
{
    auto self = TensorDesc({1, 1, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({1, 1, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(self).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// DOUBLE (unsupported on current platform)
TEST_F(l2_log_add_exp_test, logaddexp2_case_double)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto out = TensorDesc(self);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// INT32 input (unsupported)
TEST_F(l2_log_add_exp_test, logaddexp2_case_int32_unsupported)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto out = TensorDesc(self);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// INT input -> FLOAT output (type promotion)
TEST_F(l2_log_add_exp_test, logaddexp2_case_int_out_fp)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// BOOL input -> FLOAT16 output (type promotion)
TEST_F(l2_log_add_exp_test, logaddexp2_case_bool_out_fp16)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Promote type ok: FP16 + FP32 -> FP32
TEST_F(l2_log_add_exp_test, logaddexp2_case_promote_type_ok)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Promote type not ok: FP16 + FP32 -> INT32
TEST_F(l2_log_add_exp_test, logaddexp2_case_promote_type_nok)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 1D shape
TEST_F(l2_log_add_exp_test, logaddexp2_case_dim_1)
{
    auto self = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc(self).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 4D shape
TEST_F(l2_log_add_exp_test, logaddexp2_case_dim_4)
{
    auto self = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto other = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto out = TensorDesc(self).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 8D shape (max supported)
TEST_F(l2_log_add_exp_test, logaddexp2_case_dim_8)
{
    auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc(self).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 9D shape (exceeds max)
TEST_F(l2_log_add_exp_test, logaddexp2_case_dim_9_error)
{
    auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc(self).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Broadcast ok
TEST_F(l2_log_add_exp_test, logaddexp2_case_broadcast_ok)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Broadcast not ok
TEST_F(l2_log_add_exp_test, logaddexp2_case_broadcast_nok)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({3, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Empty tensor (all empty, broadcastable)
TEST_F(l2_log_add_exp_test, logaddexp2_case_empty_empty_empty)
{
    auto self = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other = TensorDesc({0, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({0, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull: input nullptr
TEST_F(l2_log_add_exp_test, logaddexp2_case_nullptr_input)
{
    auto tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull: output nullptr
TEST_F(l2_log_add_exp_test, logaddexp2_case_nullptr_output)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnLogAddExp2, INPUT(self, other), OUTPUT((aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
