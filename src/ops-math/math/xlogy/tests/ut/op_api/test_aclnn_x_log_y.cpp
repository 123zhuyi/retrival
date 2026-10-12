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

#include "math/xlogy/op_api/aclnn_x_log_y_tensor.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_x_log_y_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "x_log_y_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "x_log_y_test TearDown" << endl; }
};

// test FLOAT
TEST_F(l2_x_log_y_test, case_x_log_y_tensor_float)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto other_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto out_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnXLogYTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test FLOAT16
TEST_F(l2_x_log_y_test, case_x_log_y_tensor_float16)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto other_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto out_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnXLogYTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test BF16
TEST_F(l2_x_log_y_test, case_x_log_y_tensor_bf16)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto other_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto out_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnXLogYTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test broadcast
TEST_F(l2_x_log_y_test, case_x_log_y_tensor_broadcast)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto other_desc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnXLogYTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 1d tensor
TEST_F(l2_x_log_y_test, case_x_log_y_tensor_1d)
{
    auto self_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto other_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto out_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnXLogYTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test nullptr
TEST_F(l2_x_log_y_test, case_x_log_y_tensor_nullptr)
{
    auto ut = OP_API_UT(aclnnXLogYTensor, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test mismatched shape
TEST_F(l2_x_log_y_test, case_x_log_y_tensor_mismatched_shape)
{
    auto self_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto other_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 2.0);
    auto out_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnXLogYTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
