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

#include "../../../op_api/aclnn_tan.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_tan_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "tan_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "tan_test TearDown" << endl; }
};

// test aicore type: FLOAT
TEST_F(l2_tan_test, case_tan_for_float_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: FLOAT16
TEST_F(l2_tan_test, case_tan_for_float16_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: DOUBLE
TEST_F(l2_tan_test, case_tan_for_double_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: BF16
TEST_F(l2_tan_test, case_tan_for_bf16_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 1d tensor
TEST_F(l2_tan_test, case_tan_for_1d_tensor)
{
    auto self_tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 5d tensor
TEST_F(l2_tan_test, case_tan_for_5d_tensor)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test empty tensor
TEST_F(l2_tan_test, case_tan_for_empty_tensor)
{
    auto self_tensor_desc = TensorDesc({0, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test nullptr input
TEST_F(l2_tan_test, case_tan_nullptr_input)
{
    auto ut = OP_API_UT(aclnnTan, INPUT((aclTensor*)nullptr), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test nullptr output
TEST_F(l2_tan_test, case_tan_nullptr_output)
{
    auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test mismatched shape
TEST_F(l2_tan_test, case_tan_mismatched_shape)
{
    auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
