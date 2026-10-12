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

#include "../../../op_api/aclnn_argmin.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_arg_min_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "arg_min_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "arg_min_test TearDown" << endl; }
};

// test FLOAT with INT64 output
TEST_F(l2_arg_min_test, case_arg_min_float_int64)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);

    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test FLOAT16
TEST_F(l2_arg_min_test, case_arg_min_float16)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);

    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test BF16
TEST_F(l2_arg_min_test, case_arg_min_bf16)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);

    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test INT32
TEST_F(l2_arg_min_test, case_arg_min_int32)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);

    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test keepdim=true
TEST_F(l2_arg_min_test, case_arg_min_keepdim)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3, 1}, ACL_INT64, ACL_FORMAT_ND);

    int64_t dim = 2;
    bool keepdim = true;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 1d tensor
TEST_F(l2_arg_min_test, case_arg_min_1d)
{
    auto self_tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);

    int64_t dim = 0;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test nullptr
TEST_F(l2_arg_min_test, case_arg_min_nullptr)
{
    int64_t dim = 0;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT((aclTensor*)nullptr, dim, keepdim), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test unsupported dtype
TEST_F(l2_arg_min_test, case_arg_min_unsupported_dtype)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test unsupported output dtype
TEST_F(l2_arg_min_test, case_arg_min_unsupported_out_dtype)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test dim out of range
TEST_F(l2_arg_min_test, case_arg_min_dim_out_of_range)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 5;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test negative dim
TEST_F(l2_arg_min_test, case_arg_min_neg_dim)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = -1;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test int32 input
TEST_F(l2_arg_min_test, case_arg_min_int32_input)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND);
    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test int64 input
TEST_F(l2_arg_min_test, case_arg_min_int64)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test int8
TEST_F(l2_arg_min_test, case_arg_min_int8)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test uint8
TEST_F(l2_arg_min_test, case_arg_min_uint8)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test double
TEST_F(l2_arg_min_test, case_arg_min_double)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test int16
TEST_F(l2_arg_min_test, case_arg_min_int16)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 2;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test empty tensor (reduce dim size is 0, expected to fail)
TEST_F(l2_arg_min_test, case_arg_min_empty)
{
    auto self_tensor_desc = TensorDesc({3, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 1;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test scalar (0-dim) input
TEST_F(l2_arg_min_test, case_arg_min_scalar)
{
    auto self_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 0;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 8 dims double (reshape path)
TEST_F(l2_arg_min_test, case_arg_min_8d_double)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 7;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 8 dims double with dim > 5
TEST_F(l2_arg_min_test, case_arg_min_8d_double_dim_high)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 3, 1}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 6;
    bool keepdim = false;
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
