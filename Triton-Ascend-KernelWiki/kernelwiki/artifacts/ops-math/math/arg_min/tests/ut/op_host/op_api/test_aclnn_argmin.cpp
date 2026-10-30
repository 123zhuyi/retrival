/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "aclnn_argmin.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"

using namespace std;

class l2_argmin_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_argmin_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_argmin_test TearDown" << endl;
    }
};

TEST_F(l2_argmin_test, aclnnArgMin_float_nd_0_nd_2_dims)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_32_8_10_float16_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({32, 8, 10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({32, 10}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, ascend910B2_aclnnArgMin_32_8_10_bf16_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({32, 8, 10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({32, 10}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_64_50_2_float_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 50, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({64, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_64_16_2_double_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 16, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({16, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_32_8_10_double_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({32, 8, 10}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_64_16_2_int8_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 16, 2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({64, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_64_16_2_int16_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 16, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({16, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_64_16_2_int32_nd_1_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 16, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({64, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_int64_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 16, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({16, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_uint8_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 16, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({16, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_float_out_INT32_nd)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 5, 7, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({3, 7, 6}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// keepdim
TEST_F(l2_argmin_test, aclnnArgMin_float_out_INT64_nd_keepdim)
{
    const int64_t dim = 1;
    const bool keepdim = true;

    auto self_tensor_desc = TensorDesc({3, 5, 7, 6}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({3, 1, 7, 6}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_float16_out_INT32_nd_keepdim)
{
    const int64_t dim = 1;
    const bool keepdim = true;

    auto self_tensor_desc = TensorDesc({3, 5, 7, 6}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({3, 1, 7, 6}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_argmin_test, aclnnArgMin_float16_out_INT64_NCHW_keepdim)
{
    const int64_t dim = 2;
    const bool keepdim = true;

    auto self_tensor_desc = TensorDesc({3, 5, 7, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({3, 5, 1, 6}, ACL_INT64, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

// multidimensional
TEST_F(l2_argmin_test, aclnnArgMin_float_nd_0_nd_6_dims_keepdim)
{
    const int64_t dim = 5;
    const bool keepdim = true;

    auto self_tensor_desc = TensorDesc({3, 4, 6, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({3, 4, 6, 2, 2, 1}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

// discontiues
TEST_F(l2_argmin_test, aclnnArgMin_float_nd_discontinues_keepdim)
{
    const int64_t dim = 0;
    const bool keepdim = true;

    auto self_tensor_desc =
        TensorDesc({3, 5, 7, 6}, ACL_FLOAT, ACL_FORMAT_ND, {210, 42, 1, 7}, 0, {3, 5, 6, 7}).ValueRange(0, 2);
    auto out_tensor_desc =
        TensorDesc({1, 5, 7, 6}, ACL_INT64, ACL_FORMAT_ND, {210, 42, 1, 7}, 0, {1, 5, 6, 7}).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_argmin_test, aclnnArgMin_float_nd_discontinues)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc =
        TensorDesc({3, 5, 7, 6}, ACL_FLOAT, ACL_FORMAT_ND, {210, 42, 1, 7}, 0, {3, 5, 6, 7}).ValueRange(0, 2);
    auto out_tensor_desc =
        TensorDesc({5, 7, 6}, ACL_INT64, ACL_FORMAT_ND, {42, 1, 7}, 0, {5, 6, 7}).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_argmin_test, aclnnArgMin_float_nd_same_value)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    auto out_tensor_desc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_float_nd_same_value_keepdim)
{
    const int64_t dim = 0;
    const bool keepdim = true;

    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    auto out_tensor_desc = TensorDesc({1, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_null_tensor)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({12, 10, 0, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({12, 0, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
}

TEST_F(l2_argmin_test, aclnnArgMin_8_dims)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    // not keepdim
    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));
    // keepdim
    const int64_t bigdim = 7;
    auto out_tensor_desc_keepdim =
        TensorDesc({2, 2, 2, 2, 2, 2, 2, 1}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut_keepdim = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, bigdim, true), OUTPUT(out_tensor_desc_keepdim));
}

TEST_F(l2_argmin_test, aclnnArgMin_self_scalar)
{
    const int64_t dim = -1;
    const bool keepdim = true;

    auto self_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Abnormal Scenarios
TEST_F(l2_argmin_test, aclnnArgMin_self_bool_dtype_not_support)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 16, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({64, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_test, aclnnArgMin_self_uint64_dtype_not_support)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 16, 2}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({64, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_test, aclnnArgMin_out_dtype_not_support)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 16, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({64, 2}, ACL_INT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_test, aclnnArgMin_dim_value_not_support)
{
    const int64_t dim = 3;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({64, 16, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({64, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_test, aclnnArgMin_nullptr)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = nullptr;
    auto out_tensor_desc = TensorDesc({64, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
// 10dims
TEST_F(l2_argmin_test, aclnnArgMin_10_dims_not_support)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({12, 10, 1, 2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({12, 1, 2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_test, aclnnArgMin_self_dim_zero)
{
    const int64_t dim = 1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({12, 0, 1, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({12, 1, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_test, aclnnArgMin_self_dim_negative)
{
    const int64_t dim = -1;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({12, 1, 1, 0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({12, 1, 1}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmin_test, ascend310P_float16_nd_0_nd)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({4, 6}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnArgMin, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
