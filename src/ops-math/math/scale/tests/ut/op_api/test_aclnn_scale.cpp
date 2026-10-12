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

#include "../../../op_api/aclnn_scale.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_scale_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_scale_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_scale_test TearDown" << std::endl; }
};

// float类型，bias为空
TEST_F(l2_scale_test, scale_float_no_bias)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float类型，带bias
TEST_F(l2_scale_test, scale_float_with_bias)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto bias = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, bias, (int64_t)1, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16类型
TEST_F(l2_scale_test, scale_float16)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bf16类型
TEST_F(l2_scale_test, scale_bf16)
{
    auto x = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// numAxes=0, scale为标量
TEST_F(l2_scale_test, scale_num_axes_zero)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)0, (int64_t)0, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// scaleFromBlob=false
TEST_F(l2_scale_test, scale_not_from_blob)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, false), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 多维scale，axis=0
TEST_F(l2_scale_test, scale_multi_dim_axis0)
{
    auto x = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)0, (int64_t)2, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空指针校验: x为空
TEST_F(l2_scale_test, scale_x_nullptr)
{
    auto scale = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnScale, INPUT((aclTensor*)nullptr, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针校验: scale为空
TEST_F(l2_scale_test, scale_scale_nullptr)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, (aclTensor*)nullptr, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true),
                        OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针校验: y为空
TEST_F(l2_scale_test, scale_y_nullptr)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true),
                        OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持的数据类型
TEST_F(l2_scale_test, scale_unsupported_dtype)
{
    auto x = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x和y shape不匹配
TEST_F(l2_scale_test, scale_xy_shape_mismatch)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x和scale dtype不匹配
TEST_F(l2_scale_test, scale_x_scale_dtype_mismatch)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// axis越界
TEST_F(l2_scale_test, scale_axis_out_of_range)
{
    auto x = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scale = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)5, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非连续tensor
TEST_F(l2_scale_test, scale_non_contiguous)
{
    auto x = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-1, 1);
    auto scale = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor
TEST_F(l2_scale_test, scale_empty_tensor)
{
    auto x = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scale = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnScale, INPUT(x, scale, (aclTensor*)nullptr, (int64_t)1, (int64_t)1, true), OUTPUT(y));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
