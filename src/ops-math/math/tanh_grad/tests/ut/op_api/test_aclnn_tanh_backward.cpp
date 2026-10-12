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

#include "../../../op_api/aclnn_tanh_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_tanh_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_tanh_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_tanh_backward_test TearDown" << std::endl; }
};

// float类型
TEST_F(l2_tanh_backward_test, tanh_backward_float)
{
    auto grad_output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_input = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16类型
TEST_F(l2_tanh_backward_test, tanh_backward_float16)
{
    auto grad_output = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto output = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_input = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bf16类型
TEST_F(l2_tanh_backward_test, tanh_backward_bf16)
{
    auto grad_output = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto output = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_input = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 不同shape
TEST_F(l2_tanh_backward_test, tanh_backward_different_shape)
{
    auto grad_output = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto output = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_input = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 1维tensor
TEST_F(l2_tanh_backward_test, tanh_backward_1d)
{
    auto grad_output = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto output = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_input = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空指针校验: gradOutput为空
TEST_F(l2_tanh_backward_test, tanh_backward_grad_output_nullptr)
{
    auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_input = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT((aclTensor*)nullptr, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针校验: output为空
TEST_F(l2_tanh_backward_test, tanh_backward_output_nullptr)
{
    auto grad_output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto grad_input = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, (aclTensor*)nullptr), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针校验: gradInput为空
TEST_F(l2_tanh_backward_test, tanh_backward_grad_input_nullptr)
{
    auto grad_output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持的数据类型
TEST_F(l2_tanh_backward_test, tanh_backward_unsupported_dtype)
{
    auto grad_output = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto output = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_input = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOutput和output shape不匹配
TEST_F(l2_tanh_backward_test, tanh_backward_shape_mismatch)
{
    auto grad_output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto output = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_input = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非连续tensor
TEST_F(l2_tanh_backward_test, tanh_backward_non_contiguous)
{
    auto grad_output = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto output = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-1, 1);
    auto grad_input = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor
TEST_F(l2_tanh_backward_test, tanh_backward_empty_tensor)
{
    auto grad_output = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto output = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_input = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 超过8维
TEST_F(l2_tanh_backward_test, tanh_backward_big_dim)
{
    auto grad_output = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto output = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto grad_input = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnTanhBackward, INPUT(grad_output, output), OUTPUT(grad_input));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
