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

#include "math/less/op_api/aclnn_lt_tensor.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_inplace_lt_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "inplace_lt_tensor_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "inplace_lt_tensor_test TearDown" << endl;
    }
};

// aclnnInplaceLtTensor_001:lt.Tensor_out输入支持INT16
// 走AICPU
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_001_int16_HWCN)
{
    auto tensor_self =
        TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto tensor_other = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtTensor_002:lt.Tensor_out输入支持BOOL
// 走AICPU
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_002_bool_ND)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.001, 0.001);
    auto tensor_other = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtTensor_003:lt.Tensor_out输入支持DOUBLE
// 走AICPU
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_003_double)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto tensor_other = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_NHWC).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtTensor_004:lt.Tensor_out支持输入类型不一致
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_004_float16_float_NHWC)
{
    auto tensor_self =
        TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-10, 10).Precision(0.001, 0.001);
    auto tensor_other = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtTensor_005:lt.Tensor_out支持输入类型不一致
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_005_int32_float16_ND)
{
    auto tensor_self =
        TensorDesc({7, 9, 11, 3, 4, 6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto tensor_other = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtTensor_006:lt.Tensor_out支持输入的类型不一致
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_006_float_int32_bool_broadcast_ND)
{
    auto tensor_self =
        TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto tensor_other = TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtTensor_007:lt.Tensor_out支持输入为空Tensor
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_007_input_empty_tensor)
{
    auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// aclnnInplaceLtTensor_008:lt.Tensor_out支持输入为非连续Tensor
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_008_input_not_contiguous)
{
    auto tensor_self =
        TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto tensor_other = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtTensor_009：lt.Tensor_out支持other为非连续Tensor
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_009_output_not_contiguous)
{
    auto tensor_self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto tensor_other = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 其他用例
// 入参为other为null场景
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_010_test_lt_tensor_other_is_null)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{9, 3, 1, 1, 2, 3});

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, (aclTensor*)nullptr), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参为空指针的场景
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_011_test_lt_tensor_input_out_is_null)
{
    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 入参self&other都为空tensor的场景
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_012_test_lt_tensor_input_is_empty)
{
    auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto tensor_other = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 入参other为空tensor的场景
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_013_test_lt_tensor_input_other_is_empty)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto tensor_other = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参self数据类型不支持的场景
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_014_test_lt_tensor_self_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{9, 3, 1, 1, 2, 3});

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参other数据类型不支持的场景
TEST_F(l2_inplace_lt_tensor_test, aclnnInplaceLtTensor_015_test_lt_tensor_other_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{9, 3, 1, 1, 2, 3});

    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}
