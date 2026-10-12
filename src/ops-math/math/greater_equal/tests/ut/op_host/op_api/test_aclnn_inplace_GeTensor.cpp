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

#include "aclnn_ge_tensor.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_inplace_ge_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ge_tensor_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "ge_tensor_test TearDown" << endl;
    }
};

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_normal_support_uint88)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{3, 4, 9, 6, 7, 6});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{3, 4, 4, 6, 7, 6});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_normal_out_int8)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_normal_input_float)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.000001, 4, 9, 6, 7, 11});

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<float>{3.00000099, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_normal_out_double)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_normal_out_int64)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_normal_out_int32)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_normal_out_int16)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_normal_out_float16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_normal_input_bool)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, true, true, false, false, false});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<int8_t>{false, false, false, true, true, true});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_normal_bool_to_float)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 入参为self为null场景
TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_self_is_null)
{
    auto tensor_self = nullptr;

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参为other为null场景
TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_other_is_null)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto tensor_other = nullptr;

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参self&other都为空tensor的场景
TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_input_is_empty)
{
    auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto tensor_other = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_EQ(workspace_size, 0u);
}

// 入参self数据类型不支持的场景
TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_self_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参oter数据类型不支持的场景
TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_other_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_input_shape_greater_than_8)
{
    auto tensor_self = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参无法broadcast的场景
TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_input_cannot_broadcast)
{
    auto tensor_self = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// complex64 不支持
TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_not_support_complex64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非连续
TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_to_contiguous)
{
    auto tensor_self =
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Value(vector<float>{3, 4, 9, 6, 7, 11, 5, 89});

    auto tensor_other =
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Value(vector<float>{3, 4, 8, 6, 5, 11, 5, 80});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 数据范围[-1,1]
TEST_F(l2_inplace_ge_tensor_test, test_inplace_ge_tensor_date_range_f1_1)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(-1, 1)
            .Value(vector<float>{
                -6.75468990e-01, 7.37894118e-01, -7.84537017e-01, 9.64608252e-01, 9.09311235e-01, -9.51533616e-01});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(-1, 1)
            .Value(vector<float>{
                -6.75468990e-01, 7.37894118e-01, -7.94537017e-01, 9.64608252e-01, 9.09211235e-01, -9.51533616e-01});

    auto ut = OP_API_UT(aclnnInplaceGeTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
