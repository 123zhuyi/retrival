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

#include "math/less/op_api/aclnn_lt_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_inplace_lt_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "lt_scalar_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "lt_scalar_test TearDown" << endl;
    }
};

// aclnnInplaceLtScalar_001:lt.Scalar_out输入支持BOOL
// 走AICPU
TEST_F(l2_inplace_lt_scalar_test, aclnnInplaceLtScalar_001_bool_ND)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.001, 0.001);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(1));

    auto ut = OP_API_UT(aclnnInplaceLtScalar, INPUT(tensor_self, scalar_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtScalar_002:lt.Scalar_out输入支持DOUBLE
// 走AICPU
TEST_F(l2_inplace_lt_scalar_test, aclnnInplaceLtScalar_002_double)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));

    auto ut = OP_API_UT(aclnnInplaceLtScalar, INPUT(tensor_self, scalar_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtScalar_003:lt.Scalar_out支持输入为空Tensor
TEST_F(l2_inplace_lt_scalar_test, aclnnInplaceLtScalar_003_input_empty_tensor)
{
    auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));

    auto ut = OP_API_UT(aclnnInplaceLtScalar, INPUT(tensor_self, scalar_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtScalar_004:lt.Scalar_out支持输入为非连续Tensor
TEST_F(l2_inplace_lt_scalar_test, aclnnInplaceLtScalar_004_input_not_contiguous)
{
    auto tensor_self =
        TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto scalar_other = ScalarDesc(1.2f);

    auto ut = OP_API_UT(aclnnInplaceLtScalar, INPUT(tensor_self, scalar_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceLtScalar_005：lt.Scalar_out空指针场景测试
TEST_F(l2_inplace_lt_scalar_test, aclnnInplaceLtScalar_005_null_pointer)
{
    auto scalar_other = ScalarDesc(1.2f);

    auto ut = OP_API_UT(aclnnInplaceLtScalar, INPUT((aclTensor*)nullptr, scalar_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// 其他用例
// 入参为other为null场景
TEST_F(l2_inplace_lt_scalar_test, aclnnInplaceLtScalar_006_test_lt_scalar_other_is_null)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{9, 3, 1, 1, 2, 3});

    auto ut = OP_API_UT(aclnnInplaceLtScalar, INPUT(tensor_self, nullptr), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}