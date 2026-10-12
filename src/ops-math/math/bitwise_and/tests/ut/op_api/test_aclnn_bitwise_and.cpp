/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_bitwise_and_tensor.h"
#include "../../../op_api/aclnn_bitwise_and_scalar.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_bitwise_and_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "bitwise_and_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "bitwise_and_test TearDown" << endl; }
};

// === Tensor version ===
// test INT8
TEST_F(l2_bitwise_and_test, case_bitwise_and_tensor_int8)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_desc = TensorDesc({3, 3, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({3, 3, 3}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test UINT8
TEST_F(l2_bitwise_and_test, case_bitwise_and_tensor_uint8)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10);
    auto other_desc = TensorDesc({3, 3, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10);
    auto out_desc = TensorDesc({3, 3, 3}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test INT16
TEST_F(l2_bitwise_and_test, case_bitwise_and_tensor_int16)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_desc = TensorDesc({3, 3, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({3, 3, 3}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test INT32
TEST_F(l2_bitwise_and_test, case_bitwise_and_tensor_int32)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_desc = TensorDesc({3, 3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({3, 3, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test INT64
TEST_F(l2_bitwise_and_test, case_bitwise_and_tensor_int64)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_desc = TensorDesc({3, 3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({3, 3, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test BOOL
TEST_F(l2_bitwise_and_test, case_bitwise_and_tensor_bool)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto other_desc = TensorDesc({3, 3, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto out_desc = TensorDesc({3, 3, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test broadcast
TEST_F(l2_bitwise_and_test, case_bitwise_and_tensor_broadcast)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_desc = TensorDesc({1, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 1d tensor
TEST_F(l2_bitwise_and_test, case_bitwise_and_tensor_1d)
{
    auto self_desc = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_desc = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// === Scalar version ===
// test scalar with INT32
TEST_F(l2_bitwise_and_test, case_bitwise_and_scalar_int32)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(5);
    auto out_desc = TensorDesc({3, 3, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_desc, scalar_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test scalar with INT64
TEST_F(l2_bitwise_and_test, case_bitwise_and_scalar_int64)
{
    auto self_desc = TensorDesc({3, 3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(int64_t(3));
    auto out_desc = TensorDesc({3, 3, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_desc, scalar_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// === Error cases ===
// test nullptr
TEST_F(l2_bitwise_and_test, case_bitwise_and_tensor_nullptr)
{
    auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr),
                        OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test mismatched shape
TEST_F(l2_bitwise_and_test, case_bitwise_and_tensor_mismatched_shape)
{
    auto self_desc = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_desc = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({2, 5}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
