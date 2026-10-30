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

#include "../../../op_api/aclnn_addcmul.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_inplace_addcmul_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "inplace_addcmul_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "inplace_addcmul_test TearDown" << endl;
    }
};

// Test for Empty tensor
TEST_F(l2_inplace_addcmul_test, test_empty_tensor)
{
    auto self_tensor_desc = TensorDesc({4, 0, 4}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto other_tensor_desc = TensorDesc({4, 0, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(
        aclnnInplaceAddcmul, INPUT(self_tensor_desc, other_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for promote
TEST_F(l2_inplace_addcmul_test, test_promote)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar_desc = ScalarDesc(5.9f);

    auto ut = OP_API_UT(
        aclnnInplaceAddcmul, INPUT(self_tensor_desc, other_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for Null Scalar Inputs
TEST_F(l2_inplace_addcmul_test, test_null_scalar)
{
    auto tensor_desc = TensorDesc({10, 20}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Value(vector<float>(200, 1.0f));
    auto other_tensor_desc =
        TensorDesc({10, 20}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Value(vector<float>(200, 2.0f));
    auto scalar_desc = nullptr;

    auto ut =
        OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, other_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// Test for Invalid DataType
TEST_F(l2_inplace_addcmul_test, test_invalid_data_type)
{
    auto tensor_desc =
        TensorDesc({10, 20}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<uint8_t>(200, 1));
    auto other_tensor_desc =
        TensorDesc({10, 20}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-2, 2).Value(vector<float>(200, 2.0f));
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));

    auto ut =
        OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, other_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Test for Invalid DataType
TEST_F(l2_inplace_addcmul_test, test_invalid_data_type_1)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({4, 4}, ACL_UINT32, ACL_FORMAT_ND);

    auto scalar_desc = ScalarDesc(5.9f);

    auto ut = OP_API_UT(
        aclnnInplaceAddcmul, INPUT(self_tensor_desc, other_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Test for Invalid DataType
TEST_F(l2_inplace_addcmul_test, test_invalid_data_type_2)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc2 = TensorDesc({4, 4}, ACL_UINT32, ACL_FORMAT_ND);

    auto scalar_desc = ScalarDesc(5.9f);

    auto ut = OP_API_UT(
        aclnnInplaceAddcmul, INPUT(self_tensor_desc, other_tensor_desc, other_tensor_desc2, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Test for Format Err
TEST_F(l2_inplace_addcmul_test, test_format_err)
{
    auto tensor_desc = TensorDesc({10, 5, 2, 10, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto scalar_desc = ScalarDesc(5.9f);

    auto ut = OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, tensor_desc, tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    // SAMPLE: only test GetWorkspaceSize
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Test for BroadcastShape Err
TEST_F(l2_inplace_addcmul_test, test_broadcat_err)
{
    auto self_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);

    auto scalar_desc = ScalarDesc(5.9f);

    auto ut = OP_API_UT(
        aclnnInplaceAddcmul, INPUT(self_tensor_desc, other_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Test for BroadcastShape
TEST_F(l2_inplace_addcmul_test, test_broadcast)
{
    auto self_tensor_desc = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);

    auto scalar_desc = ScalarDesc(5.9f);

    auto ut = OP_API_UT(
        aclnnInplaceAddcmul, INPUT(self_tensor_desc, other_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_addcmul_test, test_int32)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-2, 2);

    auto scalar_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, tensor_desc, tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for shape err
TEST_F(l2_inplace_addcmul_test, test_shape_err)
{
    auto tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, tensor_desc, tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_addcmul_test, test_float)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);

    auto scalar_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, tensor_desc, tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_addcmul_test, test_float_nhwc)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);

    auto scalar_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, tensor_desc, tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_addcmul_test, test_float_nd)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto scalar_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, tensor_desc, tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_addcmul_test, test_float_ndhwc)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);

    auto scalar_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, tensor_desc, tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_addcmul_test, test_float_ncdhw)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);

    auto scalar_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, tensor_desc, tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for non-continuous
TEST_F(l2_inplace_addcmul_test, test_non_continuous)
{
    auto tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);

    auto scalar_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc, tensor_desc, tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for float16
TEST_F(l2_inplace_addcmul_test, test_float16)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto scalar_desc = ScalarDesc(5.9f);

    auto ut = OP_API_UT(
        aclnnInplaceAddcmul, INPUT(self_tensor_desc, other_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for aicpu big shape
// 该用例针对aicpu的一个bug，由于aicpu修改还未出包，先跳过
// TEST_F(l2_inplace_addcmul_test, test_aicpu_big_shape) {
//  auto tensor_desc1 = TensorDesc({3,32,1,148,2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
//  auto tensor_desc2 = TensorDesc({3,32,148,148,2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
//  auto tensor_desc3 = TensorDesc({3,1,148,148,1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
//  auto out_tensor_desc = TensorDesc(tensor_desc2).Precision(0.0001, 0.0001);
//  auto scalar_desc = ScalarDesc(1.0f);
//
//  auto ut = OP_API_UT(aclnnInplaceAddcmul, INPUT(tensor_desc1, tensor_desc2, tensor_desc3, scalar_desc),
//  OUTPUT(out_tensor_desc));
//
//  // SAMPLE: only test GetWorkspaceSize
//  uint64_t workspace_size = 0;
//  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//  EXPECT_EQ(aclRet, ACL_SUCCESS);
//
//  // SAMPLE: precision simulate
//  ut.TestPrecision();
//}