/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_isin_tensor_scalar.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_isin_tensor_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "isin_tensor_scalar_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "isin_tensor_scalar_test TearDown" << std::endl;
    }
};

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_f16)
{
    auto element_desc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto test_element_desc = ScalarDesc(1.0f);
    bool assume_unique = false;

    bool invert = false;
    auto out_desc = TensorDesc({1, 2, 3, 4}, ACL_BOOL, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_int8)
{
    auto element_desc = TensorDesc({1, 2, 3, 4}, ACL_INT8, ACL_FORMAT_HWCN);
    auto test_element_desc = ScalarDesc(1);
    bool assume_unique = false;

    bool invert = false;
    auto out_desc = TensorDesc({1, 2, 3, 4}, ACL_BOOL, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_uint8)
{
    auto element_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_UINT8, ACL_FORMAT_NDHWC);
    auto test_element_desc = ScalarDesc(static_cast<uint8_t>(1));
    bool assume_unique = false;

    bool invert = false;
    auto out_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_f32)
{
    auto element_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto test_element_desc = ScalarDesc(1.0f);
    bool assume_unique = true;

    bool invert = true;
    auto out_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_int32)
{
    auto element_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto test_element_desc = ScalarDesc(1);
    bool assume_unique = true;

    bool invert = false;
    auto out_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_int64)
{
    auto element_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto test_element_desc = ScalarDesc(1.0f);
    bool assume_unique = false;

    bool invert = true;
    auto out_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_empty)
{
    auto element_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND);
    auto test_element_desc = ScalarDesc(1);
    bool assume_unique = false;

    bool invert = false;
    auto out_desc = TensorDesc({2, 0}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_element_unsupport_dtype)
{
    bool assume_unique = false;

    bool invert = false;
    uint64_t workspace_size = 0;
    // unsupport dtype
    auto element_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND);
    auto test_element_desc = ScalarDesc(1);
    auto out_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_test_element_unsupport_dtype)
{
    bool assume_unique = false;

    bool invert = false;
    uint64_t workspace_size = 0;
    // unsupport dtype
    auto element_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto test_element_desc = ScalarDesc(false);
    auto out_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_element_null)
{
    bool assume_unique = false;

    bool invert = false;
    uint64_t workspace_size = 0;

    auto test_element_desc = ScalarDesc(1.0f);
    auto out_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnIsInTensorScalar, INPUT(nullptr, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_test_element_null)
{
    bool assume_unique = false;

    bool invert = false;
    uint64_t workspace_size = 0;

    auto element_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnIsInTensorScalar, INPUT(element_desc, nullptr, assume_unique, invert), OUTPUT(out_desc));

    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_element_out_null)
{
    bool assume_unique = false;

    bool invert = false;
    uint64_t workspace_size = 0;

    auto element_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto test_element_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(nullptr));

    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非连续
TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_not_contiguous)
{
    auto element_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto test_element_desc = ScalarDesc(1);
    bool assume_unique = false;

    bool invert = false;
    auto out_desc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// largeDim
TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_9d)
{
    auto element_desc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    auto test_element_desc = ScalarDesc(1.0f);
    bool assume_unique = false;

    bool invert = false;
    auto out_desc = TensorDesc(element_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_can_not_promote_type)
{
    auto element_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    auto test_element_desc = ScalarDesc(1.0f);
    bool assume_unique = true;

    bool invert = true;
    auto out_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_tensor_scalar_test, l2_isin_tensor_scalar_test_case_shape_invalid)
{
    auto element_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto test_element_desc = ScalarDesc(1.0f);
    bool assume_unique = true;

    bool invert = true;
    auto out_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnIsInTensorScalar, INPUT(element_desc, test_element_desc, assume_unique, invert), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
