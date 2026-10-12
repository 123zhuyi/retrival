/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_lerp_scalar.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_lerp_scalar.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_lerps_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "lerps_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "lerps_test TearDown" << endl;
    }
};

TEST_F(l2_lerps_test, case_2_self_type_unsupport)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerps_test, case_3_end_type_unsupport)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerps_test, case_4_out_type_unsupport)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerps_test, case_5_input_type_different)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerps_test, case_6_self_end_broadcast)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerps_test, case_7_input_broadcast_failed)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerps_test, case_8_out_broadcast_failed)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerps_test, case_9_self_nullptr)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLerps, INPUT(nullptr, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_lerps_test, case_10_end_nullptr)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, nullptr, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_lerps_test, case_11_weight_nullptr)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, nullptr), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_lerps_test, case_12_out_nullptr)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_lerps_test, case_13_dim9)
{
    auto self_tensor_desc = TensorDesc({2, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerps_test, case_14_empty_input)
{
    auto self_tensor_desc = TensorDesc({2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerps_test, case_15_empty_input_failed)
{
    auto self_tensor_desc = TensorDesc({2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerps_test, case_16_nonContiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto end_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto ut =
        OP_API_UT(aclnnLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerps_test, case_17_inplace)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnInplaceLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerps_test, case_18_inplace_broadcast_failed)
{
    auto self_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnInplaceLerps, INPUT(self_tensor_desc, end_tensor_desc, weight_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
