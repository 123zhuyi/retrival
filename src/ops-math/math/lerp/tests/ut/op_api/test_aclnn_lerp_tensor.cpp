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
 * \file test_aclnn_lerp_tensor.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_lerp_tensor.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_lerp_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "lerp_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "lerp_test TearDown" << endl;
    }
};

TEST_F(l2_lerp_test, ascend910A_case_1_all_support_type)
{
    vector<aclDataType> dtype_list{ACL_FLOAT, ACL_FLOAT16};
    for (auto dtype : dtype_list) {
        cout << "+++++++++++++++++++++++ start to test dtype:" << String(dtype) << endl;
        auto self_tensor_desc = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
        auto end_tensor_desc = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
        auto weight_tensor_desc = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
        auto out_tensor_desc = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
        auto ut =
            OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_lerp_test, ascend910B2_case_1_all_support_type)
{
    vector<aclDataType> dtype_list{ACL_FLOAT, ACL_FLOAT16, ACL_BF16};
    for (auto dtype : dtype_list) {
        cout << "+++++++++++++++++++++++ start to test dtype:" << String(dtype) << endl;
        auto self_tensor_desc = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
        auto end_tensor_desc = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
        auto weight_tensor_desc = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
        auto out_tensor_desc = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
        auto ut =
            OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_lerp_test, case_2_self_type_unsupport)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerp_test, case_3_end_type_unsupport)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerp_test, case_4_weight_type_unsupport)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerp_test, case_5_out_type_unsupport)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerp_test, case_6_input_type_different)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerp_test, case_7_self_end_broadcast)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerp_test, case_8_self_weight_broadcast)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerp_test, case_9_self_end_weight_broadcast)
{
    auto self_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerp_test, case_10_input_broadcast_failed)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerp_test, case_11_out_broadcast_failed)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerp_test, case_12_self_nullptr)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLerp, INPUT(nullptr, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_lerp_test, case_13_end_nullptr)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, nullptr, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_lerp_test, case_14_weight_nullptr)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, nullptr), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_lerp_test, case_15_out_nullptr)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_lerp_test, case_16_dim9)
{
    auto self_tensor_desc = TensorDesc({2, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerp_test, case_17_empty_input)
{
    auto self_tensor_desc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerp_test, case_18_empty_input_failed)
{
    auto self_tensor_desc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lerp_test, case_19_nonContiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto end_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto weight_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerp_test, case_20_inplace)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lerp_test, case_21_inplace_broadcast_failed)
{
    auto self_tensor_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto end_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceLerp, INPUT(self_tensor_desc, end_tensor_desc, weight_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
