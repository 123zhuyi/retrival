/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "gtest/gtest.h"
#include "opdev/platform.h"

#include "../../../op_api/aclnn_sqrt_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

class l2_sqrt_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }
};

TEST_F(l2_sqrt_backward_test, case_001_fp32_precision)
{
    auto yDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{1.0f, 2.0f, 4.0f, 8.0f});
    auto dyDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{2.0f, 4.0f, 8.0f, 0.0f});
    auto zDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSqrtBackward, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_sqrt_backward_test, case_002_fp16_precision)
{
    auto yDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.5f, 8.0f);
    auto dyDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(std::vector<float>{2.0f, 4.0f, 0.0f, -8.0f});
    auto zDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnSqrtBackward, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_sqrt_backward_test, case_003_bf16_precision)
{
    auto yDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0.5f, 8.0f);
    auto dyDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).Value(std::vector<float>{2.0f, 0.0f, 6.0f, -8.0f});
    auto zDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.02, 0.02);
    auto ut = OP_API_UT(aclnnSqrtBackward, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_sqrt_backward_test, case_004_scalar_success)
{
    auto yDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{4.0f});
    auto dyDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{8.0f});
    auto zDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSqrtBackward, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_sqrt_backward_test, case_005_noncontiguous_output_success)
{
    auto yDesc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {5, 4}).ValueRange(0.5f, 8.0f);
    auto dyDesc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {5, 4}).ValueRange(-8.0f, 8.0f);
    auto zDesc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {5, 4}).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSqrtBackward, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_sqrt_backward_test, case_006_shape_mismatch)
{
    auto yDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dyDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto zDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSqrtBackward, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sqrt_backward_test, case_007_dtype_mismatch)
{
    auto yDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dyDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSqrtBackward, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sqrt_backward_test, case_008_nullptr)
{
    auto tensorDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSqrtBackward, INPUT((aclTensor *)nullptr, tensorDesc), OUTPUT(tensorDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
