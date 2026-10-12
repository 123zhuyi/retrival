/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../op_host/op_api/aclnn_softshrink_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;

namespace {

class SoftshrinkBackwardApiTest : public testing::Test {};

TEST_F(SoftshrinkBackwardApiTest, Float32WorkspaceSuccess)
{
    auto gradOutput = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto self = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(1e-6, 1e-6);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(SoftshrinkBackwardApiTest, AtlasA3WorkspaceSuccess)
{
    SocVersionManager socVersionManager(SocVersion::ASCEND910_93);
    auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(1e-6, 1e-6);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(SoftshrinkBackwardApiTest, Float16UnalignedSuccess)
{
    auto gradOutput = TensorDesc({257}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto self = TensorDesc({257}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({257}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(1e-3, 1e-3);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(SoftshrinkBackwardApiTest, BFloat16Success)
{
    auto gradOutput = TensorDesc({4, 17}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto self = TensorDesc({4, 17}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({4, 17}, ACL_BF16, ACL_FORMAT_ND).Precision(1e-2, 1e-2);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(SoftshrinkBackwardApiTest, BroadcastSuccess)
{
    auto gradOutput = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto self = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto lambd = ScalarDesc(0.25f);
    auto gradInput = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(1e-6, 1e-6);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(SoftshrinkBackwardApiTest, MixedInputDtypeSuccess)
{
    auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(1e-3, 1e-3);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(SoftshrinkBackwardApiTest, EmptyTensorSuccess)
{
    auto gradOutput = TensorDesc({2, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto self = TensorDesc({2, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({2, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

TEST_F(SoftshrinkBackwardApiTest, NonContiguousSuccess)
{
    auto gradOutput = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(1e-6, 1e-6);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(SoftshrinkBackwardApiTest, RejectsNegativeLambda)
{
    auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lambd = ScalarDesc(-0.5f);
    auto gradInput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SoftshrinkBackwardApiTest, RejectsUnsupportedDtype)
{
    auto gradOutput = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SoftshrinkBackwardApiTest, RejectsUnsupportedPlatform)
{
    SocVersionManager socVersionManager(SocVersion::ASCEND910);
    auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SoftshrinkBackwardApiTest, RejectsInvalidOutputShape)
{
    auto gradOutput = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto self = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(gradOutput, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SoftshrinkBackwardApiTest, RejectsNullInput)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto lambd = ScalarDesc(0.5f);
    auto gradInput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSoftshrinkBackward, INPUT(nullptr, self, lambd), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

} // namespace
