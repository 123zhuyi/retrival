/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_fmod_tensor.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

class L2FmodTensorTest : public testing::Test {};

TEST_F(L2FmodTensorTest, float_same_shape)
{
    auto self = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 100);
    auto other = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(L2FmodTensorTest, fp16_broadcast)
{
    auto self = TensorDesc({2, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 100);
    auto other = TensorDesc({1, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({2, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(L2FmodTensorTest, invalid_broadcast)
{
    auto self = TensorDesc({20}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 100);
    auto other = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({20}, ACL_INT32, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}
