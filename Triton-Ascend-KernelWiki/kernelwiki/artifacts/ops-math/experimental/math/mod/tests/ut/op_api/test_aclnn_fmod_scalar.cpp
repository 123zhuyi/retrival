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
#include "../../../op_api/aclnn_fmod_scalar.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

class L2FmodScalarTest : public testing::Test {};

TEST_F(L2FmodScalarTest, float_scalar)
{
    auto self = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 100);
    auto scalar = ScalarDesc(3.0, ACL_FLOAT);
    auto out = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnFmodScalar, INPUT(self, scalar), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(L2FmodScalarTest, int32_scalar)
{
    auto self = TensorDesc({20}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 100);
    auto scalar = ScalarDesc(static_cast<int32_t>(7));
    auto out = TensorDesc({20}, ACL_INT32, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodScalar, INPUT(self, scalar), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}
