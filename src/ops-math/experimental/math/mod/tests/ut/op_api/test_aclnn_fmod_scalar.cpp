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

// The following cases cover the int16 same-dtype lane and tensor-scalar promotion. self is cast and the scalar is
// converted to the common calculation dtype before the two-argument Mod call; only the result is cast to out.

// K2: int16 self, int16-valued scalar -> same-dtype int16 naive lane.
TEST_F(L2FmodScalarTest, int16_scalar_same_dtype)
{
    auto self = TensorDesc({8, 8}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto scalar = ScalarDesc(static_cast<int16_t>(7));
    auto out = TensorDesc({8, 8}, ACL_INT16, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodScalar, INPUT(self, scalar), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// int16 self x fp32 scalar -> castDtype=PromoteTypeScalar(int16,float)=float -> same-dtype fp32 Mod.
TEST_F(L2FmodScalarTest, mixed_int16_self_fp32_scalar_promote)
{
    auto self = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto scalar = ScalarDesc(3.0, ACL_FLOAT);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnFmodScalar, INPUT(self, scalar), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// int16 self x fp32 scalar -> calculate with fp32 Mod, then cast the result to int16.
TEST_F(L2FmodScalarTest, mixed_scalar_int16_fp32_to_int16)
{
    auto self = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto scalar = ScalarDesc(3.0, ACL_FLOAT);
    auto out = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodScalar, INPUT(self, scalar), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// out=int16 + 普通 fp32 self/scalar (非 mixed castDtype)：CheckPromoteTypeTensorScalar 的 int16 例外
// (aclnn_fmod_tensor.cpp:283-287) 须放行该派发。
TEST_F(L2FmodScalarTest, bug_a_scalar_out_int16_from_fp32)
{
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1000, 1000);
    auto scalar = ScalarDesc(3.0, ACL_FLOAT);
    auto out = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodScalar, INPUT(self, scalar), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// fp32 self + fp32 scalar + 更窄 fp16 out：在 castDtype (fp32) 计算，仅在调用方末步 Cast(->out) 窄化。
TEST_F(L2FmodScalarTest, bug_b_scalar_same_dtype_narrow_fp16_out)
{
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto scalar = ScalarDesc(3.0, ACL_FLOAT);
    auto out = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnFmodScalar, INPUT(self, scalar), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// AICPU fallback 路径 (FmodScalarAicpuPath，上游路)：double self/scalar -> computeType
// not an AiCore-compute dtype -> InitializeTensor+ConvertToTensor+BroadcastTo+FmodMainProcess.
// Previously uncovered.
TEST_F(L2FmodScalarTest, aicpu_fallback_double_scalar)
{
    auto self = TensorDesc({4, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto scalar = ScalarDesc(static_cast<double>(3.0)); // ScalarDesc(double) overload -> ACL_DOUBLE
    auto out = TensorDesc({4, 4}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnFmodScalar, INPUT(self, scalar), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// Empty-tensor early return (FmodScalarGetWorkspaceSizeCommon's self->IsEmpty() branch) —
// previously uncovered.
TEST_F(L2FmodScalarTest, empty_tensor_early_return)
{
    auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar = ScalarDesc(3.0, ACL_FLOAT);
    auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFmodScalar, INPUT(self, scalar), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}
