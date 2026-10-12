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

// The following cases cover the int16 same-dtype lane and aclnn-side mixed-dtype normalization. Inputs are cast
// to their common calculation dtype before the two-argument Mod call; only the result is cast to out.

// K2: int16 same-dtype (self==other==out==int16) -> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST's int16
// lane (the dtype newly added to the L2 dtype support list, aclnn_fmod_tensor.cpp:38).
TEST_F(L2FmodTensorTest, int16_same_dtype)
{
    auto self = TensorDesc({8, 8}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto other = TensorDesc({8, 8}, ACL_INT16, ACL_FORMAT_ND).ValueRange(1, 7);
    auto out = TensorDesc({8, 8}, ACL_INT16, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// int16 self x fp32 other -> both inputs normalize to fp32 -> same-dtype Mod -> fp32 out.
TEST_F(L2FmodTensorTest, mixed_int16_fp32_promote_to_fp32)
{
    auto self = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto other = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// int16 self x fp32 other -> calculate with fp32 Mod, then cast the result to int16.
TEST_F(L2FmodTensorTest, mixed_int16_fp32_to_int16)
{
    auto self = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto other = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// DAV_2201 has no direct AiCore int16->bf16 Cast. aclnn must build int16->fp32->bf16 before same-dtype Mod.
TEST_F(L2FmodTensorTest, mixed_int16_bf16_uses_fp32_cast_bridge)
{
    auto self = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto other = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// Covers both bridge directions in one graph: other int16->fp32->bf16, then Mod output bf16->fp32->int16.
TEST_F(L2FmodTensorTest, mixed_bf16_int16_to_int16_uses_both_cast_bridges)
{
    auto self = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto other = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// out=int16 + 非 int16 same-dtype promote (fp32 self/other)：promoteType(fp32,fp32)=fp32 总能 cast 到
// int16；真正回归面是 int16-exception 分支 (:236) 不能误把它经非 int16 门拒掉。确认 int16-out 例外是 additive。
TEST_F(L2FmodTensorTest, bug_a_out_int16_from_fp32_same_dtype)
{
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1000, 1000);
    auto other = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// same-dtype (fp32 self/other) + 更窄 out (fp16)：self/other remain at promoteType (fp32)，仅末步 Cast(->out) 窄化。
// 两者数学结果相同，本用例钉住修正后的 compute-dtype 路由仍能成功建图。
TEST_F(L2FmodTensorTest, bug_b_same_dtype_narrow_fp16_out)
{
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto other = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// AICPU fallback path (ExecFmodTensorAicpuPath, upstream verbatim, PORT_DESIGN §3.3 "AICPU else 物化
// 分支...零回归"): double same-dtype -> computeType=double is not an IsAiCoreComputeDtype dtype ->
// InitializeTensor+BroadcastTensor+FmodMainProcess，本次增强不触及。此前本文件无 op_api UT 覆盖
// (既有 + 新增用例都用 AiCore dtype)。
TEST_F(L2FmodTensorTest, aicpu_fallback_double_same_dtype)
{
    auto self = TensorDesc({4, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto other = TensorDesc({4, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({4, 4}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// AICPU fallback path WITH genuine broadcast (other shape {1,4} broadcasts to self/out shape {4,4}):
// exercises BroadcastTensor's real-broadcast branch (l0op::BroadcastTo + Contiguous + ReFormat),
// which aicpu_fallback_double_same_dtype above does not reach (same-shape there -> broadcast a no-op).
TEST_F(L2FmodTensorTest, aicpu_fallback_broadcast_int64)
{
    auto self = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto other = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// AICPU fallback path with a 0-dim (scalar) tensor: GetTensorDimNum(out)==0 -> needUnsqueeze==true in
// RunFmodProcessAndRelease -> FmodMainProcess's SqueezeNd branch, previously uncovered (both AICPU
// tests above use rank-2 tensors).
TEST_F(L2FmodTensorTest, aicpu_fallback_0dim_int8)
{
    auto self = TensorDesc({}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto other = TensorDesc({}, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out = TensorDesc({}, ACL_INT8, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// Empty-tensor early return (ExecFmodTensorGetWorkspaceSize's self->IsEmpty()||other->IsEmpty()
// branch) — previously uncovered.
TEST_F(L2FmodTensorTest, empty_tensor_early_return)
{
    auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFmodTensor, INPUT(self, other), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}
