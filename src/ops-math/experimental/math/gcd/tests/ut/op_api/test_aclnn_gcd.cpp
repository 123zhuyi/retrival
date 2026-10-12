/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_gcd.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;

class l2_gcd_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Gcd Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "Gcd Test TearDown" << std::endl; }
};

static void ExpectWorkspaceStatus(const TensorDesc& selfDesc, const TensorDesc& otherDesc, const TensorDesc& outDesc,
                                  aclnnStatus expected)
{
    auto ut = OP_API_UT(aclnnGcd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, expected);
}

TEST_F(l2_gcd_test, float32_broadcast_success)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-15.75, 15.75);
    auto otherDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-9.5, 9.5);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(selfDesc, otherDesc, outDesc, ACL_SUCCESS);
}

TEST_F(l2_gcd_test, float16_bfloat16_success)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto fp16Self = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-7.75, 7.75);
    auto fp16Other = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-5.5, 5.5);
    auto fp16Out = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    ExpectWorkspaceStatus(fp16Self, fp16Other, fp16Out, ACL_SUCCESS);

    auto bf16Self = TensorDesc({2, 1, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-31.5, 31.5);
    auto bf16Other = TensorDesc({1, 4, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-17.5, 17.5);
    auto bf16Out = TensorDesc({2, 4, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.004, 0.004);
    ExpectWorkspaceStatus(bf16Self, bf16Other, bf16Out, ACL_SUCCESS);
}

TEST_F(l2_gcd_test, integer_task_dtypes_success)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto int8Self = TensorDesc({257}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-127, 127);
    auto int8Other = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-31, 31);
    auto int8Out = TensorDesc({257}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(int8Self, int8Other, int8Out, ACL_SUCCESS);

    auto uint8Self = TensorDesc({17, 31}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 255);
    auto uint8Other = TensorDesc({17, 1}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 255);
    auto uint8Out = TensorDesc({17, 31}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(uint8Self, uint8Other, uint8Out, ACL_SUCCESS);

    auto int16Self = TensorDesc({8, 1, 9}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-32768, 32767);
    auto int16Other = TensorDesc({1, 7, 9}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-32767, 32767);
    auto int16Out = TensorDesc({8, 7, 9}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(int16Self, int16Other, int16Out, ACL_SUCCESS);
}

TEST_F(l2_gcd_test, integer_promote_and_cast_success)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto int32Self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto int32Other = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto int32Out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(int32Self, int32Other, int32Out, ACL_SUCCESS);

    auto int64Self = TensorDesc({2, 1, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto int16Other = TensorDesc({1, 4, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto int64Out = TensorDesc({2, 4, 3}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(int64Self, int16Other, int64Out, ACL_SUCCESS);

    auto uint8Self = TensorDesc({5, 1}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 32);
    auto uint8Int32Other = TensorDesc({1, 7}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-32, 32);
    auto int16Out = TensorDesc({5, 7}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(uint8Self, uint8Int32Other, int16Out, ACL_SUCCESS);
}

TEST_F(l2_gcd_test, mixed_float_integer_promote_success)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto fpSelf = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto int16Other = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto fpOut = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(fpSelf, int16Other, fpOut, ACL_SUCCESS);

    auto int8Self = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto fp32Other = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto int8Out = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(int8Self, fp32Other, int8Out, ACL_SUCCESS);

    auto bf16Self = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto uint8Other = TensorDesc({3, 5}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10);
    auto bf16Out = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.004, 0.004);
    ExpectWorkspaceStatus(bf16Self, uint8Other, bf16Out, ACL_SUCCESS);
}

TEST_F(l2_gcd_test, rejects_bad_shape_and_rank)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto otherDesc = TensorDesc({4, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(selfDesc, otherDesc, outDesc, ACLNN_ERR_PARAM_INVALID);

    auto rank9Self = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto rank9Other = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto rank9Out = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(rank9Self, rank9Other, rank9Out, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, rejects_unsupported_input_and_output_dtypes)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto floatSelf = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto doubleOther = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto floatOther = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto floatOut = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto doubleOut = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    ExpectWorkspaceStatus(floatSelf, doubleOther, floatOut, ACLNN_ERR_PARAM_INVALID);
    ExpectWorkspaceStatus(floatSelf, floatOther, doubleOut, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gcd_test, alternate_platform_integer_promote_success)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-128, 128);
    auto otherDesc = TensorDesc({2, 2, 1, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-128, 128);
    auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(selfDesc, otherDesc, outDesc, ACL_SUCCESS);
}

TEST_F(l2_gcd_test, alternate_platform_float_success)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto otherDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    ExpectWorkspaceStatus(selfDesc, otherDesc, outDesc, ACL_SUCCESS);
}
