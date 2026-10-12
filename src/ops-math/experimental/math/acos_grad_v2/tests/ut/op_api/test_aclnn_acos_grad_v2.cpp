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
#include "../../../op_api/aclnn_acos_grad_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;

class L2AcosGradV2Test : public testing::Test {};

// 正常用例：FP32 同 shape（精度标杆 atol/rtol = 1e-4，与 acos 保持一致）
TEST_F(L2AcosGradV2Test, fp32_same_shape)
{
    auto y = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dy = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto z = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAcosGradV2, INPUT(y, dy), OUTPUT(z));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// 正常用例：FP16
TEST_F(L2AcosGradV2Test, fp16_same_shape)
{
    auto y = TensorDesc({2, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dy = TensorDesc({2, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto z = TensorDesc({2, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnAcosGradV2, INPUT(y, dy), OUTPUT(z));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// 正常用例：BF16
TEST_F(L2AcosGradV2Test, bf16_same_shape)
{
    auto y = TensorDesc({4, 8, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dy = TensorDesc({4, 8, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto z = TensorDesc({4, 8, 16}, ACL_BF16, ACL_FORMAT_ND).Precision(0.004, 0.004);
    auto ut = OP_API_UT(aclnnAcosGradV2, INPUT(y, dy), OUTPUT(z));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// 异常用例：dtype 不一致（y 为 FP32，dy 为 FP16）
TEST_F(L2AcosGradV2Test, invalid_dtype_mismatch)
{
    auto y = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dy = TensorDesc({10, 10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto z = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAcosGradV2, INPUT(y, dy), OUTPUT(z));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// 异常用例：shape 不一致
TEST_F(L2AcosGradV2Test, invalid_shape_mismatch)
{
    auto y = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dy = TensorDesc({20, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto z = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAcosGradV2, INPUT(y, dy), OUTPUT(z));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// 异常用例：不支持 INT32
TEST_F(L2AcosGradV2Test, invalid_dtype_int32)
{
    auto y = TensorDesc({10, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dy = TensorDesc({10, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto z = TensorDesc({10, 10}, ACL_INT32, ACL_FORMAT_ND).Precision(0, 0);
    auto ut = OP_API_UT(aclnnAcosGradV2, INPUT(y, dy), OUTPUT(z));
    uint64_t workspaceSize = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// 正常用例：大 shape 多核
TEST_F(L2AcosGradV2Test, fp32_large_shape)
{
    auto y = TensorDesc({1024, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dy = TensorDesc({1024, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto z = TensorDesc({1024, 1024}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAcosGradV2, INPUT(y, dy), OUTPUT(z));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}
