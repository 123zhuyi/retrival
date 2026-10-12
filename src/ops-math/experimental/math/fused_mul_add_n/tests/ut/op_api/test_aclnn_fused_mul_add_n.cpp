/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// aclnnFusedMulAddN op_api L2 UT
//   算子语义: y_i = x1_i * x3[0] + x2_i （x1/x2/x3/y 同 dtype；x1/x2/y 同 shape；
//             x3 单元素标量张量 ShapeSize==1；ND 格式；仅 ascend910b/DAV_2201）。
//
// 覆盖:
//   正常路径: 5 dtype (FLOAT / FLOAT16 / BFLOAT16 / INT32 / INT16) GetWorkspaceSize 成功。
//   参数校验失败路径:
//     · x1 nullptr               -> ACLNN_ERR_PARAM_NULLPTR
//     · dtype 不一致 (x2 != x1)  -> ACLNN_ERR_PARAM_INVALID
//     · 非法 dtype (DOUBLE)      -> ACLNN_ERR_PARAM_INVALID
//     · shape 不一致 (x1 != x2)  -> ACLNN_ERR_PARAM_INVALID
//     · x3 非单元素 (ShapeSize!=1)-> ACLNN_ERR_PARAM_INVALID
// ============================================================================

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "opdev/platform.h"
#include "../../../op_host/op_api/aclnn_fused_mul_add_n.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_fused_mul_add_n_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_fused_mul_add_n_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_fused_mul_add_n_test TearDown" << endl;
    }
};

// ---- 正常路径: 5 dtype GetWorkspaceSize 成功 ----

TEST_F(l2_fused_mul_add_n_test, case_normal_float32)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x3Desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fused_mul_add_n_test, case_normal_float16)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x3Desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fused_mul_add_n_test, case_normal_bfloat16)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x3Desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fused_mul_add_n_test, case_normal_int32)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto x2Desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto x3Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto yDesc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fused_mul_add_n_test, case_normal_int16)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({2, 3, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto x2Desc = TensorDesc({2, 3, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto x3Desc = TensorDesc({1}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto yDesc = TensorDesc({2, 3, 4}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// x3 形态 [1,1] 等价单元素标量（ShapeSize==1），仍应成功。
TEST_F(l2_fused_mul_add_n_test, case_normal_x3_shape_1x1)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x3Desc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ---- 参数校验失败路径 ----

// x1 为空指针 -> ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_fused_mul_add_n_test, case_error_x1_nullptr)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x2Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x3Desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT((aclTensor*)nullptr, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dtype 不一致 (x2 != x1) -> ACLNN_ERR_PARAM_INVALID
TEST_F(l2_fused_mul_add_n_test, case_error_dtype_mismatch)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x3Desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法 dtype (DOUBLE 不在支持集) -> ACLNN_ERR_PARAM_INVALID
TEST_F(l2_fused_mul_add_n_test, case_error_dtype_unsupported)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x3Desc = TensorDesc({1}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// shape 不一致 (x1 != x2) -> ACLNN_ERR_PARAM_INVALID
TEST_F(l2_fused_mul_add_n_test, case_error_shape_mismatch)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x3Desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x3 非单元素 (ShapeSize != 1) -> ACLNN_ERR_PARAM_INVALID
TEST_F(l2_fused_mul_add_n_test, case_error_x3_not_scalar)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x3Desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2); // ShapeSize=2，非单元素
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// y dtype 与 x1 不一致 -> ACLNN_ERR_PARAM_INVALID
TEST_F(l2_fused_mul_add_n_test, case_error_y_dtype_mismatch)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto x1Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x3Desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMulAddN, INPUT(x1Desc, x2Desc, x3Desc), OUTPUT(yDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
