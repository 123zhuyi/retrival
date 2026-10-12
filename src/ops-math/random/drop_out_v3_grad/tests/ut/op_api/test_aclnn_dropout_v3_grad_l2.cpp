/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_dropout_v3_grad_l2.cpp
 * \brief aclnnDropoutV3Grad L2 UT —— 成功路径 + 各类非法参数校验。
 *        接口：aclnnDropoutV3GradGetWorkspaceSize(gradY, mask, double scale, gradX, ...)
 *        校验：空指针 / gradY dtype / mask dtype / gradX 不可 cast / shape 不等 / mask size 不符 / 超 8 维 / 空
 * tensor。
 */

#include "random/drop_out_v3_grad/op_api/aclnn_dropout_v3_grad.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_dropout_v3_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "dropout_v3_grad_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "dropout_v3_grad_test TearDown" << endl; }
};

// mask 元素个数 = align(gradSize, 128) / 8
static inline int64_t ComputeMaskSize(const vector<int64_t>& shape)
{
    int64_t num = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        num *= shape[i];
    }
    return (num + 128 - 1) / 128 * 128 / 8;
}

// ===================== 成功用例 =====================

// case 1: float32 / uint8，scale 常规值
TEST_F(l2_dropout_v3_grad_test, case_success_float32)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto gradY = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradX = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({ComputeMaskSize(shape)}, ACL_UINT8, ACL_FORMAT_ND);
    double scale = 1.0 / 0.7;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, mask, scale), OUTPUT(gradX));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// case 2: float16 / uint8
TEST_F(l2_dropout_v3_grad_test, case_success_float16)
{
    const vector<int64_t> shape = {4, 32};
    auto gradY = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradX = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mask = TensorDesc({ComputeMaskSize(shape)}, ACL_UINT8, ACL_FORMAT_ND);
    double scale = 2.0;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, mask, scale), OUTPUT(gradX));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// case 3: scale 边界值（0 和 1，对标 torch 不做范围校验，均应成功）
TEST_F(l2_dropout_v3_grad_test, case_success_scale_boundary)
{
    const vector<int64_t> shape = {128};
    auto gradY = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradX = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({ComputeMaskSize(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    for (double scale : {0.0, 1.0}) {
        auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, mask, scale), OUTPUT(gradX));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

// case 4: 空 tensor，workspace=0，返回成功
TEST_F(l2_dropout_v3_grad_test, case_success_empty_tensor)
{
    const vector<int64_t> shape = {0};
    auto gradY = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradX = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({0}, ACL_UINT8, ACL_FORMAT_ND);
    double scale = 1.5;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, mask, scale), OUTPUT(gradX));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ===================== 非法用例 =====================

// case 5: gradY 空指针 → ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_dropout_v3_grad_test, case_fail_null_grad_y)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto gradX = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({ComputeMaskSize(shape)}, ACL_UINT8, ACL_FORMAT_ND);
    double scale = 1.5;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(nullptr, mask, scale), OUTPUT(gradX));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// case 6: mask 空指针 → ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_dropout_v3_grad_test, case_fail_null_mask)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto gradY = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradX = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    double scale = 1.5;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, nullptr, scale), OUTPUT(gradX));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// case 7: gradX 空指针 → ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_dropout_v3_grad_test, case_fail_null_grad_x)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto gradY = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({ComputeMaskSize(shape)}, ACL_UINT8, ACL_FORMAT_ND);
    double scale = 1.5;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, mask, scale), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// case 8: gradY dtype 非法（INT32，不在支持列表）→ ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_grad_test, case_fail_grad_y_dtype)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto gradY = TensorDesc(shape, ACL_INT32, ACL_FORMAT_ND);
    auto gradX = TensorDesc(shape, ACL_INT32, ACL_FORMAT_ND);
    auto mask = TensorDesc({ComputeMaskSize(shape)}, ACL_UINT8, ACL_FORMAT_ND);
    double scale = 1.5;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, mask, scale), OUTPUT(gradX));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 9: mask dtype 非法（FLOAT，对外仅支持 uint8）→ ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_grad_test, case_fail_mask_dtype)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto gradY = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradX = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({ComputeMaskSize(shape)}, ACL_FLOAT, ACL_FORMAT_ND);
    double scale = 1.5;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, mask, scale), OUTPUT(gradX));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 10: gradY / gradX shape 不一致 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_grad_test, case_fail_shape_not_equal)
{
    const vector<int64_t> shape = {2, 1, 3};
    auto gradY = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradX = TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({ComputeMaskSize(shape)}, ACL_UINT8, ACL_FORMAT_ND);
    double scale = 1.5;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, mask, scale), OUTPUT(gradX));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 11: mask size 不符 align(gradSize,128)/8 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_grad_test, case_fail_mask_size)
{
    const vector<int64_t> shape = {2, 1, 3};
    auto gradY = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradX = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({8}, ACL_UINT8, ACL_FORMAT_ND); // 正确应为 16
    double scale = 1.5;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, mask, scale), OUTPUT(gradX));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 12: gradY 维度超过 8 维 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_grad_test, case_fail_dim_exceed)
{
    const vector<int64_t> shape = {2, 9, 3, 4, 5, 6, 7, 8, 10};
    auto gradY = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradX = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({ComputeMaskSize(shape)}, ACL_UINT8, ACL_FORMAT_ND);
    double scale = 1.5;

    auto ut = OP_API_UT(aclnnDropoutV3Grad, INPUT(gradY, mask, scale), OUTPUT(gradX));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
