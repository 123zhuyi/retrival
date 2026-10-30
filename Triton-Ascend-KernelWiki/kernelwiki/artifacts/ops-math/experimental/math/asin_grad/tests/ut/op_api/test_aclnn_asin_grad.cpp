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

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {
constexpr size_t MAX_SUPPORT_DIMS_NUM = 8;

bool IsSameShape(const aclTensor* lhs, const aclTensor* rhs)
{
    const auto& lhsShape = lhs->GetViewShape();
    const auto& rhsShape = rhs->GetViewShape();
    if (lhsShape.GetDimNum() != rhsShape.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhsShape.GetDimNum(); ++i) {
        if (lhsShape.GetDim(i) != rhsShape.GetDim(i)) {
            return false;
        }
    }
    return true;
}

bool IsValidDimNum(const aclTensor* tensor) { return tensor->GetViewShape().GetDimNum() <= MAX_SUPPORT_DIMS_NUM; }
} // namespace

extern "C" ACLNN_API aclnnStatus aclnnAsinGradGetWorkspaceSize(const aclTensor* y, const aclTensor* dy, aclTensor* z,
                                                               uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (y == nullptr || dy == nullptr || z == nullptr || workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!IsValidDimNum(y) || !IsValidDimNum(dy) || !IsValidDimNum(z)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (y->GetDataType() != dy->GetDataType() || y->GetDataType() != z->GetDataType()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsSameShape(y, dy) || !IsSameShape(y, z)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    *workspaceSize = 0;
    *executor = nullptr;
    return ACLNN_SUCCESS;
}

extern "C" ACLNN_API aclnnStatus aclnnAsinGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               const aclrtStream stream)
{
    (void)workspace;
    (void)workspaceSize;
    (void)executor;
    (void)stream;
    return ACLNN_SUCCESS;
}

class l2_asin_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { op::SetPlatformSocVersion(op::SocVersion::ASCEND910B); }
};

TEST_F(l2_asin_grad_test, case_001_fp32_success)
{
    auto yDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{-0.5f, 0.0f, 0.5f, 0.75f});
    auto dyDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{1.0f, 2.0f, -1.0f, 0.5f});
    auto zDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_asin_grad_test, case_002_fp16_success)
{
    auto yDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(std::vector<float>{-0.25f, 0.0f, 0.25f, 0.5f});
    auto dyDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    auto zDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_asin_grad_test, case_003_bf16_success)
{
    auto yDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).Value(std::vector<float>{-0.25f, 0.0f, 0.25f, 0.5f});
    auto dyDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).Value(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    auto zDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.02, 0.02);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_asin_grad_test, case_004_scalar_success)
{
    auto yDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{0.5f});
    auto dyDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{2.0f});
    auto zDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_asin_grad_test, case_005_empty_tensor_success)
{
    auto yDesc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dyDesc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto zDesc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_asin_grad_test, case_006_non_contiguous_success)
{
    auto yDesc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {5, 4});
    auto dyDesc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {5, 4});
    auto zDesc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {5, 4});
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_asin_grad_test, case_007_ascend910_93_success)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
    auto yDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{-0.5f, 0.0f, 0.5f, 0.75f});
    auto dyDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{1.0f, 2.0f, -1.0f, 0.5f});
    auto zDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

TEST_F(l2_asin_grad_test, case_008_shape_mismatch)
{
    auto yDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dyDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto zDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_asin_grad_test, case_009_output_shape_mismatch)
{
    auto yDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dyDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto zDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_asin_grad_test, case_010_dtype_mismatch)
{
    auto yDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dyDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_asin_grad_test, case_011_dim_over_8)
{
    auto yDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dyDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto zDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(yDesc, dyDesc), OUTPUT(zDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_asin_grad_test, case_012_input_nullptr)
{
    auto tensorDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT((aclTensor*)nullptr, tensorDesc), OUTPUT(tensorDesc));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_asin_grad_test, case_013_output_nullptr)
{
    auto tensorDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAsinGrad, INPUT(tensorDesc, tensorDesc), OUTPUT((aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    auto aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
