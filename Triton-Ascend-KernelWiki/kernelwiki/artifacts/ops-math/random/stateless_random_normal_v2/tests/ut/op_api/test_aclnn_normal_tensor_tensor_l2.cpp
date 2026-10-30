/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "random/stateless_random_normal_v2/op_api/aclnn_normal_out.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_normal_tensor_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_normal_tensor_tensor SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "l2_normal_tensor_tensor TearDown" << std::endl; }
};

// 入参总元素个数不一致的场景
TEST_F(l2_normal_tensor_tensor_test, case_float_float64_ND_011)
{
    auto meanDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto stdDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc(meanDesc);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 入参shape不一致的场景
TEST_F(l2_normal_tensor_tensor_test, case_float_float64_ND_012)
{
    auto meanDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto stdDesc = TensorDesc({1, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc(meanDesc);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dim维度大于8维的异常场景
TEST_F(l2_normal_tensor_tensor_test, case_9dim_ND_015)
{
    auto meanDesc = TensorDesc({2, 3, 2, 2, 3, 2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto stdDesc = TensorDesc({2, 3, 2, 2, 3, 2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc(meanDesc);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mean为nullptr的异常场景
TEST_F(l2_normal_tensor_tensor_test, case_mean_nullptr_ND_016)
{
    auto meanDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto stdDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc(meanDesc);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT((aclTensor*)nullptr, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// std为nullptr的异常场景
TEST_F(l2_normal_tensor_tensor_test, case_std_nullptr_ND_017)
{
    auto meanDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto stdDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc(meanDesc);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, (aclTensor*)nullptr, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out为nullptr的异常场景
TEST_F(l2_normal_tensor_tensor_test, case_out_nullptr_ND_018)
{
    auto meanDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto stdDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
    auto outDesc = TensorDesc(meanDesc);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, stdDesc, seed, offset), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ======================== 正常场景新增 ========================

// 正常场景：float类型，走StatelessRandomNormalV3路径
TEST_F(l2_normal_tensor_tensor_test, case_float_ND_normal)
{
    auto meanDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：float16类型，走StatelessRandomNormalV3路径
TEST_F(l2_normal_tensor_tensor_test, case_float16_ND_normal)
{
    auto meanDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：double类型，走StatelessRandomNormalV2 fallback路径
TEST_F(l2_normal_tensor_tensor_test, case_double_ND_normal)
{
    auto meanDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 5);
    auto stdDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：mean和std类型不一致（需要PromoteType），float16 + float -> float
TEST_F(l2_normal_tensor_tensor_test, case_mixed_dtype_ND_normal)
{
    auto meanDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：mean需要向std广播。UT默认soc为ASCEND910B(DAV_2201)，走A2/A3分支。
// stateLessOut取mean的shape{1,3}，与out的{2,3}不一致，Mul不能复用其内存，走非inplace回退；
// Add侧则覆盖meanCast{1,3}广播进mulOutCast{2,3}
TEST_F(l2_normal_tensor_tensor_test, case_broadcast_mean_ND_normal)
{
    auto meanDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    auto stdDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：std需要向mean广播。stateLessOut取mean的shape{2,3}，与out一致且dtype相同，
// Mul可复用其内存走MulInplace
TEST_F(l2_normal_tensor_tensor_test, case_broadcast_std_ND_normal)
{
    auto meanDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    auto stdDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalTensorTensor, INPUT(meanDesc, stdDesc, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
