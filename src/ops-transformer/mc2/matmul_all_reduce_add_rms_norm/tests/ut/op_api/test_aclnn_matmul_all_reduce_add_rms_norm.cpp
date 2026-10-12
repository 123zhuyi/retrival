/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cfloat>

#include <array>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_matmul_all_reduce_add_rms_norm.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MatmulAllReduceAddRmsNormUT {

class L2MatmulAllReduceAddRmsNormTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MatmulAllReduceAddRmsNormTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MatmulAllReduceAddRmsNormTest TearDown" << endl;
    }
};

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormFirstApi)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001,
                        "test_group", "sum", 8, 1), OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNorm_empty_M)
{
    TensorDesc x1Desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001,
                        "test_group", "sum", 8, 1), OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormEmptyK)
{
    TensorDesc x1Desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001,
                        "test_group", "sum", 8, 1), OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormEmptyN)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001,
                        "test_group", "sum", 8, 1), OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormWrongK)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({30, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001,
                        "test_group", "sum", 8, 1), OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormWrongN)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001,
                        "test_group", "sum", 8, 1), OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // MatmulAllReduceAddRmsNormUT