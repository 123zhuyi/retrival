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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_grouped_mat_mul_allto_allv.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace GroupedMatMulAlltoAllvUT {
class L2GroupedMatMulAlltoAllvTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
        cout << "L2GroupedMatMulAlltoAllvTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2GroupedMatMulAlltoAllvTest TearDown" << endl;
    }
};

TEST_F(L2GroupedMatMulAlltoAllvTest, Test)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto sendCounts = nullptr;
    auto recvCounts = nullptr;
    bool transGmmWeight = false;
    bool transMmWeight = false;
    int64_t epWorldSize = 8;
    TensorDesc yDesc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnGroupedMatMulAlltoAllv,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight),
                  OUTPUT(yDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAlltoAllvTest, TestGroupNullptr)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto sendCounts = nullptr;
    auto recvCounts = nullptr;
    bool transGmmWeight = false;
    bool transMmWeight = false;
    int64_t epWorldSize = 8;
    TensorDesc yDesc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGroupedMatMulAlltoAllv,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, nullptr, epWorldSize, sendCounts,
                              recvCounts, transGmmWeight, transMmWeight),
                        OUTPUT(yDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2GroupedMatMulAlltoAllvTest, TestGroupInvalid)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sendCounts = nullptr;
    auto recvCounts = nullptr;
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool allGatherFlag = true;
    int64_t epWorldSize = 8;
    TensorDesc yDesc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGroupedMatMulAlltoAllv,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr,
                              "test_grouped_mat_mul_allto_allv_ep_group_test_grouped_mat_mul_allto_allv_ep_group_"
                              "test_grouped_mat_mul_allto_allv_ep_group_test_grouped_mat_mul_allto_allv_ep_group_"
                              "test_grouped_mat_mul_allto_allv_ep_group_test_grouped_mat_mul_allto_allv_ep_group_"
                              "test_grouped_mat_mul_allto_allv_ep_group_test_grouped_mat_mul_allto_allv_ep_group_"
                              "test_grouped_mat_mul_allto_allv_ep_group_test_grouped_mat_mul_allto_allv_ep_group",
                              epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight),
                        OUTPUT(yDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAlltoAllvTest, TestMmxInvalid)
{  // 不加这个覆盖率过不去 加了好像有问题
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX = TensorDesc({1024, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sendCounts = nullptr;
    auto recvCounts = nullptr;
    bool transGmmWeight = false;
    bool transMmWeight = false;
    int64_t epWorldSize = 8;
    TensorDesc yDesc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnGroupedMatMulAlltoAllv,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, mmX, nullptr, "test_grouped_mat_mul_allto_allv_ep_group",
                        epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight),
                  OUTPUT(yDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
} // GroupedMatMulAlltoAllvUT