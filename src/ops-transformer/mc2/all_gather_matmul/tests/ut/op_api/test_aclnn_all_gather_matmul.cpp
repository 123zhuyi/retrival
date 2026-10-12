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
#include "../../../op_api/aclnn_all_gather_matmul.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"

using namespace op;
using namespace std;

namespace AllGatherMatmulUT {

class L2AllGatherMatmulTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AllGatherMatmulTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AllGatherMatmulTest TearDown" << endl;
    }
};

TEST_F(L2AllGatherMatmulTest, TestAllGatherFirstApi)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1Desc, x2Desc, bias, "test_all_gather_group", 0, 8, 1),
                        OUTPUT(outDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllGatherMatmulTest, TestAllGatherFirstApi2)
{
    TensorDesc x1Desc = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1Desc, x2Desc, bias, "test_all_gather_group", 0, 8, 1),
                        OUTPUT(outDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllGatherMatmulTest, TestAllGatherFirstApi3)
{
    TensorDesc x1Desc = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1Desc, x2Desc, bias, "test_all_gather_group", 0, 8, 1),
                        OUTPUT(outDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllGatherMatmulTest, TestAllGatherFirstApi4)
{
    TensorDesc x1Desc = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1Desc, x2Desc, bias, "test_all_gather_group", 0, 8, 1),
                        OUTPUT(outDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllGatherMatmulTest, TestAllGatherFirstApi5)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1Desc, x2Desc, bias, "test_all_gather_group", 0, 8, 1),
                        OUTPUT(outDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllGatherMatmulTest, TestAllGatherFirstApiInputFalse)
{
    TensorDesc x1Desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1Desc, x2Desc, bias, "test_all_gather_group", 0, 8, 1),
                        OUTPUT(outDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllGatherMatmulTest, TestAllGatherFirstApiGatherOutFalse)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1Desc, x2Desc, bias, "test_all_gather_group", 0, 8, 1),
                        OUTPUT(outDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllGatherMatmulTest, TestAllGatherFirstApiX2NonContiguousWithoutTranspose)
{
    // 测试场景：非转置的连续x2输入，应返回校验失败
    TensorDesc x1Desc = TensorDesc({128, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    // 设置x2非转置非连续步长，假设每行之间间隔10个元素[64 + 10, 1]，实际内存存储storageShape为[512, 64 + 10]
    TensorDesc x2Desc = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND, {512 + 10, 1});
    TensorDesc outDesc = TensorDesc({128, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({128, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAllGatherMatmul, 
                        INPUT(x1Desc, x2Desc, nullptr, "test_all_gather_group", 0, 8, 1),
                        OUTPUT(outDesc, gatherOutDesc));

    // 预期：由于 x2 是非连续的（非转置），应该返回 ACLNN_ERR_PARAM_INVALID
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // AllGatherMatmulUT