/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_trace.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_trace_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_trace_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_trace_test TearDown" << endl;
    }
    void TearDown() override
    {
        op::SetPlatformNpuArch(NpuArch::DAV_2201);
    }
};

TEST_F(l2_trace_test, case_fp32_4x4)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrace, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trace_test, case_fp16_8x8)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrace, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trace_test, case_int32_3x5)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrace, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trace_test, case_bf16_4x4)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrace, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trace_test, case_nullptr)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrace, INPUT((aclTensor*)nullptr), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
