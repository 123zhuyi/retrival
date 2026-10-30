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
#include "../../../op_api/aclnn_roll.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;

class test_aclnn_roll : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }
};

TEST_F(test_aclnn_roll, case_basic_float)
{
    auto xDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{0, 1, 2, 3, 4, 5});
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0, 0.0);
    auto shifts = IntArrayDesc(std::vector<int64_t>{1});
    auto dims = IntArrayDesc(std::vector<int64_t>{1});

    auto ut = OP_API_UT(aclnnRoll, INPUT(xDesc, shifts, dims), OUTPUT(yDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(test_aclnn_roll, case_dims_empty)
{
    auto xDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto shifts = IntArrayDesc(std::vector<int64_t>{2});
    auto dims = IntArrayDesc(std::vector<int64_t>{});

    auto ut = OP_API_UT(aclnnRoll, INPUT(xDesc, shifts, dims), OUTPUT(yDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(test_aclnn_roll, case_invalid_dtype)
{
    auto xDesc = TensorDesc({4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto shifts = IntArrayDesc(std::vector<int64_t>{1});
    auto dims = IntArrayDesc(std::vector<int64_t>{0});

    auto ut = OP_API_UT(aclnnRoll, INPUT(xDesc, shifts, dims), OUTPUT(yDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_roll, case_invalid_dims_range)
{
    auto xDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto shifts = IntArrayDesc(std::vector<int64_t>{1});
    auto dims = IntArrayDesc(std::vector<int64_t>{2});

    auto ut = OP_API_UT(aclnnRoll, INPUT(xDesc, shifts, dims), OUTPUT(yDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
