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
#include "../../../op_api/aclnn_log.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_inplace_log_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "inplace_log_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "inplace_log_test TearDown" << endl;
    }
};

TEST_F(l2_inplace_log_test, case_001_FLOAT)
{
    auto selfTensorDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(selfTensorDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_log_test, case_002_FLOAT16)
{
    auto selfTensorDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(selfTensorDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_log_test, case_003_COMPLEX64)
{
    auto selfTensorDesc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX64, ACL_FORMAT_NHWC).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(selfTensorDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_log_test, case_004_COMPLEX128)
{
    auto selfTensorDesc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX128, ACL_FORMAT_HWCN).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(selfTensorDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_log_test, case_005_DOUBLE)
{
    auto selfTensorDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_NDHWC).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(selfTensorDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor
TEST_F(l2_inplace_log_test, case_006_EMPTY)
{
    auto selfTensorDesc = TensorDesc({7, 0, 6}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(selfTensorDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_log_test, case_007_CONTINUOUS)
{
    auto selfTensorDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(selfTensorDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_log_test, ascend910B2_case_bf16)
{
    auto selfTensorDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(selfTensorDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_log_test, case_bf16_910)
{
    auto selfTensorDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(selfTensorDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dim超出限制
TEST_F(l2_inplace_log_test, case_008_MAX_DIM)
{
    auto selfTensorDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(selfTensorDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull
TEST_F(l2_inplace_log_test, case_009_NULLPTR)
{
    auto ut = OP_API_UT(aclnnInplaceLog, INPUT((aclTensor*)nullptr), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_inplace_log_test, case_010_DTYPE)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceLog, INPUT(tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}