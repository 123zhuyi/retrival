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
#include "../../../op_api/aclnn_log2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_log2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "log2_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "log2_test TearDown" << std::endl;
    }
};

TEST_F(l2_log2_test, case_001_FLOAT)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnLog2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_log2_test, case_002_FLOAT16)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnLog2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_log2_test, case_013_ND)
{
    auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLog2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor
TEST_F(l2_log2_test, case_14_EMPTY)
{
    auto self_tensor_desc = TensorDesc({7, 0, 6}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnLog2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_log2_test, case_015_CONTINUOUS)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLog2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim超出限制
TEST_F(l2_log2_test, case_016_MAX_DIM)
{
    auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLog2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull
TEST_F(l2_log2_test, case_017_NULLPTR)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog2, INPUT(nullptr), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnLog2, INPUT(tensor_desc), OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_log2_test, case_018_DTYPE)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLog2, INPUT(tensor_desc), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat
TEST_F(l2_log2_test, case_019_FORMAT)
{
    auto self_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog2, INPUT(self_tensor_desc), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto ut_2 = OP_API_UT(aclnnLog2, INPUT(tensor_desc), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}