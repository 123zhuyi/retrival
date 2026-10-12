/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_isclose.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_isclose_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "isclose_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "isclose_test TearDown" << std::endl;
    }
};

TEST_F(l2_isclose_test, l2_isclose_test_case_f16)
{
    auto self_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    double rtol = 1.0;
    double atol = 1.0;
    bool equal_nan = false;
    auto out_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnIsClose, INPUT(self_desc, other_desc, rtol, atol, equal_nan), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isclose_test, l2_isclose_test_case_f32)
{
    auto self_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    double rtol = 1.0;
    double atol = 1.0;
    bool equal_nan = false;
    auto out_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnIsClose, INPUT(self_desc, other_desc, rtol, atol, equal_nan), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isclose_test, l2_isclose_test_case_f32_nan)
{
    auto self_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    double rtol = 1.0;
    double atol = 1.0;
    bool equal_nan = true;
    auto out_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnIsClose, INPUT(self_desc, other_desc, rtol, atol, equal_nan), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isclose_test, l2_isclose_test_case_int32)
{
    auto self_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND);
    double rtol = 1.0;
    double atol = 1.0;
    bool equal_nan = false;
    auto out_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnIsClose, INPUT(self_desc, other_desc, rtol, atol, equal_nan), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isclose_test, l2_isclose_test_case_empty)
{
    auto self_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND);
    double rtol = 1.0;
    double atol = 1.0;
    bool equal_nan = false;
    auto out_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnIsClose, INPUT(self_desc, other_desc, rtol, atol, equal_nan), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isclose_test, l2_isclose_test_case_null)
{
    double rtol = 1.0;
    double atol = 1.0;
    bool equal_nan = false;
    uint64_t workspace_size = 0;

    auto self_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto other_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto out_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut_1 = OP_API_UT(aclnnIsClose, INPUT(nullptr, other_desc_1, rtol, atol, equal_nan), OUTPUT(out_desc_1));

    aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnIsClose, INPUT(self_desc_1, nullptr, rtol, atol, equal_nan), OUTPUT(out_desc_1));

    aclnnStatus aclRet_2 = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_2, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_3 = OP_API_UT(aclnnIsClose, INPUT(self_desc_1, other_desc_1, rtol, atol, equal_nan), OUTPUT(nullptr));

    aclnnStatus aclRet_3 = ut_3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_3, ACLNN_ERR_PARAM_NULLPTR);
}

// largeDim
TEST_F(l2_isclose_test, l2_isclose_test_case_9d)
{
    auto self_desc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    auto other_desc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    double rtol = 1.0;
    double atol = 1.0;
    bool equal_nan = false;
    auto out_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnIsClose, INPUT(self_desc, other_desc, rtol, atol, equal_nan), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// broadcast
TEST_F(l2_isclose_test, l2_isclose_test_broadcast1)
{
    auto self_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    double rtol = 1.0;
    double atol = 1.0;
    bool equal_nan = false;
    auto out_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsClose, INPUT(self_desc, other_desc, rtol, atol, equal_nan), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// broadcast
TEST_F(l2_isclose_test, l2_isclose_test_broadcast2)
{
    auto self_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    double rtol = 1.0;
    double atol = 1.0;
    bool equal_nan = false;
    auto out_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsClose, INPUT(self_desc, other_desc, rtol, atol, equal_nan), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
