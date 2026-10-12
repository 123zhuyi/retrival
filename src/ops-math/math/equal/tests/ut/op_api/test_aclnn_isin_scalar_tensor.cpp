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
#include "../../../op_api/aclnn_isin.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_isin_scalar_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_isin_scalar_tensor_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_isin_scalar_tensor_test TearDown" << std::endl;
    }
};

TEST_F(l2_isin_scalar_tensor_test, case_nullptr)
{
    auto element = ScalarDesc(1.0f);
    auto testElements = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(nullptr, testElements, false, false), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, nullptr, false, false), OUTPUT(outDesc));
    workspaceSize = 0;
    aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(nullptr));
    workspaceSize = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_isin_scalar_tensor_test, case_null_tensor)
{
    auto element = ScalarDesc(1.0f);
    auto testElements = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, false), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isin_scalar_tensor_test, case_null_tensor_invert)
{
    auto element = ScalarDesc(1.0f);
    auto testElements = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_isin_scalar_tensor_test, case_error_shape)
{
    auto element = ScalarDesc(-1.0f);
    auto testElements = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_scalar_tensor_test, case_error_dtype)
{
    auto element = ScalarDesc(static_cast<uint32_t>(1));
    auto testElements = TensorDesc({2, 2, 2, 2}, ACL_UINT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_scalar_tensor_test, case_error_out_dtype)
{
    auto element = ScalarDesc(-1.0f);
    auto testElements = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isin_scalar_tensor_test, case_can_not_promote_type)
{
    auto element = ScalarDesc(-1.0f);
    auto testElements = TensorDesc({2, 2, 2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInScalarTensor, INPUT(element, testElements, false, true), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}