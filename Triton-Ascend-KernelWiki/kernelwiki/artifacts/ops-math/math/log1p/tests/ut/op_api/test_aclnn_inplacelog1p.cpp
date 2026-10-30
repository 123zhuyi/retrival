/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_log1p.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_inplacelog1p_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Inplace Log1p Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "Inplace Log1p Test TearDown" << std::endl; }
};


TEST_F(l2_inplacelog1p_test, case_shape1D)
{
    auto tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceLog1p, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplacelog1p_test, case_shape_2D)
{
    auto tensor_desc = TensorDesc({3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceLog1p, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplacelog1p_test, case_shape_3D)
{
    auto tensor_desc = TensorDesc({3,3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceLog1p, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplacelog1p_test, case_shape_4D)
{
    auto tensor_desc = TensorDesc({3,3,3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceLog1p, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplacelog1p_test, case_shape_5D)
{
    auto tensor_desc = TensorDesc({3,3,3,3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceLog1p, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplacelog1p_test, case_shape_8D)
{
    auto tensor_desc = TensorDesc({3,3,3,3,3,3,3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2)
        .Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceLog1p, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplacelog1p_test, case_invalid_shape_9D)
{
    auto tensor_desc = TensorDesc({3,3,3,3,3,3,3,3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnInplaceLog1p, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
