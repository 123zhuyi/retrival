/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_letensor.cpp
 * \brief
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "math/less_equal/op_api/aclnn_le_tensor.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

class l2_le_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "le_tensor_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "le_tensor_test TearDown" << endl; }
};

// 入参无法broadcast的场景
TEST_F(l2_le_tensor_test, test_le_tensor_input_cannot_broadcast)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_tensor_test, test_le_tensor_error_shape)
{
    auto tensor_self = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_tensor_test, test_le_tensor_notsupport_complex64)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_tensor_test, test_le_tensor_notsupport_complex128)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 入参self&other都为空tensor的场景
TEST_F(l2_le_tensor_test, test_le_tensor_emptytensor)
{
    auto tensor_self =
        TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 入参为self为null场景
TEST_F(l2_le_tensor_test, test_le_tensor_nullptr)
{
    auto tensor_self = nullptr;
    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_le_tensor_test, test_le_tensor_regbase_bool_tensor_bool_scalar)
{
    auto curSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    auto tensor_self =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, true, false, true, false});
    auto tensor_other = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(curSocVersion);
    }
}

TEST_F(l2_le_tensor_test, test_le_tensor_regbase_uint64_tensor_bool_scalar)
{
    auto curSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    auto tensor_self =
        TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{1, 2, 3, 4, 5, 6});
    auto tensor_other = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(curSocVersion);
    }
}

TEST_F(l2_le_tensor_test, test_le_tensor_regbase_bf16_tensor_float_scalar)
{
    auto curSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    auto tensor_self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_other = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(curSocVersion);
    }
}
