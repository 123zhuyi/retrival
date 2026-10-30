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
 * \file test_aclnn_inplace_lescalar.cpp
 * \brief
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "math/less_equal/op_api/aclnn_le_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class inplaceLeScalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "inplaceLeScalar_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "inplaceLeScalar_test TearDown" << endl;
    }
};

TEST_F(inplaceLeScalar_test, case_nullptr_self)
{
    auto tensor_desc = nullptr;
    auto scalar_desc = ScalarDesc(0.5f);

    auto ut = OP_API_UT(aclnnInplaceLeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(inplaceLeScalar_test, case_nullptr_other)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = nullptr;

    auto ut = OP_API_UT(aclnnInplaceLeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(inplaceLeScalar_test, case_dtype_invalid_self)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(8.0f);

    auto ut = OP_API_UT(aclnnInplaceLeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(inplaceLeScalar_test, case_promote_invalid)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_UINT16, ACL_FORMAT_ND)
                           .Value(vector<short>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto scalar_desc = ScalarDesc(8.0f);

    auto ut = OP_API_UT(aclnnInplaceLeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(inplaceLeScalar_test, case_format_internal)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(8.0f);

    auto ut = OP_API_UT(aclnnInplaceLeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceLeScalar_test, case_dim_invalid)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(8.0f);

    auto ut = OP_API_UT(aclnnInplaceLeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(inplaceLeScalar_test, case_format_abnormal)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_UNDEFINED).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(0.5f);

    auto ut = OP_API_UT(aclnnInplaceLeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceLeScalar_test, case_dtype_abnormal)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_DT_UNDEFINED, ACL_FORMAT_NHWC);
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(5));

    auto ut = OP_API_UT(aclnnInplaceLeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
