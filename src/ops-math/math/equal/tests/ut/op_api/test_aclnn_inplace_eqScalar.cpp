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

#include "../../../op_api/aclnn_eq_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_inplace_eq_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "inplace_eq_scalar_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "inplace_eq_scalar_test TearDown" << std::endl;
    }
};

// int8
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_input_int8)
{
    auto self_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto other_scalar_desc = ScalarDesc(static_cast<int64_t>(2));

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// uint8
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_input_uint8)
{
    auto self_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
    auto other_scalar_desc = ScalarDesc(static_cast<int64_t>(2));

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// int32
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_input_int32)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto other_scalar_desc = ScalarDesc(static_cast<int64_t>(2));

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// int64
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_input_int64)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto other_scalar_desc = ScalarDesc(static_cast<int64_t>(5));

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_input_float16)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(0, 100);
    auto other_scalar_desc = ScalarDesc(1.2f);

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_input_float)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bool
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_input_bool)
{
    auto self_tensor_desc =
        TensorDesc({1, 2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, false, false, true, false});
    auto other_scalar_desc = ScalarDesc(static_cast<int64_t>(2));

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 空tensor
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_empty)
{
    auto self_tensor_desc = TensorDesc({2, 3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t value = 10;
    auto other_scalar_desc = ScalarDesc(value);

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常场景 空指针
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_nullptr)
{
    auto self_tensor_desc = nullptr;

    int64_t value = 10;
    auto other_scalar_desc = ScalarDesc(value);

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景,维度
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_abnormal_shape)
{
    auto self_tensor_desc = TensorDesc({2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t value = 10;
    auto other_scalar_desc = ScalarDesc(value);

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 不校验 私有format
TEST_F(l2_inplace_eq_scalar_test, test_inplace_eq_scalar_NC1HWC0)
{
    auto self_tensor_desc = TensorDesc({2, 3, 1}, ACL_BF16, ACL_FORMAT_NC1HWC0)
                                .ValueRange(-10, 10)
                                .Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    float value = 10;
    auto other_scalar_desc = ScalarDesc(value);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnInplaceEqScalar, INPUT(self_tensor_desc, other_scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}
