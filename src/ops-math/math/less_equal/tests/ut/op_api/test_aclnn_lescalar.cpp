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
 * \file test_aclnn_lescalar.cpp
 * \brief
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "math/less_equal/op_api/aclnn_le_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_le_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "le_scalar_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "le_scalar_test TearDown" << std::endl;
    }
};

TEST_F(l2_le_scalar_test, test_le_scalar_notsupport_complex64)
{
    auto tensor_self = TensorDesc({2, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<float>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_le_scalar_test, test_le_scalar_notsupport_complex128)
{
    auto tensor_self = TensorDesc({2, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<float>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 shape不一致
TEST_F(l2_le_scalar_test, test_le_scalar_diffshape)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<float>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 空指针
TEST_F(l2_le_scalar_test, test_le_scalar_nullptr)
{
    auto tensor_self = nullptr;

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_le_scalar_test, test_le_scalar_error_shape)
{
    auto self_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLeScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
