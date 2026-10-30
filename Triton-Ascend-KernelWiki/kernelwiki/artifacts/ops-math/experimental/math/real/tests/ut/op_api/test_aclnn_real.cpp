/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_real.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_real_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "real_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "real_test TearDown" << endl; }
};

// test aicore type: ACL_COMPLEX64
TEST_F(l2_real_test, case_aicore_real_for_complex64_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: ACL_COMPLEX32
TEST_F(l2_real_test, case_aicore_real_for_complex32_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX32, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: ACL_COMPLEX128
TEST_F(l2_real_test, case_aicore_real_for_complex128_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: FLOAT/FLOAT32
TEST_F(l2_real_test, case_aicore_real_for_float_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: FLOAT16
TEST_F(l2_real_test, case_aicore_real_for_float16_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test shape: 1D
TEST_F(l2_real_test, case_aicore_real_1d_shape)
{
    auto self_tensor_desc = TensorDesc({10}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test shape: 2D
TEST_F(l2_real_test, case_aicore_real_2d_shape)
{
    auto self_tensor_desc = TensorDesc({5, 10}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test shape: 4D
TEST_F(l2_real_test, case_aicore_real_4d_shape)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReal, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
