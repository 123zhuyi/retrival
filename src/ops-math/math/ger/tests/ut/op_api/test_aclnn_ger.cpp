/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_ger.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_ger_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Ger Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "Ger Test TearDown" << std::endl; }
};

// FLOAT
TEST_F(l2_ger_test, case_float)
{
    auto self = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// FLOAT16
TEST_F(l2_ger_test, case_float16)
{
    auto self = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// INT8
TEST_F(l2_ger_test, case_int8)
{
    auto self = TensorDesc({4}, ACL_INT8, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// UINT8
TEST_F(l2_ger_test, case_uint8)
{
    auto self = TensorDesc({4}, ACL_UINT8, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// INT32
TEST_F(l2_ger_test, case_int32)
{
    auto self = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// INT64
TEST_F(l2_ger_test, case_int64)
{
    auto self = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// BOOL
TEST_F(l2_ger_test, case_bool)
{
    auto self = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// COMPLEX64
TEST_F(l2_ger_test, case_complex64)
{
    auto self = TensorDesc({4}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// Mixed dtype: self FLOAT16, vec2 FLOAT
TEST_F(l2_ger_test, case_mixed_fp16_fp32)
{
    auto self = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// Different sizes
TEST_F(l2_ger_test, case_different_sizes)
{
    auto self = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({3, 7}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// Empty 1D vectors
TEST_F(l2_ger_test, case_empty_vector)
{
    auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull: self nullptr
TEST_F(l2_ger_test, case_self_nullptr)
{
    auto vec2 = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(nullptr, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull: vec2 nullptr
TEST_F(l2_ger_test, case_vec2_nullptr)
{
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, nullptr), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull: out nullptr
TEST_F(l2_ger_test, case_out_nullptr)
{
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// Error: self is 2D (not 1D)
TEST_F(l2_ger_test, case_error_self_not_1d)
{
    auto self = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Error: vec2 is 2D (not 1D)
TEST_F(l2_ger_test, case_error_vec2_not_1d)
{
    auto self = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Error: wrong output shape
TEST_F(l2_ger_test, case_error_out_shape)
{
    auto self = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Error: self is scalar (empty shape)
TEST_F(l2_ger_test, case_error_self_scalar)
{
    auto self = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto vec2 = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGer, INPUT(self, vec2), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
