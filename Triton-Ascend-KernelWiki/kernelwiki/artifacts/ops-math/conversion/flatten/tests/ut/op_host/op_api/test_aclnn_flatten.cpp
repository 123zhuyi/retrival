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
#include "opdev/op_log.h"
#include "aclnn_flatten.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_flatten_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Flatten Test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "Flatten Test TearDown" << std::endl;
    }
};

TEST_F(l2_flatten_test, flatten_testcase_001_normal_float32)
{
    auto outDesc = TensorDesc({6, 20}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t axis = 2;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// empty
TEST_F(l2_flatten_test, flatten_testcase_003_normal_empty_tensor)
{
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(outDesc);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull
TEST_F(l2_flatten_test, flatten_testcase_004_exception_null_out)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_flatten_test, flatten_testcase_005_exception_null_self)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT((aclTensor*)nullptr, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtype different dtype of input
TEST_F(l2_flatten_test, flatten_testcase_006_normal_dtype_not_the_same)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype wrong out shape
TEST_F(l2_flatten_test, flatten_testcase_007_wrong_out_shape)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid
TEST_F(l2_flatten_test, flatten_testcase_008_exception_complex64_dtype_not_supported)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid
TEST_F(l2_flatten_test, flatten_testcase_009_exception_complex128_dtype_not_supported)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// largeDim
TEST_F(l2_flatten_test, flatten_testcase_010_normal_large_dims)
{
    auto outDesc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// negative axis
TEST_F(l2_flatten_test, flatten_testcase_011_negative_axis)
{
    auto outDesc = TensorDesc({6, 20}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t axis = -2;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// empty
TEST_F(l2_flatten_test, flatten_testcase_014_exception_out_empty_tensor)
{
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_flatten_test, flatten_testcase_015_normal_int64)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto outDesc = TensorDesc({6, 20}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t axis = 2;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_flatten_test, flatten_testcase_016_normal_bool)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto outDesc = TensorDesc({6, 20}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t axis = 2;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// axis boundary
TEST_F(l2_flatten_test, flatten_testcase_017_axis_boundary)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({1, 120}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_flatten_test, flatten_testcase_018_normal_float32_self_dim1)
{
    auto outDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_flatten_test, flatten_testcase_019_exception_float32_self_dim1)
{
    auto outDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t axis = 1;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_flatten_test, Ascend910PR_9589_flatten_testcase_1_normal_float32_self_dim1)
{
    auto outDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t axis = 0;
    auto ut = OP_API_UT(aclnnFlatten, INPUT(selfDesc, axis), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}