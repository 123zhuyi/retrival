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
#include "opdev/op_log.h"
#include "math/expm1/op_api/aclnn_expm1.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_expm1_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Expm1 Test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "Expm1 Test TearDown" << std::endl;
    }
};

TEST_F(l2_expm1_test, expm1_testcase_001_normal_float32)
{
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16
TEST_F(l2_expm1_test, expm1_testcase_002_normal_float16)
{
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bfloat16
TEST_F(l2_expm1_test, ascend910B2_expm1_testcase_002_normal_float16)
{
    auto outDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto selfDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// empty
TEST_F(l2_expm1_test, expm1_testcase_003_normal_empty_tensor)
{
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(outDesc);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull
TEST_F(l2_expm1_test, expm1_testcase_004_exception_null_out)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_expm1_test, expm1_testcase_005_exception_null_self)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExpm1, INPUT((aclTensor*)nullptr), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtype different dtype of input
TEST_F(l2_expm1_test, expm1_testcase_007_normal_dtype_not_the_same)
{
    auto outDesc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto selfDesc = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckDtype different dtype of input
TEST_F(l2_expm1_test, expm1_testcase_008_normal_dtype_not_the_same)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckShape
TEST_F(l2_expm1_test, expm1_testcase_009_exception_different_shape)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// not contiguous
TEST_F(l2_expm1_test, expm1_testcase_010_normal_not_contiguous_float)
{
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {5, 2})
                       .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {5, 2}).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// not contiguous
TEST_F(l2_expm1_test, expm1_testcase_011_normal_not_contiguous_float16)
{
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {5, 2}).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// not contiguous
TEST_F(l2_expm1_test, expm1_testcase_012_normal_not_contiguous_float16)
{
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {5, 2})
                       .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckDtypeValid
TEST_F(l2_expm1_test, expm1_testcase_013_exception_complex64_dtype_not_supported)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid
TEST_F(l2_expm1_test, expm1_testcase_014_exception_complex128_dtype_not_supported)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// largeDim
TEST_F(l2_expm1_test, expm1_testcase_015_normal_large_dims)
{
    auto outDesc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// empty
TEST_F(l2_expm1_test, expm1_testcase_016_normal_self_empty_tensor)
{
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// empty
TEST_F(l2_expm1_test, expm1_testcase_017_exception_out_empty_tensor)
{
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype can cast to out
TEST_F(l2_expm1_test, expm1_testcase_018_can_cast_out)
{
    auto selfDesc = TensorDesc({5, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({5, 5}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype can cast to out
TEST_F(l2_expm1_test, expm1_testcase_019_can_cast_out)
{
    auto selfDesc = TensorDesc({5, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype can cast to out
TEST_F(l2_expm1_test, expm1_testcase_020_can_cast_out)
{
    auto selfDesc = TensorDesc({5, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_expm1_test, expm1_testcase_021_normal_int64_float32)
{
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({2, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_expm1_test, expm1_testcase_022_normal_bool_float32)
{
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({2, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype cannot cast to out
TEST_F(l2_expm1_test, expm1_testcase_023_cannot_cast_out)
{
    auto selfDesc = TensorDesc({5, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({5, 5}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExpm1, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}