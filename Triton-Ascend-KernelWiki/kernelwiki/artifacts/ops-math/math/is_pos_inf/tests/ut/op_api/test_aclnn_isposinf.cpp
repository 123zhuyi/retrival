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
 * \file test_aclnn_isposinf.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "opdev/op_log.h"
#include "math/is_pos_inf/op_api/aclnn_isposinf.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_is_pos_inf_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "IsPosInf Test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsPosInf Test TearDown" << std::endl;
    }
};

// empty
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_003_normal_empty_tensor)
{
    auto outDesc = TensorDesc({0}, ACL_BOOL, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(outDesc);

    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_004_exception_null_out)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_005_exception_null_self)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsPosInf, INPUT((aclTensor*)nullptr), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_006_exception_float64_dtype_unsupported)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_008_normal_dtype_not_the_same)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckShape
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_009_exception_different_shape)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// not contiguous
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_010_normal_not_contiguous_float)
{
    auto outDesc = TensorDesc({2, 5}, ACL_BOOL, ACL_FORMAT_ND, {1, 2}, 0, {5, 2});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {5, 2}).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// not contiguous
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_011_normal_not_contiguous_float16)
{
    auto outDesc = TensorDesc({2, 5}, ACL_BOOL, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {5, 2}).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// not contiguous
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_012_normal_not_contiguous_float16)
{
    auto outDesc = TensorDesc({2, 5}, ACL_BOOL, ACL_FORMAT_ND, {1, 2}, 0, {5, 2});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckDtypeValid
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_013_exception_complex64_dtype_not_supported)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_BOOL, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_014_exception_complex128_dtype_not_supported)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// largeDim
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_015_normal_large_dims)
{
    auto outDesc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// empty
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_016_normal_self_empty_tensor)
{
    auto outDesc = TensorDesc({0}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// empty
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_017_exception_out_empty_tensor)
{
    auto outDesc = TensorDesc({0}, ACL_BOOL, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype can cast to out
TEST_F(l2_is_pos_inf_test, ascend910B2_is_pos_inf_testcase_018_can_cast_out)
{
    auto selfDesc = TensorDesc({5, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({5, 5}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsPosInf, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
