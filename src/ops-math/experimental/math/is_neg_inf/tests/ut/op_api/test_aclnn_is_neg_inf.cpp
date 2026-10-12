/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_is_neg_inf.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;

class l2_is_neg_inf_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "is_neg_inf_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "is_neg_inf_test TearDown" << std::endl;
    }
};

TEST_F(l2_is_neg_inf_test, case_support_float32)
{
    auto selfDesc =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(std::vector<float>{-INFINITY, -1.0f, 0.0f, 1.0f, -INFINITY, 2.0f});
    auto outDesc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{true, false, false, false, true, false});
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_is_neg_inf_test, case_support_float16)
{
    auto selfDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(std::vector<float>{-INFINITY, 2.0f, -3.0f, -INFINITY});
    auto outDesc = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{true, false, false, true});
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_is_neg_inf_test, case_support_bf16)
{
    auto selfDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).Value(std::vector<double>{-INFINITY, 2.0, -3.0, -INFINITY});
    auto outDesc = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{true, false, false, true});
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_is_neg_inf_test, case_support_integer_all_false)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(std::vector<int32_t>{-3, -2, -1, 0, 1, 2});
    auto outDesc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_is_neg_inf_test, case_support_bool_all_false)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{true, false, true, false});
    auto outDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false});
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_is_neg_inf_test, case_non_contiguous_input_output)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {5, 2})
                        .Value(std::vector<float>{-INFINITY, 1.0f, 2.0f, 3.0f, -INFINITY, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    auto outDesc = TensorDesc({2, 5}, ACL_BOOL, ACL_FORMAT_ND, {1, 2}, 0, {5, 2});
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_is_neg_inf_test, case_null_self)
{
    auto outDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(nullptr), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_is_neg_inf_test, case_null_out)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_is_neg_inf_test, case_shape_mismatch)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({3, 2}, ACL_BOOL, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_is_neg_inf_test, case_out_dtype_invalid)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_is_neg_inf_test, case_complex_invalid)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_is_neg_inf_test, case_rank_too_large)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_is_neg_inf_test, case_empty_tensor)
{
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0}, ACL_BOOL, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIsNegInf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0u);
}
