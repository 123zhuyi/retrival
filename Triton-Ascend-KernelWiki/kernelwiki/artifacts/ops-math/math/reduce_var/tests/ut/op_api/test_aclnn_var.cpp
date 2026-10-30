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
#include "math/reduce_var/op_api/aclnn_var.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_var_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_var_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_var_test TearDown" << std::endl;
    }
};

// 正常场景_float32_dim为0_keepdim为true
TEST_F(l2_var_test, var_dtype_float32_dim_0_keepdim_true)
{
    auto selfDesc = TensorDesc({2, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_float32_dim为1_keepdim为false
TEST_F(l2_var_test, var_dtype_float32_dim_1_keepdim_false)
{
    auto selfDesc = TensorDesc({1, 2, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_float32_dim为2_keepdim为false
TEST_F(l2_var_test, var_dtype_float32_dim_2_keepdim_false)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_float16_dim为2_keepdim为true
TEST_F(l2_var_test, var_dtype_float16_dim_2_keepdim_true)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_float32_dim为2_3_keepdim为true
TEST_F(l2_var_test, var_dtype_float32_dim_2_3_keepdim_true)
{
    auto selfDesc = TensorDesc({1, 2, 8, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_float16_dim为1_4_keepdim为true
TEST_F(l2_var_test, var_dtype_float16_dim_1_4_keepdim_true)
{
    auto selfDesc = TensorDesc({1, 2, 6, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1, 4});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 6, 2, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_unbiased为false
TEST_F(l2_var_test, var_unbiased_0)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1, 2});
    bool unbiased = false;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 所有维度都做reduce_keepdim为false
TEST_F(l2_var_test, var_all_reduce_keepdim_false)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入dim为空_keepdim为false
TEST_F(l2_var_test, var_all_reduce_empty_dim_keepdim_false)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入dim为空_keepdim为true
TEST_F(l2_var_test, var_all_reduce_empty_dim_keepdim_true)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 0维tensor场景
TEST_F(l2_var_test, var_shape_dim_0)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 8维tensor场景
TEST_F(l2_var_test, var_shape_dim_8)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9, 4, 3, 9, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0, 3, 4});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 9, 3, 9, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim为空指针
TEST_F(l2_var_test, var_dim_nullptr)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, nullptr, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor场景
TEST_F(l2_var_test, var_empty_tensor)
{
    auto selfDesc = TensorDesc({2, 0, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({2, 0, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim为-1
TEST_F(l2_var_test, var_dim_negative)
{
    auto selfDesc = TensorDesc({1, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-1});
    bool unbiased = false;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim负数值和其他值对应了同一个维度
TEST_F(l2_var_test, var_negative_dim_same)
{
    auto selfDesc = TensorDesc({1, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-1, 2});
    bool unbiased = false;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dim多个负值和正值
TEST_F(l2_var_test, var_dim_more_negative)
{
    auto selfDesc = TensorDesc({2, 3, 5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-1, 1, -2});
    bool unbiased = false;
    bool keepdim = true;
    auto outDesc = TensorDesc({2, 1, 5, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull_1
TEST_F(l2_var_test, var_self_nullptr)
{
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnVar, INPUT(nullptr, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_2
TEST_F(l2_var_test, var_out_nullptr)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    bool unbiased = true;
    bool keepdim = false;

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_1
TEST_F(l2_var_test, var_dtype_int32)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_2
TEST_F(l2_var_test, var_dtype_uint8)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 1}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDimValid_1
TEST_F(l2_var_test, var_dim_out_of_range)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{7});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDimValid_2
TEST_F(l2_var_test, var_dim_repeated)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 2});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDimValid_3
TEST_F(l2_var_test, var_dim_neg_out_of_range)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-5});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 9维场景
TEST_F(l2_var_test, var_shape_dim_greater_than_threshold)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9, 4, 3, 9, 9, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3, 5, 7});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({2, 1, 1, 1, 4, 1, 9, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景_float32_dim为0_keepdim为true_correction为1
TEST_F(l2_var_test, var_dtype_float32_dim_0_keepdim_true_correction_1)
{
    auto selfDesc = TensorDesc({2, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVarCorrection, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_float32_dim为1_keepdim为false_correction为1
TEST_F(l2_var_test, var_dtype_float32_dim_1_keepdim_false_correction_1)
{
    auto selfDesc = TensorDesc({1, 2, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVarCorrection, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_float32_dim为2_3_keepdim为true_correction=0
TEST_F(l2_var_test, var_dtype_float32_dim_2_3_keepdim_true_correction_0)
{
    auto selfDesc = TensorDesc({1, 2, 8, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVarCorrection, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_correction大小超出对应dim且对应dim不为1
TEST_F(l2_var_test, var_correction_large)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1});
    uint64_t correction = 4;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVarCorrection, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_correction大小超出对应dim且对应dim为1
TEST_F(l2_var_test, var_correction_large_1)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    uint64_t correction = 3;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVarCorrection, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim为-1
TEST_F(l2_var_test, var_dim_negative_correction_0)
{
    auto selfDesc = TensorDesc({1, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-1});
    int64_t correction = 0;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVarCorrection, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_float16_dim为1_4_keepdim为true
TEST_F(l2_var_test, ascend950_var_dtype_float16_dim_1_4)
{
    auto selfDesc = TensorDesc({1, 2, 6, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1, 4});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 6, 2, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    ut.TestGetWorkspaceSize(&workspaceSize);
}

// 正常场景_float16_dim为1_4_keepdim为true
TEST_F(l2_var_test, ascend950_var_dtype_float16_dim_1_4_correction)
{
    auto selfDesc = TensorDesc({1, 2, 6, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1, 4});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 6, 2, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnVarCorrection, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    ut.TestGetWorkspaceSize(&workspaceSize);
}

// shapeProd==1 && shapeProd<=correction 返回NAN场景
TEST_F(l2_var_test, var_correction_return_nan)
{
    auto selfDesc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVarCorrection, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// correction>1 && shapeProd<=correction 返回INF场景
TEST_F(l2_var_test, var_correction_return_inf)
{
    auto selfDesc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
    int64_t correction = 2;
    bool keepdim = false;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVarCorrection, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// BF16数据类型测试
TEST_F(l2_var_test, var_dtype_bfloat16)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 6, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim为负数边界值-1
TEST_F(l2_var_test, var_dim_neg_one)
{
    auto selfDesc = TensorDesc({1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-1});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim为负数边界值-2
TEST_F(l2_var_test, var_dim_neg_two)
{
    auto selfDesc = TensorDesc({1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-2});
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 标量tensor场景（0维）
TEST_F(l2_var_test, var_scalar_tensor)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    bool unbiased = true;
    bool keepdim = false;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, nullptr, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 单维度tensor场景
TEST_F(l2_var_test, var_single_dim_tensor)
{
    auto selfDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    bool unbiased = true;
    bool keepdim = true;
    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnVar, INPUT(selfDesc, dimDesc, unbiased, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
