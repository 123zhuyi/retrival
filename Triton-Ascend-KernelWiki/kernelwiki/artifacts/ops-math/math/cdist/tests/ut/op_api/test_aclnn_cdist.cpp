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

#include "opdev/platform.h"
#include "../../../op_api/aclnn_cdist.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_cdist_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_cdist_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_cdist_test TearDown" << endl;
    }
};

TEST_F(l2_cdist_test, case_01_float)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 2;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_cdist_test, case_02_float16)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 2;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_cdist_test, case_04_empty_tensor)
{
    auto x1Desc = TensorDesc({0, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({0, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 2;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常场景：x1为空指针
TEST_F(l2_cdist_test, case_05_x1_null)
{
    auto x1Desc = nullptr;
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：x2为空指针
TEST_F(l2_cdist_test, case_06_x2_null)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = nullptr;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：out为空指针
TEST_F(l2_cdist_test, case_07_out_null)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = nullptr;
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：x1不在API支持的数据类型范围之内
TEST_F(l2_cdist_test, case_08_x1_not_in_dtype_support_list)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：x2不在API支持的数据类型范围之内
TEST_F(l2_cdist_test, case_09_x2_not_in_dtype_support_list)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：out不在API支持的数据类型范围之内
TEST_F(l2_cdist_test, case_10_out_not_in_dtype_support_list)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：out和x1的数据类型不一致
TEST_F(l2_cdist_test, case_11_x1_out_diff_dtype)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：out和x2的数据类型不一致
TEST_F(l2_cdist_test, case_12_x2_out_diff_dtype)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：x1维度大于8
TEST_F(l2_cdist_test, case_13_x1_dim_gt_8)
{
    auto x1Desc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：x2维度大于8
TEST_F(l2_cdist_test, case_13_x2_dim_gt_8)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：out维度大于8
TEST_F(l2_cdist_test, case_14_out_dim_gt_8)
{
    auto x1Desc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：x1维度小于2
TEST_F(l2_cdist_test, case_15_x1_dim_lt_2)
{
    auto x1Desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：x2维度小于2
TEST_F(l2_cdist_test, case_16_x2_dim_lt_2)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：out维度小于2
TEST_F(l2_cdist_test, case_17_out_dim_lt_2)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：x2的点特征维度与x1不一致
TEST_F(l2_cdist_test, case_18_x1_x2_diff_lastdim)
{
    auto x1Desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：p范数为负数
TEST_F(l2_cdist_test, case_19_p_lt_0)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = -2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：p范数为nan
TEST_F(l2_cdist_test, case_20_p_is_nan)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = std::nan("");
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：x1和x2不符合广播规则
TEST_F(l2_cdist_test, case_21_not_broadcast)
{
    auto x1Desc = TensorDesc({1, 5, 3, 1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({7, 2, 3, 2, 6, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({7, 5, 3, 2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：p=2.0但compute_mode!=2
TEST_F(l2_cdist_test, case_22_p2_compute_mode_not_2)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景：p=0
TEST_F(l2_cdist_test, case_23_p_zero)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 0.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景：p=1.0
TEST_F(l2_cdist_test, case_24_p_one)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 1.0;
    int64_t compute_mode = 0;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景：x1 batch维度为0
TEST_F(l2_cdist_test, case_25_x1_batch_zero)
{
    auto x1Desc = TensorDesc({0, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({0, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 2;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景：bf16数据类型(ASCEND950)
TEST_F(l2_cdist_test, case_26_bf16)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 2;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景：x1最后一维为0
TEST_F(l2_cdist_test, case_27_x1_lastdim_zero)
{
    auto x1Desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 2;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// 正常场景：广播测试
TEST_F(l2_cdist_test, case_28_broadcast)
{
    auto x1Desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({4, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 2;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// 正常场景：多维广播测试
TEST_F(l2_cdist_test, case_29_multidim_broadcast)
{
    auto x1Desc = TensorDesc({2, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({1, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 2;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// 异常场景：x2 batch维度为0
TEST_F(l2_cdist_test, case_30_x2_batch_zero)
{
    auto x1Desc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({0, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 2;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常场景：workspaceSize为空指针
TEST_F(l2_cdist_test, case_31_workspace_null)
{
    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float p = 2.0;
    int64_t compute_mode = 2;

    auto ut = OP_API_UT(aclnnCdist, INPUT(x1Desc, x2Desc, p, compute_mode), OUTPUT(outDesc));

    uint64_t* workspace_size = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
