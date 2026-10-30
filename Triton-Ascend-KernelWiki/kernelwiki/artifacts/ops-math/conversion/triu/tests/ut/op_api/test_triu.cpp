/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include "conversion/triu/op_api/aclnn_triu.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_triu_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "triu_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "triu_test TearDown" << endl;
    }
};

// 测试所有支持的类型
TEST_F(l2_triu_test, case_1_all_dtype)
{
    vector<aclDataType> dtype_list{ACL_FLOAT, ACL_FLOAT16, ACL_INT64, ACL_INT32,  ACL_INT16,  ACL_UINT8,
                                   ACL_INT8,  ACL_DOUBLE,  ACL_BOOL,  ACL_UINT64, ACL_UINT32, ACL_UINT16};
    for (auto dtype : dtype_list) {
        cout << "+++++++++++++++++++++++ start to test dtype: " << String(dtype) << endl;
        auto inputDesc = TensorDesc({3, 5}, dtype, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        auto outDesc = TensorDesc({3, 5}, dtype, ACL_FORMAT_ND);
        int64_t diagonal = -3;
        auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

TEST_F(l2_triu_test, case_2_in_out_different_dtype)
{
    auto inputDesc =
        TensorDesc({3, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto outDesc = TensorDesc({3, 4}, ACL_INT16, ACL_FORMAT_ND);
    int64_t diagonal = -2;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triu_test, case_3_all_format)
{
    vector<aclFormat> format_list{ACL_FORMAT_NC1HWC0, ACL_FORMAT_NCHW,  ACL_FORMAT_NHWC, ACL_FORMAT_ND,
                                  ACL_FORMAT_HWCN,    ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};
    for (auto format : format_list) {
        cout << "+++++++++++++++++++++++ start to test format: " << format << endl;
        auto inputDesc =
            TensorDesc({3, 4}, ACL_INT32, format).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        auto outDesc = TensorDesc({3, 4}, ACL_INT32, format);
        int64_t diagonal = 0;
        auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        if (format == ACL_FORMAT_NC1HWC0) {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        } else {
            EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        }
    }
}

TEST_F(l2_triu_test, case_4_different_format)
{
    auto inputDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto outDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_NCHW);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triu_test, case_5_different_shape)
{
    auto inputDesc =
        TensorDesc({3, 4}, ACL_INT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto outDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triu_test, case_6_input_dim_lt_2)
{
    auto inputDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND).Value(vector<int>{1, 2});
    auto outDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triu_test, case_7_input_dim_gt_8)
{
    auto inputDesc = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triu_test, case_8_all_diagonal)
{
    auto inputDesc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto outDesc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND);
    for (int64_t diagonal = -5; diagonal <= 5; diagonal++) {
        auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

TEST_F(l2_triu_test, case_9_nullptr)
{
    auto outDesc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(aclnnTriu, INPUT(nullptr, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_triu_test, case_10_empty)
{
    auto inputDesc = TensorDesc({0}, ACL_INT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0}, ACL_INT16, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_triu_test, case_11_non_contiguous)
{
    auto inputDesc = TensorDesc({5, 5}, ACL_BOOL, ACL_FORMAT_ND, {1, 5}, 0, {5, 5});
    auto outDesc = TensorDesc({5, 5}, ACL_BOOL, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 数据范围[-1, 1]
TEST_F(l2_triu_test, case_12_data_value_between_negative_1_and_positive_1)
{
    auto inputDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t diagonal = -1;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_triu_test, case_13_nullptr_out)
{
    auto inputDesc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_triu_test, case_14_dtype_not_support)
{
    auto inputDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triu_test, case_15_inplace_dtype_not_support)
{
    auto inputDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int64_t diagonal = -2;
    auto ut = OP_API_UT(aclnnInplaceTriu, INPUT(inputDesc, diagonal), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triu_test, Ascend950_input_dim_gt_8)
{
    auto inputDesc = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t diagonal = 1;
    auto ut = OP_API_UT(aclnnTriu, INPUT(inputDesc, diagonal), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}