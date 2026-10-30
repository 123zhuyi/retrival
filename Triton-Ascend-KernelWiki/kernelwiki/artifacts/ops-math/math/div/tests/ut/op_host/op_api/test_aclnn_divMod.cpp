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

#include "aclnn_div.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_div_mod_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "div_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "div_test TearDown" << endl;
    }
};

// 测试所有数据格式支持
TEST_F(l2_div_mod_test, case_dtype_all_format)
{
    vector<aclFormat> format_list{ACL_FORMAT_ND,   ACL_FORMAT_NCHW,  ACL_FORMAT_NHWC,
                                  ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};
    for (auto format : format_list) {
        auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, format).ValueRange(10, 100);
        auto other_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, format).ValueRange(10, 100);
        auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, format);

        int mode = 2;
        auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// 测试类型转换支持
TEST_F(l2_div_mod_test, case_dtype_cast_support)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int mode = 0;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试非连续支持
TEST_F(l2_div_mod_test, case_NonContiguous)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND, {1, 2}, 0, {3, 2})
                                .Value(vector<float>{-10.5, -20, -30, -40, 50, 60});
    auto other_tensor_desc =
        TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND, {1, 2}, 0, {3, 2}).Value(vector<float>{-2, -3, -2, -4, 2, 4});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int mode = 0;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试broadcast
TEST_F(l2_div_mod_test, case_broadcast)
{
    auto self_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 100);
    auto other_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 100);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    int mode = 1;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试空tensor
TEST_F(l2_div_mod_test, case_empty_tensors)
{
    auto self_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    int mode = 2;
    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试不支持类型CheckDtypeValid
TEST_F(l2_div_mod_test, case_CheckDtypeValid)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);
    int mode = 1;
    auto ut = OP_API_UT(aclnnDivMod, INPUT(tensor_desc, tensor_desc, mode), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试入参空指针
TEST_F(l2_div_mod_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int mode = 2;
    auto ut = OP_API_UT(aclnnDivMod, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr, mode), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试other为scalar,mode为2立即数输入
TEST_F(l2_div_mod_test, case_other_scalar_floor_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int mode = 2;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试other为scalar,mode为1立即数输入
TEST_F(l2_div_mod_test, case_other_scalar_trunc_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int mode = 1;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试other为scalar超过8维的tensor
TEST_F(l2_div_mod_test, case_scalar_shape_dim_9)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int mode = 2;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试超过8维的tensor
TEST_F(l2_div_mod_test, case_shape_dim_9)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int mode = 2;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, self_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试div_:other为scalar立即数输入
TEST_F(l2_div_mod_test, case_inplace_other_scalar_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);
    int mode = 1;

    auto ut = OP_API_UT(aclnnInplaceDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试div_:other为tensor输入
TEST_F(l2_div_mod_test, case_inplace_other_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    int mode = 2;

    auto ut = OP_API_UT(aclnnInplaceDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_div_mod_test, case_complex_mod_1_invalid)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 2);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int mode = 1;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_div_mod_test, case_complex_mod_2_invalid)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 2);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int mode = 2;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试mode异常值拦截
TEST_F(l2_div_mod_test, case_mode_invalid)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int mode = 3;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Ascend950PR_9589
// 测试场景mode为0时数据类型支持
TEST_F(l2_div_mod_test, Ascend950PR_89_case_real_div_dtype_modeNone)
{
    vector<aclDataType> dtype_list{ACL_FLOAT16};
    for (auto dtype : dtype_list) {
        auto self_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).ValueRange(10, 100);
        auto other_tensor_desc = ScalarDesc(2);
        auto out_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
        int mode = 0;

        auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_div_mod_test, Ascend950PR_89_case_divmods_trunc_div)
{
    vector<aclDataType> dtype_list{ACL_INT8};
    for (auto dtype : dtype_list) {
        auto self_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).ValueRange(10, 100);
        auto other_tensor_desc = ScalarDesc(2);
        auto out_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
        int mode = 1;

        auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_div_mod_test, Ascend950PR_89_case_divmods_floor_div)
{
    vector<aclDataType> dtype_list{ACL_INT16};
    for (auto dtype : dtype_list) {
        auto self_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).ValueRange(10, 100);
        auto other_tensor_desc = ScalarDesc(2);
        auto out_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
        int mode = 2;

        auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_div_mod_test, Ascend950PR_89_case_divmod_trunc_div)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 2);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int mode = 1;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}