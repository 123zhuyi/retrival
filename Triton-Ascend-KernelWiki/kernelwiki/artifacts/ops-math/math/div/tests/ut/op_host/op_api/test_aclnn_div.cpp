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
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_div_test : public testing::Test {
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

// 测试aicore:FLOAT,FLOAT32类型支持
TEST_F(l2_div_test, case_dtype_aicore_support)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试aicpu:ACL_INT8,ACL_INT16类型支持
TEST_F(l2_div_test, case_int8_aicpu_support)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试aicpu:ACL_INT32,ACL_INT64类型支持
TEST_F(l2_div_test, case_int32_dtype_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = TensorDesc({4, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(10, 100);
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试aicpu:ACL_UINT8类型支持
TEST_F(l2_div_test, case_uint8_dtype_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = TensorDesc({4, 5}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(10, 100);
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试aicpu:ACL_BOOL类型支持
TEST_F(l2_div_test, case_bool_dtype_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = TensorDesc({4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(10, 100);
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试支持数据格式
TEST_F(l2_div_test, case_dtype_all_format)
{
    vector<aclFormat> format_list{ACL_FORMAT_ND,   ACL_FORMAT_NCHW,  ACL_FORMAT_NHWC,
                                  ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};
    for (auto format : format_list) {
        auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, format).ValueRange(10, 100);
        auto other_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, format).ValueRange(10, 100);
        auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, format);

        auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_div_test, case_dtype_wrong_self_format)
{
    auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ).ValueRange(10, 100);
    auto other_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_div_test, case_dtype_wrong_other_format)
{
    auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ).ValueRange(10, 100);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_div_test, case_dtype_wrong_out_format)
{
    auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_div_test, case_dtype_wrong_self_format_scalar)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDivs, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_div_test, case_dtype_wrong_out_format_scalar)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);

    auto ut = OP_API_UT(aclnnDivs, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试非连续支持
TEST_F(l2_div_test, case_NonContiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试不支持类型CheckDtypeValid
TEST_F(l2_div_test, case_CheckDtypeValid)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDiv, INPUT(tensor_desc, tensor_desc), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试broadcast
TEST_F(l2_div_test, case_broadcast)
{
    auto self_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试入参空指针
TEST_F(l2_div_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnDiv, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试空tensor
TEST_F(l2_div_test, case_empty_tensors)
{
    auto self_tensor_desc = TensorDesc({7, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({7, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试other为scalar立即数输入
TEST_F(l2_div_test, case_other_scalar_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnDivs, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试混合精度
TEST_F(l2_div_test, case_other_scalar_fp16_fp32_910B)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(10, 100);
        auto other_tensor_desc = ScalarDesc(2.0f);
        auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

        auto ut = OP_API_UT(aclnnDivs, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_div_test, case_other_scalar_bf16_fp32_910B)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        auto self_tensor_desc = TensorDesc({4, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(10, 100);
        auto other_tensor_desc = ScalarDesc(2.0f);
        auto out_tensor_desc = TensorDesc({4, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

        auto ut = OP_API_UT(aclnnDivs, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_div_test, case_other_scalar_fp16_fp32)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnDivs, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试other为scalar 输入
TEST_F(l2_div_test, case_other_scalar_mod)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, 0), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试other为scalar入参空指针
TEST_F(l2_div_test, case_other_scalar_nullptr)
{
    auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDivs, INPUT((aclTensor*)nullptr, (aclScalar*)nullptr), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试other为scalar不支持类型
TEST_F(l2_div_test, case_other_scalar_CheckDtypeValid)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);
    auto other_tensor_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnDivs, INPUT(tensor_desc, other_tensor_desc), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试other为scalar超过8维的tensor
TEST_F(l2_div_test, case_scalar_shape_dim_9)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnDivs, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试超过8维的tensor
TEST_F(l2_div_test, case_shape_dim_9)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnDiv, INPUT(self_tensor_desc, self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试div_:other为scalar立即数输入
TEST_F(l2_div_test, case_inplace_other_scalar_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnInplaceDivs, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试div_:other为tensor输入
TEST_F(l2_div_test, case_inplace_other_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);

    auto ut = OP_API_UT(aclnnInplaceDiv, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_div_test, Ascend950PR_89_case_inplace_other_scalar_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnInplaceDivs, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}