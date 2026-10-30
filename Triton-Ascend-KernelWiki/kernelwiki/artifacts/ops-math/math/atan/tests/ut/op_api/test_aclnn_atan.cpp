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

#include "math/atan/op_api/aclnn_atan.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_atan_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_atan_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_atan_test TearDown" << endl;
    }
};

TEST_F(l2_atan_test, ascend910B2_atan_dtype_all)
{
    vector<aclDataType> input_vaild_dtype_list{ACL_INT8, ACL_INT32, ACL_UINT8,   ACL_INT16,  ACL_INT64,
                                               ACL_BOOL, ACL_FLOAT, ACL_FLOAT16, ACL_DOUBLE, ACL_BF16};
    vector<aclDataType> output_vaild_dtype_list{ACL_FLOAT, ACL_FLOAT16, ACL_DOUBLE};
    vector<aclDataType> invaild_dtype_list{ACL_COMPLEX64, ACL_COMPLEX128};
    for (auto dtype1 : input_vaild_dtype_list) {
        auto self_tensor_desc = TensorDesc({3, 5}, dtype1, ACL_FORMAT_ND).ValueRange(-20, 20);
        for (auto dtype2 : output_vaild_dtype_list) {
            auto out_tensor_desc = TensorDesc({3, 5}, dtype2, ACL_FORMAT_ND).Precision(0.001, 0.001);

            auto ut = OP_API_UT(aclnnAtan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
            uint64_t workspace_size = 0;
            aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
            EXPECT_EQ(aclRet, ACL_SUCCESS);
        }
    }
    for (auto dtype : invaild_dtype_list) {
        auto self_tensor_desc = TensorDesc({3, 5}, dtype, ACL_FORMAT_ND).ValueRange(-20, 20);
        auto out_tensor_desc = TensorDesc({3, 5}, dtype, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnAtan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_atan_test, atan_nullptr)
{
    auto ut = OP_API_UT(aclnnAtan, INPUT((aclTensor*)nullptr), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_atan_test, atan_precision)
{
    auto self_tensor_desc = TensorDesc({13, 16, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({13, 16, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAtan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atan_test, ascend910B2_atan_bf16_precision)
{
    auto self_tensor_desc = TensorDesc({13, 16, 9}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({13, 16, 9}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAtan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atan_test, atan_empty_tensor)
{
    auto self_tensor_desc = TensorDesc({13, 0, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({13, 0, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAtan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atan_test, atan_non_contiguous)
{
    auto self_tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {3, 5}).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {3, 5}).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAtan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atan_test, atan_bigDim)
{
    auto self_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAtan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_atan_test, atan_bigDim_non_contiguous)
{
    auto self_tensor_desc = TensorDesc(
                                {2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {128, 64, 32, 16, 8, 4, 2, 1, 0},
                                0, {2, 2, 2, 2, 2, 2, 2, 2, 1})
                                .ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc(
                               {2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {128, 64, 32, 16, 8, 4, 2, 1, 0},
                               0, {2, 2, 2, 2, 2, 2, 2, 2, 1})
                               .Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAtan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_atan_test, atan_inplace)
{
    auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto ut = OP_API_UT(aclnnInplaceAtan, INPUT(self_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
