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

#include "aclnn_fill_diagonal.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_inplace_fill_diagonal_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_inplace_fill_diagonal_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_inplace_fill_diagonal_test TearDown" << endl;
    }
};

TEST_F(l2_inplace_fill_diagonal_test, case_normal)
{
    auto tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_format_nd_normal)
{
    vector<aclFormat> ValidList = {
        ACL_FORMAT_NCHW, ACL_FORMAT_NHWC, ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};
    int length = ValidList.size();
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    for (int i = 0; i < length; i++) {
        auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ValidList[i]).ValueRange(-1, 1).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_inplace_fill_diagonal_test, case_float_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_float16_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.001, 0.001);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_double_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_int8_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_int16_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_int32_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_int64_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_uint8_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_bool_normal)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_invalid_dtype_abnormal)
{
    vector<aclDataType> ValidList = {ACL_BF16, ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128};
    int length = ValidList.size();
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    for (int i = 0; i < length; i++) {
        auto self_tensor_desc =
            TensorDesc({2, 3}, ValidList[i], ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_inplace_fill_diagonal_test, case_1d_abnormal)
{
    auto self_tensor_desc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_fill_diagonal_test, case_2d_normal)
{
    auto self_tensor_desc = TensorDesc({8, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_4d_normal)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_9d_abnormal)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_fill_diagonal_test, case_uncontinuous_normal)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_tensor_nullptr_abnormal)
{
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;
    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT((aclTensor*)nullptr, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_inplace_fill_diagonal_test, case_scalar_nullptr_abnormal)
{
    auto self_tensor_desc = TensorDesc({8, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    bool wrap = false;
    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, (aclScalar*)nullptr, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_inplace_fill_diagonal_test, case_int32_int64_normal)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3, 3}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_3_4_5_abnormal)
{
    auto self_tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto scalar_desc = ScalarDesc(1.0f);
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2_inplace_fill_diagonal_test, case_empty_tensor_0d_abnormal)
{
    auto self_tensor_desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
    bool wrap = false;
    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_fill_diagonal_test, case_empty_tensor_2d_normal)
{
    auto self_tensor_desc = TensorDesc({7, 0}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_3_3_wrap_normal)
{
    auto self_tensor_desc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(5));
    bool wrap = true;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_8_3_wrap_normal)
{
    auto self_tensor_desc = TensorDesc({8, 3}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(5));
    bool wrap = true;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_3_8_wrap_normal)
{
    auto self_tensor_desc = TensorDesc({3, 8}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(5));
    bool wrap = true;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_3_3_3_wrap_normal)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(5));
    bool wrap = true;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_3_3_3_3_wrap_normal)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3, 3}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(5));
    bool wrap = true;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_test, case_uint8_overflow_abnormal)
{
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(921474836479));
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_fill_diagonal_test, case_int8_overflow_abnormal)
{
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(921474836479));
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_fill_diagonal_test, case_int16_overflow_abnormal)
{
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_INT16, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(921474836479));
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_fill_diagonal_test, case_int32_overflow_abnormal)
{
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(921474836479));
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_fill_diagonal_test, case_float16_overflow_abnormal)
{
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-20, 2);
    auto scalar_desc = ScalarDesc(static_cast<double>(921474836.9));
    bool wrap = false;

    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, wrap), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
