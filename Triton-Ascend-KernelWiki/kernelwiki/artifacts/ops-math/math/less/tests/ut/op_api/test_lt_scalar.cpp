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

#include "math/less/op_api/aclnn_lt_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_lt_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "lt_scalar_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "lt_scalar_test TearDown" << endl;
    }
};

// aclnnLtScalar_001:lt.Scalar_out输入支持FLOAT
TEST_F(l2_lt_scalar_test, aclnnLtScalar_001_float_ND)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_other = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_002:lt.Scalar_out输入支持FLOAT16
TEST_F(l2_lt_scalar_test, aclnnLtScalar_002_float16_NHWC)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto scalar_other = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NHWC).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_003:lt.Scalar_out输入支持INT32
TEST_F(l2_lt_scalar_test, aclnnLtScalar_003_int32_NHWC)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NHWC).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_004:lt.Scalar_out输入支持INT64
TEST_F(l2_lt_scalar_test, aclnnLtScalar_004_int64_NHWC)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(5));
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_005:lt.Scalar_out输入支持INT16
// 走AICPU
TEST_F(l2_lt_scalar_test, clnnLessScalar_005_int16_HWCN)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_006:lt.Scalar_out输入支持INT8
TEST_F(l2_lt_scalar_test, aclnnLtScalar_006_int8_NDHWC)
{
    auto tensor_self = TensorDesc({5, 2, 5, 6, 3}, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_BOOL, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_007:lt.Scalar_out输入支持UINT8
TEST_F(l2_lt_scalar_test, aclnnLtScalar_007_uint8_NCDHW)
{
    auto tensor_self = TensorDesc({5, 2, 5, 6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_BOOL, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_008:lt.Scalar_out输入支持BOOL
// 走AICPU
TEST_F(l2_lt_scalar_test, aclnnLtScalar_008_bool_ND)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(1));
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lt_scalar_test, ascend910B2_aclnnLtScalar_bf16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_009:lt.Scalar_out输入支持BFLOAT16
// 走AICPU
TEST_F(l2_lt_scalar_test, aclnnLtScalar_009_bf16)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_other = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnLtScalar_010:lt.Scalar_out输入支持DOUBLE
// 走AICPU
TEST_F(l2_lt_scalar_test, aclnnLtScalar_010_double)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Less算子不支持COMPLEX64
// aclnnLtScalar_011:lt.Scalar_out输入支持COMPLEX64
// 走AICPU
TEST_F(l2_lt_scalar_test, aclnnLtScalar_011_complex64)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Less算子不支持COMPLEX128
// aclnnLtScalar_012:lt.Scalar_out输入支持COMPLEX128
// 走AICPU
TEST_F(l2_lt_scalar_test, aclnnLtScalar_012_complex128)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnLtScalar_013:lt.Scalar_out支持输入和输出类型不一致
TEST_F(l2_lt_scalar_test, aclnnLtScalar_013_float16_float16_bool_NHWC)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NHWC).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_015:lt.Scalar_out支持输入的类型不一致
TEST_F(l2_lt_scalar_test, aclnnLtScalar_015_float_int32_bool_broadcast_ND)
{
    auto tensor_self = TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_016:lt.Scalar_out支持输入为空Tensor
TEST_F(l2_lt_scalar_test, aclnnLtScalar_016_input_empty_tensor)
{
    auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnLtScalar_017:lt.Scalar_out支持输入为非连续Tensor
TEST_F(l2_lt_scalar_test, aclnnLtScalar_017_input_not_contiguous)
{
    auto tensor_self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(1.2f);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_019：lt.Scalar_out异常shape测试
TEST_F(l2_lt_scalar_test, aclnnLtScalar_019_error_format)
{
    auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(1.2f);
    auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(self_tensor_desc, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnLtScalar_020：lt.Scalar_out空指针场景测试
TEST_F(l2_lt_scalar_test, aclnnLtScalar_020_null_pointer)
{
    auto tensor_self = nullptr;
    auto scalar_other = ScalarDesc(1.2f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// aclnnLtScalar_021：lt.Scalar_out异常数据类型测试
TEST_F(l2_lt_scalar_test, aclnnLtScalar_021_error_dtype)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(1.2f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnLtScalar_023：lt.Scalar_out异常shape测试
TEST_F(l2_lt_scalar_test, aclnnLtScalar_023_error_shape)
{
    auto self_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(1.2f);
    auto out_tensor_desc =
        TensorDesc({10, 5, 10, 5}, ACL_BOOL, ACL_FORMAT_NHWC).ValueRange(-10, 10).Precision(0.0001, 0.0001);

    auto ut_2 = OP_API_UT(aclnnLtScalar, INPUT(self_tensor_desc, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 其他用例
// 入参为other为null场景
TEST_F(l2_lt_scalar_test, test_lt_scalar_other_is_null)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{9, 3, 1, 1, 2, 3});

    auto scalar_other = nullptr;

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// 入参out为空指针的场景(结果待定，cpu输出空指针)
TEST_F(l2_lt_scalar_test, test_lt_scalar_input_out_is_null)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto scalar_other = ScalarDesc(1.2f);

    auto out_tensor_desc = nullptr;

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 入参self数据类型不支持的场景
TEST_F(l2_lt_scalar_test, test_lt_scalar_self_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto scalar_other = ScalarDesc(1.2f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 入参out数据类型不支持的场景
TEST_F(l2_lt_scalar_test, test_lt_scalar_out_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto scalar_other = ScalarDesc(1.2f);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_lt_scalar_test, aclnnLtScalar_input_all_bool1)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(static_cast<bool>(1));
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lt_scalar_test, aclnnLtScalar_input_all_bool2)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(static_cast<bool>(0));
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtScalar_013:lt.Scalar_out支持输入和输出类型不一致
TEST_F(l2_lt_scalar_test, Ascend950PR_89_aclnnLtScalar_013_float16_float16_bool_NHWC)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto scalar_other = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NHWC).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtScalar, INPUT(tensor_self, scalar_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}