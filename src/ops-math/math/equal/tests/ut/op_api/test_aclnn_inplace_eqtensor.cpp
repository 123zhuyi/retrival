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
#include "../../../op_api/aclnn_eq_tensor.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_inplace_eq_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "inplace_eq_tensor_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "inplace_eq_tensor_test TearDown" << std::endl;
    }
};

// 正常场景 int8
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_int8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 int16
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_int16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 int32
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_int32)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 int64
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_int64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 uint8
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_uint8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 float16
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_float16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 float32
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_float)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 double
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_double)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_1971support_input_bf16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 9, 6, 7, 6});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 4, 6, 7, 6});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
    } else {
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 5;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(workspace_size, 5u);
    }
}

// 正常场景 complex64
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_complex64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 complex128
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_complex128)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// float16转bf16
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_float16_bf16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 9, 6, 7, 6});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 4, 6, 7, 6});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
    } else {
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 5;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_float_bf16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 9, 6, 7, 6});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 4, 6, 7, 6});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
    } else {
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 5;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_input_decimal_fraction)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.000001, 4, 9, 6, 7, 11});

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<float>{3.00000099, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 format_not_equal_HWCN
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_format_not_equal)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_HWCN).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 ACL_FORMAT_NCDHW
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_format_NCDHW)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NCDHW)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NCDHW)
                            .ValueRange(-10, 10)
                            .Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 ACL_FORMAT_NDHWC
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_format_NDHWC)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NDHWC)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NDHWC)
                            .ValueRange(-10, 10)
                            .Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 ACL_FORMAT_HWCN
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_format_HWCN)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_HWCN)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_HWCN).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 ACL_FORMAT_NHWC
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_format_nhwc)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NHWC)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NHWC).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 ACL_FORMAT_NCHW
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal_format_nchw)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NCHW)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal01)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景,做broadcast
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal02)
{
    auto tensor_self = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(vector<float>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 2, 1});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & int16，走cpu
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal03)
{
    auto tensor_self = TensorDesc({2, 3, 2}, ACL_INT16, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(vector<int16_t>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int16_t>{1, 2, 2, 1});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 做broadcast & double，走cpu
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal04)
{
    auto tensor_self = TensorDesc({2, 3, 2}, ACL_DOUBLE, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(vector<double>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{1, 2, 2, 1});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());
}

// 正常场景 做broadcast & bool
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal05)
{
    auto tensor_self =
        TensorDesc({2, 3, 2}, ACL_BOOL, ACL_FORMAT_ND)
            .ValueRange(-10, 10)
            .Value(vector<bool>{false, true, true, true, true, true, false, true, true, true, true, true});

    auto tensor_other =
        TensorDesc({2, 1, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<bool>{true, true, true, false});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & INT64
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal06)
{
    auto tensor_self = TensorDesc({2, 3, 2}, ACL_INT64, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(vector<int64_t>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int64_t>{1, 2, 2, 1});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & UINT8
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal07)
{
    auto tensor_self = TensorDesc({2, 3, 2}, ACL_UINT8, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(vector<uint8_t>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint8_t>{1, 2, 2, 1});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & float16
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal08)
{
    auto tensor_self = TensorDesc({2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(vector<float>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 2, 1});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & (2,2,3,2)NCHW & ND
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal09)
{
    auto tensor_self =
        TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10).Value(vector<float>{1, 2, 4, 3, 2, 1,
                                                                                                     1, 2, 4, 3, 2, 1,
                                                                                                     1, 2, 4, 3, 2, 1,
                                                                                                     1, 2, 4, 3, 2, 1});

    auto tensor_other = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 全部为空Tensor
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_normal10)
{
    auto tensor_self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 入参self为null场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_self_is_null)
{
    auto tensor_self = nullptr;

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 入参other为null场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_other_is_null)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto tensor_other = nullptr;

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参self&other都为空tensor的场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_input_is_empty)
{
    auto tensor_self =
        TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_EQ(workspace_size, 0u);
}

// 入参self为空tensor的场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_input_self_is_empty)
{
    auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参other为空tensor的场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_input_other_is_empty)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto tensor_other = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// shape异常，入参Tensor为9维
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_input_rank_9)
{
    auto tensor_self = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 异常数据类型，入参self数据类型不支持的场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_self_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 异常数据类型，入参oter数据类型不支持的场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_other_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// broadcast异常场景，入参无法broadcast
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_input_cannot_broadcast)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 异常shape场景,broadcast正常，但outShape与self的shape不一致
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_boradcast_not_equal_out)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 4, 3, 2, 1});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 异常场景，入参self dtype不支持场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_self_dtype_undefined)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景，入参other dtype不支持场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_other_dtype_undefined)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非连续场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_to_contiguous)
{
    auto tensor_self =
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Value(vector<float>{3, 4, 9, 6, 7, 11, 5, 89});

    auto tensor_other =
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Value(vector<float>{3, 4, 8, 6, 5, 11, 5, 80});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 数据范围[-1,1]场景
TEST_F(l2_inplace_eq_tensor_test, test_inplace_eq_tensor_date_range_f1_1)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(-1, 1)
            .Value(vector<float>{
                -6.75468990e-01, 7.37894118e-01, -7.84537017e-01, 9.64608252e-01, 9.09311235e-01, -9.51533616e-01});
    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(-1, 1)
            .Value(vector<float>{
                -6.75468990e-01, 7.37894118e-01, -7.94537017e-01, 9.64608252e-01, 9.09211235e-01, -9.51533616e-01});

    auto ut = OP_API_UT(aclnnInplaceEqTensor, INPUT(tensor_self, tensor_other), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}