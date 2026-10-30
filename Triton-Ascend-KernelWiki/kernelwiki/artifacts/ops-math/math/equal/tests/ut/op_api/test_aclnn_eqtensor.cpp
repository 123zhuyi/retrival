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

class l2_eq_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "eq_tensor_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "eq_tensor_test TearDown" << std::endl; }
};

// self Bool转complex64 目前aicpuu支持
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_self_to_bool_complex64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 6});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 9, 6, 7, 6});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// Bool转complex128 目前aicpuu支持
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_bool_complex128)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// float16转complex64
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_float16_complex64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// float16转complex128
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_float16_complex128)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// complex64转complex128 aicpuu支持
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_double_complex64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_suppport_complex128)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 8, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_suppport_complex64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 8, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_bf16_to_float16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 9, 6, 7, 6});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 4, 6, 7, 6});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
    } else {
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 5;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_bf16_to_float)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 9, 6, 7, 6});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 4, 6, 7, 6});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
    } else {
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 5;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_1971support_bf16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 9, 6, 7, 6});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{39, 4, 4, 6, 7, 6});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
    } else {
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 5;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(workspace_size, 5u);
    }
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_support_uint64)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(vector<uint64_t>{3999999, 4, 9, 6, 7, 6});

    auto tensor_other = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<uint64_t>{3999999, 4, 4, 6, 7, 6});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
    } else {
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 5;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(workspace_size, 5u);
    }
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_support_uint32)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(vector<uint32_t>{3999999, 4, 9, 6, 7, 6});

    auto tensor_other = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<uint32_t>{3999999, 4, 4, 6, 7, 6});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
    } else {
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 5;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(workspace_size, 5u);
    }
}

// double转complex128 aicpuu支持
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_double_complex128)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// uint64转complex128 aicpuu支持
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_uint64_complex128)
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        auto tensor_self =
            TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

        auto tensor_other = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND)
                                .ValueRange(-10, 10)
                                .Value(vector<float>{3, 4, 5, 6, 7, 8});

        auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
        // .Value(vector<bool>{false, false, false, false, false, false});

        auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
    }
}

// uint32转complex64
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_uint32_complex64)
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        auto tensor_self =
            TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

        auto tensor_other =
            TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

        auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
        // .Value(vector<bool>{false, false, false, false, false, false});

        auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
    }
}

// uint32转complex128
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_uint32_complex128)
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        auto tensor_self =
            TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

        auto tensor_other = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND)
                                .ValueRange(-10, 10)
                                .Value(vector<float>{3, 4, 5, 6, 7, 8});

        auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
        // .Value(vector<bool>{false, false, false, false, false, false});

        auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
    }
}

// golden支持uint32
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_uint32)
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        auto tensor_self =
            TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

        auto tensor_other =
            TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

        auto out_tensor_desc = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);
        // .Value(vector<bool>{false, false, false, false, false, false});

        auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
    }
}

// 正常场景 uint32
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_format_uint32)
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        auto tensor_self = TensorDesc({3}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9});

        auto tensor_other =
            TensorDesc({3}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5});

        auto out_tensor_desc = TensorDesc({3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false});

        auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
    }
}

// golden支持uint64
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_uint64)
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        auto tensor_self =
            TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

        auto tensor_other =
            TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

        auto out_tensor_desc = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
        // .Value(vector<bool>{false, false, false, false, false, false});

        auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
    }
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_input_xiaoshu)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.000001, 4, 9, 6, 7, 11});

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<float>{3.00000099, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_float16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_float)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_int64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_int32)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_int8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_uint8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_double)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_out_int16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 format_not_equal   HWCN
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_format_not_equal)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_HWCN).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 ACL_FORMAT_NCDHW
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_format_NCDHW)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NCDHW)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NCDHW)
                            .ValueRange(-10, 10)
                            .Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3, 1, 1, 1}, ACL_BOOL, ACL_FORMAT_NCDHW)
                               .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 ACL_FORMAT_NDHWC
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_format_NDHWC)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NDHWC)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_NDHWC)
                            .ValueRange(-10, 10)
                            .Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3, 1, 1, 1}, ACL_BOOL, ACL_FORMAT_NDHWC)
                               .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 ACL_FORMAT_HWCN
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_format_HWCN)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_HWCN)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_HWCN).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3, 1, 1}, ACL_BOOL, ACL_FORMAT_HWCN)
                               .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 ACL_FORMAT_NHWC
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_format_nhwc)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NHWC)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NHWC).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3, 1, 1}, ACL_BOOL, ACL_FORMAT_NHWC)
                               .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 ACL_FORMAT_NCHW
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_format_nchw)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NCHW)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3, 1, 1}, ACL_BOOL, ACL_FORMAT_NCHW)
                               .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 int8
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_int8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 float16
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_int32)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 float16
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_float16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 complex64
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_complex64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 complex128
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_complex128)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_complex128_float)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal_complex128_int32)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{3, 4, 5, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal01)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal02)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(
                                   vector<bool>{
                                       false, false, false, false, false, false, false, false, false, false, false,
                                       false, false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & dtype cast
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal03)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(
                                   vector<bool>{
                                       false, false, false, false, false, false, false, false, false, false, false,
                                       false, false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & dtype cast & out INT32
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal04)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND)
                               .Value(vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & int16 走cpu
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal05)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int16_t>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int16_t>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(
                                   vector<bool>{
                                       false, false, false, false, false, false, false, false, false, false, false,
                                       false, false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 做broadcast & double 走cpu
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal05_2)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<double>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(
                                   vector<bool>{
                                       false, false, false, false, false, false, false, false, false, false, false,
                                       false, false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 正常场景 做broadcast & bool
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal06)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(vector<bool>{false, true, true, true, true, true});

    auto tensor_other = TensorDesc({2, 1, 3}, ACL_BOOL, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<bool>{true, true, true, true, true, true});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(
                                   vector<bool>{
                                       false, false, false, false, false, false, false, false, false, false, false,
                                       false, false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & INT64
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal07)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int64_t>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int64_t>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(
                                   vector<bool>{
                                       false, false, false, false, false, false, false, false, false, false, false,
                                       false, false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & UINT8
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal08)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint8_t>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint8_t>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(
                                   vector<bool>{
                                       false, false, false, false, false, false, false, false, false, false, false,
                                       false, false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast & float16
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal09)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(
                                   vector<bool>{
                                       false, false, false, false, false, false, false, false, false, false, false,
                                       false, false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做broadcast (2)ND & (2,2,3,2)NCHW -> (2,2,3,2)NCHW
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal10)
{
    auto tensor_self = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2});

    auto tensor_other =
        TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10).Value(vector<float>{1, 2, 4, 3, 2, 1,
                                                                                                     1, 2, 4, 3, 2, 1,
                                                                                                     1, 2, 4, 3, 2, 1,
                                                                                                     1, 2, 4, 3, 2, 1});

    auto out_tensor_desc =
        TensorDesc({2, 2, 3, 2}, ACL_BOOL, ACL_FORMAT_NCHW).Value(vector<bool>{false, false, false, false, false,
                                                                               false, false, false, false, false,
                                                                               false, false, false, false, false,
                                                                               false, false, false, false, false,
                                                                               false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 全部为空Tensor 更新后放开
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal11)
{
    auto tensor_self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({0}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 待castOp支持int16转float时候放开
TEST_F(l2_eq_tensor_test, test_eq_tensor_normal12)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 入参为self为null场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_self_is_null)
{
    auto tensor_self = nullptr;

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 入参为other为null场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_other_is_null)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto tensor_other = nullptr;

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参out为空指针的场景(结果待定，cpu输出空指针)
TEST_F(l2_eq_tensor_test, test_eq_tensor_input_out_is_null)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = nullptr;

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(workspace_size, 5u);
}

// // 入参self&other都为空tensor的场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_input_is_empty)
{
    auto tensor_self =
        TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_EQ(workspace_size, 0u);

    // 打开精度 golden报错 np.array(eval(value)).astype(dtype, copy=False).reshape(*storage_shape)
    // cannot reshape array of size 6 into shape (2,0)
}

// 入参self为空tensor的场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_input_self_is_empty)
{
    auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto tensor_other = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto out_tensor_desc = TensorDesc({2, 0}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_EQ(workspace_size, 0u);
}

// 入参Tensor 9维
TEST_F(l2_eq_tensor_test, test_eq_tensor_input_rank_9)
{
    auto tensor_self = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参self数据类型不支持的场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_self_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参oter数据类型不支持的场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_other_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参out数据类型不支持的场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_out_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_input_shape_greater_than_8)
{
    auto tensor_self = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参无法broadcast的场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_input_cannot_broadcast)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 正常场景 做broadcast后 outShape与broadcastShape不一致
TEST_F(l2_eq_tensor_test, test_eq_tensor_boradcast_not_equal_out)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 3, 1, 2, 3});

    auto tensor_other =
        TensorDesc({2, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 4, 3, 2, 1});

    auto out_tensor_desc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参dtype不支持场景
TEST_F(l2_eq_tensor_test, test_eq_tensor_self_dtype_undefined)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_other_dtype_undefined)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_out_dtype_undefined)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_self_format_undefined_normal)
{
    auto tensor_self = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_UNDEFINED)
                           .ValueRange(-10, 10)
                           .Value(vector<int8_t>{3, 4, 9, 6, 7, 11});

    auto tensor_other = TensorDesc({2, 3, 1, 1, 1}, ACL_INT8, ACL_FORMAT_UNDEFINED)
                            .ValueRange(-10, 10)
                            .Value(vector<int8_t>{3, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3, 1, 1, 1}, ACL_BOOL, ACL_FORMAT_NCDHW)
                               .Value(vector<bool>{false, false, false, false, false, false});
    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

// 非连续

TEST_F(l2_eq_tensor_test, test_eq_tensor_to_contiguous)
{
    auto tensor_self =
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Value(vector<float>{3, 4, 9, 6, 7, 11, 5, 89});

    auto tensor_other =
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Value(vector<float>{3, 4, 8, 6, 5, 11, 5, 80});

    auto out_tensor_desc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 数据范围[-1,1]
TEST_F(l2_eq_tensor_test, test_eq_tensor_date_range_f1_1)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(-1, 1)
            .Value(
                vector<float>{
                    -6.75468990e-01, 7.37894118e-01, -7.84537017e-01, 9.64608252e-01, 9.09311235e-01, -9.51533616e-01});
    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(-1, 1)
            .Value(
                vector<float>{
                    -6.75468990e-01, 7.37894118e-01, -7.94537017e-01, 9.64608252e-01, 9.09211235e-01, -9.51533616e-01});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(vector<bool>{false, false, false, false, false, false, false});
    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_eq_tensor_test, Ascend950PR_89_test_eq_tensor_normal_out_int32)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.1, 4, 9, 6, 7, 11});

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3.2, 4, 5, 6, 7, 8});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    // .Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_regbase_float16_tensor_int32_scalar)
{
    auto curSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_other = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(curSocVersion);
    }
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_regbase_complex64_tensor_float_scalar)
{
    auto curSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(vector<float>{1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0});
    auto tensor_other = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(curSocVersion);
    }
}

TEST_F(l2_eq_tensor_test, test_eq_tensor_regbase_bf16_tensor_bool_scalar)
{
    auto curSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    auto tensor_self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_other = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEqTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(curSocVersion);
    }
}