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

#include "../../../op_api/aclnn_ne_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;

class l2_ne_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ne_scalar_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ne_scalar_test TearDown" << std::endl;
    }
};

// 正常场景
TEST_F(l2_ne_scalar_test, test_ne_scalar_int8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_ne_scalar_test, test_ne_scalar_uint8)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_ne_scalar_test, test_ne_scalar_int16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_ne_scalar_test, test_ne_scalar_int32)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_ne_scalar_test, test_ne_scalar_int64)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_ne_scalar_test, test_ne_scalar_bool)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, true, false, true, false, false});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_ne_scalar_test, test_ne_scalar_float16)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_ne_scalar_test, test_ne_scalar_float)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景910B SocVersion bf16
TEST_F(l2_ne_scalar_test, ascend910B2_support_bf16_910B)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(std::vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast
TEST_F(l2_ne_scalar_test, test_ne_scalar_input_diffdtype)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<int32_t>{1, 2, 3, 1, 2, 3});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast
TEST_F(l2_ne_scalar_test, test_ne_scalar_output_diffdtype)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<int32_t>{1, 2, 3, 1, 2, 3});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast
TEST_F(l2_ne_scalar_test, test_ne_scalar_int64_int32)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_INT64, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<int32_t>{1, 2, 3, 1, 2, 3});

    int64_t value = 3;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast
TEST_F(l2_ne_scalar_test, test_ne_scalar_uint8_bool)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_UINT8, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<int32_t>{1, 2, 3, 1, 2, 3});

    int64_t value = 3;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast
TEST_F(l2_ne_scalar_test, test_ne_scalar_float_bool)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<int32_t>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 空tensor
TEST_F(l2_ne_scalar_test, test_ne_scalar_empty_tensor)
{
    auto tensor_self = TensorDesc({2, 3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 0}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常场景 shape不一致
TEST_F(l2_ne_scalar_test, test_ne_scalar_diffshape)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND)
                           .ValueRange(-10, 10)
                           .Value(std::vector<int32_t>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 空指针
TEST_F(l2_ne_scalar_test, test_ne_scalar_nullptr)
{
    auto tensor_self = nullptr;

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND)
                               .Value(std::vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非连续测试
TEST_F(l2_ne_scalar_test, case_non_contiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_BOOL, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_ne_scalar_test, test_ne_scalar_error_shape)
{
    auto self_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ne_scalar_test, test_ne_scalar_float_double)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_ne_scalar_test, Ascend950PR_89_case_non_contiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_BOOL, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_ne_scalar_test, Ascend950PR_89_case_norm_float32_with_double)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(static_cast<double>(0.5));

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ne_scalar_test, Ascend950PR_89_case_norm_float16_with_double)
{
    auto tensor_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto scalar_desc = ScalarDesc(static_cast<double>(2.0));

    auto ut = OP_API_UT(aclnnNeScalar, INPUT(tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
