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

#include "math/less/op_api/aclnn_lt_tensor.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_lt_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "lt_tensor_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "lt_tensor_test TearDown" << std::endl; }
};

// aclnnLtTensor_001:lt.Tensor_out输入支持FLOAT
TEST_F(l2_lt_tensor_test, aclnnLtTensor_001_float_ND)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_002:lt.Tensor_out输入支持FLOAT16
TEST_F(l2_lt_tensor_test, aclnnLtTensor_002_float16_NHWC)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NHWC).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_003:lt.Tensor_out输入支持INT32
TEST_F(l2_lt_tensor_test, aclnnLtTensor_003_int32_NCHW)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_004:lt.Tensor_out输入支持INT64
TEST_F(l2_lt_tensor_test, aclnnLtTensor_004_int64_NCHW_ND)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_005:lt.Tensor_out输入支持INT16
// 走AICPU
TEST_F(l2_lt_tensor_test, aclnnLtTensor_005_int16_HWCN)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto tensor_other = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_006:lt.Tensor_out输入支持INT8
TEST_F(l2_lt_tensor_test, aclnnLtTensor_006_int8_NDHWC)
{
    auto tensor_self = TensorDesc({5, 2, 5, 6, 3}, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({5, 2, 5, 6, 3}, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_BOOL, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_007:lt.Tensor_out输入支持UINT8
TEST_F(l2_lt_tensor_test, aclnnLtTensor_007_uint8_NCDHW)
{
    auto tensor_self = TensorDesc({5, 2, 5, 6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({5, 2, 5, 6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({5, 2, 5, 6, 3}, ACL_BOOL, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_008:lt.Tensor_out输入支持BOOL
// 走AICPU
TEST_F(l2_lt_tensor_test, aclnnLtTensor_008_bool_ND)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_009:lt.Tensor_out输入支持BFLOAT16
// 走AICPU
TEST_F(l2_lt_tensor_test, ascend910B2_aclnnLtTensor_009_bf16)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensor_other = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_lt_tensor_test, aclnnLtTensor_009_bf16)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensor_other = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnLtTensor_010:lt.Tensor_out输入支持DOUBLE
// 走AICPU
TEST_F(l2_lt_tensor_test, aclnnLtTensor_010_double)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensor_other = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_013:lt.Tensor_out支持输入和输出类型不一致
TEST_F(l2_lt_tensor_test, aclnnLtTensor_013_float16_float16_bool_NHWC)
{
    auto tensor_self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_BOOL, ACL_FORMAT_NHWC).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_014:lt.Tensor_out支持输入和输出类型不一致
TEST_F(l2_lt_tensor_test, aclnnLtTensor_014_int32_int32_bool_ND)
{
    auto tensor_self = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_015:lt.Tensor_out支持输入的类型不一致
TEST_F(l2_lt_tensor_test, aclnnLtTensor_015_float_int32_bool_broadcast_ND)
{
    auto tensor_self = TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_016:lt.Tensor_out支持输入为空Tensor
TEST_F(l2_lt_tensor_test, aclnnLtTensor_016_input_empty_tensor)
{
    auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// aclnnLtTensor_017:lt.Tensor_out支持输入为非连续Tensor
TEST_F(l2_lt_tensor_test, aclnnLtTensor_017_input_not_contiguous)
{
    auto tensor_self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnLtTensor_019：lt.Tensor_out异常shape测试
TEST_F(l2_lt_tensor_test, aclnnLtTensor_019_error_format)
{
    auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnLtTensor_020：lt.Tensor_out空指针场景测试
TEST_F(l2_lt_tensor_test, aclnnLtTensor_020_null_pointer)
{
    auto tensor_self = nullptr;
    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(workspace_size, 5u);
}

// aclnnLtTensor_021：lt.Tensor_out异常数据类型测试
TEST_F(l2_lt_tensor_test, aclnnLtTensor_021_error_dtype)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// aclnnLtTensor_022：lt.Tensor_out异常shape测试
TEST_F(l2_lt_tensor_test, aclnnLtTensor_022_error_shape)
{
    auto self_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto other_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnLtTensor_023：lt.Tensor_out异常shape测试
TEST_F(l2_lt_tensor_test, aclnnLtTensor_023_error_shape)
{
    auto self_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto other_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto out_tensor_desc =
        TensorDesc({10, 5, 10, 5}, ACL_BOOL, ACL_FORMAT_NHWC).ValueRange(-10, 10).Precision(0.0001, 0.0001);

    auto ut_2 = OP_API_UT(aclnnLtTensor, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 其他用例
// 入参为other为null场景
TEST_F(l2_lt_tensor_test, test_lt_tensor_other_is_null)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{9, 3, 1, 1, 2, 3});

    auto tensor_other = nullptr;

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参out为空指针的场景(结果待定，cpu输出空指针)
TEST_F(l2_lt_tensor_test, test_lt_tensor_input_out_is_null)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = nullptr;

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 入参self&other都为空tensor的场景
TEST_F(l2_lt_tensor_test, test_lt_tensor_input_is_empty)
{
    auto tensor_self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto tensor_other = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 入参other为空tensor的场景
TEST_F(l2_lt_tensor_test, test_lt_tensor_input_other_is_empty)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto tensor_other = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {});

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参self数据类型不支持的场景
TEST_F(l2_lt_tensor_test, test_lt_tensor_self_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{9, 3, 1, 1, 2, 3});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

TEST_F(l2_lt_tensor_test, test_lt_tensor_regbase_bool_tensor_bool_scalar)
{
    auto curSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    auto tensor_self =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, true, false, true, false});
    auto tensor_other = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(curSocVersion);
    }
}

TEST_F(l2_lt_tensor_test, test_lt_tensor_regbase_float16_tensor_int32_scalar)
{
    auto curSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(curSocVersion);
    }
}

TEST_F(l2_lt_tensor_test, test_lt_tensor_regbase_float_tensor_bool_scalar_out_float)
{
    auto curSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tensor_other = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(curSocVersion);
    }
}

// 入参oter数据类型不支持的场景
TEST_F(l2_lt_tensor_test, test_lt_tensor_other_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other =
        TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{9, 3, 1, 1, 2, 3});

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参out数据类型不支持的场景
TEST_F(l2_lt_tensor_test, test_lt_tensor_out_dtype_invalid)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_STRING, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}

// 入参无法broadcast的场景
TEST_F(l2_lt_tensor_test, test_lt_tensor_input_cannot_broadcast)
{
    auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto tensor_other = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnLtTensor, INPUT(tensor_self, tensor_other), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspace_size, 5u);
}
