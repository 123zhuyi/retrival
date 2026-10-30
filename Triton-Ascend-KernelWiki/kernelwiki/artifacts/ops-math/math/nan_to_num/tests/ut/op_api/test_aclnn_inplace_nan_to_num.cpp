/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_nan_to_num.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2InplaceNanToNumTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2InplaceNanToNumTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2InplaceNanToNumTest TearDown" << std::endl;
    }
};

// 异常场景：self为空指针，返回ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2InplaceNanToNumTest, l2_nan_to_num_test_err_self_null)
{
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT((aclTensor*)nullptr, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：self的数据类型不在支持范围内，返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceNanToNumTest, l2_nan_to_num_test_err_self_double)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self的数据类型不在支持范围内，返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceNanToNumTest, l2_nan_to_num_test_err_self_complex)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：Tensor为9维、返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceNanToNumTest, l2_nan_to_num_test_err_dim_over)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景：空tensor，返回ACLNN_SUCCESS
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_err_empty_tensor)
{
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：float16、6维、nan/posinf/neginf为int，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_fp16_dim6_para_int)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7, 9, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    int nan = 0;
    int posinf = 1;
    int neginf = -1;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：float16、6维，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_fp16_dim6)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7, 9, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：float32、7维，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_fp32_dim7)
{
    auto selfDesc = TensorDesc({2, 4, 7, 9, 1, 2, 12}, ACL_FLOAT, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：bfloat16、8维，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_bf16_dim8)
{
    auto selfDesc = TensorDesc({2, 4, 7, 9, 1, 2, 12, 23}, ACL_BF16, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：ACL_INT8、1维，ACLNN_SUCCESS
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_int8_dim1)
{
    auto selfDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：ACL_INT16、2维，ACLNN_SUCCESS
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_int16_dim2)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：ACL_INT32、3维，ACLNN_SUCCESS
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_int32_dim3)
{
    auto selfDesc = TensorDesc({2, 4, 12}, ACL_INT32, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：ACL_INT64、4维，ACLNN_SUCCESS
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_int64_dim4)
{
    auto selfDesc = TensorDesc({2, 4, 12, 9}, ACL_INT64, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：ACL_UINT8、5维，ACLNN_SUCCESS
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_uint8_dim5)
{
    auto selfDesc = TensorDesc({2, 4, 12, 9, 21}, ACL_UINT8, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：ACL_BOOL、6维，ACLNN_SUCCESS
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_bool_dim6)
{
    auto selfDesc = TensorDesc({2, 4, 12, 9, 21, 32}, ACL_BOOL, ACL_FORMAT_ND);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：float32、4维、ACL_FORMAT_NCHW，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_float32_nchw)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：float32、4维、ACL_FORMAT_NHWC，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_float32_nhwc)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：float32、4维、ACL_FORMAT_NC1HWC0，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_float32_nc1hwc0)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常场景：float32、4维、ACL_FORMAT_HWCN，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_float32_hwcn)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 成功场景：浮点型、non-continuous，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_float32_non_continuous)
{
    auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 成功场景：整型、non-continuous，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceNanToNumTest, ascend910B2_l2_nan_to_num_test_int_non_continuous)
{
    auto selfDesc = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    float nan = 0;
    float posinf = 1.0;
    float neginf = -1.0;

    auto ut = OP_API_UT(aclnnInplaceNanToNum, INPUT(selfDesc, nan, posinf, neginf), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
