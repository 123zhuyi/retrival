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

#include "conversion/im2col/op_api/aclnn_im2col.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;
class l2_im2col_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "im2col_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "im2col_test TearDown" << endl;
    }
};

TEST_F(l2_im2col_test, case_BF16)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_BF16, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, case_FLOAT)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, case_FLOAT16)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, case_INT8)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_test, ascend950_case_INT8)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_UINT8)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_UINT8, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_INT16)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_INT16, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_UINT16)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_UINT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_UINT16, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_INT32)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_INT32, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_UINT32)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_UINT32, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_UINT32, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_INT64)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_INT64, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_UINT64)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_UINT64, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_UINT64, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_BF16)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_BF16, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_FLOAT)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_FLOAT16)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_DOUBLE)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_BOOL)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_BOOL, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_COMPLEX32)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_COMPLEX32, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_COMPLEX32, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_COMPLEX64)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, ascend950_case_COMPLEX128)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_test, case_range_value)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);

    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, case_dim3_FLOAT)
{
    auto tensor_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({8, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, case_not_contiguous)
{
    auto tensor_desc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {12, 1, 3}, 0, {1, 4, 3});
    auto out_desc = TensorDesc({1, 12}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> kernel = {1, 1};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, case_NHWC)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_im2col_test, case_error_tensor_size)
{
    auto tensor_desc = TensorDesc({0, -2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> size = {2, 2};
    auto array_dec = IntArrayDesc(size);

    auto ut1 = OP_API_UT(aclnnIm2col, INPUT(tensor_desc, array_dec, array_dec, array_dec, array_dec), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_test, case_error_array_size)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> size = {2, 2};
    vector<int64_t> incorrect_size = {4, 5, 1};
    auto array_dec = IntArrayDesc(size);
    auto incorrect_array_dec = IntArrayDesc(incorrect_size);

    auto ut1 = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, incorrect_array_dec, array_dec, array_dec, array_dec), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut2 = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, array_dec, incorrect_array_dec, array_dec, array_dec), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut3 = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, array_dec, array_dec, incorrect_array_dec, array_dec), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut4 = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, array_dec, array_dec, array_dec, incorrect_array_dec), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_test, case_error_array_num)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> size = {2, 2};
    vector<int64_t> incorrect_num = {-1, -1};
    auto array_dec = IntArrayDesc(size);
    auto incorrect_array_dec = IntArrayDesc(incorrect_num);

    auto ut1 = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, incorrect_array_dec, array_dec, array_dec, array_dec), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut2 = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, array_dec, incorrect_array_dec, array_dec, array_dec), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut3 = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, array_dec, array_dec, incorrect_array_dec, array_dec), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut4 = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, array_dec, array_dec, array_dec, incorrect_array_dec), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_test, case_NULLPTR)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut1 =
        OP_API_UT(aclnnIm2col, INPUT(nullptr, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 =
        OP_API_UT(aclnnIm2col, INPUT(tensor_desc, nullptr, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 =
        OP_API_UT(aclnnIm2col, INPUT(tensor_desc, kernel_desc, nullptr, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut4 =
        OP_API_UT(aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, nullptr, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut5 =
        OP_API_UT(aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, nullptr), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut5.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut6 = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut6.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_im2col_test, case_error_output_shape)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> kernel = {5, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_test, case_error_output_shape2)
{
    auto tensor_desc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 10, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {1, 1};
    vector<int64_t> stride = {2, 2};
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(
        aclnnIm2col, INPUT(tensor_desc, kernel_desc, dilation_desc, padding_desc, stride_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
