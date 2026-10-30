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

#include "aclnn_rsqrt.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_rsqrt_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Rsqrt Test Setup" << endl; }
    static void TearDownTestCase() { cout << "Rsqrt Test TearDown" << endl; }
};

// UT for aclnnRsqrt

TEST_F(l2_rsqrt_test, case_normal)
{
    auto tensor_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsqrt, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 空tensor
TEST_F(l2_rsqrt_test, case_empty_tensor)
{
    auto self_tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnRsqrt, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull self
TEST_F(l2_rsqrt_test, case_nullptr_self)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRsqrt, INPUT(nullptr), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull out
TEST_F(l2_rsqrt_test, case_nullptr_out)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRsqrt, INPUT(tensor_desc), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_rsqrt_test, case_dtype_valid)
{
    vector<aclDataType> ValidList = {
        ACL_FLOAT,
        ACL_FLOAT16,
        ACL_DOUBLE,
        ACL_COMPLEX64,
        ACL_COMPLEX128};
    auto length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({1, 1, 4, 4}, ValidList[i], ACL_FORMAT_ND);
        auto out_desc = TensorDesc({1, 1, 4, 4}, ValidList[i], ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRsqrt, INPUT(self_desc), OUTPUT(out_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

// CheckDtypeValid cast
TEST_F(l2_rsqrt_test, case_dtype_valid_cast)
{
    vector<aclDataType> ValidList = {
        ACL_UINT8,
        ACL_INT8,
        ACL_INT16,
        ACL_INT32,
        ACL_INT64};
    auto length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({1, 1, 4, 4}, ValidList[i], ACL_FORMAT_ND);
        auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRsqrt, INPUT(self_desc), OUTPUT(out_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

// Check invalid out dtype
TEST_F(l2_rsqrt_test, case_dtype_invalid_out)
{
    vector<aclDataType> ValidList = {
        ACL_UINT8,
        ACL_INT8,
        ACL_INT16,
        ACL_INT32,
        ACL_INT64};
    auto length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
        auto out_desc = TensorDesc({1, 1, 4, 4}, ValidList[i], ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRsqrt, INPUT(self_desc), OUTPUT(out_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// check format
TEST_F(l2_rsqrt_test, case_format)
{
    vector<aclFormat> ValidList = {
        ACL_FORMAT_UNDEFINED,
        ACL_FORMAT_NCHW,
        ACL_FORMAT_NHWC,
        ACL_FORMAT_ND,
        ACL_FORMAT_NC1HWC0,
        ACL_FORMAT_FRACTAL_Z,
        ACL_FORMAT_NC1HWC0_C04,
        ACL_FORMAT_HWCN,
        ACL_FORMAT_NDHWC,
        ACL_FORMAT_FRACTAL_NZ,
        ACL_FORMAT_NCDHW,
        ACL_FORMAT_NDC1HWC0,
        ACL_FRACTAL_Z_3D};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ValidList[i]);
        auto out_desc = TensorDesc({1, 16, 1, 1},  ACL_FLOAT, ValidList[i]);

        auto ut = OP_API_UT(aclnnRsqrt, INPUT(self_desc), OUTPUT(out_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

// check not contiguous AICORE
TEST_F(l2_rsqrt_test, case_not_contiguous_aicore)
{
    auto self_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 1000);
    auto out_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsqrt, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// check not contiguous AICPU
TEST_F(l2_rsqrt_test, case_not_contiguous_aicpu)
{
    auto self_desc = TensorDesc({5, 4}, ACL_DOUBLE, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 1000);
    auto out_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsqrt, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// check dim num valid
TEST_F(l2_rsqrt_test, case_dim_8)
{
    auto self_tensor_desc = TensorDesc({1,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnRsqrt, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// check dim num invalid
TEST_F(l2_rsqrt_test, case_dim_9)
{
    auto self_tensor_desc = TensorDesc({1,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnRsqrt, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


// UT for aclnnInplaceRsqrt

TEST_F(l2_rsqrt_test, case_inplace_normal)
{
    auto tensor_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
        .Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnInplaceRsqrt, INPUT(tensor_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 空tensor
TEST_F(l2_rsqrt_test, case_inplace_empty_tensor)
{
    auto self_tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceRsqrt, INPUT(self_tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckDtypeValid
TEST_F(l2_rsqrt_test, case_inplace_dtype_valid)
{
    vector<aclDataType> ValidList = {
        ACL_FLOAT,
        ACL_FLOAT16,
        ACL_DOUBLE,
        ACL_COMPLEX64,
        ACL_COMPLEX128};
    auto length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({1, 1, 4, 4}, ValidList[i], ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnInplaceRsqrt, INPUT(self_desc), OUTPUT());

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

// Check inplace invalid dtype 
TEST_F(l2_rsqrt_test, case_inplace_dtype_invalid)
{
    vector<aclDataType> ValidList = {
        ACL_UINT8,
        ACL_INT8,
        ACL_INT16,
        ACL_INT32,
        ACL_INT64};
    auto length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({1, 1, 4, 4}, ValidList[i], ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnInplaceRsqrt, INPUT(self_desc), OUTPUT());

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// self的数据类型不在支持范围内
TEST_F(l2_rsqrt_test, l2_inplace_Sqrt_test_uint64) {
  auto selfDesc = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceRsqrt, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}