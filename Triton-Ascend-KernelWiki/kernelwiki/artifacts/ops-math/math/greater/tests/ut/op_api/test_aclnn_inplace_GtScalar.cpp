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

#include "math/greater/op_api/aclnn_gt_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class inplaceGtScalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "inplaceGtScalar_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "inplaceGtScalar_test TearDown" << endl;
    }
};

TEST_F(inplaceGtScalar_test, case_norm_float32)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(0.5f);

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceGtScalar_test, case_norm_float16)
{
    auto tensor_desc = TensorDesc({1, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto scalar_desc = ScalarDesc(2.0f); // 当前scalar不支持float16

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceGtScalar_test, case_norm_int64)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_INT64, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceGtScalar_test, case_norm_int32)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_INT32, ACL_FORMAT_HWCN)
                           .Value(vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(5));

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceGtScalar_test, case_norm_int8)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_INT8, ACL_FORMAT_ND)
                           .Value(vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    int8_t val = 5;
    auto scalar_desc = ScalarDesc(val);

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceGtScalar_test, case_norm_uint8)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1, 1}, ACL_UINT8, ACL_FORMAT_NDHWC).ValueRange(0, 100);
    uint8_t val = 10;
    auto scalar_desc = ScalarDesc(val);

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceGtScalar_test, case_nullptr_self)
{
    auto tensor_desc = nullptr;
    auto scalar_desc = ScalarDesc(0.5f);

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(inplaceGtScalar_test, case_nullptr_other)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = nullptr;

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(inplaceGtScalar_test, case_dtype_invalid_self)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(8.0f);

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(inplaceGtScalar_test, case_format_internal)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(8.0f);

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceGtScalar_test, case_empty_tensor)
{
    auto tensor_desc = TensorDesc({1, 16, 0, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(8.0f);

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceGtScalar_test, case_dim_invalid)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(8.0f);

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(inplaceGtScalar_test, case_format_abnormal)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_UNDEFINED).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(0.5f);

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(inplaceGtScalar_test, case_dtype_abnormal)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_DT_UNDEFINED, ACL_FORMAT_NHWC);
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(5));

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(inplaceGtScalar_test, case_strided)
{
    auto tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto scalar_desc = ScalarDesc(0.0f);

    auto ut = OP_API_UT(aclnnInplaceGtScalar, INPUT(tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
