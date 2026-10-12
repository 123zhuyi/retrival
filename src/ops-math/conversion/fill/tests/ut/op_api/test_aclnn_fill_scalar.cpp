/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_fill_scalar.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_fill_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "fill_scalar_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "fill_scalar_test TearDown" << endl;
    }
};

TEST_F(l2_fill_scalar_test, case_001_FLOAT)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_002_FLOAT16)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_003_INT32)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_004_INT64)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_005_INT16)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_006_INT8)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_007_UINT8)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_008_BOOL)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 2);
    auto scalar_desc = ScalarDesc(1);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_009_DOUBLE)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_010_COMPLEX64)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_011_COMPLEX128)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_012_NHWC)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_013_NCHW)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_014_HWCN)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_015_NDHWC)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_016_NCDHW)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_017_NC1HWC0)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT32, ACL_FORMAT_NC1HWC0).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_018_EMPTY)
{
    auto self_tensor_desc = TensorDesc({2, 0, 3}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_019_CONTINUOUS)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_INT8, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-1, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_020_MAX_DIM)
{
    auto self_tensor_desc = TensorDesc({2, 3, 2, 3, 2, 2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fill_scalar_test, case_021_DIM_0)
{
    auto self_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_022_DIM_1)
{
    auto self_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_023_DIM_3)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_024_DIM_4)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_025_DIM_5)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_026_DIM_6)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_027_DIM_7)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_028_DIM_8)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_029_SCALAR_INT)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_030_SCALAR_DOUBLE)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_031_SCALAR_BOOL)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto scalar_desc = ScalarDesc(true);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_scalar_test, case_032_NOT_SUPPORT_DTYPE)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fill_scalar_test, case_033_SELF_NULLPTR)
{
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(nullptr, scalar_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fill_scalar_test, case_034_VALUE_NULLPTR)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(self_tensor_desc, nullptr), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
