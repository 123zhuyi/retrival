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

#include "../../../op_api/aclnn_masked_fill_scalar.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_masked_fill_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "masked_fill_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "masked_fill_test TearDown" << endl;
    }
};

TEST_F(l2_masked_fill_scalar_test, case_001_FLOAT)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_fill_scalar_test, case_002_FLOAT16)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    ;
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常场景 bf16
TEST_F(l2_masked_fill_scalar_test, test_not_support_bf16)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 2);
    ;
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景910B SocVersion bf16
TEST_F(l2_masked_fill_scalar_test, ascend910B2_support_bf16_910B)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    ;
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 走aicpu，暂时不支持，先注释掉
// TEST_F(l2_masked_fill_scalar_test, case_003_INT64)
// {
//     auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto scalar_desc = ScalarDesc(8.88888f);

//     auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

TEST_F(l2_masked_fill_scalar_test, case_004_INT32)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_fill_scalar_test, case_006_INT8)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_fill_scalar_test, case_008_BOOL)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 2);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(0);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_fill_scalar_test, case_013_NHWC)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor
TEST_F(l2_masked_fill_scalar_test, case_014_EMPTY)
{
    auto self_tensor_desc = TensorDesc({2, 0, 3}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({2, 0, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_fill_scalar_test, case_015_CONTINUOUS)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_INT8, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-1, 10);
    auto mask_tensor_desc = TensorDesc({5, 4}, ACL_BOOL, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim超出限制
TEST_F(l2_masked_fill_scalar_test, case_016_MAX_DIM)
{
    auto self_tensor_desc = TensorDesc({2, 3, 2, 3, 2, 2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({2, 3, 2, 3, 2, 2, 2, 2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull
TEST_F(l2_masked_fill_scalar_test, case_017_NULLPTR)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(nullptr, mask_tensor_desc, scalar_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, nullptr, scalar_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size2 = 0;
    auto aclRet2 = ut.TestGetWorkspaceSize(&workspace_size2);
    EXPECT_EQ(aclRet2, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, nullptr), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size3 = 0;
    auto aclRet3 = ut.TestGetWorkspaceSize(&workspace_size3);
    EXPECT_EQ(aclRet3, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_masked_fill_scalar_test, case_018_DTYPE)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_masked_fill_scalar_test, case_mask_float)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat
TEST_F(l2_masked_fill_scalar_test, case_019_FORMAT)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT32, ACL_FORMAT_NC1HWC0).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 满足broadcast
TEST_F(l2_masked_fill_scalar_test, case_020_BROADCAST)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({3, 4}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 不满足broadcast
TEST_F(l2_masked_fill_scalar_test, case_021_BROADCAST_NOK)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({2, 3, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dim=0
TEST_F(l2_masked_fill_scalar_test, case_022_DIM_0)
{
    auto self_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1.0, 1.5);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim=1
TEST_F(l2_masked_fill_scalar_test, case_023_DIM_1)
{
    auto self_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({100}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_fill_scalar_test, case_024_FORMAT_NCHW)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_fill_scalar_test, case_025_FORMAT_NHWC)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_fill_scalar_test, case_026_FORMAT_HWCN)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_fill_scalar_test, case_027_FORMAT_NDHWC)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_fill_scalar_test, case_028_FORMAT_NCDHW)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-10, 10);
    auto mask_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scalar_desc = ScalarDesc(8.88888f);

    auto ut = OP_API_UT(aclnnInplaceMaskedFillScalar, INPUT(self_tensor_desc, mask_tensor_desc, scalar_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
