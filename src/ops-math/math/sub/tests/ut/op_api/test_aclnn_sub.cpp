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
#include "../../../op_api/aclnn_sub.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_sub_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_sub_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_sub_test TearDown" << std::endl;
    }
};

// 正常场景_float32_nd
TEST_F(l2_sub_test, normal_dtype_float32_format_nd)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto scalarDesc = ScalarDesc(1.2f);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_float16_nhwc
TEST_F(l2_sub_test, normal_dtype_float16_format_nhwc)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto otherDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto scalarDesc = ScalarDesc(1.2f);
    auto outDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_NHWC).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // // ut.TestPrecision();
}

// 正常场景_double_nchw
// TEST_F(l2_sub_test, normal_dtype_double_format_nchw)
// {
//     auto selfDesc = TensorDesc({2, 5, 6, 3}, ACL_DOUBLE, ACL_FORMAT_NCHW).ValueRange(-2, 2);
//     auto otherDesc = TensorDesc({2, 5, 6, 3}, ACL_DOUBLE, ACL_FORMAT_NCHW).ValueRange(-2, 2);
//     auto scalarDesc = ScalarDesc(2.4f);
//     auto outDesc = TensorDesc({2, 5, 6, 3}, ACL_DOUBLE, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

//     auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// 正常场景_uint8_ncdhw
TEST_F(l2_sub_test, normal_dtype_uint8_format_ncdhw)
{
    auto selfDesc = TensorDesc({6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
    auto otherDesc = TensorDesc({2, 5, 6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
    auto scalarDesc = ScalarDesc(static_cast<int64_t>(2));
    auto outDesc = TensorDesc({2, 5, 6, 3}, ACL_UINT8, ACL_FORMAT_NCDHW).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_int8_ndhwc
TEST_F(l2_sub_test, normal_dtype_int8_format_ndhwc)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto otherDesc = TensorDesc({2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto scalarDesc = ScalarDesc(static_cast<int64_t>(2));
    auto outDesc = TensorDesc({2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_NDHWC).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_int16_hwcn
// TEST_F(l2_sub_test, normal_dtype_int16_format_hwcn)
// {
//     auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-2, 2);
//     auto otherDesc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-2, 2);
//     auto scalarDesc = ScalarDesc(static_cast<int64_t>(2));
//     auto outDesc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).Precision(0.001, 0.001);

//     auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// 正常场景_int32
TEST_F(l2_sub_test, normal_dtype_int32)
{
    auto selfDesc = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto otherDesc = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto scalarDesc = ScalarDesc(static_cast<int64_t>(2));
    auto outDesc = TensorDesc({7, 9, 11, 3, 4, 6}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_bool
TEST_F(l2_sub_test, normal_dtype_bool)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(static_cast<int64_t>(2));
    auto outDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_int64
TEST_F(l2_sub_test, normal_dtype_int64)
{
    auto selfDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto otherDesc = TensorDesc({7, 9, 11, 3, 4, 6, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto scalarDesc = ScalarDesc(static_cast<int64_t>(2));
    auto outDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // // ut.TestPrecision();  // comment bcz of timeout in model tests (316883 ms)
}

// 正常场景_alpha为1
TEST_F(l2_sub_test, normal_alpha_1)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_broadcast
TEST_F(l2_sub_test, normal_broadcast)
{
    auto selfDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(1.2f);
    auto outDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 空tensor场景
TEST_F(l2_sub_test, normal_empty_tensor)
{
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// Other为scalar立即数输入
TEST_F(l2_sub_test, normal_other_scalar)
{
    auto selfDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = ScalarDesc(1.0f);
    auto scalarDesc = ScalarDesc(1.2f);
    auto outDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSubs, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Other为scalar立即数输入
// TEST_F(l2_sub_test, normal_othercomplex_scalar)
// {
//     auto selfDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
//     float src_value = 1.0f;
//     aclScalar* otherDesc = aclCreateScalar(&src_value, aclDataType::ACL_COMPLEX64);
//     auto scalarDesc = ScalarDesc(1.2f);
//     auto outDesc = TensorDesc({10, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnSubs, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// Other为scalar立即数输入
TEST_F(l2_sub_test, Ascend910B2_normal_other_scalar_bf16)
{
    auto selfDesc = TensorDesc({10, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 100);
    auto otherDesc = ScalarDesc(1.0f);
    auto scalarDesc = ScalarDesc(1.2f);
    auto outDesc = TensorDesc({10, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 100);

    auto ut = OP_API_UT(aclnnSubs, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 1971
TEST_F(l2_sub_test, Ascend910B2_case)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 100);
    auto otherDesc = TensorDesc({2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 100);
    auto scalarDesc = ScalarDesc(1.2f);
    auto outDesc = TensorDesc({2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 100).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull_1
TEST_F(l2_sub_test, abnormal_self_nullptr)
{
    auto selfDesc = nullptr;
    auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_2
TEST_F(l2_sub_test, abnormal_other_nullptr)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = nullptr;
    auto scalarDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_3
TEST_F(l2_sub_test, abnormal_out_nullptr)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(1.0f);
    auto outDesc = nullptr;

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_1
TEST_F(l2_sub_test, abnormal_dtype_alpha_int64)
{
    auto selfDesc = TensorDesc({2}, ACL_UINT32, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_UINT32, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(static_cast<int64_t>(5));
    auto outDesc = TensorDesc({2}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_2
TEST_F(l2_sub_test, abnormal_dtype_other_float)
{
    auto selfDesc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto otherDesc = ScalarDesc(1.0f);
    auto scalarDesc = ScalarDesc(static_cast<int64_t>(5));
    auto outDesc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSubs, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// PromoteDtype_1
TEST_F(l2_sub_test, abnormal_dtype_promote)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(1.2f);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_1
TEST_F(l2_sub_test, abnormal_shape_self_other_not_broadcast)
{
    auto selfDesc = TensorDesc({5, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(1.2f);
    auto outDesc = TensorDesc({5, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_2
TEST_F(l2_sub_test, abnormal_shape_dim_greater_than_threshold)
{
    auto selfDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(1.2f);
    auto outDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSub, INPUT(selfDesc, otherDesc, scalarDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sub_test, Ascend950PR_89_case_001)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnSub, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_sub_test, Ascend950PR_89_case_axpy)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto scalar_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnSub, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_sub_test, Ascend950PR_89_case_002)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnSub, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_sub_test, Ascend950PR_89_case_axpy_v2)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(2.0f));

    auto ut = OP_API_UT(aclnnSub, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_sub_test, Ascend950PR_89_case_003)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_desc = ScalarDesc(1.0f);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnSubs, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_sub_test, Ascend950PR_89_case_004)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto scalar_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnSubs, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_sub_test, Ascend950PR_89_case_005)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_desc = ScalarDesc(static_cast<int32_t>(2.0f));
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(2.0f));

    auto ut = OP_API_UT(aclnnSubs, INPUT(self_tensor_desc, other_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}