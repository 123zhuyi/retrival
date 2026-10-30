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
#include "../../../op_api/aclnn_rsub.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_rsub_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_rsub_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_rsub_test TearDown" << std::endl;
    }
};

TEST_F(l2_rsub_test, case_nullptr)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRsub, INPUT(nullptr, otherDesc, alphaDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnRsub, INPUT(selfDesc, nullptr, alphaDesc), OUTPUT(outDesc));
    workspaceSize = 0;
    aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, nullptr), OUTPUT(outDesc));
    workspaceSize = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(nullptr));
    workspaceSize = 0;
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_rsub_test, case_self_null_tensor)
{
    auto selfDesc = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_rsub_test, case_other_null_tensor)
{
    auto selfDesc = TensorDesc({2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({2, 1, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_rsub_test, case_dim1_float_nd)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(0.5f);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_rsub_test, case_dim2_float16_nd)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(-1.2f, ACL_FLOAT16);
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// TEST_F(l2_rsub_test, case_dim3_double_nd)
// {
//     auto selfDesc = TensorDesc({2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
//     auto otherDesc = TensorDesc({2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
//     auto alphaDesc = ScalarDesc(2.4);
//     auto outDesc = TensorDesc({2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);

//     auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

TEST_F(l2_rsub_test, case_dim4_uint8_nchw)
{
    auto selfDesc = TensorDesc({2, 3, 3, 2}, ACL_UINT8, ACL_FORMAT_NCHW).ValueRange(-5, 5);
    auto otherDesc = TensorDesc({2, 3, 3, 2}, ACL_UINT8, ACL_FORMAT_NCHW).ValueRange(-5, 5);
    auto alphaDesc = ScalarDesc(static_cast<uint8_t>(2));
    auto outDesc = TensorDesc({2, 3, 3, 2}, ACL_UINT8, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_rsub_test, case_dim4_int8_nhwc)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_NHWC).ValueRange(-5, 5);
    auto otherDesc = TensorDesc({2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_NHWC).ValueRange(-5, 5);
    auto alphaDesc = ScalarDesc(static_cast<int8_t>(2));
    auto outDesc = TensorDesc({2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_NHWC).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// TEST_F(l2_rsub_test, case_dim4_int16_hwcn)
// {
//     auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-10, 10);
//     auto otherDesc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).ValueRange(-10, 10);
//     auto alphaDesc = ScalarDesc(static_cast<int16_t>(2));
//     auto outDesc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_HWCN).Precision(0.001, 0.001);

//     auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

TEST_F(l2_rsub_test, case_dim5_int32_ndhwc)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_INT32, ACL_FORMAT_NDHWC).ValueRange(-10, 10);
    auto otherDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_INT32, ACL_FORMAT_NDHWC).ValueRange(-10, 10);
    auto alphaDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_INT32, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_rsub_test, case_dim5_int64_ncdhw)
{
    auto selfDesc = TensorDesc({2, 3, 4, 6, 1}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(-20, 20);
    auto otherDesc = TensorDesc({2, 3, 4, 6, 1}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(-20, 20);
    auto alphaDesc = ScalarDesc(static_cast<int64_t>(2));
    auto outDesc = TensorDesc({2, 3, 4, 6, 1}, ACL_INT64, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_rsub_test, case_dim6_complex64_nd)
{
    auto selfDesc = TensorDesc({2, 3, 4, 6, 1, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto otherDesc = TensorDesc({2, 3, 4, 6, 1, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto alphaDesc = ScalarDesc(2.0f);
    auto outDesc = TensorDesc({2, 3, 4, 6, 1, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_rsub_test, case_dim7_complex128_nd)
// {
//     auto selfDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-5, 5);
//     auto otherDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-5, 5);
//     auto alphaDesc = ScalarDesc(2.1);
//     auto outDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.001, 0.001);

//     auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

TEST_F(l2_rsub_test, case_dim8_float_nd)
{
    auto selfDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto otherDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto alphaDesc = ScalarDesc(2.1f);
    auto outDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_rsub_test, case_error_self_shape)
{
    auto selfDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto otherDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto alphaDesc = ScalarDesc(2.1f);
    auto outDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rsub_test, case_error_other_shape)
{
    auto selfDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto otherDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto alphaDesc = ScalarDesc(2.1f);
    auto outDesc = TensorDesc({2, 3, 4, 6, 1, 5, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rsub_test, case_error_out_shape)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto otherDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto alphaDesc = ScalarDesc(2.1f);
    auto outDesc = TensorDesc({2, 3, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rsub_test, case_error_self_dtype)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto otherDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto alphaDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outDesc = TensorDesc({2, 3, 4}, ACL_UINT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rsub_test, case_error_other_dtype)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto otherDesc = TensorDesc({2, 3, 4}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto alphaDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rsub_test, case_can_not_promote_type)
{
    auto selfDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rsub_test, case_alpha_can_not_cast)
{
    auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rsub_test, case_can_not_cast_out)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rsub_test, case_alpha_is_1)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// TEST_F(l2_rsub_test, case_diff_dtype)
// {
//     auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto otherDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto alphaDesc = ScalarDesc(1.2f);
//     auto outDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

//     auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_rsub_test, case_not_contiguous)
// {
//     auto selfDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5});
//     auto otherDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5});
//     auto alphaDesc = ScalarDesc(1.2f);
//     auto outDesc = TensorDesc({3, 5, 6}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

//     auto ut = OP_API_UT(aclnnRsub, INPUT(selfDesc, otherDesc, alphaDesc), OUTPUT(outDesc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

TEST_F(l2_rsub_test, Ascend950PR_89_case_001)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnRsub, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_rsub_test, Ascend950PR_89_case_002)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto scalar_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnRsub, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_rsub_test, Ascend950PR_89_case_003)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto scalar_desc = ScalarDesc(static_cast<int32_t>(2.0f));

    auto ut = OP_API_UT(aclnnRsub, INPUT(self_tensor_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}