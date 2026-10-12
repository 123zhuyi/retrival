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

#include "math/cross/op_api/aclnn_linalg_cross.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_linalg_cross_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LinalgCross Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "LinalgCross Test TearDown" << std::endl; }
};

// 不支持类型校验
TEST_F(l2_linalg_cross_test, case_invalid_dtype_bool)
{
    auto self_tensor_desc = TensorDesc({1, 1, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    int64_t dim = 2;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2_linalg_cross_test, case_empty_tensor)
{
    auto self_tensor_desc = TensorDesc({1, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 2;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor
TEST_F(l2_linalg_cross_test, case_empty_tensor_other_broadcast)
{
    auto self_tensor_desc = TensorDesc({1, 1, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 3;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor
TEST_F(l2_linalg_cross_test, case_empty_tensor_self_broadcast)
{
    auto self_tensor_desc = TensorDesc({1, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(other_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 2;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor
TEST_F(l2_linalg_cross_test, case_empty_tensor_invalid_broadcast)
{
    auto self_tensor_desc = TensorDesc({1, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(other_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 2;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2_linalg_cross_test, case_empty_tensor_empty_other_broadcast)
{
    auto self_tensor_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(other_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 2;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// CheckNotNull self
TEST_F(l2_linalg_cross_test, case_nullptr_self)
{
    auto other_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(other_tensor_desc);
    int64_t dim = 3;
    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(nullptr, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull other
TEST_F(l2_linalg_cross_test, case_nullptr_other)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    int64_t dim = 3;
    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, nullptr, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull out
TEST_F(l2_linalg_cross_test, case_nullptr_out)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 3;
    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// check 1D tensor
TEST_F(l2_linalg_cross_test, case_1D_tensor)
{
    auto self_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// check 3D tensor
TEST_F(l2_linalg_cross_test, case_3D_tensor)
{
    auto self_tensor_desc = TensorDesc({1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 2;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// check 4D tensor
TEST_F(l2_linalg_cross_test, case_4D_tensor)
{
    auto self_tensor_desc = TensorDesc({1,1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1,1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 3;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// check 5D tensor
TEST_F(l2_linalg_cross_test, case_5D_tensor)
{
    auto self_tensor_desc = TensorDesc({1,1,1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1,1,1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 4;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// check 8D tensor
TEST_F(l2_linalg_cross_test, case_8D_tensor)
{
    auto self_tensor_desc = TensorDesc({1,1,1,1,1,1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1,1,1,1,1,1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 7;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// check invalid 9D tensor
TEST_F(l2_linalg_cross_test, case_invalid_9D_tensor)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 8;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDiffDtype
TEST_F(l2_linalg_cross_test, case_diff_type_self_other)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_INT16, ACL_FORMAT_ND);
    int64_t dim = 3;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDiffDtype
TEST_F(l2_linalg_cross_test, case_diff_type_self_out)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_INT16, ACL_FORMAT_ND);
    int64_t dim = 3;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeInvalid
TEST_F(l2_linalg_cross_test, case_invalid_shape_1)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 3;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeInvalid
TEST_F(l2_linalg_cross_test, case_invalid_shape_2)
{
    auto self_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = -1;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeInvalid
TEST_F(l2_linalg_cross_test, case_invalid_shape_3)
{
    auto self_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeInvalid
TEST_F(l2_linalg_cross_test, case_invalid_shape_4)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1, 1, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 3;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeInvalid
TEST_F(l2_linalg_cross_test, case_invalid_shape_5)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 3;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeBroadcast
TEST_F(l2_linalg_cross_test, case_self_expand_dim_broadcast)
{
    auto self_tensor_desc = TensorDesc({1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1,1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1,1,1,3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.1, 0.1);
    int64_t dim = 2;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckShapeBroadcast
TEST_F(l2_linalg_cross_test, case_other_expand_dim_broadcast)
{
    auto self_tensor_desc = TensorDesc({1,1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1,1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1,1,1,3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.1, 0.1);
    int64_t dim = 3;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckShapeBroadcast
TEST_F(l2_linalg_cross_test, case_self_expand_dim_other_expand_shape_broadcast)
{
    auto self_tensor_desc = TensorDesc({1,3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1,3,1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1,3,3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.1, 0.1);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckDimValid
TEST_F(l2_linalg_cross_test, case_dim_out_of_range)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 4;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDimValid
TEST_F(l2_linalg_cross_test, case_dim_shape_not_3)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 2;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// AICPU contiguous
TEST_F(l2_linalg_cross_test, case_contiguous_float)
{
    auto self_tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {3, 5}).ValueRange(-20, 20);
    auto other_tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {3, 5}).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.1, 0.1);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_linalg_cross_test, case_ascend910B_bf16_1) {
    auto self_tensor_desc = TensorDesc({5, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto other_tensor_desc = TensorDesc({5, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    int64_t dim = 1;

    auto ut = OP_API_UT(aclnnLinalgCross, INPUT(self_tensor_desc, other_tensor_desc, dim), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
