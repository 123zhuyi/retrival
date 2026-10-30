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

#include "../../../op_api/aclnn_trunc.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_trunc_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Trunc Test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "Trunc Test TearDown" << std::endl;
    }
};

TEST_F(l2_trunc_test, case_dtype_float16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({1, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trunc_test, case_dtype_float32)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trunc_test, case_dtype_bfloat16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trunc_test, case_shape_1D)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trunc_test, case_shape_2D)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trunc_test, case_shape_3D)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trunc_test, case_shape_4D)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trunc_test, case_shape_5D)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trunc_test, case_shape_8D)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trunc_test, case_invalid_shape_9D)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trunc_test, case_empty_tensor)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(tensor_desc);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_trunc_test, case_null_self)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(nullptr), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_trunc_test, case_null_out)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_trunc_test, case_invalid_dtype_self)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(tensor_desc);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trunc_test, case_invalid_dtype_mismatch)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trunc_test, case_invalid_shape_diff)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trunc_test, case_not_contiguous)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {2, 2}).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTrunc, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}