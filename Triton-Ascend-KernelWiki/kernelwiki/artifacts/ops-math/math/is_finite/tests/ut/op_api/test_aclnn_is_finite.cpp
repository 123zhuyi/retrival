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

#include "opdev/platform.h"
#include "../../../op_api/aclnn_isfinite.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_isfinite_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_isfinite_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_isfinite_test TearDown" << endl;
    }
};

TEST_F(l2_isfinite_test, case_01_float)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc =
        TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1.023, 2023.08, 3.14, 10987654321.0});
    auto outDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_02_float16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);

    auto selfDesc =
        TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<double>{-1, 2.718, 3.14, 49999});
    auto outDesc = TensorDesc({1, 1, 2, 2}, ACL_BOOL, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_03_bfloat16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc =
        TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{-1.023, 2023.08, 3.14, 10987654321.0});
    auto outDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_04_double)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_05_int32)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc =
        TensorDesc({1, 1, 2, 2}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{-1, 2023, 314, 987654321});
    auto outDesc = TensorDesc({1, 1, 2, 2}, ACL_BOOL, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_06_int64)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{-1, 2023, 314, 987654321});
    auto outDesc = TensorDesc({1, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_07_int16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({2, 2}, ACL_INT16, ACL_FORMAT_ND).Value(vector<int16_t>{-1, 2023, -32768, 32767});
    auto outDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_08_int8)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({5}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{-1, 0, -128, 127, 88});
    auto outDesc = TensorDesc({5}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_09_uint8)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({5}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{255, 0, 1, 127, 88});
    auto outDesc = TensorDesc({5}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_10_bool)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({5}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, false, false, false});
    auto outDesc = TensorDesc({5}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_11_empty_tensor)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_isfinite_test, case_12_self_out_diff_shape)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isfinite_test, case_13_self_null)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = nullptr;
    auto outDesc = TensorDesc({3, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_isfinite_test, case_14_out_null)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = nullptr;

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_isfinite_test, case_15_self_dim_gt_8)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isfinite_test, case_16_out_not_bool)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_isfinite_test, case_17_not_contiguous)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {2, 2}).Value(vector<float>{1, 2, 3, 4});
    auto outDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsFinite, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}