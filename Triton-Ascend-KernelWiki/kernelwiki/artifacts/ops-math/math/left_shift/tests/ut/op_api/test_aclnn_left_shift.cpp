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
#include "../../../op_api/aclnn_left_shift.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_left_shift_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "left_shift_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "left_shift_test TearDown" << endl;
    }
};

// 正常场景_INT8_ND
TEST_F(l2_left_shift_test, normal_INT8_ND)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_INT16_NHWC
TEST_F(l2_left_shift_test, normal_INT16_NHWC)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_NHWC);
    auto shiftBitsDesc = TensorDesc({2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_NHWC);
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_INT32_HWCN
TEST_F(l2_left_shift_test, normal_INT32_HWCN)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_HWCN);
    auto shiftBitsDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_HWCN);
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_INT64_NDHWC
TEST_F(l2_left_shift_test, normal_INT64_NDHWC)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_NDHWC);
    auto shiftBitsDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_UINT16_ND
TEST_F(l2_left_shift_test, normal_UINT16_ND)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_UINT16, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_UINT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空tensor场景
TEST_F(l2_left_shift_test, normal_empty_tensor)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull_self_nullptr
TEST_F(l2_left_shift_test, abnormal_self_nullptr)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = nullptr;
    auto shiftBitsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_other_nullptr
TEST_F(l2_left_shift_test, abnormal_other_nullptr)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto shiftBitsDesc = nullptr;
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_out_nullptr
TEST_F(l2_left_shift_test, abnormal_out_nullptr)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = nullptr;

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_self_BF16
TEST_F(l2_left_shift_test, abnormal_self_BF16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_FLOAT
TEST_F(l2_left_shift_test, abnormal_self_FLOAT)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_FLOAT16
TEST_F(l2_left_shift_test, abnormal_self_FLOAT16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_DOUBLE
TEST_F(l2_left_shift_test, abnormal_self_DOUBLE)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_COMPLEX64
TEST_F(l2_left_shift_test, abnormal_self_COMPLEX64)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_COMPLEX128
TEST_F(l2_left_shift_test, abnormal_self_COMPLEX128)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_UNDEFINED
TEST_F(l2_left_shift_test, abnormal_dtype_UNDEFINED)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_other_FLOAT
TEST_F(l2_left_shift_test, abnormal_other_FLOAT)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_dtype_unequal
TEST_F(l2_left_shift_test, normal_dtype_unequal)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckPromoteType_can_cast
TEST_F(l2_left_shift_test, normal_dtype_can_cast)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckShape_unequal
TEST_F(l2_left_shift_test, abnormal_shape_unequal)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_10D
TEST_F(l2_left_shift_test, abnormal_shape_10D)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat_self_NC1HWC0
TEST_F(l2_left_shift_test, abnormal_format_self_NC1HWC0)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NC1HWC0);
    auto shiftBitsDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NC1HWC0);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat_out_NC1HWC0
TEST_F(l2_left_shift_test, abnormal_format_out_NC1HWC0)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto shiftBitsDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NC1HWC0);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据范围[-1，1]
TEST_F(l2_left_shift_test, normal_valuerange)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto shiftBitsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 非连续
TEST_F(l2_left_shift_test, normal_uncontiguous)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto selfDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto shiftBitsDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto outDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto ut = OP_API_UT(aclnnLeftShift, INPUT(selfDesc, shiftBitsDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}