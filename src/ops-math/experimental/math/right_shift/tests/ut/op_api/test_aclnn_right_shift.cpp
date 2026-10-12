/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_right_shift.h"
#include "acl/acl.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

namespace {
aclnnStatus aclnnRightShiftForUtGetWorkspaceSize(const aclTensor* input, const aclTensor* shiftBits, aclTensor* out,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnRightShiftGetWorkspaceSize(input, shiftBits, out, workspaceSize, executor);
}

aclnnStatus aclnnRightShiftForUt(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnRightShift(workspace, workspaceSize, executor, stream);
}
} // namespace

class l2_right_shift_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_right_shift_test SetUp" << endl;
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }

    static void TearDownTestCase() { cout << "l2_right_shift_test TearDown" << endl; }

    void TestRun(const vector<int64_t>& inputDims, aclDataType inputDtype, const vector<int64_t>& shiftDims,
                 aclDataType shiftDtype, const vector<int64_t>& outDims, aclDataType outDtype, aclnnStatus expectStatus)
    {
        auto input = TensorDesc(inputDims, inputDtype, ACL_FORMAT_ND).ValueRange(-10, 10);
        auto shiftBits = TensorDesc(shiftDims, shiftDtype, ACL_FORMAT_ND).ValueRange(0, 3);
        auto out = TensorDesc(outDims, outDtype, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnRightShiftForUt, INPUT(input, shiftBits), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, expectStatus);
    }
};

TEST_F(l2_right_shift_test, case_01_int32_same_shape)
{
    TestRun({2, 3}, ACL_INT32, {2, 3}, ACL_INT32, {2, 3}, ACL_INT32, ACLNN_SUCCESS);
}

TEST_F(l2_right_shift_test, case_02_uint8_y_scalar)
{
    TestRun({2, 3}, ACL_UINT8, {1}, ACL_UINT8, {2, 3}, ACL_UINT8, ACLNN_SUCCESS);
}

TEST_F(l2_right_shift_test, case_03_int64_x_scalar)
{
    TestRun({1}, ACL_INT64, {2, 3}, ACL_INT64, {2, 3}, ACL_INT64, ACLNN_SUCCESS);
}

TEST_F(l2_right_shift_test, case_04_int16_broadcast)
{
    TestRun({2, 1, 4}, ACL_INT16, {1, 3, 4}, ACL_INT16, {2, 3, 4}, ACL_INT16, ACLNN_SUCCESS);
}

TEST_F(l2_right_shift_test, case_05_mixed_dtype_promote)
{
    TestRun({2, 3}, ACL_INT16, {2, 3}, ACL_INT32, {2, 3}, ACL_INT32, ACLNN_SUCCESS);
}

TEST_F(l2_right_shift_test, case_06_empty_tensor)
{
    TestRun({2, 0, 3}, ACL_INT32, {2, 0, 3}, ACL_INT32, {2, 0, 3}, ACL_INT32, ACLNN_SUCCESS);
}

TEST_F(l2_right_shift_test, case_07_output_shape_mismatch)
{
    TestRun({2, 3}, ACL_INT32, {2, 3}, ACL_INT32, {2, 1}, ACL_INT32, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_right_shift_test, case_08_unbroadcastable_shape)
{
    TestRun({2, 3}, ACL_INT32, {4, 3}, ACL_INT32, {2, 3}, ACL_INT32, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_right_shift_test, case_09_unsupported_float_dtype)
{
    TestRun({2, 3}, ACL_FLOAT, {2, 3}, ACL_FLOAT, {2, 3}, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID);
}
