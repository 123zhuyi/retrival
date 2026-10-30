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

#include "../../../op_api/aclnn_isclose.h"
#include "acl/acl.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

namespace {
aclnnStatus aclnnIsCloseForUtGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                              double rtol, double atol, bool equalNan, uint64_t* workspaceSize,
                                              aclOpExecutor** executor)
{
    return aclnnIsCloseGetWorkspaceSize(self, other, rtol, atol, equalNan, out, workspaceSize, executor);
}

aclnnStatus aclnnIsCloseForUt(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnIsClose(workspace, workspaceSize, executor, stream);
}
} // namespace

class l2_is_close_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_is_close_test SetUp" << endl;
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }

    static void TearDownTestCase()
    {
        cout << "l2_is_close_test TearDown" << endl;
    }

    void TestRun(const vector<int64_t>& selfDims, aclDataType selfDtype, const vector<int64_t>& otherDims,
                 aclDataType otherDtype, const vector<int64_t>& outDims, aclnnStatus expectStatus,
                 double rtol = 1e-5, double atol = 1e-8, bool equalNan = false)
    {
        auto self = TensorDesc(selfDims, selfDtype, ACL_FORMAT_ND).ValueRange(-10, 10);
        auto other = TensorDesc(otherDims, otherDtype, ACL_FORMAT_ND).ValueRange(-10, 10);
        auto out = TensorDesc(outDims, ACL_BOOL, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnIsCloseForUt, INPUT(self, other), OUTPUT(out), rtol, atol, equalNan);
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, expectStatus);
    }
};

TEST_F(l2_is_close_test, case_01_float32_same_shape)
{
    TestRun({2, 3, 4}, ACL_FLOAT, {2, 3, 4}, ACL_FLOAT, {2, 3, 4}, ACLNN_SUCCESS);
}
