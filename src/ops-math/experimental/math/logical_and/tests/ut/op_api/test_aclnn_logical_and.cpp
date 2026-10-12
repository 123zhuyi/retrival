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
#include "../../../op_api/aclnn_logical_and.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_logical_and_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_logical_and_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_logical_and_test TearDown" << endl; }
};

TEST_F(l2_logical_and_test, case_01_bool)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto selfDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, true, false, false});
    auto otherDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, false, true, false});

    auto outDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogicalAnd, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
