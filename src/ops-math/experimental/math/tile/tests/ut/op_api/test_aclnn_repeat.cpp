/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_repeat.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/platform.h"
#include "../../../op_api/aclnn_repeat.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_repeat_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_repeat_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_repeat_test TearDown" << endl; }
};

TEST_F(l2_repeat_test, case_01_float_2d)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto repeatsDesc = IntArrayDesc(vector<int64_t>{3, 2});
    auto outDesc = TensorDesc({6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRepeat, INPUT(selfDesc, repeatsDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_repeat_test, case_02_float16_1d)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto selfDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<double>{1.0, 2.0, 3.0, 4.0});
    auto repeatsDesc = IntArrayDesc(vector<int64_t>{3});
    auto outDesc = TensorDesc({12}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRepeat, INPUT(selfDesc, repeatsDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
