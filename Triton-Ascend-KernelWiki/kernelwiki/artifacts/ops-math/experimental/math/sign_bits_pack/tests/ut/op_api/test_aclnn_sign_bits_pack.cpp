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
 * \file test_aclnn_sign_bits_pack.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/platform.h"
#include "../../../op_api/aclnn_sign_bits_pack.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_sign_bits_pack_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_sign_bits_pack_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_sign_bits_pack_test TearDown" << endl; }
};

TEST_F(l2_sign_bits_pack_test, case_01_float)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto selfDesc =
        TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{5, 4, 3, 2, 0, -1, -2, 4, 3, 2, 1, 0, -1, -2});
    int64_t size = 2;
    auto outDesc = TensorDesc({2, 1}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(selfDesc, size), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
