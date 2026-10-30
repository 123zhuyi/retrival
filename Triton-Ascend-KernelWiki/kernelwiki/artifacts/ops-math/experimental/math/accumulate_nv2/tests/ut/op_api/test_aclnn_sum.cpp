/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_api/aclnn_sum.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_sum_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_sum_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_sum_test TearDown" << endl; }
};

TEST_F(l2_sum_test, case_01_float)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto selfDesc =
        TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1.023, 2023.08, 3.14, 10987654321.0});
    auto otherDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-3.023, 20.09, 5.197, 109.0888});

    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<TensorDesc> tmp{selfDesc, otherDesc};

    auto tensorList = TensorListDesc(tmp);

    auto ut = OP_API_UT(aclnnSum, INPUT(tensorList), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
