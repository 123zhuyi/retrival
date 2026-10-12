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

#include "math/cumprod/op_api/aclnn_cumprod.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_cumprod_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "cumprod_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "cumprod_test TearDown" << endl; }
};

// CheckNotNull_self_nullptr
TEST_F(l2_cumprod_test, l2_cumprod_abnormal_self_nullptr)
{
    auto selfDesc = nullptr;
    auto scalar_desc = ScalarDesc(1.0f);
    aclDataType dtypeDesc = ACL_FLOAT;
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCumprod, INPUT(selfDesc, scalar_desc, dtypeDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
