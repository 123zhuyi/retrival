/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_cdist_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_cdist_grad_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "aclnnCdistBackward_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "aclnnCdistBackward_test TearDown" << std::endl; }
};

TEST_F(l2_cdist_grad_test, ascend910B_case_1) {
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCdistBackward, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, 0.0f), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cdist_grad_test, ascend910B_case_2) {
    auto grad_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto x1_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto x2_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto cdist_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnCdistBackward, INPUT(grad_desc, x1_desc, x2_desc, cdist_desc, 0.0f), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
