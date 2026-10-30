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

#include "conversion/diag_v2/op_api/aclnn_diag.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2_diag_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "diag_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "diag_test TearDown" << endl;
  }
};

TEST_F(l2_diag_test, case_input_empty) {
    auto input_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(aclnnDiag, INPUT(input_tensor_desc, diagonal), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_diag_test, case_nullptr) {
    auto input_tensor_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut1 = OP_API_UT(aclnnDiag, INPUT(nullptr, diagonal), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet1, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_diag_test, case_input_empty_1) {
    auto input_tensor_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t diagonal = 2;
    auto ut1 = OP_API_UT(aclnnDiag, INPUT(input_tensor_desc, diagonal), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet1, ACL_SUCCESS);
}