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

#include "aclnn_sinkhorn.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include <cstdlib>
#include <ctime>


using namespace std;

class l2_sinkhorn_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "sinkhorn_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "sinkhorn_test TearDown" << endl;
  }
};

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_float_8_2) {
  const vector<int64_t>& costShape = {8, 2};
  aclDataType dType = ACL_FLOAT;
  aclFormat dFormat = ACL_FORMAT_ND;

  const vector<int64_t>& pShape = {8, 2};

  auto costTensorDesc = TensorDesc(costShape, dType, dFormat)
    .Value(vector<float>{45.0f, 48.0f, 65.0f, 68.0f, 68.0f, 10.0f, 84.0f, 22.0f, 37.0f, 71.0f, 13.0f, 59.0f, 66.0f, 40.0f, 47.0f, 82.0f});
  auto pTensorDesc = TensorDesc(pShape, dType, dFormat).ValidCount(16);
  auto tolScalarDesc = ScalarDesc(0.0001f);

  auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT(pTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_sinkhorn_test, ascend910B2_aclnnSinkhorn_bf16_8_2) {
  const vector<int64_t>& costShape = {8, 2};
  aclDataType dType = ACL_BF16;
  aclFormat dFormat = ACL_FORMAT_ND;

  const vector<int64_t>& pShape = {8, 2};

  auto costTensorDesc = TensorDesc(costShape, dType, dFormat).ValueRange(0.0, 1.0);
  auto pTensorDesc = TensorDesc(pShape, dType, dFormat).ValidCount(16);
  auto tolScalarDesc = ScalarDesc(0.0001f);

  auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT(pTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_sinkhorn_test, ascend910B2_aclnnSinkhorn_float16_8_2) {
  const vector<int64_t>& costShape = {8, 2};
  aclDataType dType = ACL_FLOAT16;
  aclFormat dFormat = ACL_FORMAT_ND;

  const vector<int64_t>& pShape = {8, 2};

  auto costTensorDesc = TensorDesc(costShape, dType, dFormat)
    .ValueRange(0, 1);
  auto pTensorDesc = TensorDesc(pShape, dType, dFormat).ValidCount(16);
  auto tolScalarDesc = ScalarDesc(0.0001f);

  auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT(pTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
