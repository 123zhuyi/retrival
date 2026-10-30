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

#include "math/cumsum/op_api/aclnn_cumsum.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_cumsum_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "cumsum_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "cumsum_test TearDown" << endl; }
};

// 正常场景_FLOAT_ND
TEST_F(l2_cumsum_test, l2_cumsum_normal_FLOAT_ND) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 正常场景_FLOAT_ND_CUBE实现
TEST_F(l2_cumsum_test, ascend910B2_l2_cumsum_normal_FLOAT_ND) {
  auto selfDesc = TensorDesc({32, 16 * 256}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({32, 16 * 256}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_FLOAT16_NCHW
TEST_F(l2_cumsum_test, l2_cumsum_normal_FLOAT16_NCHW) {
  auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT16;
  auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 正常场景_FLOAT16_NCHW_CUBE实现
TEST_F(l2_cumsum_test, ascend910B2_l2_cumsum_normal_FLOAT16_ND) {
  auto selfDesc = TensorDesc({32, 16, 256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT16;
  auto outDesc = TensorDesc({32, 16, 256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 大shape场景
TEST_F(l2_cumsum_test, ascend910B2_l2_cumsum_large_FLOAT32_ND) {
  auto selfDesc = TensorDesc({256, 256, 70000}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 2;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({256, 256, 70000}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
}

// 正常场景_INT32_NHWC
TEST_F(l2_cumsum_test, l2_cumsum_normal_INT32_NHWC) {
  auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NHWC);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_INT32;
  auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NHWC);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 正常场景_BFLOAT16_ND
TEST_F(l2_cumsum_test, l2_cumsum_normal_dtype_BFLOAT16_ND) {
  auto selfDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_BF16;
  auto outDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  } else {
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

// 0维场景
TEST_F(l2_cumsum_test, l2_cumsum_normal_0dim_tensor) {
  auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 空tensor场景
TEST_F(l2_cumsum_test, l2_cumsum_normal_empty_tensor) {
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// CheckNotNull_self_nullptr
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_self_nullptr) {
  auto selfDesc = nullptr;
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_out_nullptr
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_out_nullptr) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = nullptr;

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_BOOL
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_dtype_BOOL) {
  auto selfDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_BOOL;
  auto outDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_UNDEFINED
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_dtype_UNDEFINED) {
  auto selfDesc = TensorDesc({2, 2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_DT_UNDEFINED;
  auto outDesc = TensorDesc({2, 2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_shape_unequal) {
  auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDim = 1
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_dim_correct) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 1;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckDim_1
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_dim_incorrect_1) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 2;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDim_2
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_dim_incorrect_2) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = -3;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_10D
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_shape_10D) {
  auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据范围[-1，1]
TEST_F(l2_cumsum_test, l2_cumsum_normal_valuerange) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t dimDesc = 0;
  aclDataType dtypeDesc = ACL_FLOAT;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnCumsum, INPUT(selfDesc, dimDesc, dtypeDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// CheckExclusiveReverse
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_exclusive_reverse) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  bool exclusive = true;
  bool reverse = true;

  auto ut = OP_API_UT(aclnnCumsumV2, INPUT(selfDesc, dimDesc, exclusive, reverse), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckExclusive
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_exclusive) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  bool exclusive = true;
  bool reverse = false;

  auto ut = OP_API_UT(aclnnCumsumV2, INPUT(selfDesc, dimDesc, exclusive, reverse), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckReverse
TEST_F(l2_cumsum_test, l2_cumsum_abnormal_reverse) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  bool exclusive = false;
  bool reverse = true;

  auto ut = OP_API_UT(aclnnCumsumV2, INPUT(selfDesc, dimDesc, exclusive, reverse), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空tensor场景
TEST_F(l2_cumsum_test, l2_cumsumv2_normal_empty_tensor) {
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 0;
  auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  bool exclusive = false;
  bool reverse = true;

  auto ut = OP_API_UT(aclnnCumsumV2, INPUT(selfDesc, dimDesc, exclusive, reverse), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dim = 1场景
TEST_F(l2_cumsum_test, l2_cumsumv2_normal_dim_one_tensor) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dimDesc = 1;
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  bool exclusive = false;
  bool reverse = true;

  auto ut = OP_API_UT(aclnnCumsumV2, INPUT(selfDesc, dimDesc, exclusive, reverse), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}