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

#include "../../../op_api/aclnn_dot.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;

class l2_dot_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "dot_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "dot_test TearDown" << endl; }
};

// 正常场景_FLOAT
TEST_F(l2_dot_test, l2_dot_normal_dtype_FLOAT) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 正常场景_FLOAT16
TEST_F(l2_dot_test, l2_dot_normal_dtype_FLOAT16) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 空tensor场景_FLOAT
TEST_F(l2_dot_test, l2_dot_normal_empty_FLOAT) {
  auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 空tensor场景_FLOAT16
TEST_F(l2_dot_test, l2_dot_normal_empty_FLOAT16) {
  auto selfDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// CheckNotNull_self
TEST_F(l2_dot_test, l2_dot_abnormal_self_nullptr) {
  auto selfDesc = nullptr;
  auto tensorDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_other
TEST_F(l2_dot_test, l2_dot_abnormal_other_nullptr) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorDesc = nullptr;
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_out
TEST_F(l2_dot_test, l2_dot_abnormal_out_nullptr) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = nullptr;

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_BF16
TEST_F(l2_dot_test, ascend910B2_l2_dot_dtype_BF16) {
  auto selfDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// CheckDtypeValid_INT8
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_INT8) {
  auto selfDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckDtypeValid_INT32
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_INT32) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckDtypeValid_UINT8
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_UINT8) {
  auto selfDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_UINT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckDtypeValid_DOUBLE
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_DOUBLE) {
  auto selfDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_INT64
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_INT64) {
  auto selfDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_INT16
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_INT16) {
  auto selfDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_INT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_BOOL
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_BOOL) {
  auto selfDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_COMPLEX64
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_COMPLEX64) {
  auto selfDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_COMPLEX64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_COMPLEX128
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_COMPLEX128) {
  auto selfDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_COMPLEX128, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_UNDEFINED
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_UNDEFINED) {
  auto selfDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_other_unequal
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_self_other_unequal) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_out_unequal
TEST_F(l2_dot_test, l2_dot_abnormal_dtype_self_out_unequal) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_self_not_1D
TEST_F(l2_dot_test, l2_dot_abnormal_shape_self_not_1D) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_other_not_1D
TEST_F(l2_dot_test, l2_dot_abnormal_shape_other_not_1D) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_self_other_unequal
TEST_F(l2_dot_test, l2_dot_abnormal_shape_self_other_unequal) {
  auto selfDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_out_not_0D
TEST_F(l2_dot_test, l2_dot_abnormal_shape_out_not_0D) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据范围[-1，1]
TEST_F(l2_dot_test, l2_dot_normal_valuerange) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 非连续
TEST_F(l2_dot_test, l2_dot_normal_uncontiguous) {
  auto selfDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 0, {8});
  auto tensorDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 0, {8});
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnDot, INPUT(selfDesc, tensorDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}