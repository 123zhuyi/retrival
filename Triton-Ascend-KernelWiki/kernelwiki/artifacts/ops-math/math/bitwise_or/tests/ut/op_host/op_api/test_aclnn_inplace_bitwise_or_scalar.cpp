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

#include "aclnn_bitwise_or_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_inplace_bitwise_or_scalar_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "inplace_bitwise_or_scalar_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "inplace_bitwise_or_scalar_test TearDown" << endl;
  }
};

// 正常场景_BOOL_ND
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_BOOL_ND) {
  auto selfDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
  bool value = true;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 空tensor场景
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_empty_tensor) {
  auto selfDesc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}
// CheckFormat_self_NCL
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_test_format_self_NCL) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_INT32, ACL_FORMAT_NCL);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 数据范围[-1，1]
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_testvaluerange) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// 非连续
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_self_uncontiguous) {
  auto selfDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// CheckShape_detype_unequal
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_abnormal_shape_unequal) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  int64_t value = 1;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

// abmnormal
// CheckNotNull_self_nullptr
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_abnormal_self_nullptr) {
  auto selfDesc = nullptr;
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_other_nullptr
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_abnormal_other_nullptr) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  auto otherDesc = nullptr;

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckShape_10D
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_abnormal_shape_10D) {
  auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_self_dtype_not_support) {
  vector<aclDataType> dtypes{ACL_DT_UNDEFINED, ACL_COMPLEX128, ACL_COMPLEX64, ACL_DOUBLE,
                             ACL_FLOAT16,      ACL_FLOAT,      ACL_BF16};
  for (auto dtype : dtypes) {
    auto selfDesc = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_NCHW);
    int8_t value = 1;
    auto otherDesc = ScalarDesc(value);

    auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

// CheckPromoteType_cannot_cast
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_abnormal_dtype_cannot_cast) {
  auto selfDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_other_FLOAT
TEST_F(l2_inplace_bitwise_or_scalar_test, l2_inplace_bitwise_or_scalar_abnormal_other_FLOAT) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  float value = 1.0;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseOrScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
