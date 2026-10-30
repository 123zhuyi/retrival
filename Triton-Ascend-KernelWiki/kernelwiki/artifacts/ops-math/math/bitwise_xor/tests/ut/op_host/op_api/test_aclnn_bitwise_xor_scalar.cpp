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

#include "aclnn_bitwise_xor_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;

class l2_bitwise_xor_scalar_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "bitwise_xor_scalar_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "bitwise_xor_scalar_test TearDown" << endl; }
};

// 正常场景_BOOL_ND
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_BOOL_ND) {
  auto selfDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
  bool value = true;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// 正常场景_INT8_NCHW
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_INT8_NCHW) {
  auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_INT8, ACL_FORMAT_NCHW);
  int8_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_INT8, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// 正常场景_INT16_NHWC
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_INT16_NHWC) {
  auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_NHWC);
  int16_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_INT16, ACL_FORMAT_NHWC);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// 正常场景_INT32_HWCN
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_INT32_HWCN) {
  auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_HWCN);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_HWCN);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// 正常场景_INT64_NDHWC
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_INT64_NDHWC) {
  auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_NDHWC);
  int64_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_NDHWC);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// 正常场景_UINT8_NCDHW
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_UINT8_NCDHW) {
  auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_UINT8, ACL_FORMAT_NCDHW);
  uint8_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_UINT8, ACL_FORMAT_NCDHW);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// 空tensor场景
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_empty_tensor) {
  auto selfDesc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// CheckNotNull_self_nullptr
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_self_nullptr) {
  auto selfDesc = nullptr;
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_other_nullptr
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_other_nullptr) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  auto otherDesc = nullptr;
  auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_out_nullptr
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_out_nullptr) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = nullptr;

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_self_BF16
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_self_BF16) {
  auto selfDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_FLOAT
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_self_FLOAT) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_FLOAT16
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_self_FLOAT16) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_DOUBLE
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_self_DOUBLE) {
  auto selfDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_COMPLEX64
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_self_COMPLEX64) {
  auto selfDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_COMPLEX128
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_self_COMPLEX128) {
  auto selfDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_UNDEFINED
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_dtype_UNDEFINED) {
  auto selfDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_other_FLOAT
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_other_FLOAT) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  float value = 1.0;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_dtype_unequal
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_dtype_unequal) {
  auto selfDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
  int64_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// CheckPromoteType_can_cast
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_dtype_can_cast) {
  auto selfDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);
  int16_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// CheckPromoteType_cannot_cast
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_dtype_cannot_cast) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_unequal
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_shape_unequal) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_10D
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_abnormal_shape_10D) {
  auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据范围[-1，1]
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_valuerange) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// 非连续
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_normal_uncontiguous) {
  auto selfDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// 正常场景_BOOL_ND_INPLACE
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_inplace_BOOL) {
  auto selfDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
  bool value = true;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// 正常场景_INT32_ND_INPLACE
TEST_F(l2_bitwise_xor_scalar_test, l2_bitwise_xor_scalar_inplace_INT32) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  int32_t value = 1;
  auto otherDesc = ScalarDesc(value);

  auto ut = OP_API_UT(aclnnInplaceBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}

// 正常场景_910B2_INT64_ND
TEST_F(l2_bitwise_xor_scalar_test, ascend910B2_normal_INT64_ND) {
  auto selfDesc = TensorDesc({3, 4}, ACL_INT64, ACL_FORMAT_ND);
  int64_t value = 1;
  auto otherDesc = ScalarDesc(value);
  auto outDesc = TensorDesc({3, 4}, ACL_INT64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(selfDesc, otherDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  // ut.TestPrecision();
}
