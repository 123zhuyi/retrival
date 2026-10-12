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

#include "../../../op_api/aclnn_pow.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_pow_scalar_tensor_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "pow_scalar_tensor_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "pow_scalar_tensor_test TearDown" << endl; }
};

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_null_0) {
  // input
  const vector<int64_t>& expShape = {0};
  aclDataType expDtype = ACL_FLOAT;
  aclFormat expFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {0};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat).ValueRange(0,1);
  auto selfScalarDesc = ScalarDesc(3);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_nullptr2) {
  // input
  const vector<int64_t>& expShape = {5};
  aclDataType expDtype = ACL_FLOAT;
  aclFormat expFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {5};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat).ValueRange(0,1);
  auto selfScalarDesc = ScalarDesc(3);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(nullptr));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_8_float_nd_8_float_nd) {
  // input
  const vector<int64_t>& expShape = {8};
  aclDataType expDtype = ACL_FLOAT;
  aclFormat expFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {8};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat).ValueRange(0,1);
  auto selfScalarDesc = ScalarDesc(3);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_1_11_34_22_19_6_5_1_shape9_nd_23_11_34_22_1_1_5_5_shape9_nd) {
  // input
  const vector<int64_t>& expShape = {1,11,34,22,19,6,5,1,3};
  aclDataType expDtype = ACL_FLOAT;
  aclFormat expFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {1,11,34,22,19,6,5,1,3};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat).ValueRange(0,1);
  auto selfScalarDesc = ScalarDesc(3);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_223_32_17_17_float16_nchw_223_1_1_17_float16_nchw) {
  // left input
  const vector<int64_t>& expShape = {5,3};
  aclDataType expDtype = ACL_FLOAT16;
  aclFormat expFormat = ACL_FORMAT_NCHW;

  // output
  const vector<int64_t>& outShape = {5,3};
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_NCHW;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat).ValueRange(0,1);
  auto selfScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
/* 
TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_11_32_1_12_float64_nchw_11_32_1_12_float64_nchw) {
  // input
  const vector<int64_t>& expShape = {5,3};
  aclDataType expDtype = ACL_DOUBLE;
  aclFormat expFormat = ACL_FORMAT_ND;

  // output
  const vector<int64_t>& outShape = {5,3};
  aclDataType outDtype = ACL_DOUBLE;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat).ValueRange(0,1);
  auto selfScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
 */
TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_1_1_16_129_int8_hwcn_1_1_16_129_int8_hwcn_int32) {
  // input
  const vector<int64_t>& expShape = {5,3};
  aclDataType expDtype = ACL_INT8;
  aclFormat expFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {5,3};
  aclDataType outDtype = ACL_INT8;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_1_1_16_129_in32_hwcn_1_1_16_129_int8_hwcn_int32) {
  // input
  const vector<int64_t>& expShape = {5,3};
  aclDataType expDtype = ACL_INT32;
  aclFormat expFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {5,3};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
/* 
TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_1_1_16_129_in64_hwcn_1_1_16_129_int8_hwcn_int64) {
  // input
  const vector<int64_t>& expShape = {5,3};
  aclDataType expDtype = ACL_INT64;
  aclFormat expFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {5,3};
  aclDataType outDtype = ACL_INT64;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_1_1_16_129_complex64_hwcn_1_1_16_129_int8_hwcn_complex64) {
  // input
  const vector<int64_t>& expShape = {5,3};
  aclDataType expDtype = ACL_COMPLEX64;
  aclFormat expFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {5,3};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_1_1_16_129_complex128_hwcn_1_1_16_129_int8_hwcn_complex128) {
  // input
  const vector<int64_t>& expShape = {5,3};
  aclDataType expDtype = ACL_COMPLEX128;
  aclFormat expFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {5,3};
  aclDataType outDtype = ACL_COMPLEX128;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_1_1_16_129_bf16_hwcn_1_1_16_129_int8_hwcn_bf16) {
  // input
  const vector<int64_t>& expShape = {5,3};
  aclDataType expDtype = ACL_BF16;
  aclFormat expFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {5,3};
  aclDataType outDtype = ACL_BF16;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
 */

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_1_1_16_129_bool_hwcn_1_1_16_129_int8_hwcn_bool) {
  // input
  const vector<int64_t>& expShape = {5,3};
  aclDataType expDtype = ACL_BOOL;
  aclFormat expFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {5,3};
  aclDataType outDtype = ACL_BOOL;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
/* 
TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_1_1_16_129_int16_hwcn_1_1_16_129_int8_hwcn_int16) {
  // input
  const vector<int64_t>& expShape = {5,3};
  aclDataType expDtype = ACL_INT16;
  aclFormat expFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {5,3};
  aclDataType outDtype = ACL_INT16;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  
  auto selfScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_scalar_tensor_test, aclnnPowScalarTensor_1_complex_hwcn_1_int8_hwcn_float) {
  // input
  const vector<int64_t>& expShape = {1};
  aclDataType expDtype = ACL_FLOAT;
  aclFormat expFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  float src_value = 1.0f;
  auto selfScalarDesc = aclCreateScalar(&src_value, aclDataType::ACL_COMPLEX64);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
 */

TEST_F(l2_pow_scalar_tensor_test, Ascend950PR_89_aclnnPowScalarTensor_fill_1) {
  // input
  const vector<int64_t>& expShape = {5};
  aclDataType expDtype = ACL_INT32;
  aclFormat expFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {5};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(1);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_scalar_tensor_test, Ascend950PR_89_aclnnPowScalarTensor_scalar_float16) {
  // input
  const vector<int64_t>& expShape = {5, 3};
  aclDataType expDtype = ACL_FLOAT16;
  aclFormat expFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {5, 3};
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(3.0);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
/* 
TEST_F(l2_pow_scalar_tensor_test, Ascend950PR_89_aclnnPowScalarTensor_exp_complex64) {
  // input
  const vector<int64_t>& expShape = {5, 3};
  aclDataType expDtype = ACL_COMPLEX64;
  aclFormat expFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {5, 3};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(3);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_scalar_tensor_test, Ascend950PR_89_aclnnPowScalarTensor_scalar_float_exp_int32) {
  // input
  const vector<int64_t>& expShape = {5, 3};
  aclDataType expDtype = ACL_INT32;
  aclFormat expFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {5, 3};
  aclDataType outDtype = ACL_DOUBLE;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto expTensorDesc = TensorDesc(expShape, expDtype, expFormat);
  auto selfScalarDesc = ScalarDesc(3.0);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowScalarTensor, INPUT(selfScalarDesc, expTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
 */
