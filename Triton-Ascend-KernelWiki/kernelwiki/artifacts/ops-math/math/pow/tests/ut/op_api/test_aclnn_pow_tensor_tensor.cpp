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

#include "../../../op_api/aclnn_pow_tensor_tensor.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_pow_tensor_tensor_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "pow_tensor_tnesor_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "pow_tensor_tnesor_test TearDown" << endl; }
};

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_1_11_float_nd_23_11_float_nd) {
  // left input
  const vector<int64_t>& selfShape = {1,11};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {23,11};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {23,11};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(0,1);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_17_17_float16_nchw_17_17_float16_nchw) {
  // left input
  const vector<int64_t>& selfShape = {17,17};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // right input
  const vector<int64_t>& otherShape = {17,17};
  aclDataType otherDtype = ACL_FLOAT16;
  aclFormat otherFormat = ACL_FORMAT_NCHW;
  // output
  const vector<int64_t>& outShape = {17,17};
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(0,1);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
/* 
TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_1_12_float64_nchw_1_12_float64_nchw) {
  // left input
  const vector<int64_t>& selfShape = {1,12};
  aclDataType selfDtype = ACL_DOUBLE;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {1,12};
  aclDataType otherDtype = ACL_DOUBLE;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {1,12};
  aclDataType outDtype = ACL_DOUBLE;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(0,1);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
 */
TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_16_31_31_16_uint8_nhwc_16_31_1_1_uint8_nhwc_float16) {
  // left input
  const vector<int64_t>& selfShape = {16,31,31,16};
  aclDataType selfDtype = ACL_UINT8;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // right input
  const vector<int64_t>& otherShape = {16,31,1,1};
  aclDataType otherDtype = ACL_UINT8;
  aclFormat otherFormat = ACL_FORMAT_NHWC;
  // output
  const vector<int64_t>& outShape = {16,31,31,16};
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_1_1_16_12_int8_hwcn_32_32_16_1_int8_hwcn_int32) {
  // left input
  const vector<int64_t>& selfShape = {1,1,16,12};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // right input
  const vector<int64_t>& otherShape = {32,32,16,1};
  aclDataType otherDtype = ACL_INT8;
  aclFormat otherFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {32,32,16,12};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
/* 
TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_2_7_int64_ncdhw_2_7_int64_ndhwc) {
  // left input
  const vector<int64_t>& selfShape = {2,7};
  aclDataType selfDtype = ACL_INT64;
  aclFormat selfFormat = ACL_FORMAT_NCDHW;
  // right input
  const vector<int64_t>& otherShape = {2,7};
  aclDataType otherDtype = ACL_INT64;
  aclFormat otherFormat = ACL_FORMAT_NCDHW;
  // output
  const vector<int64_t>& outShape = {2,7};
  aclDataType outDtype = ACL_INT64;
  aclFormat outFormat = ACL_FORMAT_NCDHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(1, 5);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(1, 5);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_2_7_complex64_nd_2_7_complex64_nd) {
  // left input
  const vector<int64_t>& selfShape = {2,7};
  aclDataType selfDtype = ACL_COMPLEX64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {2,7};
  aclDataType otherDtype = ACL_COMPLEX64;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {2,7};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

//TODO: COMPLEX128 NOT SUPPORT NOW
TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_1_5_complex128_nd_1_5_complex128_nd) {
  // left input
  const vector<int64_t>& selfShape = {1,5};
  aclDataType selfDtype = ACL_COMPLEX128;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {1,5};
  aclDataType otherDtype = ACL_COMPLEX128;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {1,5};
  aclDataType outDtype = ACL_COMPLEX128;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(0,1);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
 */
TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_1_11_21_6_float32_nd_12_1_1_1_float16_nchw) {
  // left input
  const vector<int64_t>& selfShape = {1,11,21,16};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {12,1,1,1};
  aclDataType otherDtype = ACL_FLOAT16;
  aclFormat otherFormat = ACL_FORMAT_NCHW;
  // output
  const vector<int64_t>& outShape = {12,11,21,16};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(1,5);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(1,5);;
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_0_2_float32_nd_1_2_float32_nd_empty_tensor) {
  // left input
  const vector<int64_t>& selfShape = {0, 2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {1, 2};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {0, 2};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_100_int8_100_int8_in_overflow) {
  // left input
  const vector<int64_t>& selfShape = {100};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {100};
  aclDataType otherDtype = ACL_INT8;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {100};
  aclDataType outDtype = ACL_INT8;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(257, 300);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(1, 1);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_100_int32_100_int32) {
  // left input
  const vector<int64_t>& selfShape = {100};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {100};
  aclDataType otherDtype = ACL_INT32;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {100};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(1, 5);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(1, 5);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_1_2_float16_2_1_float16_boundary_value) {
  // left input
  const vector<int64_t>& selfShape = {1,2};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {2,1};
  aclDataType otherDtype = ACL_FLOAT16;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {2, 2};
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{65504.0, -65504.0});
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).Value(vector<float>{1, -1});
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// not contiguous
TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_5_4_float32_not_contiguous) {
  // left input
  const vector<int64_t>& selfShape = {5, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  const vector<int64_t>& selfViewDim = {1, 5};
  int64_t selfOffset = 0;
  const vector<int64_t>& selfStorageDim = {4, 5};

  // right input
  const vector<int64_t>& otherShape = {5,4};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  const vector<int64_t>& otherViewDim = {1, 5};
  int64_t otherOffset = 0;
  const vector<int64_t>& otherStorageDim = {4, 5};
  // output
  const vector<int64_t>& outShape = {5, 4};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim).ValueRange(0, 2);
  auto otherTensorDesc =
      TensorDesc(otherShape, otherDtype, otherFormat, otherViewDim, otherOffset, otherStorageDim).ValueRange(0, 2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_input_output_nullptr) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut_l = OP_API_UT(aclnnPowTensorTensor, INPUT(nullptr, tensor_desc), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_r = OP_API_UT(aclnnPowTensorTensor, INPUT(tensor_desc, nullptr), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_r.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_o = OP_API_UT(aclnnPowTensorTensor, INPUT(tensor_desc, tensor_desc), OUTPUT(nullptr));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_o.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_input_error_shape) {
  // left input
  const vector<int64_t>& selfShape = {11,2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {11,2};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {11,2};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_input_output_diff_shape) {
  // left input
  const vector<int64_t>& selfShape = {11,2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {11,2};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {8,2};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_input_error_shape_len) {
  // left input
  const vector<int64_t>& selfShape = {2,3,4,5,6,7,8,9,10};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {2,3,4,5,6,7,8,9,1};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {2,3,4,5,6,7,8,9,10};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_error_input_dtype) {
  // left input
  const vector<int64_t>& selfShape = {11,16};
  aclDataType selfDtype = ACL_UINT64;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // right input
  const vector<int64_t>& otherShape = {11,16};
  aclDataType otherDtype = ACL_UINT64;
  aclFormat otherFormat = ACL_FORMAT_NHWC;
  // output
  const vector<int64_t>& outShape = {11,16};
  aclDataType outDtype = ACL_UINT64;
  aclFormat outFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_output_error_cast_dtype) {
  // left input
  const vector<int64_t>& selfShape = {11,1};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // right input
  const vector<int64_t>& otherShape = {11,1};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_NHWC;
  // output
  const vector<int64_t>& outShape = {11,1};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
/* 
TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_2_7_int16_nd_2_7_int16_nd) {
  // left input
  const vector<int64_t>& selfShape = {2,7};
  aclDataType selfDtype = ACL_INT16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {2,7};
  aclDataType otherDtype = ACL_INT16;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {2,7};
  aclDataType outDtype = ACL_INT16;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
 */
TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_2_7_float32_nd_2_7_bool_nd) {
  // left input
  const vector<int64_t>& selfShape = {2,7};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {2,7};
  aclDataType otherDtype = ACL_BOOL;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {2,7};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_tensor_test, aclnnPowTensorTensor_2_7_bool_nd_2_7_bool_nd) {
  // left input
  const vector<int64_t>& selfShape = {2,7};
  aclDataType selfDtype = ACL_BOOL;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {2,7};
  aclDataType otherDtype = ACL_BOOL;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {2,7};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorTensor, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
