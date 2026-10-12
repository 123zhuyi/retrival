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

#include "opdev/platform.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_pow_tensor_scalar_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "pow_tensor_scalar_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "pow_tensor_scalar_test TearDown" << endl; }
};

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_null_0) {
  // input
  const vector<int64_t>& selfShape = {0};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {0};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto expScalarDesc = ScalarDesc(3);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_null_1) {
  // input
  const vector<int64_t>& selfShape = {1,11,34,22,19,6,5,1};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {1,11,34,22,19,6,5,1};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto expScalarDesc = ScalarDesc(3);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(nullptr, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_nullptr2) {
  // input
  const vector<int64_t>& selfShape = {1,11,34,22,19,6,5,1};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {1,11,34,22,19,6,5,1};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto expScalarDesc = ScalarDesc(3);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(nullptr));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_1_2_2_2_1_1_float_nd_1_1_1_2_2_2_1_1_float_nd) {
  // input
  const vector<int64_t>& selfShape = {1,1,1,2,2,2,1,1};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {1,1,1,2,2,2,1,1};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto expScalarDesc = ScalarDesc(3);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_2_3_4_5_6_7_8_9_shape9_nd_1_2_3_4_5_6_7_8_9_shape9_nd) {
  // input
  const vector<int64_t>& selfShape = {1,2,3,4,5,6,7,8,9};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {1,2,3,4,5,6,7,8,9};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto expScalarDesc = ScalarDesc(3);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_223_32_17_17_float16_nchw_223_1_1_17_float16_nchw) {
  // left input
  const vector<int64_t>& selfShape = {223,32,17,17};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NCHW;

  // output
  const vector<int64_t>& outShape = {223,32,17,17};
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
/* 
TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_11_32_1_12_float64_nchw_11_32_1_12_float64_nchw) {
  // input
  const vector<int64_t>& selfShape = {11,32,1,12};
  aclDataType selfDtype = ACL_DOUBLE;
  aclFormat selfFormat = ACL_FORMAT_ND;

  // output
  const vector<int64_t>& outShape = {11,32,1,12};
  aclDataType outDtype = ACL_DOUBLE;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
 */

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_int8_hwcn_1_1_16_129_int8_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_INT8;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_in32_hwcn_1_1_16_129_int32_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_in64_hwcn_1_1_16_129_int64_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_INT64;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_INT64;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_scalar_test, Ascend950_aclnnPowTensorScalar_square_complex64) {
  op::SetPlatformNpuArch(NpuArch::DAV_3510);

  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_COMPLEX64;
  aclFormat selfFormat = ACL_FORMAT_HWCN;

  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  op::SetPlatformNpuArch(NpuArch::DAV_2201);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, Ascend950_aclnnPowTensorScalar_square_complex128) {
  op::SetPlatformNpuArch(NpuArch::DAV_3510);

  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_COMPLEX128;
  aclFormat selfFormat = ACL_FORMAT_HWCN;

  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_COMPLEX128;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  op::SetPlatformNpuArch(NpuArch::DAV_2201);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
/* 
TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_complex64_hwcn_1_1_16_129_complex64_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_COMPLEX64;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_complex128_hwcn_1_1_16_129_complex128_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_COMPLEX128;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_COMPLEX128;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_bf16_hwcn_1_1_16_129_bf16_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_BF16;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_BF16;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
 */
TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_bool_hwcn_1_1_16_129_int64_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_BOOL;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_INT64;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_int16_hwcn_1_1_16_129_int16_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_INT16;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_INT16;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_undefined_hwcn_1_1_16_129_undefined_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_DT_UNDEFINED;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_DT_UNDEFINED;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
/* 
TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_float_hwcn_1_complex64_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  float src_value = 1.0f;
  auto expScalarDesc = aclCreateScalar(&src_value, aclDataType::ACL_COMPLEX64);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Ascend950PR_9589测试float16 tensor + complex scalar
TEST_F(l2_pow_tensor_scalar_test, Ascend950PR_89_aclnnPowTensorScalar_base_fp16_scalar_complex) {
  // input
  const vector<int64_t>& selfShape = {4, 5};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_ND;

  // output
  const vector<int64_t>& outShape = {4, 5};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto expScalarDesc = ScalarDesc(2.5 + 2.5j);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
} */
