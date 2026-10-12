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

#include "../../../../op_api/aclnn_logical_or.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_logical_or_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "logical_or_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "logical_or_test TearDown" << endl; }
};

// TEST_F(l2_logical_or_test, aclnnLogicalOr_1_11_34_22_19_6_5_1_float_nd_23_11_34_22_1_1_5_5_float_nd_testcase001) {
//   // left input
//   const vector<int64_t>& selfShape = {1,11,34,22,19,6,5,1};
//   aclDataType selfDtype = ACL_FLOAT;
//   aclFormat selfFormat = ACL_FORMAT_ND;
//   // right input
//   const vector<int64_t>& otherShape = {23,11,34,22,1,1,5,5};
//   aclDataType otherDtype = ACL_FLOAT;
//   aclFormat otherFormat = ACL_FORMAT_ND;
//   // output
//   const vector<int64_t>& outShape = {23,11,34,22,19,6,5,5};
//   aclDataType outDtype = ACL_FLOAT;
//   aclFormat outFormat = ACL_FORMAT_ND;

//   auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
//   auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
//   auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

//   auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspaceSize = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }

TEST_F(l2_logical_or_test, aclnnLogicalOr_223_32_17_17_float16_nchw_223_1_1_17_float16_nchw_testcase002) {
  // left input
  const vector<int64_t>& selfShape = {223,32,17,17};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // right input
  const vector<int64_t>& otherShape = {223,1,1,17};
  aclDataType otherDtype = ACL_FLOAT16;
  aclFormat otherFormat = ACL_FORMAT_NCHW;
  // output
  const vector<int64_t>& outShape = {223,32,17,17};
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision(); // comment bcz of timeout in model tests (104717 ms)
}

// 异常场景 bf16
TEST_F(l2_logical_or_test, aclnnLogicalOr_bfloat16_nd_not_support_testcase003) {
  // left input
  const vector<int64_t>& selfShape = {32,1,12};
  aclDataType selfDtype = ACL_BF16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {1,1,12};
  aclDataType otherDtype = ACL_BF16;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {32,1,12};
  aclDataType outDtype = ACL_BF16;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景910B SocVersion bf16
TEST_F(l2_logical_or_test, ascend910B2_support_bf16_910B) {
  // left input
  const vector<int64_t>& selfShape = {32,1,12};
  aclDataType selfDtype = ACL_BF16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {1,1,12};
  aclDataType otherDtype = ACL_BF16;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {32,1,12};
  aclDataType outDtype = ACL_BF16;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_11_32_1_12_double_nchw_11_1_1_12_double_nchw_testcase004) {
  // left input
  const vector<int64_t>& selfShape = {11,32,1,12};
  aclDataType selfDtype = ACL_DOUBLE;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {11,1,1,12};
  aclDataType otherDtype = ACL_DOUBLE;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {11,32,1,12};
  aclDataType outDtype = ACL_DOUBLE;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_1_5_1_16_129_int32_ndhwc_32_5_32_16_1_int32_ndhwc_testcase005) {
  // left input
  const vector<int64_t>& selfShape = {1,5,1,16,129};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_NDHWC;
  // right input
  const vector<int64_t>& otherShape = {32,5,32,16,1};
  aclDataType otherDtype = ACL_INT32;
  aclFormat otherFormat = ACL_FORMAT_NDHWC;
  // output
  const vector<int64_t>& outShape = {32,5,32,16,129};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_NDHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();  // comment bcz of timeout in model tests (149086 ms)
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_23_71_16_5_1_int64_ncdhw_1_1_16_5_23_int64_ndhwc_testcase006) {
  // left input
  const vector<int64_t>& selfShape = {23,71,16,5,1};
  aclDataType selfDtype = ACL_INT64;
  aclFormat selfFormat = ACL_FORMAT_NCDHW;
  // right input
  const vector<int64_t>& otherShape = {1,1,16,5,23};
  aclDataType otherDtype = ACL_INT64;
  aclFormat otherFormat = ACL_FORMAT_NCDHW;
  // output
  const vector<int64_t>& outShape = {23,71,16,5,23};
  aclDataType outDtype = ACL_INT64;
  aclFormat outFormat = ACL_FORMAT_NCDHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();  // comment bcz of timeout in model tests (147883 ms)
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_1_5_1_16_129_int16_ndhwc_32_5_32_16_1_int16_ndhwc_testcase007) {
  // left input
  const vector<int64_t>& selfShape = {1,5,1,16,129};
  aclDataType selfDtype = ACL_INT16;
  aclFormat selfFormat = ACL_FORMAT_NDHWC;
  // right input
  const vector<int64_t>& otherShape = {32,5,32,16,1};
  aclDataType otherDtype = ACL_INT16;
  aclFormat otherFormat = ACL_FORMAT_NDHWC;
  // output
  const vector<int64_t>& outShape = {32,5,32,16,129};
  aclDataType outDtype = ACL_INT16;
  aclFormat outFormat = ACL_FORMAT_NDHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_1_1_16_129_int8_hwcn_32_32_16_1_int8_hwcn_testcase008) {
  // left input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // right input
  const vector<int64_t>& otherShape = {32,32,16,1};
  aclDataType otherDtype = ACL_INT8;
  aclFormat otherFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {32,32,16,129};
  aclDataType outDtype = ACL_INT8;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_64_31_31_16_uint8_nhwc_64_31_1_1_uint8_nhwc_testcase009) {
  // left input
  const vector<int64_t>& selfShape = {64,31,31,16};
  aclDataType selfDtype = ACL_UINT8;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // right input
  const vector<int64_t>& otherShape = {64,31,1,1};
  aclDataType otherDtype = ACL_UINT8;
  aclFormat otherFormat = ACL_FORMAT_NHWC;
  // output
  const vector<int64_t>& outShape = {64,31,31,16};
  aclDataType outDtype = ACL_UINT8;
  aclFormat outFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_23_71_16_5_1_bool_nd_1_1_16_5_23_bool_nd_testcase010) {
  // left input
  const vector<int64_t>& selfShape = {23,71,16,5,1};
  aclDataType selfDtype = ACL_BOOL;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {1,1,16,5,23};
  aclDataType otherDtype = ACL_BOOL;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {23,71,16,5,23};
  aclDataType outDtype = ACL_BOOL;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision(); // comment bcz of timeout in model tests (64132 ms)
}

// //TODO: COMPLEX64 NOT SUPPORT NOW
// TEST_F(l2_logical_or_test, aclnnLogicalOr_23_71_complex64_nd_23_1_complex64_nd_testcase011) {
//   // left input
//   const vector<int64_t>& selfShape = {23,71};
//   aclDataType selfDtype = ACL_COMPLEX64;
//   aclFormat selfFormat = ACL_FORMAT_ND;
//   // right input
//   const vector<int64_t>& otherShape = {23,1};
//   aclDataType otherDtype = ACL_COMPLEX64;
//   aclFormat otherFormat = ACL_FORMAT_ND;
//   // output
//   const vector<int64_t>& outShape = {23,71};
//   aclDataType outDtype = ACL_COMPLEX64;
//   aclFormat outFormat = ACL_FORMAT_ND;

//   auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
//   auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
//   auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

//   auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspaceSize = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }

// //TODO: COMPLEX128 NOT SUPPORT NOW
// TEST_F(l2_logical_or_test, aclnnLogicalOr_1_65_23_complex128_nd_55_1_1_complex128_nd_testcase012) {
//   // left input
//   const vector<int64_t>& selfShape = {1,65,23};
//   aclDataType selfDtype = ACL_COMPLEX128;
//   aclFormat selfFormat = ACL_FORMAT_ND;
//   // right input
//   const vector<int64_t>& otherShape = {55,1,1};
//   aclDataType otherDtype = ACL_COMPLEX128;
//   aclFormat otherFormat = ACL_FORMAT_ND;
//   // output
//   const vector<int64_t>& outShape = {55,65,23};
//   aclDataType outDtype = ACL_COMPLEX128;
//   aclFormat outFormat = ACL_FORMAT_ND;

//   auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
//   auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
//   auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

//   auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspaceSize = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }

TEST_F(l2_logical_or_test, aclnnLogicalOr_64_31_31_16_uint8_nhwc_64_31_1_1_uint8_nhwc_float16_testcase013) {
  // left input
  const vector<int64_t>& selfShape = {64,31,31,16};
  aclDataType selfDtype = ACL_UINT8;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // right input
  const vector<int64_t>& otherShape = {64,31,1,1};
  aclDataType otherDtype = ACL_UINT8;
  aclFormat otherFormat = ACL_FORMAT_NHWC;
  // output
  const vector<int64_t>& outShape = {64,31,31,16};
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_1_1_16_129_int8_hwcn_32_32_16_1_int8_hwcn_int32_testcase014) {
  // left input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // right input
  const vector<int64_t>& otherShape = {32,32,16,1};
  aclDataType otherDtype = ACL_INT8;
  aclFormat otherFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {32,32,16,129};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_1_5_1_16_129_int16_ndhwc_32_5_32_16_1_int32_ndhwc_float32_testcase015) {
  // left input
  const vector<int64_t>& selfShape = {1,5,1,16,129};
  aclDataType selfDtype = ACL_INT16;
  aclFormat selfFormat = ACL_FORMAT_NDHWC;
  // right input
  const vector<int64_t>& otherShape = {32,5,32,16,1};
  aclDataType otherDtype = ACL_INT32;
  aclFormat otherFormat = ACL_FORMAT_NDHWC;
  // output
  const vector<int64_t>& outShape = {32,5,32,16,129};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_NDHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_1_11_21_6_float_nd_12_1_1_1_float16_nchw_testcase016) {
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

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_1_11_21_6_float_nd_12_1_1_1_double_nchw_testcase017) {
  // left input
  const vector<int64_t>& selfShape = {1,11,21,16};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {12,1,1,1};
  aclDataType otherDtype = ACL_DOUBLE;
  aclFormat otherFormat = ACL_FORMAT_NCHW;
  // output
  const vector<int64_t>& outShape = {12,11,21,16};
  aclDataType outDtype = ACL_INT32;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_1_11_21_6_float_nd_12_1_1_1_int32_nchw_testcase018) {
  // left input
  const vector<int64_t>& selfShape = {1,11,21,16};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {12,1,1,1};
  aclDataType otherDtype = ACL_INT32;
  aclFormat otherFormat = ACL_FORMAT_NCHW;
  // output
  const vector<int64_t>& outShape = {12,11,21,16};
  aclDataType outDtype = ACL_BOOL;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// TEST_F(l2_logical_or_test, aclnnLogicalOr_1_11_34_22_19_6_5_1_float_nd_23_11_34_22_1_1_5_5_float_ValueRange_testcase019) {
//   // left input
//   const vector<int64_t>& selfShape = {1,11,34,22,19,6,5,1};
//   aclDataType selfDtype = ACL_FLOAT;
//   aclFormat selfFormat = ACL_FORMAT_ND;
//   // right input
//   const vector<int64_t>& otherShape = {23,11,34,22,1,1,5,5};
//   aclDataType otherDtype = ACL_FLOAT;
//   aclFormat otherFormat = ACL_FORMAT_ND;
//   // output
//   const vector<int64_t>& outShape = {23,11,34,22,19,6,5,5};
//   aclDataType outDtype = ACL_FLOAT;
//   aclFormat outFormat = ACL_FORMAT_ND;

//   auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
//   auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(-1, 1);
//   auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

//   auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspaceSize = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }

TEST_F(l2_logical_or_test, aclnnLogicalOr_223_32_17_17_float16_nchw_223_1_1_17_float16_ValueRange_testcase020) {
  // left input
  const vector<int64_t>& selfShape = {223,32,17,17};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // right input
  const vector<int64_t>& otherShape = {223,1,1,17};
  aclDataType otherDtype = ACL_FLOAT16;
  aclFormat otherFormat = ACL_FORMAT_NCHW;
  // output
  const vector<int64_t>& outShape = {223,32,17,17};
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(-1, 1);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision(); // comment bcz of timeout in model tests (114518 ms)
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_0_2_float32_nd_1_2_float32_nd_empty_tensor_testcase021) {
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

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_100_int8_100_int8_in_overflow_testcase022) {
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

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_100_int32_100_int32_out_overflow_testcase023) {
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

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(65536, 66000);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(65536, 66000);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_1_2_float16_2_1_float16_boundary_value_testcase024) {
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

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// // TODO: special data value
// TEST_F(l2_logical_or_test, aclnnLogicalOr_6_1_float32_1_6_float32_special_value_testcase025) {
//   // left input
//   const vector<int64_t>& selfShape = {6,1};
//   aclDataType selfDtype = ACL_FLOAT;
//   aclFormat selfFormat = ACL_FORMAT_ND;
//   // right input
//   const vector<int64_t>& otherShape = {1,6};
//   aclDataType otherDtype = ACL_FLOAT;
//   aclFormat otherFormat = ACL_FORMAT_ND;
//   // output
//   const vector<int64_t>& outShape = {6,6};
//   aclDataType outDtype = ACL_FLOAT;
//   aclFormat outFormat = ACL_FORMAT_ND;

//   auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat)
//                               .Value(vector<float>{float("inf"), float("-inf"), float("nan"), 0, 1, -1});
//   auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat)
//                               .Value(vector<float>{float("inf"), float("-inf"), float("nan"), 0, 1, -1});
//   auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

//   auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspaceSize = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }

// not contiguous
TEST_F(l2_logical_or_test, aclnnLogicalOr_5_4_float32_not_contiguous_testcase026) {
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
      TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim).ValueRange(-2, 2);
  auto otherTensorDesc =
      TensorDesc(otherShape, otherDtype, otherFormat, otherViewDim, otherOffset, otherStorageDim).ValueRange(-2, 2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_input_output_nullptr_testcase027) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut_l = OP_API_UT(aclnnLogicalOr, INPUT((aclTensor*)nullptr, tensor_desc), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);

  auto ut_r = OP_API_UT(aclnnLogicalOr, INPUT(tensor_desc, (aclTensor*)nullptr), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_r.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);

  auto ut_o = OP_API_UT(aclnnLogicalOr, INPUT(tensor_desc, tensor_desc), OUTPUT((aclTensor*)nullptr));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_o.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_input_error_shape_testcase028) {
  // left input
  const vector<int64_t>& selfShape = {123,11,2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {123,8,2};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {123,11,2};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_input_output_diff_shape_testcase029) {
  // left input
  const vector<int64_t>& selfShape = {123,11,2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // right input
  const vector<int64_t>& otherShape = {123,11,2};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {123,8,2};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_input_error_shape_len_testcase030) {
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

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logical_or_test, aclnnLogicalOr_error_input_dtype_testcase031) {
  // left input
  const vector<int64_t>& selfShape = {6,12,11,16};
  aclDataType selfDtype = ACL_UINT64;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // right input
  const vector<int64_t>& otherShape = {6,12,11,1};
  aclDataType otherDtype = ACL_UINT64;
  aclFormat otherFormat = ACL_FORMAT_NHWC;
  // output
  const vector<int64_t>& outShape = {6,12,11,16};
  aclDataType outDtype = ACL_UINT64;
  aclFormat outFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnLogicalOr, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景_INT32_ND_INPLACE
TEST_F(l2_logical_or_test, inplace_INT32) {
  auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  auto otherDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceLogicalOr, INPUT(selfDesc, otherDesc), OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // precision simulate
  ut.TestPrecision();
}