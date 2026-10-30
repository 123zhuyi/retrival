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

#include "../../../../op_host/op_api/aclnn_atan2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include <cstdlib>
#include <ctime>

using namespace op;
using namespace std;

class l2_atan2_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "atan2_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "atan2_test TearDown" << std::endl;
  }
};

// 基础用例
TEST_F(l2_atan2_test, ascend910B2_aclnnAtan2_base_case_1) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {2, 4};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {2, 4};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).Value(vector<float>{8, 7, 6, 5, 4, 3, 2, 1});
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Value(vector<float>{0, 0, 0, 0, 0, 0, 0, 0});

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_atan2_test, ascend910B2_aclnnAtan2_base_case_2) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {2, 4};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {2, 4};
  aclDataType outDtype = ACL_DOUBLE;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).Value(vector<float>{8, 7, 6, 5, 4, 3, 2, 1});
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Value(vector<float>{0, 0, 0, 0, 0, 0, 0, 0});

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

/* 各元素基本类型覆盖用例
 * 维度：1-8维
 * float16,bfloat16,float,float64,int8,int16,int32,int64,uint8,bool
 * 数据格式：ND、NCHW、NHWC、HWCN、NDHWC、NCDHW
 */

TEST_F(l2_atan2_test, aclnnAtan2_1_2_3_4_5_6_7_8_float_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {1, 2, 3, 4, 5, 6, 7, 8};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = selfDtype;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atan2_test, aclnnAtan2_1_2_3_4_5_6_7_float16_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {1, 2, 3, 4, 5, 6, 7};
  aclDataType otherDtype = ACL_FLOAT16;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = selfDtype;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atan2_test, aclnnAtan2_1_2_3_4_int32_nchw) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // other input
  const vector<int64_t>& otherShape = {1, 2, 3, 4};
  aclDataType otherDtype = ACL_INT32;
  aclFormat otherFormat = ACL_FORMAT_NCHW;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atan2_test, aclnnAtan2_1_2_3_4_int8_float_nchw) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // other input
  const vector<int64_t>& otherShape = {1, 2, 3, 4};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_NCHW;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atan2_test, aclnnAtan2_1_2_3_4_double_uint8_nhwc) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_DOUBLE;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // other input
  const vector<int64_t>& otherShape = {4};
  aclDataType otherDtype = ACL_UINT8;
  aclFormat otherFormat = ACL_FORMAT_NHWC;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = selfDtype;
  aclFormat outFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_atan2_test, ascend910B2_aclnnAtan2_1_bool_nhwc) {
  // self input
  const vector<int64_t>& selfShape = {3};
  aclDataType selfDtype = ACL_BOOL;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // other input
  const vector<int64_t>& otherShape = {3};
  aclDataType otherDtype = ACL_BOOL;
  aclFormat otherFormat = ACL_FORMAT_NHWC;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_atan2_test, aclnnAtan2_1_2_3_4_int8_int32_hwcn) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // other input
  const vector<int64_t>& otherShape = {1, 2, 3};
  aclDataType otherDtype = ACL_INT32;
  aclFormat otherFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = ACL_FLOAT16;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 各元素特殊类型覆盖用例
// 空tensor
TEST_F(l2_atan2_test, aclnnAtan2_float_nd_empty_tensor) {
  // self input
  const vector<int64_t>& selfShape = {0};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {0};

  aclDataType otherDtype = ACL_INT64;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {0};
  aclDataType outDtype = selfDtype;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// 边界值
TEST_F(l2_atan2_test, aclnnAtan2_float32_nd_boundary_value) {
  // self input
  const vector<int64_t>& selfShape = {1, 2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {1, 2};
  aclDataType otherDtype = ACL_INT64;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output3
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = selfDtype;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{65504.0, -65504.0});
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).Value(vector<int32_t>{1, 0});
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// 不连续
TEST_F(l2_atan2_test, aclnnAtan2_5_4_float_nd_not_contiguous) {
  // self input
  const vector<int64_t>& selfShape = {5, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  const vector<int64_t>& selfViewDim = {1, 5};
  int64_t selfOffset = 0;
  const vector<int64_t>& selfStorageDim = {4, 5};
  // other input
  const vector<int64_t>& otherShape = {5, 4};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  const vector<int64_t>& otherViewDim = {1, 5};
  int64_t otherOffset = 0;
  const vector<int64_t>& otherStorageDim = {4, 5};
  // output
  const vector<int64_t>& outShape = {5, 4};
  aclDataType outDtype = selfDtype;
  aclFormat outFormat = ACL_FORMAT_ND;
  const vector<int64_t>& outViewDim = {1, 5};
  int64_t sourceOffset = 0;
  const vector<int64_t>& outStorageDim = {4, 5};

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim).ValueRange(-2, 2);

  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat, otherViewDim, otherOffset, otherStorageDim);

  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat, outViewDim, sourceOffset, outStorageDim);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 报错类型覆盖用例
// 空指针
TEST_F(l2_atan2_test, aclnnAtan2_input_nullptr) {
  auto tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut_self_nullptr = OP_API_UT(aclnnAtan2, INPUT((aclTensor*)nullptr, tensor_desc), OUTPUT(tensor_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut_self_nullptr.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_index_nullptr = OP_API_UT(aclnnAtan2, INPUT(tensor_desc, (aclTensor*)nullptr), OUTPUT(tensor_desc));

  aclRet = ut_index_nullptr.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_source_nullptr = OP_API_UT(aclnnAtan2, INPUT(tensor_desc, tensor_desc), OUTPUT((aclTensor*)nullptr));

  aclRet = ut_source_nullptr.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// self类型不满足
TEST_F(l2_atan2_test, aclnnAtan2_self_dtype_error) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_COMPLEX64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {2, 4};
  aclDataType otherDtype = ACL_INT64;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// index类型不满足
TEST_F(l2_atan2_test, aclnnAtan2_index_dtype_error) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_INT64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {2, 4};
  aclDataType otherDtype = ACL_COMPLEX64;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 形状大于8
TEST_F(l2_atan2_test, aclnnAtan2_self_shape_out_of_8) {
  auto out_shape_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND);
  auto normal_tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

  // self形状大于8
  auto self_out_of_shape_ut =
      OP_API_UT(aclnnAtan2, INPUT(out_shape_tensor_desc, normal_tensor_desc), OUTPUT(normal_tensor_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = self_out_of_shape_ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // other形状大于8
  auto other_out_of_shape_ut =
      OP_API_UT(aclnnAtan2, INPUT(normal_tensor_desc, out_shape_tensor_desc), OUTPUT(normal_tensor_desc));
  aclRet = other_out_of_shape_ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_atan2_test, ascend910B2_aclnnAtan2_bf16_nhwc) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_BF16;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // other input
  const vector<int64_t>& otherShape = {4};
  aclDataType otherDtype = ACL_BF16;
  aclFormat otherFormat = ACL_FORMAT_NHWC;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = selfDtype;
  aclFormat outFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_atan2_test, aclnnAtan2_bf16_nhwc) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_BF16;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // other input
  const vector<int64_t>& otherShape = {4};
  aclDataType otherDtype = ACL_BF16;
  aclFormat otherFormat = ACL_FORMAT_NHWC;
  // output
  const vector<int64_t>& outShape = selfShape;
  aclDataType outDtype = selfDtype;
  aclFormat outFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TEST BROADCAST
TEST_F(l2_atan2_test, ascend910B2_atan2_self_broadcast) {
  // self input
  const vector<int64_t>& selfShape = {2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {4, 2};
  aclDataType otherDtype = ACL_FLOAT;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {4, 2};
  aclDataType outDtype = ACL_FLOAT;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// 形状大于8
TEST_F(l2_atan2_test, ascend910B2_atan2_inferdtype_test) {
  // self input
  const vector<int64_t>& selfShape = {2};
  aclDataType selfDtype = ACL_COMPLEX64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {4, 2};
  aclDataType otherDtype = ACL_INT32;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {4, 2};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TEST Cast
TEST_F(l2_atan2_test, ascend910B2_atan2_self_int16_other_int64_out_bool) {
  // self input
  const vector<int64_t>& selfShape = {2};
  aclDataType selfDtype = ACL_INT64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // other input
  const vector<int64_t>& otherShape = {4, 2};
  aclDataType otherDtype = ACL_INT16;
  aclFormat otherFormat = ACL_FORMAT_ND;
  // output
  const vector<int64_t>& outShape = {4, 2};
  aclDataType outDtype = ACL_BOOL;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnAtan2, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
