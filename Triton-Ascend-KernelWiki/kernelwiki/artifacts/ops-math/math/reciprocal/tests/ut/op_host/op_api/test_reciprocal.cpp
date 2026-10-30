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

#include "aclnn_reciprocal.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_reciprocal_test : public testing::Test {
 protected:
  static void SetUpTestCase() {cout << "reciprocal_test SetUp" << endl;}

  static void TearDownTestCase() { cout << "reciprocal_test TearDown" << endl; }

  // FP16 or BF16
  void test_run_low_precision(aclDataType input_test_dtype, aclDataType output_test_dtype, aclFormat test_format,
                              vector<int64_t> view_dims, bool need_precision=false){
    const vector<int64_t>& view_dims_final= const_cast <vector<int64_t>&>(view_dims);

    auto self = TensorDesc(view_dims_final, input_test_dtype, test_format).ValueRange(-100, 100);
    auto out = TensorDesc(view_dims_final, output_test_dtype, test_format).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    if (need_precision) {
      ut.TestPrecision();
    }
  }

  // FP32
  void test_run_high_precision(aclDataType input_test_dtype, aclDataType output_test_dtype, aclFormat test_format,
                               vector<int64_t> view_dims, bool need_precision=false){
    const vector<int64_t>& view_dims_final= const_cast <vector<int64_t>&>(view_dims);

    auto self = TensorDesc(view_dims_final, input_test_dtype, test_format).ValueRange(-100, 100);
    auto out = TensorDesc(view_dims_final, output_test_dtype, test_format).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    if (need_precision) {
      ut.TestPrecision();
    }
  }

  void test_run_invalid(aclDataType input_test_dtype, aclDataType output_test_dtype, aclFormat test_format, vector<int64_t> view_dims){
    const vector<int64_t>& view_dims_final= const_cast <vector<int64_t>&>(view_dims);

    auto self = TensorDesc(view_dims_final, input_test_dtype, test_format).ValueRange(-100, 100);
    auto out = TensorDesc(view_dims_final, output_test_dtype, test_format).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
  }


  void test_run_nullptr(aclDataType input_test_dtype, aclDataType output_test_dtype, aclFormat test_format, vector<int64_t> view_dims){
    const vector<int64_t>& view_dims_final= const_cast <vector<int64_t>&>(view_dims);

    auto self = TensorDesc(view_dims_final, input_test_dtype, test_format).ValueRange(-100, 100);
    auto out = TensorDesc(view_dims_final, output_test_dtype, test_format).Precision(0.0001, 0.0001);

    uint64_t workspaceSize = 0;

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(nullptr), OUTPUT(out));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(nullptr));
    aclnnStatus getWorkspaceResult1 = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult1, ACLNN_ERR_PARAM_NULLPTR);
  }
};

// 检查支持的数据类型
TEST_F(l2_reciprocal_test, case_1) {
  test_run_low_precision(ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_INT8, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_UINT8, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_INT32, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_INT64, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_INT16, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_BOOL, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
}
/*
// 特殊检查：芯片版本符合要求时，支持BF16数据类型
TEST_F(l2_reciprocal_test, case_2) {
  test_run_low_precision(ACL_BF16, ACL_BF16, ACL_FORMAT_ND, {1, 2, 3});
}
*/
// 检查不支持的数据类型
TEST_F(l2_reciprocal_test, case_3) {
  test_run_high_precision(ACL_DOUBLE, ACL_DOUBLE, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_COMPLEX64, ACL_COMPLEX64, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_COMPLEX128, ACL_COMPLEX128, ACL_FORMAT_ND, {1, 2, 3});
}

// 检查支持的数据格式
TEST_F(l2_reciprocal_test, case_4) {
  test_run_high_precision(ACL_INT16, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_INT16, ACL_FLOAT, ACL_FORMAT_NCHW, {1, 2, 3, 4});
  test_run_high_precision(ACL_INT16, ACL_FLOAT, ACL_FORMAT_NHWC, {1, 2, 3, 4});
  test_run_high_precision(ACL_INT16, ACL_FLOAT, ACL_FORMAT_NDHWC, {1, 2, 3, 4, 5});
  test_run_high_precision(ACL_INT16, ACL_FLOAT, ACL_FORMAT_NCDHW, {1, 2, 3, 4, 5});
  test_run_high_precision(ACL_INT16, ACL_FLOAT, ACL_FORMAT_HWCN, {1, 2, 3, 4});
  test_run_high_precision(ACL_INT16, ACL_FLOAT, ACL_FORMAT_NCL, {1, 2, 3});
}

// 检查空Tensor和1-8维的输入
TEST_F(l2_reciprocal_test, case_6) {
  test_run_high_precision(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {});
  test_run_high_precision(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1});
  test_run_high_precision(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2});
  test_run_high_precision(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3, 4});
  test_run_high_precision(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3, 4, 5});
  test_run_high_precision(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3, 4, 5, 6});
  test_run_high_precision(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3, 4, 5, 6, 7});
  test_run_high_precision(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3, 4, 5, 6, 7, 8});
}

// 检查超过8维的输入
TEST_F(l2_reciprocal_test, case_7) {
  test_run_invalid(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test_run_invalid(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
}

//检查空指针
TEST_F(l2_reciprocal_test, case_8) {
  test_run_nullptr(ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3, 4, 5});
}

// 检查非连续Tensor
TEST_F(l2_reciprocal_test, case_9) {
  auto self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-100, 100);
  auto out = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

  //ut.TestPrecision();
}

// 检查输入self在支持数据类型内，输出out为fp16
TEST_F(l2_reciprocal_test, case_10) {
  test_run_low_precision(ACL_FLOAT, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_INT8, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_UINT8, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_INT32, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_INT64, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_INT16, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_BOOL, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2, 3});
}

// 检查输入self在支持数据类型内，输出out为fp32
TEST_F(l2_reciprocal_test, case_11) {
  test_run_high_precision(ACL_FLOAT16, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_INT8, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_UINT8, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_INT32, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_INT64, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_INT16, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_high_precision(ACL_BOOL, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
}
/*
// 检查输入self在支持数据类型内，输出out为bf16
TEST_F(l2_reciprocal_test, case_12) {
  test_run_low_precision(ACL_FLOAT16, ACL_BF16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_FLOAT, ACL_BF16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_INT8, ACL_BF16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_UINT8, ACL_BF16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_INT32, ACL_BF16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_INT64, ACL_BF16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_INT16, ACL_BF16, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_BOOL, ACL_BF16, ACL_FORMAT_ND, {1, 2, 3});
}

// 特殊检查：芯片版本符合要求时，支持BF16数据类型cast成FP16或FP32
TEST_F(l2_reciprocal_test, case_13) {
  test_run_high_precision(ACL_BF16, ACL_FLOAT, ACL_FORMAT_ND, {1, 2, 3});
  test_run_low_precision(ACL_BF16, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2, 3});
}
*/
// 检查输入self在支持数据类型范围内，输出out不在支持数据类型范围内
TEST_F(l2_reciprocal_test, case_14) {
  test_run_invalid(ACL_FLOAT, ACL_INT8, ACL_FORMAT_ND, {1, 2, 3, 4, 5});
  test_run_invalid(ACL_FLOAT, ACL_UINT8, ACL_FORMAT_ND, {1, 2, 3, 4, 5});
  test_run_invalid(ACL_FLOAT, ACL_INT32, ACL_FORMAT_ND, {1, 2, 3, 4, 5});
  test_run_invalid(ACL_FLOAT, ACL_INT64, ACL_FORMAT_ND, {1, 2, 3, 4, 5});
  test_run_invalid(ACL_FLOAT, ACL_INT16, ACL_FORMAT_ND, {1, 2, 3, 4, 5});
  test_run_invalid(ACL_FLOAT, ACL_BOOL, ACL_FORMAT_ND, {1, 2, 3, 4, 5});
}

// 检查输入self的元素全为0
TEST_F(l2_reciprocal_test, case_15) {
  auto self = TensorDesc({5, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,0);
  auto out = TensorDesc({5, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

  //ut.TestPrecision();
}

// 检查原地操作
TEST_F(l2_reciprocal_test, case_16) {
  auto self = TensorDesc({5, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-100,100);

  auto ut = OP_API_UT(aclnnInplaceReciprocal, INPUT(self), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

  //ut.TestPrecision();
}

// 检查空Tensor输入
TEST_F(l2_reciprocal_test, case_17) {
  auto self = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,0);
  auto out = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

  //ut.TestPrecision();
}

// double
TEST_F(l2_reciprocal_test, case_double) {
  auto self = TensorDesc({5, 4, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-5,5);
  auto out = TensorDesc({5, 4, 3}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

  ut.TestPrecision();
}

// complex64
TEST_F(l2_reciprocal_test, case_complex64) {
  auto self = TensorDesc({5, 4, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-5,5);
  auto out = TensorDesc({5, 4, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

  //ut.TestPrecision();
}

// complex128
TEST_F(l2_reciprocal_test, case_complex128) {
  auto self = TensorDesc({5, 4, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-5,5);
  auto out = TensorDesc({5, 4, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

  //ut.TestPrecision();
}

// bf16_Ascend910
TEST_F(l2_reciprocal_test, case_bf16_ascend910) {
  auto self = TensorDesc({5, 4, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5,5);
  auto out = TensorDesc({5, 4, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// bf16_Ascend910B2
TEST_F(l2_reciprocal_test, case_bf16_ascend910B2) {
  auto self = TensorDesc({5, 4, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5,5);
  auto out = TensorDesc({5, 4, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

  //ut.TestPrecision();
}
