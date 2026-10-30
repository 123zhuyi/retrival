/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../../op_api/aclnn_reduce_nansum.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_reduce_nansum_test : public testing::Test {
protected:
  static void SetUpTestCase() {cout << "l2_reduce_nansum_test SetUp" << endl;}

  static void TearDownTestCase() { cout << "l2_reduce_nansum_test TearDown" << endl; }
};

// self为空指针
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_nullptr_self) {
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(nullptr, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// dim为空指针
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_nullptr_dim) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, nullptr, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// out为空指针
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_nullptr_out) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(nullptr));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// self为空tensor
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_empty_self) {
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常路径，float16
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dtype_float16) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT16), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常路径，float32
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dtype_float32) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 不支持类型complex64
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dtype_complex64) {
  auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0});
  bool keep_dim = true;
  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_COMPLEX64), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持类型complex128
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dtype_complex128) {
  auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0});
  bool keep_dim = true;
  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_COMPLEX128), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

//  不支持类型double
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dtype_double) {
  auto selfDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0});
  bool keep_dim = true;
  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_DOUBLE), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，integral type, int64
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dtype_int64) {
  auto selfDesc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 1}, ACL_INT64, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{1});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_INT64), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常路径，integral type, int8
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dtype_int8) {
  auto selfDesc = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 1}, ACL_INT8, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{1});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_INT8), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常路径，integral type, bool
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dtype_bool) {
  auto selfDesc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 1}, ACL_BOOL, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{1});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_BOOL), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常路径，bf16
TEST_F(l2_reduce_nansum_test, ascend910B2_case_self_bf16) {
    auto input_tensor_desc = TensorDesc({1, 8, 2, 3, 6}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto dim_desc = IntArrayDesc(vector<int64_t>{2});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_BF16;
    auto out_desc = TensorDesc({1, 8, 3, 6}, ACL_BF16, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnReduceNansum, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

TEST_F(l2_reduce_nansum_test, case_self_bf16) {
    auto input_tensor_desc = TensorDesc({1, 8, 2, 3, 6}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto dim_desc = IntArrayDesc(vector<int64_t>{2});
    bool keepDim = false;
    aclDataType dtype_desc = ACL_BF16;
    auto out_desc = TensorDesc({1, 8, 3, 6}, ACL_BF16, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnReduceNansum, INPUT(input_tensor_desc, dim_desc, keepDim, dtype_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，dim为1
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dim_1) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{1});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常路径，dim为-1
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dim_neg1) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{-1});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常路径，dim为-1,0
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dim_neg1_0) {
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{-1, 0});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常路径，dim为空
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dim_empty) {
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// dim轴重复
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_repeat_dim) {
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{1, 1});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// dim轴超过shape范围
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_invalid_dim) {
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{1, 3});
  bool keep_dim = true;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，keep_dim为false
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_keepdim_false) {
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{-1, 0});
  bool keep_dim = false;

  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

//  用例不支持，最大维度超过8
TEST_F(l2_reduce_nansum_test, ascend910B2_case_test_dimension) {
  auto selfDesc = TensorDesc({2, 2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0});
  bool keep_dim = true;
  auto ut = OP_API_UT(aclnnReduceNansum, INPUT(selfDesc, dim, keep_dim, ACL_FLOAT), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
