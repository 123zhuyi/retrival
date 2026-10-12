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

#include "../../../op_api/aclnn_expand.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_expand_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "expand_test SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "expand_test TearDown" << std::endl; }
};

// precision
TEST_F(l2_expand_test, case_precisions) {
  auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self nullptr
TEST_F(l2_expand_test, case_self_nullptr) {
  auto self = nullptr;
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_expand_test, case_out_nullptr) {
  auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = nullptr;
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// self float16
TEST_F(l2_expand_test, case_self_float16) {
  auto self = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self bfloat16
TEST_F(l2_expand_test, ascend910B2_case_self_float16) {
  auto self = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self uint8
TEST_F(l2_expand_test, case_self_uint8) {
  auto self = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self int8
TEST_F(l2_expand_test, case_self_int8) {
  auto self = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self int32
TEST_F(l2_expand_test, case_self_int32) {
  auto self = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self bool
TEST_F(l2_expand_test, case_self_bool) {
  auto self = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self float64
TEST_F(l2_expand_test, case_self_float64) {
  auto self = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self and out dtype not match
TEST_F(l2_expand_test, case_dtype_not_match) {
  auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self nchw
TEST_F(l2_expand_test, case_self_nchw) {
  auto self = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto size = IntArrayDesc(vector<int64_t>{2, 2, 2, 2});
  auto out = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self nhwc
TEST_F(l2_expand_test, case_self_nhwc) {
  auto self = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);
  auto size = IntArrayDesc(vector<int64_t>{2, 2, 2, 2});
  auto out = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self hwcn
TEST_F(l2_expand_test, case_self_hwcn) {
  auto self = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN);
  auto size = IntArrayDesc(vector<int64_t>{2, 2, 2, 2});
  auto out = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self ncdhw
TEST_F(l2_expand_test, case_self_ncdhw) {
  auto self = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto size = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2});
  auto out = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self ndhwc
TEST_F(l2_expand_test, case_self_ndhwc) {
  auto self = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC);
  auto size = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2});
  auto out = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// self and size not match
TEST_F(l2_expand_test, case_self_and_size_not_match) {
  auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 3});
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out and size not match
TEST_F(l2_expand_test, case_out_and_size_not_match) {
  auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self size greater than size
TEST_F(l2_expand_test, case_self_size_greater_than_size) {
  auto self = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2});
  auto out = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// not contiguous
TEST_F(l2_expand_test, case_not_contiguous) {
  auto self = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
  auto size = IntArrayDesc(vector<int64_t>{2, 4});
  auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// empty tensor
TEST_F(l2_expand_test, case_emtpy_tensor) {
  auto self = TensorDesc({2, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 0, 2});
  auto out = TensorDesc({2, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// max dim greater than 8
TEST_F(l2_expand_test, case_max_dim_greater_than_8) {
  auto self = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto size = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2, 2, 2, 2});
  auto out = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnExpand, INPUT(self, size), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}