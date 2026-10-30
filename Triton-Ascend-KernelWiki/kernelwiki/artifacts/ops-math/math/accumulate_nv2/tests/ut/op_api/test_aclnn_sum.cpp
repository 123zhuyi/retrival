/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "math/accumulate_nv2/op_api/aclnn_sum.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"

using namespace std;

class l2_sum_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_sum_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_sum_test TearDown" << endl;
    }
};

// 输入为空指针
TEST_F(l2_sum_test, l2_sum_test_nullptr_input) {
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnSum, INPUT(nullptr), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 输出为空指针
TEST_F(l2_sum_test, l2_sum_test_nullptr_out) {
  auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_2_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(nullptr));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空tensors
TEST_F(l2_sum_test, l2_sum_test_empty_tensors) {
  auto tensor_1_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_2_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
  auto out_tensor_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输出shape不符合推导得到的shape
 TEST_F(l2_sum_test, l2_sum_test_int8_out_shape_fail) {
  auto tensor_1_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto tensor_2_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND)
                           .Value(vector<float>{7, 8, 9, 10, 11, 12});
  // OUT SHAPE EXPECT (2,3)
  auto out_tensor_desc = TensorDesc({1, 3}, ACL_INT8, ACL_FORMAT_ND);
  auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，int8
TEST_F(l2_sum_test, l2_sum_test_dtype_int8) {
  auto tensor_1_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto tensor_2_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND)
                           .Value(vector<float>{7, 8, 9, 10, 11, 12});
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
  auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，int32
TEST_F(l2_sum_test, l2_sum_test_dtype_int32) {
  auto tensor_1_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto tensor_2_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                           .Value(vector<float>{7, 8, 9, 10, 11, 12});
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
  auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，uint8
TEST_F(l2_sum_test, l2_sum_test_dtype_uint8) {
  auto tensor_1_desc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto tensor_2_desc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND)
                           .Value(vector<float>{7, 8, 9, 10, 11, 12});
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
  auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 用例不支持，double
TEST_F(l2_sum_test, l2_sum_test_dtype_double) {
  auto tensor_1_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto tensor_2_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND)
                           .Value(vector<float>{7, 8, 9, 10, 11, 12});
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，输入1个tensor
TEST_F(l2_sum_test, l2_sum_test_tensors_1) {
  auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_list_desc = TensorListDesc(1, tensor_1_desc);
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，输入200个tensor
TEST_F(l2_sum_test, l2_sum_test_tensors_200) {
  auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_list_desc = TensorListDesc(200, tensor_1_desc);
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 用例不支持，输入tensor维度超过8
TEST_F(l2_sum_test, l2_sum_test_9dim) {
  auto tensor_1_desc = TensorDesc({2, 3, 2, 3, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_2_desc = TensorDesc({2, 3, 2, 3, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 3, 2, 3, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 0维Tensor用例
TEST_F(l2_sum_test, l2_sum_test_0dim) {
  auto tensor_1_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1.2});
  auto tensor_2_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2.2});
  auto out_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.0});
  auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
  auto ut = OP_API_UT(aclnnSum, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
