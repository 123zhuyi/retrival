/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "math/cummin/op_api/aclnn_cummin.h"
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "acl/acl.h"

using namespace std;

class l2_cummin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "cummin_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "cummin_test TearDown" << endl;
  }
 public:
  template <typename T>
  int CreateAclScalarTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(0, 0, dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), nullptr);
    return 0;
  }
};

// 正常场景
TEST_F(l2_cummin_test, l2_cummin_all_datatype_out_int64) {
  vector<aclDataType> dtypes{ACL_FLOAT, ACL_FLOAT16, ACL_UINT8, ACL_INT8, ACL_INT32};
  int64_t dim = 0;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto values_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({2, 2, 2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  }
}

TEST_F(l2_cummin_test, l2_cummin_all_datatype_out_int32) {
  vector<aclDataType> dtypes{ACL_FLOAT, ACL_FLOAT16, ACL_UINT8, ACL_INT8, ACL_INT32};
  int64_t dim = 0;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto values_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({2, 2, 2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  }
}

TEST_F(l2_cummin_test, l2_cummin_all_Format) {
  vector<aclFormat> formats{ACL_FORMAT_ND, ACL_FORMAT_NCHW, ACL_FORMAT_NC, ACL_FORMAT_NCL};
  int64_t dim = 0;
  aclDataType dtype = ACL_FLOAT;
  for (auto format : formats) {
    auto self_tensor = TensorDesc({2, 2, 2, 3}, dtype, format).ValueRange(-50, 50);
    auto values_tensor = TensorDesc({2, 2, 2, 3}, dtype, format).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({2, 2, 2, 3}, ACL_INT32, format).ValueRange(-50, 50);

    auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  }
}

TEST_F(l2_cummin_test, l2_cummin_all_datatype_not_same_out_int64) {
  vector<aclDataType> dtypes{ACL_FLOAT, ACL_FLOAT16, ACL_UINT8, ACL_INT8, ACL_INT32};
  int64_t dim = 0;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({2, 2, 2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-50, 50);
    for (auto dtype_ : dtypes) {
      auto values_tensor = TensorDesc({2, 2, 2, 3}, dtype_, ACL_FORMAT_ND).ValueRange(-50, 50);

      auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
    }
  }
}

TEST_F(l2_cummin_test, l2_cummin_all_datatype_not_same_out_int32) {
  vector<aclDataType> dtypes{ACL_FLOAT, ACL_FLOAT16, ACL_UINT8, ACL_INT8, ACL_INT32};
  int64_t dim = 0;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({2, 2, 2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-50, 50);
    for (auto dtype_ : dtypes) {
      auto values_tensor = TensorDesc({2, 2, 2, 3}, dtype_, ACL_FORMAT_ND).ValueRange(-50, 50);

      auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
    }
  }
}

TEST_F(l2_cummin_test, l2_cummin_all_datatype_out_int32_8_dim_0) {
  vector<aclDataType> dtypes{ACL_FLOAT, ACL_FLOAT16, ACL_UINT8, ACL_INT8, ACL_INT32};
  int64_t dim = 0;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto values_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  }
}

TEST_F(l2_cummin_test, l2_cummin_all_datatype_out_int64_8_dim_0) {
  vector<aclDataType> dtypes{ACL_FLOAT, ACL_FLOAT16, ACL_UINT8, ACL_INT8, ACL_INT32};
  int64_t dim = 0;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto values_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  }
}

TEST_F(l2_cummin_test, l2_cummin_all_datatype_out_int32_8_dim_1) {
  vector<aclDataType> dtypes{ACL_FLOAT, ACL_FLOAT16, ACL_UINT8, ACL_INT8, ACL_INT32};
  int64_t dim = 1;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto values_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  }
}

TEST_F(l2_cummin_test, l2_cummin_out_int64_8_dim_1) {
  vector<aclDataType> dtypes{ACL_FLOAT, ACL_FLOAT16, ACL_UINT8, ACL_INT8, ACL_INT32};
  int64_t dim = 1;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto values_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({1, 1, 1, 1, 2, 2, 2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  }
}

TEST_F(l2_cummin_test, l2_cummin_out_int32_negative_one) {
  int64_t dim = -1;
  aclDataType dtype = ACL_FLOAT;
  auto self_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto values_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto indices_tensor = TensorDesc({2, 2, 2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-50, 50);

  auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
}

TEST_F(l2_cummin_test, l2_cummin_null_tensor) {
  int64_t dim = 1;
  aclDataType dtype = ACL_FLOAT;
  auto self_tensor = TensorDesc({2, 2, 0, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto values_tensor = TensorDesc({2, 2, 0, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto indices_tensor = TensorDesc({2, 2, 0, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-50, 50);

  auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// discontinues
TEST_F(l2_cummin_test, l2_cummin_out_int64_self_discontinues) {
  int64_t dim = 0;
  aclDataType dtype = ACL_FLOAT;
  auto self_tensor =
      TensorDesc({2, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NCHW, {40, 20, 1, 5}, 0, {2, 2, 4, 5}).ValueRange(-50, 50);
  auto values_tensor = TensorDesc({2, 2, 5, 4}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
  auto indices_tensor = TensorDesc({2, 2, 5, 4}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(-50, 50);

  auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
}
TEST_F(l2_cummin_test, l2_cummin_out_int32_self_discontinues) {
  int64_t dim = 0;
  aclDataType dtype = ACL_FLOAT;
  auto self_tensor =
      TensorDesc({2, 2, 5, 4}, dtype, ACL_FORMAT_ND, {40, 20, 1, 5}, 0, {2, 2, 4, 5}).ValueRange(-50, 50);
  auto values_tensor = TensorDesc({2, 2, 5, 4}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto indices_tensor = TensorDesc({2, 2, 5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-50, 50);

  auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
}

TEST_F(l2_cummin_test, l2_cummin_out_int64_out_discontinues) {
  int64_t dim = 0;
  aclDataType dtype = ACL_FLOAT;
  auto self_tensor = TensorDesc({2, 2, 5, 4}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto values_tensor =
      TensorDesc({2, 2, 5, 4}, dtype, ACL_FORMAT_ND, {40, 20, 1, 5}, 0, {2, 2, 4, 5}).ValueRange(-50, 50);
  auto indices_tensor = TensorDesc({2, 2, 5, 4}, ACL_INT64, ACL_FORMAT_ND, {40, 20, 1, 5}, 0, {2, 2, 4, 5}).ValueRange(-50, 50);

  auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
}

// Abnormal Scenarios

TEST_F(l2_cummin_test, l2_cummin_self_datatype_not_support) {
  vector<aclDataType> dtypes{ACL_COMPLEX64, ACL_COMPLEX128};
  int64_t dim = 1;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto values_tensor = TensorDesc({2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({2, 2, 2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(l2_cummin_test, l2_cummin_values_datatype_not_support) {
  vector<aclDataType> dtypes{ACL_COMPLEX64, ACL_COMPLEX128};
  int64_t dim = 1;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto values_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({2, 2, 2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(l2_cummin_test, l2_cummin_indices_datatype_not_support) {
  vector<aclDataType> dtypes{ACL_FLOAT, ACL_FLOAT16, ACL_DOUBLE,    ACL_UINT8,     ACL_INT8,
                             ACL_INT16, ACL_BOOL,    ACL_COMPLEX64, ACL_COMPLEX128};
  int64_t dim = 1;
  for (auto dtype : dtypes) {
    auto self_tensor = TensorDesc({2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto values_tensor = TensorDesc({2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto indices_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(l2_cummin_test, l2_cummin_values_shape_not_support_out_int32) {
  aclDataType dtype = ACL_FLOAT;
  int64_t dim = 1;

  auto self_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto values_tensor = TensorDesc({2, 2, 2, 4}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto indices_tensor = TensorDesc({2, 2, 2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-50, 50);

  auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cummin_test, l2_cummin_values_shape_not_support_out_int64) {
  aclDataType dtype = ACL_FLOAT;
  int64_t dim = 1;

  auto self_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto values_tensor = TensorDesc({2, 2, 2, 4}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto indices_tensor = TensorDesc({2, 2, 2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-50, 50);

  auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cummin_test, l2_cummin_indices_shape_not_support_out_int32) {
  aclDataType dtype = ACL_FLOAT;
  int64_t dim = 1;

  auto self_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto values_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto indices_tensor = TensorDesc({2, 2, 2, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-50, 50);

  auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cummin_test, l2_cummin_self_nullptr) {
  aclDataType dtype = ACL_FLOAT;
  int64_t dim = 1;

  auto self_tensor = nullptr;
  auto values_tensor = TensorDesc({2, 2, 2, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
  auto indices_tensor = TensorDesc({2, 2, 2, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-50, 50);

  auto ut = OP_API_UT(aclnnCummin, INPUT(self_tensor, dim), OUTPUT(values_tensor, indices_tensor));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cummin_test, l2_cummin_self_scalar_tensor) {
  std::vector<int64_t> selfShape = {1};
  std::vector<int64_t> valuesOutShape = {1};
  std::vector<int64_t> indicesOutShape = {1};
  int64_t dim = 0;
  aclTensor* self = nullptr;
  aclTensor* values = nullptr;
  aclTensor* indices = nullptr;
  void* selfDeviceAddr = nullptr;
  void* valuesOutDeviceAddr = nullptr;
  void* indicesOutDeviceAddr = nullptr;
  std::vector<float> selfHostData = {0.0};
  std::vector<float> valuesOutHostData(0.0);
  std::vector<int64_t> indicesOutHostData(0);
  auto ret = CreateAclScalarTensor<float>(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  std::cout<<"create self success"<<ret<<std::endl;
  ret = CreateAclScalarTensor<float>(valuesOutHostData, valuesOutShape, &valuesOutDeviceAddr, aclDataType::ACL_FLOAT, &values);
  std::cout<<"create values success"<<ret<<std::endl;
  ret = CreateAclScalarTensor<int64_t>(indicesOutHostData, indicesOutShape, &indicesOutDeviceAddr, aclDataType::ACL_INT64, &indices);
  std::cout<<"create indices success"<<ret<<std::endl;
  auto ut = OP_API_UT(aclnnCummin, INPUT(self, dim), OUTPUT(values, indices));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
