/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include <cmath>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_Asin_UT : public testing::Test {};

auto CreateAsinNodeDef(const vector<vector<int64_t>> &shapes, const vector<DataType> &data_types,
                       const vector<void *> &datas) {
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "Asin", "Asin")
      .Input({"x", data_types[0], shapes[0], datas[0]})
      .Output({"y", data_types[1], shapes[1], datas[1]});
  return node_def;
}

template <typename T>
void RunAsinKernel(vector<DataType> data_types, vector<vector<int64_t>> &shapes,
                   const T *input_data, const T *output_exp_data) {
  uint64_t input_size = CalTotalElements(shapes, 0);
  T *input = new T[input_size];
  for (uint64_t i = 0; i < input_size; ++i) {
    input[i] = input_data[i];
  }
  uint64_t output_size = CalTotalElements(shapes, 1);
  T *output = new T[output_size];
  vector<void *> datas = {(void *)input, (void *)output};

  auto node_def = CreateAsinNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  T *output_exp = new T[output_size];
  for (uint64_t i = 0; i < output_size; ++i) {
    output_exp[i] = output_exp_data[i];
  }
  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_Asin_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  const float input_data[] = {0.0f, 0.5f, 1.0f, -0.5f, -1.0f, 0.25f};
  const float output_exp_data[] = {std::asin(0.0f), std::asin(0.5f), std::asin(1.0f),
                                   std::asin(-0.5f), std::asin(-1.0f), std::asin(0.25f)};
  RunAsinKernel<float>(data_types, shapes, input_data, output_exp_data);
}

TEST_F(TEST_Asin_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{4}, {4}};
  const double input_data[] = {0.1, 0.2, 0.3, 0.4};
  const double output_exp_data[] = {std::asin(0.1), std::asin(0.2), std::asin(0.3), std::asin(0.4)};
  RunAsinKernel<double>(data_types, shapes, input_data, output_exp_data);
}

TEST_F(TEST_Asin_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3}, {3}};
  const Eigen::half input_data[] = {static_cast<Eigen::half>(0.0f),
                                    static_cast<Eigen::half>(0.5f),
                                    static_cast<Eigen::half>(-0.5f)};
  const Eigen::half output_exp_data[] = {static_cast<Eigen::half>(std::asin(0.0f)),
                                         static_cast<Eigen::half>(std::asin(0.5f)),
                                         static_cast<Eigen::half>(std::asin(-0.5f))};
  RunAsinKernel<Eigen::half>(data_types, shapes, input_data, output_exp_data);
}

TEST_F(TEST_Asin_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  float output[6] = {0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  auto node_def = CreateAsinNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Asin_UT, DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  int32_t input[6] = {0};
  int32_t output[6] = {0};
  vector<void *> datas = {(void *)input, (void *)output};
  auto node_def = CreateAsinNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Asin_UT, DTYPE_MISMATCH_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  float input[6] = {0};
  double output[6] = {0};
  vector<void *> datas = {(void *)input, (void *)output};
  auto node_def = CreateAsinNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
