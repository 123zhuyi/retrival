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
#include <complex>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_TanhGrad_UT : public testing::Test {};

auto CreateTanhGradNodeDef(const vector<vector<int64_t>> &shapes, const vector<DataType> &data_types,
                           const vector<void *> &datas, bool has_conj_attr = false, bool conj = false) {
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  auto builder = NodeDefBuilder(node_def.get(), "TanhGrad", "TanhGrad")
                     .Input({"y", data_types[0], shapes[0], datas[0]})
                     .Input({"dy", data_types[1], shapes[1], datas[1]});
  if (has_conj_attr) {
    builder.Attr("complex_conj", conj);
  }
  builder.Output({"z", data_types[2], shapes[2], datas[2]});
  return node_def;
}

template <typename T>
void RunTanhGradKernel(vector<DataType> data_types, vector<vector<int64_t>> &shapes,
                       const T *input_y_data, const T *input_dy_data,
                       const T *output_exp_data) {
  uint64_t input_y_size = CalTotalElements(shapes, 0);
  T *input_y = new T[input_y_size];
  for (uint64_t i = 0; i < input_y_size; ++i) {
    input_y[i] = input_y_data[i];
  }
  uint64_t input_dy_size = CalTotalElements(shapes, 1);
  T *input_dy = new T[input_dy_size];
  for (uint64_t i = 0; i < input_dy_size; ++i) {
    input_dy[i] = input_dy_data[i];
  }
  uint64_t output_size = CalTotalElements(shapes, 2);
  T *output = new T[output_size];
  vector<void *> datas = {(void *)input_y, (void *)input_dy, (void *)output};

  auto node_def = CreateTanhGradNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  T *output_exp = new T[output_size];
  for (uint64_t i = 0; i < output_size; ++i) {
    output_exp[i] = output_exp_data[i];
  }
  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input_y;
  delete[] input_dy;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_TanhGrad_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};
  const float input_y_data[] = {0.5f, 0.3f, 0.7f, 0.1f};
  const float input_dy_data[] = {1.0f, 1.0f, 2.0f, 2.0f};
  // dy * (1 - y * y)
  const float output_exp_data[] = {1.0f * (1.0f - 0.25f), 1.0f * (1.0f - 0.09f),
                                   2.0f * (1.0f - 0.49f), 2.0f * (1.0f - 0.01f)};
  RunTanhGradKernel<float>(data_types, shapes, input_y_data, input_dy_data, output_exp_data);
}

TEST_F(TEST_TanhGrad_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}};
  const double input_y_data[] = {0.5, 0.3, 0.7};
  const double input_dy_data[] = {1.0, 1.0, 2.0};
  const double output_exp_data[] = {1.0 * (1.0 - 0.25), 1.0 * (1.0 - 0.09), 2.0 * (1.0 - 0.49)};
  RunTanhGradKernel<double>(data_types, shapes, input_y_data, input_dy_data, output_exp_data);
}

TEST_F(TEST_TanhGrad_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
  const Eigen::half input_y_data[] = {static_cast<Eigen::half>(0.5f), static_cast<Eigen::half>(0.3f)};
  const Eigen::half input_dy_data[] = {static_cast<Eigen::half>(1.0f), static_cast<Eigen::half>(1.0f)};
  const Eigen::half output_exp_data[] = {static_cast<Eigen::half>(1.0f * (1.0f - 0.25f)),
                                         static_cast<Eigen::half>(1.0f * (1.0f - 0.09f))};
  RunTanhGradKernel<Eigen::half>(data_types, shapes, input_y_data, input_dy_data, output_exp_data);
}

TEST_F(TEST_TanhGrad_UT, DATA_TYPE_COMPLEX128_NO_CONJ_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
  std::complex<double> input_y[2] = {std::complex<double>(4.0, 5.0), std::complex<double>(5.0, -6.0)};
  std::complex<double> input_dy[2] = {std::complex<double>(4.0, 5.0), std::complex<double>(5.0, -6.0)};
  std::complex<double> output[2] = {0.0, 0.0};
  vector<void *> datas = {(void *)input_y, (void *)input_dy, (void *)output};

  auto node_def = CreateTanhGradNodeDef(shapes, data_types, datas, true, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  std::complex<double> output_exp[2] = {std::complex<double>(240.0, -110.0), std::complex<double>(420.0, 228.0)};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TanhGrad_UT, DATA_TYPE_COMPLEX128_CONJ_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
  std::complex<double> input_y[2] = {std::complex<double>(4.0, 5.0), std::complex<double>(5.0, -6.0)};
  std::complex<double> input_dy[2] = {std::complex<double>(4.0, 5.0), std::complex<double>(5.0, -6.0)};
  std::complex<double> output[2] = {0.0, 0.0};
  vector<void *> datas = {(void *)input_y, (void *)input_dy, (void *)output};

  auto node_def = CreateTanhGradNodeDef(shapes, data_types, datas, true, true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  std::complex<double> output_exp[2] = {std::complex<double>(-160.0, 210.0), std::complex<double>(-300.0, -372.0)};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TanhGrad_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
  float output[6] = {0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  auto node_def = CreateTanhGradNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TanhGrad_UT, DTYPE_MISMATCH_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
  float input_y[6] = {0};
  double input_dy[6] = {0};
  double output[6] = {0};
  vector<void *> datas = {(void *)input_y, (void *)input_dy, (void *)output};
  auto node_def = CreateTanhGradNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TanhGrad_UT, EMPTY_TENSOR_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{0}, {0}, {0}};
  float input_y[1] = {0};
  float input_dy[1] = {0};
  float output[1] = {0};
  vector<void *> datas = {(void *)input_y, (void *)input_dy, (void *)output};
  auto node_def = CreateTanhGradNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_TanhGrad_UT, OUTPUT_TOO_SMALL_EXCEPTION) {
  // canndev semantics: compute over min(y, dy). Sanity check rejects only when output cannot hold that range.
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 2}};
  float input_y[6] = {0};
  float input_dy[6] = {0};
  float output[4] = {0};
  vector<void *> datas = {(void *)input_y, (void *)input_dy, (void *)output};
  auto node_def = CreateTanhGradNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TanhGrad_UT, Y_DY_SHAPE_MISMATCH_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {3, 2}, {2, 3}};
  float input_y[6] = {0};
  float input_dy[6] = {0};
  float output[6] = {0};
  // Same NumElements (6==6) -> canndev computes over min(y,dy) and returns OK.
  vector<void *> datas = {(void *)input_y, (void *)input_dy, (void *)output};
  auto node_def = CreateTanhGradNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}
