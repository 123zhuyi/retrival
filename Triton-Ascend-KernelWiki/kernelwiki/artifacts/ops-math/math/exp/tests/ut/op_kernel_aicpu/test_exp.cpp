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
#include <cmath>

using namespace std;
using namespace aicpu;

class TEST_Exp_UT : public testing::Test {};

auto CreateExpNodeDef(const vector<vector<int64_t>> &shapes, const vector<DataType> &data_types,
                      const vector<void *> &datas) {
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "Exp", "Exp")
      .Input({"x", data_types[0], shapes[0], datas[0]})
      .Output({"output", data_types[1], shapes[1], datas[1]});
  return node_def;
}

TEST_F(TEST_Exp_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3}, {3}};
  std::complex<float> input_data[] = {std::complex<float>(0.0f, 0.0f),
                                      std::complex<float>(1.0f, 0.0f),
                                      std::complex<float>(0.0f, 1.0f)};
  std::complex<float> output[3];
  std::complex<float> output_exp[] = {std::exp(input_data[0]), std::exp(input_data[1]), std::exp(input_data[2])};
  vector<void *> datas = {(void *)input_data, (void *)output};
  auto node_def = CreateExpNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  EXPECT_EQ(CompareResult(output, output_exp, 3), true);
}

TEST_F(TEST_Exp_UT, DTYPE_FLOAT_UNSUPPORTED) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2}, {2}};
  float input_data[] = {0.0f, 1.0f};
  float output[2] = {0};
  vector<void *> datas = {(void *)input_data, (void *)output};
  auto node_def = CreateExpNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
