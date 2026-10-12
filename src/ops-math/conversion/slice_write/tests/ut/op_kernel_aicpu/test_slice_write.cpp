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
#include "utils/aicpu_read_file.h"
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_SLICE_WRITE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "SliceWrite", "SliceWrite")       \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Input({"begin", data_types[1], shapes[1], datas[1]})        \
      .Input({"value", data_types[2], shapes[2], datas[2]})        \
      .Output({"x", data_types[3], shapes[3], datas[3]})

TEST_F(TEST_SLICE_WRITE_UT, Dim2_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2,2}, {2}, {1,2}, {2,2}};
  float input0[4] = {5, 4, 6, 7};
  int32_t input1[2] = {1,0};
  float input2[2] = {8, 9};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)input0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float output_exp[4] = {5, 4, 8, 9};
  bool compare = CompareResult(input0, output_exp, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SLICE_WRITE_UT, Dim1_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4}, {1}, {2}, {4}};
  float input0[4] = {5, 4, 6, 7};
  int32_t input1[2] = {1};
  float input2[2] = {8, 9};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)input0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float output_exp[4] = {5, 8, 9, 7};
  bool compare = CompareResult(input0, output_exp, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SLICE_WRITE_UT, Dim1_Failed) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4}, {1}, {2}, {4}};
  float input0[4] = {5, 4, 6, 7};
  int32_t input1[2] = {3};
  float input2[2] = {8, 9};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)input0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SLICE_WRITE_UT, Dim2_double_SUCCESS) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2,2}, {2}, {1,2}, {2,2}};
  double input0[4] = {5, 4, 6, 7};
  int32_t input1[2] = {1,0};
  double input2[2] = {8, 9};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)input0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  double output_exp[4] = {5, 4, 8, 9};
  bool compare = CompareResult(input0, output_exp, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SLICE_WRITE_UT, Dim2_int32_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2,2}, {2}, {1,2}, {2,2}};
  int32_t input0[4] = {5, 4, 6, 7};
  int64_t input1[2] = {1,0};
  int32_t input2[2] = {8, 9};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)input0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[4] = {5, 4, 8, 9};
  bool compare = CompareResult(input0, output_exp, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SLICE_WRITE_UT, Dim2_int64_SUCCESS) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{2,2}, {2}, {1,2}, {2,2}};
  int64_t input0[4] = {5, 4, 6, 7};
  int32_t input1[2] = {1,0};
  int64_t input2[2] = {8, 9};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)input0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int64_t output_exp[4] = {5, 4, 8, 9};
  bool compare = CompareResult(input0, output_exp, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SLICE_WRITE_UT, Dim2_uint32_Failed) {
  vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{2,2}, {2}, {1,2}, {2,2}};
  uint32_t input0[4] = {5, 4, 6, 7};
  int32_t input1[2] = {1,0};
  uint32_t input2[2] = {8, 9};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)input0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}