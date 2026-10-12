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
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_SQRT_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CreateNodeDef();                 \
  NodeDefBuilder(node_def.get(), "Sqrt", "Sqrt")                   \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})

// ---- float32 basic test ----
TEST_F(TEST_SQRT_UT, TestSqrt_FLOAT) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
  // seed=23457, uniform(1,100), shape=[5,5]
  float input[25] = {81.52432f, 43.482735f, 67.64165f, 23.798012f, 36.49648f,
                     23.168648f, 11.171817f, 62.294262f, 83.37125f, 50.089794f,
                     69.03212f, 67.75925f, 46.901024f, 54.83537f, 98.68752f,
                     71.01781f, 92.65018f, 92.62283f, 99.74749f, 98.54601f,
                     47.414738f, 15.765511f, 78.63351f, 46.978813f, 94.61364f};
  float output[25] = {0.0f};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float output_exp[25] = {9.029082f, 6.594144f, 8.224454f, 4.8783207f, 6.0412316f,
                           4.813382f, 3.3424268f, 7.8926716f, 9.130786f, 7.0774145f,
                           8.308557f, 8.231601f, 6.848432f, 7.405091f, 9.934159f,
                           8.427206f, 9.625496f, 9.624076f, 9.987367f, 9.927034f,
                           6.885836f, 3.9705806f, 8.867554f, 6.8541093f, 9.726954f};
  EXPECT_EQ(CompareResult<float>(output, output_exp, 25), true);
}

// ---- float64 basic test (use perfect squares for exact results) ----
TEST_F(TEST_SQRT_UT, TestSqrt_DOUBLE) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
  // use perfect squares so expected output is exact
  double input[25] = {1.0,  4.0,  9.0,  16.0, 25.0,
                      36.0, 49.0, 64.0, 81.0, 100.0,
                      121.0, 144.0, 169.0, 196.0, 225.0,
                      256.0, 289.0, 324.0, 361.0, 400.0,
                      441.0, 484.0, 529.0, 576.0, 625.0};
  double output[25] = {0.0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  double output_exp[25] = {1.0,  2.0,  3.0,  4.0,  5.0,
                            6.0,  7.0,  8.0,  9.0,  10.0,
                            11.0, 12.0, 13.0, 14.0, 15.0,
                            16.0, 17.0, 18.0, 19.0, 20.0,
                            21.0, 22.0, 23.0, 24.0, 25.0};
  EXPECT_EQ(CompareResult<double>(output, output_exp, 25), true);
}

// ---- float16 basic test ----
TEST_F(TEST_SQRT_UT, TestSqrt_FLOAT16) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
  // seed=3457, randint(1,100), shape=[5,5]
  Eigen::half input[25] = {
    Eigen::half(15.0f), Eigen::half(46.0f), Eigen::half(8.0f), Eigen::half(95.0f), Eigen::half(72.0f),
    Eigen::half(54.0f), Eigen::half(20.0f), Eigen::half(59.0f), Eigen::half(98.0f), Eigen::half(23.0f),
    Eigen::half(98.0f), Eigen::half(63.0f), Eigen::half(76.0f), Eigen::half(37.0f), Eigen::half(72.0f),
    Eigen::half(13.0f), Eigen::half(59.0f), Eigen::half(70.0f), Eigen::half(77.0f), Eigen::half(88.0f),
    Eigen::half(85.0f), Eigen::half(96.0f), Eigen::half(28.0f), Eigen::half(2.0f),  Eigen::half(7.0f)};
  Eigen::half output[25] = {Eigen::half(0.0f)};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  Eigen::half output_exp[25] = {
    Eigen::half(3.873f), Eigen::half(6.78f),  Eigen::half(2.828f), Eigen::half(9.75f),  Eigen::half(8.484f),
    Eigen::half(7.348f), Eigen::half(4.473f),  Eigen::half(7.68f),  Eigen::half(9.9f),   Eigen::half(4.797f),
    Eigen::half(9.9f),   Eigen::half(7.938f),  Eigen::half(8.72f),  Eigen::half(6.082f), Eigen::half(8.484f),
    Eigen::half(3.605f), Eigen::half(7.68f),   Eigen::half(8.37f),  Eigen::half(8.77f),  Eigen::half(9.38f),
    Eigen::half(9.22f),  Eigen::half(9.8f),    Eigen::half(5.293f), Eigen::half(1.414f), Eigen::half(2.646f)};
  EXPECT_EQ(CompareResult<Eigen::half>(output, output_exp, 25), true);
}

// ---- exception: mismatched data type ----
TEST_F(TEST_SQRT_UT, TestSqrt_InputDtypeException) {
  vector<DataType> data_types = {DT_DOUBLE, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  double input[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  float output[6] = {0.0f};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ---- exception: mismatched shape ----
TEST_F(TEST_SQRT_UT, TestSqrt_InputShapeException) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 4}};
  float input[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float output[8] = {0.0f};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ---- exception: unsupported type (INT32) ----
TEST_F(TEST_SQRT_UT, TestSqrt_UnsupportedType) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  int32_t input[6] = {1, 2, 3, 4, 5, 6};
  int32_t output[6] = {0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ---- null input ----
TEST_F(TEST_SQRT_UT, TestSqrt_NullInput) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  float output[6] = {0.0f};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
