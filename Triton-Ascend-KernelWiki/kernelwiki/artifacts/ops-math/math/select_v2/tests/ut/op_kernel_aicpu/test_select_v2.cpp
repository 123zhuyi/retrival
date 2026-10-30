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

class TEST_SELECTV2_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CreateNodeDef();                 \
  NodeDefBuilder(node_def.get(), "SelectV2", "SelectV2")           \
      .Input({"condition", data_types[0], shapes[0], datas[0]})    \
      .Input({"then", data_types[1], shapes[1], datas[1]})         \
      .Input({"else", data_types[2], shapes[2], datas[2]})         \
      .Output({"result", data_types[3], shapes[3], datas[3]})

// ---- float32 basic test ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_FLOAT) {
  vector<DataType> data_types = {DT_BOOL, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}, {4}};
  bool condition[4] = {true, false, true, false};
  float then_val[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float else_val[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float output[4] = {0.0f};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float output_exp[4] = {1.0f, 6.0f, 3.0f, 8.0f};
  EXPECT_EQ(CompareResult<float>(output, output_exp, 4), true);
}

// ---- int32 basic test ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_INT32) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}, {3}};
  bool condition[3] = {true, true, false};
  int32_t then_val[3] = {10, 20, 30};
  int32_t else_val[3] = {40, 50, 60};
  int32_t output[3] = {0};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[3] = {10, 20, 60};
  EXPECT_EQ(CompareResult<int32_t>(output, output_exp, 3), true);
}

// ---- scalar test ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_SCALAR) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
  bool condition[1] = {true};
  int32_t then_val[1] = {2};
  int32_t else_val[1] = {1};
  int32_t output[1] = {0};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[1] = {2};
  EXPECT_EQ(CompareResult<int32_t>(output, output_exp, 1), true);
}

// ---- bool type test ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_BOOL) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
  bool condition[1] = {true};
  bool then_val[1] = {true};
  bool else_val[1] = {false};
  bool output[1] = {false};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool output_exp[1] = {true};
  EXPECT_EQ(CompareResult<bool>(output, output_exp, 1), true);
}

// ---- exception: mismatched data type ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_InputDtypeException) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};
  bool condition[2] = {true, false};
  int32_t then_val[2] = {1, 2};
  float else_val[2] = {3.0f, 4.0f};
  int32_t output[2] = {0};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ---- exception: null input ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_NullInput) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};
  int32_t output[2] = {0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ---- double type test (AICPU only) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_DOUBLE) {
  vector<DataType> data_types = {DT_BOOL, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}, {4}};
  bool condition[4] = {true, false, true, false};
  double then_val[4] = {1.1, 2.2, 3.3, 4.4};
  double else_val[4] = {5.5, 6.6, 7.7, 8.8};
  double output[4] = {0.0};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  double output_exp[4] = {1.1, 6.6, 3.3, 8.8};
  EXPECT_EQ(CompareResult<double>(output, output_exp, 4), true);
}

// ---- int16 type test (AICPU only) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_INT16) {
  vector<DataType> data_types = {DT_BOOL, DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}, {3}};
  bool condition[3] = {true, false, true};
  int16_t then_val[3] = {100, 200, 300};
  int16_t else_val[3] = {400, 500, 600};
  int16_t output[3] = {0};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int16_t output_exp[3] = {100, 500, 300};
  EXPECT_EQ(CompareResult<int16_t>(output, output_exp, 3), true);
}

// ---- uint16 type test (AICPU only) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_UINT16) {
  vector<DataType> data_types = {DT_BOOL, DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}, {4}};
  bool condition[4] = {false, true, false, true};
  uint16_t then_val[4] = {1000, 2000, 3000, 4000};
  uint16_t else_val[4] = {5000, 6000, 7000, 8000};
  uint16_t output[4] = {0};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint16_t output_exp[4] = {5000, 2000, 7000, 4000};
  EXPECT_EQ(CompareResult<uint16_t>(output, output_exp, 4), true);
}

// ---- uint32 type test (AICPU only) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_UINT32) {
  vector<DataType> data_types = {DT_BOOL, DT_UINT32, DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{5}, {5}, {5}, {5}};
  bool condition[5] = {true, true, false, false, true};
  uint32_t then_val[5] = {1, 2, 3, 4, 5};
  uint32_t else_val[5] = {10, 20, 30, 40, 50};
  uint32_t output[5] = {0};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint32_t output_exp[5] = {1, 2, 30, 40, 5};
  EXPECT_EQ(CompareResult<uint32_t>(output, output_exp, 5), true);
}

// ---- uint64 type test (AICPU only) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_UINT64) {
  vector<DataType> data_types = {DT_BOOL, DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}, {3}};
  bool condition[3] = {false, true, false};
  uint64_t then_val[3] = {100000ULL, 200000ULL, 300000ULL};
  uint64_t else_val[3] = {400000ULL, 500000ULL, 600000ULL};
  uint64_t output[3] = {0};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[3] = {400000ULL, 200000ULL, 600000ULL};
  EXPECT_EQ(CompareResult<uint64_t>(output, output_exp, 3), true);
}

// ---- complex64 type test (AICPU only) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_COMPLEX64) {
  vector<DataType> data_types = {DT_BOOL, DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};
  bool condition[2] = {true, false};
  std::complex<float> then_val[2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  std::complex<float> else_val[2] = {{5.0f, 6.0f}, {7.0f, 8.0f}};
  std::complex<float> output[2] = {{0.0f, 0.0f}};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::complex<float> output_exp[2] = {{1.0f, 2.0f}, {7.0f, 8.0f}};
  EXPECT_EQ(CompareResult<std::complex<float>>(output, output_exp, 2), true);
}

// ---- complex128 type test (AICPU only) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_COMPLEX128) {
  vector<DataType> data_types = {DT_BOOL, DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}, {3}};
  bool condition[3] = {true, false, true};
  std::complex<double> then_val[3] = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}};
  std::complex<double> else_val[3] = {{7.7, 8.8}, {9.9, 10.10}, {11.11, 12.12}};
  std::complex<double> output[3] = {{0.0, 0.0}};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::complex<double> output_exp[3] = {{1.1, 2.2}, {9.9, 10.10}, {5.5, 6.6}};
  EXPECT_EQ(CompareResult<std::complex<double>>(output, output_exp, 3), true);
}

// ---- bool vector test (AICPU only, comprehensive) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_BOOL_Vector) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{6}, {6}, {6}, {6}};
  bool condition[6] = {true, false, true, false, true, false};
  bool then_val[6] = {true, true, false, false, true, true};
  bool else_val[6] = {false, false, true, true, false, false};
  bool output[6] = {false};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool output_exp[6] = {true, false, false, true, true, false};
  EXPECT_EQ(CompareResult<bool>(output, output_exp, 6), true);
}

// ---- exception: shape broadcast failure ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_ShapeBroadcastFailure) {
  vector<DataType> data_types = {DT_BOOL, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 2}, {2, 2}, {2, 2}};
  bool condition[6] = {true, false, true, false, true, false};
  float then_val[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float else_val[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float output[4] = {0.0f};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ---- normal: output dtype mismatch (kernel allows type cast) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_OutputDtypeMismatch) {
  vector<DataType> data_types = {DT_BOOL, DT_FLOAT, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};
  bool condition[2] = {true, false};
  float then_val[2] = {1.0f, 2.0f};
  float else_val[2] = {3.0f, 4.0f};
  int32_t output[2] = {0};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

// ---- normal: condition not BOOL type (kernel accepts any dtype for condition) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_ConditionNotBool) {
  vector<DataType> data_types = {DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};
  int32_t condition[2] = {1, 0};
  float then_val[2] = {1.0f, 2.0f};
  float else_val[2] = {3.0f, 4.0f};
  float output[2] = {0.0f};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

// ---- normal: unsupported dtype for condition (kernel accepts any dtype) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_UnsupportedConditionDtype) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};
  Eigen::half condition[2] = {Eigen::half(1.0f), Eigen::half(0.0f)};
  float then_val[2] = {1.0f, 2.0f};
  float else_val[2] = {3.0f, 4.0f};
  float output[2] = {0.0f};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

// ---- exception: empty tensor (zero elements) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_EmptyTensor) {
  vector<DataType> data_types = {DT_BOOL, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{0}, {0}, {0}, {0}};
  bool condition[1] = {true};
  float then_val[1] = {1.0f};
  float else_val[1] = {2.0f};
  float output[1] = {0.0f};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

// ---- exception: then and else different dtypes ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_ThenElseDtypeMismatch) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_INT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};
  bool condition[2] = {true, false};
  int32_t then_val[2] = {1, 2};
  int64_t else_val[2] = {3, 4};
  int32_t output[2] = {0};
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ---- exception: large tensor test (edge case) ----
TEST_F(TEST_SELECTV2_UT, TestSelectV2_LargeTensor) {
  vector<DataType> data_types = {DT_BOOL, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1024}, {1024}, {1024}, {1024}};
  bool condition[1024];
  float then_val[1024];
  float else_val[1024];
  float output[1024];
  for (int i = 0; i < 1024; i++) {
    condition[i] = (i % 2 == 0);
    then_val[i] = static_cast<float>(i);
    else_val[i] = static_cast<float>(i + 1000);
    output[i] = 0.0f;
  }
  vector<void *> datas = {(void *)condition, (void *)then_val, (void *)else_val, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  for (int i = 0; i < 1024; i++) {
    float expected = (i % 2 == 0) ? static_cast<float>(i) : static_cast<float>(i + 1000);
    EXPECT_NEAR(output[i], expected, 1e-6);
  }
}