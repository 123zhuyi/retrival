/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * ============================================================================
 * POW 算子测试用例说明
 * ============================================================================
 *
 * 一、正常测试用例
 * --------------------------------------------------------------------------
 * 1. 基础数据类型测试
 *    - DATA_TYPE_FLOAT_SUCC: Float 类型同形状计算
 *    - DATA_TYPE_INT8_SUCC: Int8 类型同形状计算
 *    - DATA_TYPE_INT32_SUCC: Int32 类型同形状计算
 *    - DATA_TYPE_INT64_SUCC: Int64 类型同形状计算
 *    - DATA_TYPE_DOUBLE_SUCC: Double 类型同形状计算
 *    - DATA_TYPE_COMPLEX64_SUCC: Complex64 类型同形状计算
 *    - DATA_TYPE_COMPLEX128_SUCC: Complex128 类型同形状计算
 *
 * 2. 广播场景测试
 *    - DATA_TYPE_FLOAT16_BROADCAST_SUCC: Float16 广播场景
 *    - DATA_TYPE_INT32_BROADCAST_SUCC: Int32 广播场景
 *
 * 3. 单元素广播测试
 *    - DATA_TYPE_ONE_X_ELEMENT_SUCC: X 单元素广播到 Y
 *    - DATA_TYPE_ONE_Y_ELEMENT_SUCC: Y 单元素广播到 X
 *
 * 4. 大张量测试
 *    - DATA_TYPE_NOBROADCAST_LARGE_35x1024: 大张量无广播 (35x1024)
 *    - DATA_TYPE_NOBROADCAST_LARGE_141x256: 大张量无广播 (141x256)
 *    - DATA_TYPE_BROADCAST_LARGE_30x1024: 大张量广播 (30x1024)
 *    - DATA_TYPE_FLOAT16_LARGE_100: Float16 大张量 (100)
 *    - DATA_TYPE_BROADCAST_FLOAT_100x24: Float 广播 (100x24)
 *    - DATA_TYPE_DOUBLE_LARGE_100: Double 大张量 (100)
 *    - DATA_TYPE_INT32_LARGE_100: Int32 大张量 (100)
 *    - DATA_TYPE_UINT8_LARGE_10x10x100: Uint8 大张量 (10x10x100)
 *
 * 5. 边界条件测试
 *    - FLOAT_SAME_SHAPE_SUCC: Float 同形状边界测试
 *    - FLOAT16_X_ONE_ELEMENT_SUCC: Float16 X 单元素边界测试
 *    - INT32_Y_ONE_ELEMENT_SUCC: Int32 Y 单元素边界测试
 *    - INT32_POWER_0_SUCC: Int32 零次幂边界测试
 *
 * 6. 类型转换宏测试
 *    - TYPE_CAST_DOUBLE_TO_FLOAT: Double 转 Float 类型转换
 *    - TYPE_CAST_FLOAT_TO_INT32: Float 转 Int32 类型转换
 *    - TYPE_CAST_INT32_TO_INT64: Int32 转 Int64 类型转换
 *    - TYPE_CAST_INT64_TO_INT32: Int64 转 Int32 类型转换
 *    - TYPE_CAST_INT8_TO_INT32: Int8 转 Int32 类型转换
 *    - TYPE_CAST_INT16_TO_INT64: Int16 转 Int64 类型转换
 *    - TYPE_CAST_UINT8_TO_INT64: Uint8 转 Int64 类型转换
 *
 * 7. 复数类型测试
 *    - RUN_POW_CASE_*: 复数类型组合测试
 *
 * --------------------------------------------------------------------------
 * 二、异常测试用例
 * --------------------------------------------------------------------------
 * 1. 输入异常
 *    - INPUT_NULL_EXCEPTION: 空指针输入异常
 *      测试场景: 输入张量指针为 nullptr 时应返回参数错误
 *    - INPUT_DTYPE_EXCEPTION: 数据类型不支持异常
 *      测试场景: 不支持的数据类型组合 (如 BOOL) 应返回参数错误
 *
 * 2. 形状异常
 *    - SHAPE_MISMATCH_EXCEPTION: 形状不匹配异常
 *      测试场景: 无法广播的不兼容形状应返回参数错误
 *
 * 3. 数值异常
 *    - NEGATIVE_EXPONENT_EXCEPTION: 负指数异常
 *      测试场景: 整数类型的负指数运算应返回参数错误
 *    - ZERO_NEGATIVE_POWER_EXCEPTION: 零的负次幂异常
 *      测试场景: 0 的负数次幂应返回参数错误 (避免除零)
 *
 * ============================================================================
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
#include <iostream>


using namespace std;
using namespace aicpu;

class TEST_POW_UT : public testing::Test { };

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Pow", "Pow")                     \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_POW_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 3}, {3, 3}, {3, 3}};
  float input0[9] = {1.0f, 2.0f, 6.7f, 3.0f, 4.0f, 7.0f, 2.0f, 4.0f, 1.0f};
  float input1[9] = {2.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 2.0f};
  float output[9] = {0};
  float expect[9] = {1.0f, 4.0f, 300.763f, 3.0f, 4.0f, 7.0f, 2.0f, 1.0f, 1.0f};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (int i = 0; i < 9; ++i) {
    float absolute_error = std::fabs(output[i] - expect[i]);
    float relative_error = (expect[i] == 0) ? absolute_error : absolute_error / std::fabs(expect[i]);
    if ((absolute_error > 0.01f) && (relative_error > 0.01f)) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_FLOAT16_BROADCAST_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 3}, {1, 3}, {3, 3}};
  Eigen::half input0[9] = {Eigen::half(1.0f), Eigen::half(2.0f), Eigen::half(3.0f),
                           Eigen::half(3.0f), Eigen::half(4.0f), Eigen::half(2.0f),
                           Eigen::half(2.0f), Eigen::half(4.0f), Eigen::half(1.0f)};
  Eigen::half input1[3] = {Eigen::half(2.0f), Eigen::half(2.0f), Eigen::half(2.0f)};
  Eigen::half output[9] = {Eigen::half(0.0f)};
  Eigen::half expect[9] = {Eigen::half(1.0f), Eigen::half(4.0f), Eigen::half(9.0f),
                           Eigen::half(9.0f), Eigen::half(16.0f), Eigen::half(4.0f),
                           Eigen::half(4.0f), Eigen::half(16.0f), Eigen::half(1.0f)};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (int i = 0; i < 9; ++i) {
    float out_val = static_cast<float>(output[i]);
    float exp_val = static_cast<float>(expect[i]);
    float absolute_error = std::fabs(out_val - exp_val);
    float relative_error = (exp_val == 0) ? absolute_error : absolute_error / std::fabs(exp_val);
    if ((absolute_error > 0.1f) && (relative_error > 0.01f)) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_INT32_BROADCAST_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {1, 3}, {3, 3}};
  int32_t input0[9] = {1, 2, 3, 4, 5, 6, 2, 4, 1};
  int32_t input1[3] = {2, 2, 2};
  int32_t output[9] = {0};
  int32_t expect[9] = {1, 4, 9, 16, 25, 36, 4, 16, 1};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output, expect, 9);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 3}, {3, 3}, {3, 3}};
  int64_t input0[9] = {1, 2, 7, 3, 4, 7, 2, 48, 1};
  int64_t input1[9] = {2, 2, 3, 8, 8, 5, 5, 2, 2};
  int64_t output[9] = {0};
  int64_t expect[9] = {1, 4, 343, 6561, 65536, 16807, 32, 2304, 1};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output, expect, 9);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
  int8_t input0[6] = {1, 2, 1, 3, 2, 2};
  int8_t input1[6] = {2, 2, 3, 1, 2, 2};
  int8_t output[6] = {0};
  int8_t expect[6] = {1, 4, 1, 3, 4, 4};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output, expect, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
  double input0[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 2.0};
  double input1[6] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  double output[6] = {0};
  double expect[6] = {1.0, 4.0, 9.0, 16.0, 25.0, 4.0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output, expect, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};
  std::complex<float> input0[4] = {std::complex<float>(2.0f, 0.0f), std::complex<float>(3.0f, 0.0f),
                                   std::complex<float>(2.0f, 0.0f), std::complex<float>(3.0f, 0.0f)};
  std::complex<float> input1[4] = {std::complex<float>(2.0f, 0.0f), std::complex<float>(2.0f, 0.0f),
                                    std::complex<float>(2.0f, 0.0f), std::complex<float>(2.0f, 0.0f)};
  std::complex<float> output[4] = {std::complex<float>(0.0f, 0.0f)};
  std::complex<float> expect[4] = {std::complex<float>(4.0f, 0.0f), std::complex<float>(9.0f, 0.0f),
                                    std::complex<float>(4.0f, 0.0f), std::complex<float>(9.0f, 0.0f)};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (int i = 0; i < 4; ++i) {
    float absolute_error = std::abs(output[i] - expect[i]);
    if (absolute_error > 0.001f) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};
  std::complex<double> input0[4] = {std::complex<double>(2.0, 0.0), std::complex<double>(3.0, 0.0),
                                     std::complex<double>(2.0, 0.0), std::complex<double>(3.0, 0.0)};
  std::complex<double> input1[4] = {std::complex<double>(2.0, 0.0), std::complex<double>(2.0, 0.0),
                                     std::complex<double>(2.0, 0.0), std::complex<double>(2.0, 0.0)};
  std::complex<double> output[4] = {std::complex<double>(0.0, 0.0)};
  std::complex<double> expect[4] = {std::complex<double>(4.0, 0.0), std::complex<double>(9.0, 0.0),
                                     std::complex<double>(4.0, 0.0), std::complex<double>(9.0, 0.0)};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (int i = 0; i < 4; ++i) {
    double absolute_error = std::abs(output[i] - expect[i]);
    if (absolute_error > 0.001) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_ONE_X_ELEMENT_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 2}, {}, {4, 2}};
  int32_t input0[8] = {1, 2, 2, 1, 2, 1, 2, 1};
  int32_t input1[1] = {3};
  int32_t output[8] = {0};
  int32_t expect[8] = {1, 8, 8, 1, 8, 1, 8, 1};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output, expect, 8);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_ONE_Y_ELEMENT_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{}, {2, 3}, {2, 3}};
  int32_t input0[1] = {2};
  int32_t input1[6] = {1, 2, 7, 2, 8, 1};
  int32_t output[6] = {0};
  int32_t expect[6] = {2, 4, 128, 4, 256, 2};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output, expect, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_NOBROADCAST_LARGE_35x1024) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{35, 1024}, {35, 1024}, {35, 1024}};
  int32_t input0[35840];
  int32_t input1[35840];
  int32_t output[35840];
  for (size_t i = 0; i < 35840; ++i) {
    input0[i] = (i % 10) + 1;
    input1[i] = (i % 3) + 1;
  }
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (size_t i = 0; i < 35840; ++i) {
    int32_t expected = 1;
    for (int j = 0; j < input1[i]; ++j) {
      expected *= input0[i];
    }
    if (output[i] != expected) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_NOBROADCAST_LARGE_141x256) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{141, 256}, {141, 256}, {141, 256}};
  int32_t input0[36096];
  int32_t input1[36096];
  int32_t output[36096];
  for (size_t i = 0; i < 36096; ++i) {
    input0[i] = (i % 5) + 1;
    input1[i] = (i % 2) + 1;
  }
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (size_t i = 0; i < 36096; ++i) {
    int32_t expected = 1;
    for (int j = 0; j < input1[i]; ++j) {
      expected *= input0[i];
    }
    if (output[i] != expected) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_BROADCAST_LARGE_30x1024) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{30, 1024}, {1024}, {30, 1024}};
  int32_t input0[30720];
  int32_t input1[1024];
  int32_t output[30720];
  for (size_t i = 0; i < 30720; ++i) {
    input0[i] = (i % 10) + 1;
  }
  for (size_t i = 0; i < 1024; ++i) {
    input1[i] = (i % 3) + 1;
  }
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (size_t i = 0; i < 30720; ++i) {
    int32_t expected = 1;
    int32_t y = input1[i % 1024];
    for (int j = 0; j < y; ++j) {
      expected *= input0[i];
    }
    if (output[i] != expected) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_FLOAT16_LARGE_100) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{100}, {100}, {100}};
  Eigen::half input0[100];
  Eigen::half input1[100];
  Eigen::half output[100];
  for (size_t i = 0; i < 100; ++i) {
    input0[i] = Eigen::half(static_cast<float>((i % 5) + 1));
    input1[i] = Eigen::half(static_cast<float>((i % 3) + 1));
  }
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (size_t i = 0; i < 100; ++i) {
    float expected = powf(static_cast<float>(input0[i]), static_cast<float>(input1[i]));
    float actual = static_cast<float>(output[i]);
    if (std::fabs(expected - actual) > 0.01f) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_BROADCAST_FLOAT_100x24) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{100, 24}, {24}, {100, 24}};
  float input0[2400];
  float input1[24];
  float output[2400];
  for (size_t i = 0; i < 2400; ++i) {
    input0[i] = static_cast<float>((i % 5) + 1);
  }
  for (size_t i = 0; i < 24; ++i) {
    input1[i] = static_cast<float>((i % 3) + 1);
  }
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (size_t i = 0; i < 2400; ++i) {
    float expected = powf(input0[i], input1[i % 24]);
    if (std::fabs(output[i] - expected) > 0.01f) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_DOUBLE_LARGE_100) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{100}, {100}, {100}};
  double input0[100];
  double input1[100];
  double output[100];
  for (size_t i = 0; i < 100; ++i) {
    input0[i] = static_cast<double>((i % 5) + 1);
    input1[i] = static_cast<double>((i % 3) + 1);
  }
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (size_t i = 0; i < 100; ++i) {
    double expected = pow(input0[i], input1[i]);
    if (std::fabs(output[i] - expected) > 0.0001) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_INT32_LARGE_100) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{100}, {100}, {100}};
  int32_t input0[100];
  int32_t input1[100];
  int32_t output[100];
  for (size_t i = 0; i < 100; ++i) {
    input0[i] = (i % 5) + 1;
    input1[i] = (i % 3) + 1;
  }
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (size_t i = 0; i < 100; ++i) {
    int32_t expected = 1;
    for (int j = 0; j < input1[i]; ++j) {
      expected *= input0[i];
    }
    if (output[i] != expected) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, DATA_TYPE_UINT8_LARGE_10x10x100) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{10, 10, 100}, {10, 10, 100}, {10, 10, 100}};
  uint8_t input0[10000];
  uint8_t input1[10000];
  uint8_t output[10000];
  for (size_t i = 0; i < 10000; ++i) {
    input0[i] = (i % 4) + 1;
    input1[i] = (i % 2) + 1;
  }
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool result = true;
  for (size_t i = 0; i < 10000; ++i) {
    uint8_t expected = 1;
    for (int j = 0; j < input1[i]; ++j) {
      expected *= input0[i];
    }
    if (output[i] != expected) {
      result = false;
      break;
    }
  }
  EXPECT_EQ(result, true);
}

TEST_F(TEST_POW_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
 vector<vector<int64_t>> shapes = {{4,4},{4,4},{4,4}};
 int8_t input0[16]={(int32_t)0};
 int8_t input1[16]={(int32_t)0};
 int8_t output[16]={(int32_t)0};
 vector<void *> datas = {(void *)nullptr,(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_POW_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{1,2},{2,2},{2,2}};
  bool input0[2]={0,0};
  bool input1[4]={0,1,0,1};
  bool output[4]={0,1,0,1};
  vector<void *> datas = {(void *)input0,(void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_POW_UT, OUTPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4,4},{4,4},{4,4}};
  int32_t input0[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  int32_t input1[16]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
  vector<void *> datas = {(void *)input0,(void *)input1, (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_POW_UT, NEGATIVE_EXPONENT_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3},{3},{3}};
  int32_t input0[3]={2,3,4};
  int32_t input1[3]={-1,-2,-3};
  int32_t output[3]={0};
  vector<void *> datas = {(void *)input0,(void *)input1,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_POW_UT, ZERO_NEGATIVE_POWER_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3},{3},{3}};
  float input0[3]={0.0f,0.0f,0.0f};
  float input1[3]={-1.0f,-2.0f,-3.0f};
  float output[3]={0};
  vector<void *> datas = {(void *)input0,(void *)input1,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_POW_UT, INPUT_X_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4,4},{4,4},{4,4}};
  int32_t input1[16]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
  int32_t output[16]={0};
  vector<void *> datas = {(void *)nullptr,(void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_POW_UT, INPUT_Y_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4,4},{4,4},{4,4}};
  int32_t input0[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  int32_t output[16]={0};
  vector<void *> datas = {(void *)input0,(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_POW_UT, EMPTY_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{0},{1},{1}};
  int32_t input1[1]={1};
  int32_t output[1]={0};
  vector<void *> datas = {(void *)nullptr,(void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_POW_UT, FLOAT_SAME_SHAPE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}};
  float input0[3] = {2.0f, 3.0f, 4.0f};
  float input1[3] = {2.0f, 2.0f, 2.0f};
  float output[3] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float expect[3] = {4.0f, 9.0f, 16.0f};
  bool compare = CompareResult(output, expect, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, FLOAT16_X_ONE_ELEMENT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{}, {3}, {3}};
  Eigen::half input0[1] = {Eigen::half(2.0f)};
  Eigen::half input1[3] = {Eigen::half(2.0f), Eigen::half(3.0f), Eigen::half(4.0f)};
  Eigen::half output[3] = {Eigen::half(0.0f)};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  Eigen::half expect[3] = {Eigen::half(4.0f), Eigen::half(8.0f), Eigen::half(16.0f)};
  bool compare = CompareResult(output, expect, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, INT32_Y_ONE_ELEMENT_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {}, {3}};
  int32_t input0[3] = {2, 3, 4};
  int32_t input1[1] = {2};
  int32_t output[3] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t expect[3] = {4, 9, 16};
  bool compare = CompareResult(output, expect, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, INT32_POWER_0_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}};
  int32_t input0[3] = {2, 3, 4};
  int32_t input1[3] = {0, 0, 0};
  int32_t output[3] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t expect[3] = {1, 1, 1};
  bool compare = CompareResult(output, expect, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, INT32_POWER_1_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}};
  int32_t input0[3] = {2, 3, 4};
  int32_t input1[3] = {1, 1, 1};
  int32_t output[3] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t expect[3] = {2, 3, 4};
  bool compare = CompareResult(output, expect, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, INT32_POWER_NEGATIVE_1_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}};
  int32_t input0[3] = {-1, -1, -1};
  int32_t input1[3] = {-2, -3, -4};
  int32_t output[3] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t expect[3] = {1, -1, 1};
  bool compare = CompareResult(output, expect, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, INT32_POWER_NEGATIVE_OTHER_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}};
  int32_t input0[3] = {2, 3, 4};
  int32_t input1[3] = {-2, -3, -4};
  int32_t output[3] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t expect[3] = {0, 0, 0};
  bool compare = CompareResult(output, expect, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_POW_UT, EMPTY_TENSOR_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{0}, {0}, {0}};
  float input0[1] = {0};
  float input1[1] = {0};
  float output[1] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

#define RUN_POW_CASE(base_type_in1, base_type_in2, base_type_out,                            \
                     aicpu_type_in1, aicpu_type_in2, aicpu_type_out)                         \
  TEST_F(TEST_POW_UT, TestCast_##aicpu_type_in1##aicpu_type_in2####aicpu_type_out) {         \
      vector<DataType> data_types = {aicpu_type_in1, aicpu_type_in2, aicpu_type_out};        \
      base_type_in1 input1[3] = {(base_type_in1)2, (base_type_in1)3, (base_type_in1)4};      \
      base_type_in2 input2[1] = {(base_type_in2)2};                                          \
      base_type_out output[3] = {};                                                          \
      vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};               \
      vector<vector<int64_t>> shapes = {{3}, {}, {3}};                                       \
      CREATE_NODEDEF(shapes, data_types, datas);                                             \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                          \
      base_type_out expect_out[3] = {(base_type_out)4, (base_type_out)9, (base_type_out)16}; \
      EXPECT_EQ(CompareResult<base_type_out>(output, expect_out, 3), true);                  \
  }

RUN_POW_CASE(int8_t, int8_t, int8_t, DT_INT8, DT_INT8, DT_INT8);
RUN_POW_CASE(int8_t, int16_t, int8_t, DT_INT8, DT_INT16, DT_INT8);
RUN_POW_CASE(int8_t, int16_t, int16_t, DT_INT8, DT_INT16, DT_INT16);
RUN_POW_CASE(int8_t, int32_t, int8_t, DT_INT8, DT_INT32, DT_INT8);
RUN_POW_CASE(int8_t, int32_t, int32_t, DT_INT8, DT_INT32, DT_INT32);
RUN_POW_CASE(int8_t, int64_t, int8_t, DT_INT8, DT_INT64, DT_INT8);
RUN_POW_CASE(int8_t, int64_t, int64_t, DT_INT8, DT_INT64, DT_INT64);
RUN_POW_CASE(int8_t, uint8_t, int8_t, DT_INT8, DT_UINT8, DT_INT8);
RUN_POW_CASE(int8_t, uint8_t, int16_t, DT_INT8, DT_UINT8, DT_INT16);
RUN_POW_CASE(int8_t, Eigen::half, Eigen::half, DT_INT8, DT_FLOAT16, DT_FLOAT16);
RUN_POW_CASE(int8_t, float, float, DT_INT8, DT_FLOAT, DT_FLOAT);
RUN_POW_CASE(int8_t, double, double, DT_INT8, DT_DOUBLE, DT_DOUBLE);
RUN_POW_CASE(int8_t, std::complex<float>, std::complex<float>, DT_INT8, DT_COMPLEX64, DT_COMPLEX64);
RUN_POW_CASE(int8_t, std::complex<double>, std::complex<double>, DT_INT8, DT_COMPLEX128, DT_COMPLEX128);

RUN_POW_CASE(int16_t, int8_t, int16_t, DT_INT16, DT_INT8, DT_INT16);
RUN_POW_CASE(int16_t, int16_t, int16_t, DT_INT16, DT_INT16, DT_INT16);
RUN_POW_CASE(int16_t, int32_t, int16_t, DT_INT16, DT_INT32, DT_INT16);
RUN_POW_CASE(int16_t, int32_t, int32_t, DT_INT16, DT_INT32, DT_INT32);
RUN_POW_CASE(int16_t, int64_t, int16_t, DT_INT16, DT_INT64, DT_INT16);
RUN_POW_CASE(int16_t, int64_t, int64_t, DT_INT16, DT_INT64, DT_INT64);
RUN_POW_CASE(int16_t, uint8_t, int16_t, DT_INT16, DT_UINT8, DT_INT16);
RUN_POW_CASE(int16_t, Eigen::half, Eigen::half, DT_INT16, DT_FLOAT16, DT_FLOAT16);
RUN_POW_CASE(int16_t, Eigen::half, int16_t, DT_INT16, DT_FLOAT16, DT_INT16);
RUN_POW_CASE(int16_t, float, float, DT_INT16, DT_FLOAT, DT_FLOAT);
RUN_POW_CASE(int16_t, float, int16_t, DT_INT16, DT_FLOAT, DT_INT16);
RUN_POW_CASE(int16_t, double, double, DT_INT16, DT_DOUBLE, DT_DOUBLE);
RUN_POW_CASE(int16_t, double, int16_t, DT_INT16, DT_DOUBLE, DT_INT16);
RUN_POW_CASE(int16_t, std::complex<float>, std::complex<float>, DT_INT16, DT_COMPLEX64, DT_COMPLEX64);
RUN_POW_CASE(int16_t, std::complex<float>, int16_t, DT_INT16, DT_COMPLEX64, DT_INT16);
RUN_POW_CASE(int16_t, std::complex<double>, std::complex<double>, DT_INT16, DT_COMPLEX128, DT_COMPLEX128);
RUN_POW_CASE(int16_t, std::complex<double>, int16_t, DT_INT16, DT_COMPLEX128, DT_INT16);

RUN_POW_CASE(int32_t, int8_t, int32_t, DT_INT32, DT_INT8, DT_INT32);
RUN_POW_CASE(int32_t, int16_t, int32_t, DT_INT32, DT_INT16, DT_INT32);
RUN_POW_CASE(int32_t, int32_t, int32_t, DT_INT32, DT_INT32, DT_INT32);
RUN_POW_CASE(int32_t, int64_t, int32_t, DT_INT32, DT_INT64, DT_INT32);
RUN_POW_CASE(int32_t, int64_t, int64_t, DT_INT32, DT_INT64, DT_INT64);
RUN_POW_CASE(int32_t, uint8_t, int32_t, DT_INT32, DT_UINT8, DT_INT32);
RUN_POW_CASE(int32_t, Eigen::half, Eigen::half, DT_INT32, DT_FLOAT16, DT_FLOAT16);
RUN_POW_CASE(int32_t, float, float, DT_INT32, DT_FLOAT, DT_FLOAT);
RUN_POW_CASE(int32_t, double, double, DT_INT32, DT_DOUBLE, DT_DOUBLE);
RUN_POW_CASE(int32_t, std::complex<float>, std::complex<float>, DT_INT32, DT_COMPLEX64, DT_COMPLEX64);
RUN_POW_CASE(int32_t, std::complex<double>, std::complex<double>, DT_INT32, DT_COMPLEX128, DT_COMPLEX128);

RUN_POW_CASE(int64_t, int8_t, int64_t, DT_INT64, DT_INT8, DT_INT64);
RUN_POW_CASE(int64_t, int16_t, int64_t, DT_INT64, DT_INT16, DT_INT64);
RUN_POW_CASE(int64_t, int32_t, int64_t, DT_INT64, DT_INT32, DT_INT64);
RUN_POW_CASE(int64_t, int64_t, int64_t, DT_INT64, DT_INT64, DT_INT64);
RUN_POW_CASE(int64_t, uint8_t, int64_t, DT_INT64, DT_UINT8, DT_INT64);
RUN_POW_CASE(int64_t, Eigen::half, Eigen::half, DT_INT64, DT_FLOAT16, DT_FLOAT16);
RUN_POW_CASE(int64_t, float, float, DT_INT64, DT_FLOAT, DT_FLOAT);
RUN_POW_CASE(int64_t, double, double, DT_INT64, DT_DOUBLE, DT_DOUBLE);
RUN_POW_CASE(int64_t, std::complex<float>, std::complex<float>, DT_INT64, DT_COMPLEX64, DT_COMPLEX64);
RUN_POW_CASE(int64_t, std::complex<double>, std::complex<double>, DT_INT64, DT_COMPLEX128, DT_COMPLEX128);

RUN_POW_CASE(uint8_t, int8_t, uint8_t, DT_UINT8, DT_INT8, DT_UINT8);
RUN_POW_CASE(uint8_t, int8_t, int16_t, DT_UINT8, DT_INT8, DT_INT16);
RUN_POW_CASE(uint8_t, int16_t, uint8_t, DT_UINT8, DT_INT16, DT_UINT8);
RUN_POW_CASE(uint8_t, int16_t, int16_t, DT_UINT8, DT_INT16, DT_INT16);
RUN_POW_CASE(uint8_t, int32_t, uint8_t, DT_UINT8, DT_INT32, DT_UINT8);
RUN_POW_CASE(uint8_t, int32_t, int32_t, DT_UINT8, DT_INT32, DT_INT32);
RUN_POW_CASE(uint8_t, int64_t, uint8_t, DT_UINT8, DT_INT64, DT_UINT8);
RUN_POW_CASE(uint8_t, int64_t, int64_t, DT_UINT8, DT_INT64, DT_INT64);
RUN_POW_CASE(uint8_t, uint8_t, uint8_t, DT_UINT8, DT_UINT8, DT_UINT8);
RUN_POW_CASE(uint8_t, Eigen::half, Eigen::half, DT_UINT8, DT_FLOAT16, DT_FLOAT16);
RUN_POW_CASE(uint8_t, float, float, DT_UINT8, DT_FLOAT, DT_FLOAT);
RUN_POW_CASE(uint8_t, double, double, DT_UINT8, DT_DOUBLE, DT_DOUBLE);
RUN_POW_CASE(uint8_t, std::complex<float>, std::complex<float>, DT_UINT8, DT_COMPLEX64, DT_COMPLEX64);
RUN_POW_CASE(uint8_t, std::complex<double>, std::complex<double>, DT_UINT8, DT_COMPLEX128, DT_COMPLEX128);

RUN_POW_CASE(Eigen::half, int8_t, Eigen::half, DT_FLOAT16, DT_INT8, DT_FLOAT16);
RUN_POW_CASE(Eigen::half, int16_t, Eigen::half, DT_FLOAT16, DT_INT16, DT_FLOAT16);
RUN_POW_CASE(Eigen::half, int32_t, Eigen::half, DT_FLOAT16, DT_INT32, DT_FLOAT16);
RUN_POW_CASE(Eigen::half, int64_t, Eigen::half, DT_FLOAT16, DT_INT64, DT_FLOAT16);
RUN_POW_CASE(Eigen::half, uint8_t, Eigen::half, DT_FLOAT16, DT_UINT8, DT_FLOAT16);
RUN_POW_CASE(Eigen::half, Eigen::half, Eigen::half, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16);
RUN_POW_CASE(Eigen::half, float, Eigen::half, DT_FLOAT16, DT_FLOAT, DT_FLOAT16);
RUN_POW_CASE(Eigen::half, float, float, DT_FLOAT16, DT_FLOAT, DT_FLOAT);
RUN_POW_CASE(Eigen::half, double, Eigen::half, DT_FLOAT16, DT_DOUBLE, DT_FLOAT16);
RUN_POW_CASE(Eigen::half, double, double, DT_FLOAT16, DT_DOUBLE, DT_DOUBLE);
RUN_POW_CASE(Eigen::half, std::complex<float>, std::complex<float>, DT_FLOAT16, DT_COMPLEX64, DT_COMPLEX64);
RUN_POW_CASE(Eigen::half, std::complex<double>, std::complex<double>, DT_FLOAT16, DT_COMPLEX128, DT_COMPLEX128);

RUN_POW_CASE(float, int8_t, float, DT_FLOAT, DT_INT8, DT_FLOAT);
RUN_POW_CASE(float, int16_t, float, DT_FLOAT, DT_INT16, DT_FLOAT);
RUN_POW_CASE(float, int32_t, float, DT_FLOAT, DT_INT32, DT_FLOAT);
RUN_POW_CASE(float, int64_t, float, DT_FLOAT, DT_INT64, DT_FLOAT);
RUN_POW_CASE(float, uint8_t, float, DT_FLOAT, DT_UINT8, DT_FLOAT);
RUN_POW_CASE(float, Eigen::half, float, DT_FLOAT, DT_FLOAT16, DT_FLOAT);
RUN_POW_CASE(float, float, float, DT_FLOAT, DT_FLOAT, DT_FLOAT);
RUN_POW_CASE(float, double, float, DT_FLOAT, DT_DOUBLE, DT_FLOAT);
RUN_POW_CASE(float, double, double, DT_FLOAT, DT_DOUBLE, DT_DOUBLE);
RUN_POW_CASE(float, std::complex<float>, std::complex<float>, DT_FLOAT, DT_COMPLEX64, DT_COMPLEX64);
RUN_POW_CASE(float, std::complex<double>, std::complex<double>, DT_FLOAT, DT_COMPLEX128, DT_COMPLEX128);

RUN_POW_CASE(double, int8_t, double, DT_DOUBLE, DT_INT8, DT_DOUBLE);
RUN_POW_CASE(double, int16_t, double, DT_DOUBLE, DT_INT16, DT_DOUBLE);
RUN_POW_CASE(double, int32_t, double, DT_DOUBLE, DT_INT32, DT_DOUBLE);
RUN_POW_CASE(double, int64_t, double, DT_DOUBLE, DT_INT64, DT_DOUBLE);
RUN_POW_CASE(double, uint8_t, double, DT_DOUBLE, DT_UINT8, DT_DOUBLE);
RUN_POW_CASE(double, Eigen::half, double, DT_DOUBLE, DT_FLOAT16, DT_DOUBLE);
RUN_POW_CASE(double, float, double, DT_DOUBLE, DT_FLOAT, DT_DOUBLE);
RUN_POW_CASE(double, double, double, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE);
RUN_POW_CASE(double, std::complex<float>, std::complex<float>, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX64);
RUN_POW_CASE(double, std::complex<float>, std::complex<double>, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128);
RUN_POW_CASE(double, std::complex<double>, std::complex<double>, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX128);

RUN_POW_CASE(std::complex<float>, int8_t, std::complex<float>, DT_COMPLEX64, DT_INT8, DT_COMPLEX64);
RUN_POW_CASE(std::complex<float>, int16_t, std::complex<float>, DT_COMPLEX64, DT_INT16, DT_COMPLEX64);
RUN_POW_CASE(std::complex<float>, int32_t, std::complex<float>, DT_COMPLEX64, DT_INT32, DT_COMPLEX64);
RUN_POW_CASE(std::complex<float>, int64_t, std::complex<float>, DT_COMPLEX64, DT_INT64, DT_COMPLEX64);
RUN_POW_CASE(std::complex<float>, uint8_t, std::complex<float>, DT_COMPLEX64, DT_UINT8, DT_COMPLEX64);
RUN_POW_CASE(std::complex<float>, Eigen::half, std::complex<float>, DT_COMPLEX64, DT_FLOAT16, DT_COMPLEX64);
RUN_POW_CASE(std::complex<float>, float, std::complex<float>, DT_COMPLEX64, DT_FLOAT, DT_COMPLEX64);
RUN_POW_CASE(std::complex<float>, double, std::complex<float>, DT_COMPLEX64, DT_DOUBLE, DT_COMPLEX64);
RUN_POW_CASE(std::complex<float>, double, std::complex<double>, DT_COMPLEX64, DT_DOUBLE, DT_COMPLEX128);
RUN_POW_CASE(std::complex<float>, std::complex<float>, std::complex<float>, DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64);
RUN_POW_CASE(std::complex<float>, std::complex<double>, std::complex<float>, DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX64);
RUN_POW_CASE(std::complex<float>, std::complex<double>, std::complex<double>, DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX128);

RUN_POW_CASE(std::complex<double>, int8_t, std::complex<double>, DT_COMPLEX128, DT_INT8, DT_COMPLEX128);
RUN_POW_CASE(std::complex<double>, int16_t, std::complex<double>, DT_COMPLEX128, DT_INT16, DT_COMPLEX128);
RUN_POW_CASE(std::complex<double>, int32_t, std::complex<double>, DT_COMPLEX128, DT_INT32, DT_COMPLEX128);
RUN_POW_CASE(std::complex<double>, int64_t, std::complex<double>, DT_COMPLEX128, DT_INT64, DT_COMPLEX128);
RUN_POW_CASE(std::complex<double>, uint8_t, std::complex<double>, DT_COMPLEX128, DT_UINT8, DT_COMPLEX128);
RUN_POW_CASE(std::complex<double>, Eigen::half, std::complex<double>, DT_COMPLEX128, DT_FLOAT16, DT_COMPLEX128);
RUN_POW_CASE(std::complex<double>, float, std::complex<double>, DT_COMPLEX128, DT_FLOAT, DT_COMPLEX128);
RUN_POW_CASE(std::complex<double>, double, std::complex<double>, DT_COMPLEX128, DT_DOUBLE, DT_COMPLEX128);
RUN_POW_CASE(std::complex<double>, std::complex<float>, std::complex<double>, DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX128);
RUN_POW_CASE(std::complex<double>, std::complex<double>, std::complex<double>, DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128);
