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
#include <math.h>
#include <stdint.h>
#include <bitset>
#include <Eigen/Dense>
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "node_def_builder.h"
#include "cpu_kernel_utils.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;
using namespace Eigen;

namespace {
const char* Test = "Cast";
}

class TEST_CAST_UT : public testing::Test {};

template <typename Tin, typename Tout>
void CalcExpectFunc(const NodeDef& node_def, Tin input_type, Tout expect_out[]) {
  auto input = node_def.MutableInputs(0);
  auto output = node_def.MutableOutputs(0);
  Tin* input_data = (Tin*)input->GetData();
  Tout* output_data = (Tout*)output->GetData();

  int64_t input_num = input->NumElements();

  for (int i = 0; i < input_num; i++) {
    expect_out[i] = (Tout)input_data[i];
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Cast", "Cast")                   \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})           \
      .Attr("SrcT", data_types[0])                                 \
      .Attr("DstT", data_types[1]);

#define CAST_CASE_WITH_TYPE(base_type_in, aicpu_type_in, base_type_out, aicpu_type_out, is_empty) \
  TEST_F(TEST_CAST_UT, TestCast_##aicpu_type_in##_To_##aicpu_type_out) {                          \
    if (!is_empty) {                                                                              \
      vector<DataType> data_types = {aicpu_type_in, aicpu_type_out};                              \
      base_type_in input[6] = {(base_type_in)22,    (base_type_in)32, (base_type_in)-78,           \
                                (base_type_in)-28, (base_type_in)77,   (base_type_in)0};          \
      base_type_out output[6] = {(base_type_out)0};                                               \
      vector<void*> datas = {(void*)input, (void*)output};                                        \
      vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};                                          \
      CREATE_NODEDEF(shapes, data_types, datas);                                                  \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                               \
      base_type_out expect_out[6] = {(base_type_out)0};                                           \
      base_type_in input_type = (base_type_in)0;                                                  \
      CalcExpectFunc(*node_def.get(), input_type, expect_out);                                    \
      CompareResult<base_type_out>(output, expect_out, 6);                                        \
    } else {                                                                                      \
      vector<void*> datas = {nullptr, nullptr};                                                   \
      vector<vector<int64_t>> shapes = {{}, {}};                                                  \
      vector<DataType> data_types = {aicpu_type_in, aicpu_type_out};                              \
      CREATE_NODEDEF(shapes, data_types, datas);                                                  \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                               \
    }                                                                                             \
  }

CAST_CASE_WITH_TYPE(float, DT_FLOAT, int8_t, DT_INT8, false)
CAST_CASE_WITH_TYPE(float, DT_FLOAT, int16_t, DT_INT16, false)
CAST_CASE_WITH_TYPE(float, DT_FLOAT, int32_t, DT_INT32, false)
CAST_CASE_WITH_TYPE(float, DT_FLOAT, int64_t, DT_INT64, false)
CAST_CASE_WITH_TYPE(double, DT_DOUBLE, int8_t, DT_INT8, false)
CAST_CASE_WITH_TYPE(double, DT_DOUBLE, int16_t, DT_INT16, false)
CAST_CASE_WITH_TYPE(double, DT_DOUBLE, int32_t, DT_INT32, true)
CAST_CASE_WITH_TYPE(double, DT_DOUBLE, int64_t, DT_INT64, false)
CAST_CASE_WITH_TYPE(Eigen::half, DT_FLOAT16, uint8_t, DT_UINT8, false)
CAST_CASE_WITH_TYPE(float, DT_FLOAT, uint8_t, DT_UINT8, false)
CAST_CASE_WITH_TYPE(double, DT_DOUBLE, uint8_t, DT_UINT8, false)

#define CAST_CASE_WITH_TYPE_COMPLEX(base_type_in, aicpu_type_in, base_type_out, aicpu_type_out) \
  TEST_F(TEST_CAST_UT, TestCast_##aicpu_type_in##_To_##aicpu_type_out) {                        \
    vector<DataType> data_types = {aicpu_type_in, aicpu_type_out};                              \
    base_type_in input[6] = {(base_type_in)22,    (base_type_in)32, (base_type_in)-78,          \
                             (base_type_in)-28, (base_type_in)77,   (base_type_in)0};           \
    base_type_out output[6] = {};                                                               \
    vector<void*> datas = {(void*)input, (void*)output};                                        \
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};                                          \
    CREATE_NODEDEF(shapes, data_types, datas);                                                  \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                               \
    base_type_out expect_out[6] = {{22, 0}, {32, 0}, {-78, 0}, {-28, 0}, {77, 0}, {0, 0}};      \
    base_type_in input_type = (base_type_in)0;                                                  \
    CalcExpectFunc(*node_def.get(), input_type, expect_out);                                    \
    CompareResult<base_type_out>(output, expect_out, 6);                                        \
  }

CAST_CASE_WITH_TYPE_COMPLEX(int8_t, DT_INT8, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(int16_t, DT_INT16, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(int32_t, DT_INT32, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(int64_t, DT_INT64, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(uint8_t, DT_UINT8, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(uint16_t, DT_UINT16, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(uint32_t, DT_UINT32, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(uint64_t, DT_UINT64, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(float, DT_FLOAT, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(double, DT_DOUBLE, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(bool, DT_BOOL, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_COMPLEX(int8_t, DT_INT8, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX(int16_t, DT_INT16, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX(int32_t, DT_INT32, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX(int64_t, DT_INT64, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX(uint8_t, DT_UINT8, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX(uint16_t, DT_UINT16, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX(uint32_t, DT_UINT32, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX(uint64_t, DT_UINT64, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX(float, DT_FLOAT, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX(double, DT_DOUBLE, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX(bool, DT_BOOL, std::complex<float>, DT_COMPLEX64)

#define CAST_CASE_WITH_TYPE_COMPLEX_TO_COMPLEX(base_type_in, aicpu_type_in, base_type_out, aicpu_type_out) \
  TEST_F(TEST_CAST_UT, TestCast_##aicpu_type_in##_To_##aicpu_type_out) {                                   \
    vector<DataType> data_types = {aicpu_type_in, aicpu_type_out};                                         \
    base_type_in input[6] = {{22, 0}, {32, 0}, {-78, 0}, {-28, 0}, {77, 0}, {0, 0}};                       \
    base_type_out output[6] = {};                                                                          \
    vector<void*> datas = {(void*)input, (void*)output};                                                   \
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};                                                     \
    CREATE_NODEDEF(shapes, data_types, datas);                                                             \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                          \
    base_type_out expect_out[6] = {{22, 0}, {32, 0}, {-78, 0}, {-28, 0}, {77, 0}, {0, 0}};                 \
    base_type_in input_type = (base_type_in)0;                                                             \
    CalcExpectFunc(*node_def.get(), input_type, expect_out);                                               \
    CompareResult<base_type_out>(output, expect_out, 6);                                                   \
  }

CAST_CASE_WITH_TYPE_COMPLEX_TO_COMPLEX(std::complex<double>, DT_COMPLEX128, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_COMPLEX_TO_COMPLEX(std::complex<float>, DT_COMPLEX64, std::complex<double>, DT_COMPLEX128)

template <typename Tin, typename Tout>
void CalcFp16ToComplexExpectFunc(const NodeDef& node_def, Tin input_type, Tout expect_out[]) {
  auto input = node_def.MutableInputs(0);
  auto output = node_def.MutableOutputs(0);
  Tin* input_data = (Tin*)input->GetData();
  Tout* output_data = (Tout*)output->GetData();

  int64_t input_num = input->NumElements();

  for (int i = 0; i < input_num; i++) {
    expect_out[i] = (Tout)(float)input_data[i];
  }
}

#define CAST_CASE_WITH_TYPE_FP16_TO_COMPLEX(base_type_in, aicpu_type_in, base_type_out, aicpu_type_out) \
  TEST_F(TEST_CAST_UT, TestCast_##aicpu_type_in##_To_##aicpu_type_out) { \
    vector<DataType> data_types = {aicpu_type_in, aicpu_type_out}; \
    base_type_in input[6] = {(base_type_in)22, (base_type_in)32.3, (base_type_in)-78, \
                             (base_type_in)-28.5, (base_type_in)77, (base_type_in)0}; \
    base_type_out output[6] = {}; \
    vector<void*> datas = {(void*)input, (void*)output}; \
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}}; \
    CREATE_NODEDEF(shapes, data_types, datas); \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK); \
    base_type_out expect_out[6] = {{22, 0}, {32, 0}, {-78, 0}, {-28, 0}, {77, 0}, {0, 0}}; \
    base_type_in input_type = (base_type_in)0; \
    CalcFp16ToComplexExpectFunc(*node_def.get(), input_type, expect_out); \
    CompareResult<base_type_out>(output, expect_out, 6); \
  }

CAST_CASE_WITH_TYPE_FP16_TO_COMPLEX(Eigen::half, DT_FLOAT16, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_FP16_TO_COMPLEX(Eigen::half, DT_FLOAT16, std::complex<double>, DT_COMPLEX128)

TEST_F(TEST_CAST_UT, TestCast_DT_INT64_To_DT_FLOAT_MIN) {
  vector<DataType> data_types = {DT_INT64, DT_FLOAT};
  int64_t input[1] = {INT64_MIN};
  float output[1] = {0.0f};
  vector<void*> datas = {(void*)input, (void*)output};
  vector<vector<int64_t>> shapes = {{1}, {1}};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float expect_out[1] = {static_cast<float>(INT64_MIN)};
  CompareResult<float>(output, expect_out, 1);
}

#define CAST_CASE_WITH_TYPE_BF16(base_type_in, aicpu_type_in, base_type_out, aicpu_type_out) \
  TEST_F(TEST_CAST_UT, TestCast_##aicpu_type_in##_To_##aicpu_type_out) {                     \
    vector<DataType> data_types = {aicpu_type_in, aicpu_type_out};                           \
    base_type_in input[6] = {(base_type_in)22, (base_type_in)32, (base_type_in)-78,           \
                             (base_type_in)-28, (base_type_in)77, (base_type_in)0};           \
    base_type_out output[6] = {(base_type_out)0};                                            \
    vector<void*> datas = {(void*)input, (void*)output};                                     \
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};                                       \
    CREATE_NODEDEF(shapes, data_types, datas);                                               \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                            \
    base_type_out expect_out[6] = {(base_type_out)0};                                        \
    base_type_in input_type = (base_type_in)0;                                               \
    CalcExpectFunc(*node_def.get(), input_type, expect_out);                                 \
    CompareResult<base_type_out>(output, expect_out, 6);                                     \
  }

CAST_CASE_WITH_TYPE_BF16(bool, DT_BOOL, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(uint8_t, DT_UINT8, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(uint16_t, DT_UINT16, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(uint32_t, DT_UINT32, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(uint64_t, DT_UINT64, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(int8_t, DT_INT8, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(int16_t, DT_INT16, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(int32_t, DT_INT32, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(int64_t, DT_INT64, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(float, DT_FLOAT, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(double, DT_DOUBLE, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, bool, DT_BOOL)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, uint8_t, DT_UINT8)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, uint16_t, DT_UINT16)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, uint32_t, DT_UINT32)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, uint64_t, DT_UINT64)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, int8_t, DT_INT8)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, int16_t, DT_INT16)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, int32_t, DT_INT32)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, int64_t, DT_INT64)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, float, DT_FLOAT)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, double, DT_DOUBLE)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, Eigen::half, DT_FLOAT16)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, std::complex<float>, DT_COMPLEX64)
CAST_CASE_WITH_TYPE_BF16(Eigen::bfloat16, DT_BFLOAT16, std::complex<double>, DT_COMPLEX128)
CAST_CASE_WITH_TYPE_BF16(std::complex<float>, DT_COMPLEX64, Eigen::bfloat16, DT_BFLOAT16)
CAST_CASE_WITH_TYPE_BF16(std::complex<double>, DT_COMPLEX128, Eigen::bfloat16, DT_BFLOAT16)

bool CompareBinResult(int8_t *output, bitset<8> binary_expect[], int32_t out_size) {
  for (int32_t i = 0; i < out_size; i++) {
    bitset<8> binary(output[i]);
    if (binary.to_string() != binary_expect[i].to_string()) {
      return false;
    }
  }
  return true;
}

TEST_F(TEST_CAST_UT, TestCast_DT_FLOAT_To_DT_HIFLOAT8) {
  vector<DataType> data_types = {DT_FLOAT, DT_HIFLOAT8};
  float input[12] = {(float)pow(2.0, 16), (float)3.1415926, (float)0.125999, (float)9.9e-06, (float)9.9e-08, (float)24,
                     (float)-1.25 * pow(2.0, -15), (float)1.5 * pow(2.0, -15), (float)0, (float)NAN, (float)INFINITY, (float)(-INFINITY)};
  int8_t output[12] = {(int8_t)0};
  vector<void*> datas = {(void*)input, (void*)output};
  vector<vector<int64_t>> shapes = {{1, 12}, {1, 12}};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bitset<8> binary_expect[12] = {0b01101111, 0b00010101, 0b00111000, 0b00000110, 0b00000000, 0b01000010,
                                 0b11111111, 0b01111111, 0b00000000, 0b10000000, 0b01101111, 0b11101111};
  bool status = CompareBinResult(output, binary_expect, 12);
  EXPECT_EQ(status, true);
}

TEST_F(TEST_CAST_UT, TestCast_DT_FLOAT16_To_DT_HIFLOAT8) {
  vector<DataType> data_types = {DT_FLOAT16, DT_HIFLOAT8};
  Eigen::half input[12] = {static_cast<Eigen::half>(pow(2.0, 16)), static_cast<Eigen::half>(3.1415926), static_cast<Eigen::half>(0.125999),
                           static_cast<Eigen::half>(9.9e-06), static_cast<Eigen::half>(9.9e-08), static_cast<Eigen::half>(24),
                           static_cast<Eigen::half>(-1.25 * pow(2.0, -15)), static_cast<Eigen::half>(1.5 * pow(2.0, -15)), static_cast<Eigen::half>(0),
                           static_cast<Eigen::half>(NAN), static_cast<Eigen::half>(INFINITY), static_cast<Eigen::half>(-INFINITY)};
  int8_t output[12] = {(int8_t)0};
  vector<void*> datas = {(void*)input, (void*)output};
  vector<vector<int64_t>> shapes = {{1, 12}, {1, 12}};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bitset<8> binary_expect[12] = {0b01101111, 0b00010101, 0b00111000, 0b00000110, 0b00000001, 0b01000010,
                                 0b11111111, 0b01111111, 0b00000000, 0b10000000, 0b01101111, 0b11101111};
  bool status = CompareBinResult(output, binary_expect, 12);
  EXPECT_EQ(status, true);
}

TEST_F(TEST_CAST_UT, TestCast_DT_BFLOAT16_To_DT_HIFLOAT8) {
  vector<DataType> data_types = {DT_BFLOAT16, DT_HIFLOAT8};
  Eigen::bfloat16 input[12] = {static_cast<Eigen::bfloat16>(pow(2.0, 16)), static_cast<Eigen::bfloat16>(3.1415926), static_cast<Eigen::bfloat16>(0.125999),
                               static_cast<Eigen::bfloat16>(9.9e-06), static_cast<Eigen::bfloat16>(9.9e-08), static_cast<Eigen::bfloat16>(24),
                               static_cast<Eigen::bfloat16>(-1.25 * pow(2.0, -15)), static_cast<Eigen::bfloat16>(1.5 * pow(2.0, -15)), static_cast<Eigen::bfloat16>(0),
                               static_cast<Eigen::bfloat16>(NAN), static_cast<Eigen::bfloat16>(INFINITY), static_cast<Eigen::bfloat16>(-INFINITY)};
  int8_t output[12] = {(int8_t)0};
  vector<void*> datas = {(void*)input, (void*)output};
  vector<vector<int64_t>> shapes = {{1, 12}, {1, 12}};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bitset<8> binary_expect[12] = {0b01101111, 0b00010101, 0b00111000, 0b00000110, 0b00000000, 0b01000010,
                                 0b11111111, 0b01111111, 0b00000000, 0b10000000, 0b01101111, 0b11101111};
  bool status = CompareBinResult(output, binary_expect, 12);
  EXPECT_EQ(status, true);
}