/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <math.h>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_REALDIV_UT : public testing::Test {};

template <typename T>
void CalcExpectFunc(const NodeDef &node_def, T expect_out[]) {
  auto output = node_def.MutableOutputs(0);
  int64_t output_num = output->NumElements();
  for (int i = 0; i < output_num; i++) {
    expect_out[i] = 1;
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                                 \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                \
  NodeDefBuilder(node_def.get(), "RealDiv", "RealDiv")                            \
      .Input({"x1", data_types[0], shapes[0], datas[0]})                          \
      .Input({"x2", data_types[1], shapes[1], datas[1]})                          \
      .Output({"y", data_types[2], shapes[2], datas[2]})

#define ADD_CASE(base_type, aicpu_type)                                           \
  TEST_F(TEST_REALDIV_UT, TestRealDiv_##aicpu_type) {                             \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};           \
    vector<vector<int64_t>> shapes = {{24}, {24}, {24}};                          \
    base_type input_x1[24];                                                       \
    SetRandomValue<base_type>(input_x1, 24);                                      \
    base_type input_x2[24];                                                       \
    SetRandomValue<base_type>(input_x2, 24, 1);                                   \
    base_type output[24] = {(base_type)0};                                        \
    vector<void *> datas = {(void *)input_x1, (void *)input_x2, (void *)output};  \
    CREATE_NODEDEF(shapes, data_types, datas);                                    \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                 \
  }

#define REALDIV_CASE_WITH_SHAPE(case_name, base_type, aicpu_type,                  \
                                shapes, data_num)                                  \
  TEST_F(TEST_REALDIV_UT, TestRealdiv_##case_name) {                               \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};            \
    std::vector<base_type> input1(data_num[0], (base_type)1);                      \
    std::vector<base_type> input2(data_num[0], (base_type)1);                      \
    std::vector<base_type> output(data_num[2], (base_type)0);                      \
    vector<void*> datas = {input1.data(), input2.data(), output.data()};           \
    CREATE_NODEDEF(shapes, data_types, datas);                                     \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                  \
    std::vector<base_type> expect_out(data_num[2], (base_type)0);                  \
    CalcExpectFunc(*node_def.get(), expect_out.data());                            \
    CompareResult<base_type>(output.data(), expect_out.data(), data_num[2]);       \
  }

#define ADD_ZERO_CASE(base_type, aicpu_type)                                      \
  TEST_F(TEST_REALDIV_UT, TestRealDiv_ZeroInput_##aicpu_type) {                   \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};           \
    vector<vector<int64_t>> shapes = {{24}, {24}, {24}};                          \
    base_type input_x1[24];                                                       \
    SetRandomValue<base_type>(input_x1, 24);                                      \
    base_type input_x2[24];                                                       \
    SetRandomValue<base_type>(input_x2, 24, 0, 0);                                \
    base_type output[24] = {(base_type)0};                                        \
    vector<void *> datas = {(void *)input_x1, (void *)input_x2, (void *)output};  \
    CREATE_NODEDEF(shapes, data_types, datas);                                    \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                      \
  }

#define ADD_CASE_DIM0(base_type, aicpu_type)                                      \
  TEST_F(TEST_REALDIV_UT, TestRealDiv_DIM0_##aicpu_type) {                        \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};           \
    vector<vector<int64_t>> shapes = {{}, {}, {}};                                \
    base_type input_x1[1] = {(base_type)24};                                      \
    base_type input_x2[1] = {(base_type)24};                                      \
    base_type output[1] = {(base_type)1};                                         \
    vector<void *> datas = {(void *)input_x1, (void *)input_x2, (void *)output};  \
    CREATE_NODEDEF(shapes, data_types, datas);                                    \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                 \
  }

#define ADD_CASE_EMPTY_SHAPE(base_type, aicpu_type)                               \
  TEST_F(TEST_REALDIV_UT, TestRealDiv_EMPTY_##aicpu_type) {                       \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};           \
    vector<vector<int64_t>> shapes = {{0}, {}, {}};                               \
    base_type input_x1 = 24;                                                      \
    base_type input_x2 = 24;                                                      \
    base_type output = {(base_type)1};                                            \
    vector<void *> datas = {(void *)input_x1, (void *)input_x2, (void *)output};  \
    CREATE_NODEDEF(shapes, data_types, datas);                                    \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                 \
  }

#define ADD_CASE_BOOL(base_type, aicpu_type)                                      \
  TEST_F(TEST_REALDIV_UT, TestRealDiv_##aicpu_type) {                             \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};           \
    vector<vector<int64_t>> shapes = {{24}, {24}, {24}};                          \
    base_type input_x1[24];                                                       \
    SetRandomValue<base_type>(input_x1, 24);                                      \
    base_type input_x2[24];                                                       \
    SetRandomValue<base_type>(input_x2, 24, 1);                                   \
    base_type output[24] = {(base_type)0};                                        \
    vector<void *> datas = {(void *)input_x1, (void *)input_x2, (void *)output};  \
    CREATE_NODEDEF(shapes, data_types, datas);                                    \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                      \
  }

ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)

ADD_CASE(double, DT_DOUBLE)

ADD_CASE(int8_t, DT_INT8)

ADD_CASE(int16_t, DT_INT16)

ADD_CASE(int32_t, DT_INT32)

ADD_CASE(uint32_t, DT_UINT32)

ADD_CASE(int64_t, DT_INT64)

ADD_CASE(uint64_t, DT_UINT64)

ADD_CASE(uint8_t, DT_UINT8)

ADD_CASE(uint16_t, DT_UINT16)

ADD_CASE_DIM0(int64_t, DT_INT64)

ADD_CASE_EMPTY_SHAPE(int32_t, DT_INT32)

ADD_CASE_BOOL(bool, DT_BOOL)

ADD_CASE(std::complex<float>, DT_COMPLEX64)

ADD_CASE(std::complex<double>, DT_COMPLEX128)

vector<vector<int64_t>> shapes_realdiv_dim3 = {{2, 1, 1}, {1, 1, 1}, {2, 1, 1}};
vector<int64_t> data_num_realdiv_dim3 = {2, 1, 2};
REALDIV_CASE_WITH_SHAPE(int32_realdiv_vector_dim_3, int32_t, DT_INT32, shapes_realdiv_dim3, data_num_realdiv_dim3)

vector<vector<int64_t>> shapes_realdiv_dim7 = {{2, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1}};
vector<int64_t> data_num_realdiv_dim7 = {2, 1, 2};
REALDIV_CASE_WITH_SHAPE(int16_realdiv_vector_dim_7, int16_t, DT_INT16, shapes_realdiv_dim7, data_num_realdiv_dim7)

vector<vector<int64_t>> shapes_realdiv_dim8 = {{2, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1}};
vector<int64_t> data_num_realdiv_dim8 = {2, 1, 2};
REALDIV_CASE_WITH_SHAPE(int16_realdiv_vector_dim_8, int16_t, DT_INT16, shapes_realdiv_dim8, data_num_realdiv_dim8)

ADD_ZERO_CASE(int8_t, DT_INT8)

ADD_ZERO_CASE(int16_t, DT_INT16)

ADD_ZERO_CASE(int32_t, DT_INT32)

ADD_ZERO_CASE(int64_t, DT_INT64)

ADD_ZERO_CASE(uint8_t, DT_UINT8)

ADD_ZERO_CASE(uint16_t, DT_UINT16)

// ==================== coverage: large same-shape parallel path ====================
// data_num >= kParallelDataNumSameShape (7*1024=7168) triggers ParallelFor in NoBcastComputeImpl
TEST_F(TEST_REALDIV_UT, TestRealDiv_LargeSameShape_Parallel) {
    const int64_t num = 8192;
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{num}, {num}, {num}};
    std::vector<float> input1(num, 2.0f);
    std::vector<float> input2(num, 2.0f);
    std::vector<float> output(num, 0.0f);
    vector<void *> datas = {input1.data(), input2.data(), output.data()};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    std::vector<float> expect_out(num, 1.0f);
    CompareResult<float>(output.data(), expect_out.data(), num);
}

// data_num in [kParallelDataNumSameShape, kParallelDataNumSameShapeMid] → max 4 cores
TEST_F(TEST_REALDIV_UT, TestRealDiv_LargeSameShape_MidThreshold) {
    const int64_t num = 16384;
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{num}, {num}, {num}};
    std::vector<float> input1(num, 6.0f);
    std::vector<float> input2(num, 3.0f);
    std::vector<float> output(num, 0.0f);
    vector<void *> datas = {input1.data(), input2.data(), output.data()};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    std::vector<float> expect_out(num, 2.0f);
    CompareResult<float>(output.data(), expect_out.data(), num);
}

// ==================== coverage: X_ONE_ELEMENT path (scalar / vector) ====================
TEST_F(TEST_REALDIV_UT, TestRealDiv_ScalarDivVector) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1}, {4}, {4}};
    float input1[1] = {12.0f};
    float input2[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0};
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect_out[4] = {12.0f, 6.0f, 4.0f, 3.0f};
    CompareResult<float>(output, expect_out, 4);
}

// ==================== coverage: Y_ONE_ELEMENT path (vector / scalar) ====================
TEST_F(TEST_REALDIV_UT, TestRealDiv_VectorDivScalar) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{4}, {1}, {4}};
    float input1[4] = {4.0f, 8.0f, 12.0f, 16.0f};
    float input2[1] = {4.0f};
    float output[4] = {0};
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect_out[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    CompareResult<float>(output, expect_out, 4);
}

// ==================== coverage: broadcast parallel path (>= kParallelDataNum=2048) ====================
// Broadcast with large output to trigger ParallelFor in BcastComputeImpl
TEST_F(TEST_REALDIV_UT, TestRealDiv_BcastLargeParallel) {
    const int64_t rows = 64;
    const int64_t cols = 64;
    const int64_t out_num = rows * cols;
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{rows, cols}, {1, cols}, {rows, cols}};
    std::vector<float> input1(out_num, 10.0f);
    std::vector<float> input2(cols, 5.0f);
    std::vector<float> output(out_num, 0.0f);
    vector<void *> datas = {input1.data(), input2.data(), output.data()};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    std::vector<float> expect_out(out_num, 2.0f);
    CompareResult<float>(output.data(), expect_out.data(), out_num);
}

// ==================== coverage: broadcast y_inner==0 (Y broadcast on innermost dim) ====================
// x shape (3,4), y shape (3,1) → output (3,4), y broadcasts on innermost → y_inner=0
TEST_F(TEST_REALDIV_UT, TestRealDiv_BcastYInner0) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{3, 4}, {3, 1}, {3, 4}};
    float input1[12] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
    float input2[3] = {2, 2, 2};
    float output[12] = {0};
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect_out[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    CompareResult<float>(output, expect_out, 12);
}

// ==================== coverage: broadcast x_inner==0 (X broadcast on innermost dim) ====================
// x shape (3,1), y shape (3,4) → output (3,4), x broadcasts on innermost → x_inner=0
TEST_F(TEST_REALDIV_UT, TestRealDiv_BcastXInner0) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{3, 1}, {3, 4}, {3, 4}};
    float input1[3] = {12.0f, 24.0f, 36.0f};
    float input2[12] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    float output[12] = {0};
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect_out[12] = {12, 6, 4, 3, 24, 12, 8, 6, 36, 18, 12, 9};
    CompareResult<float>(output, expect_out, 12);
}

// ==================== coverage: broadcast both contiguous innermost (outer dim broadcast) ====================
// x (4,3), y (1,3) → output (4,3), x_inner=1, y_inner=1, broadcast only on dim 0
TEST_F(TEST_REALDIV_UT, TestRealDiv_BcastBothContiguous) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{4, 3}, {1, 3}, {4, 3}};
    float input1[12] = {3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36};
    float input2[3] = {3, 3, 3};
    float output[12] = {0};
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect_out[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    CompareResult<float>(output, expect_out, 12);
}

// ==================== coverage: "Div" op type registration ====================
TEST_F(TEST_REALDIV_UT, TestDiv_OpType) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    float input1[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float input2[4] = {2.0f, 4.0f, 5.0f, 8.0f};
    float output[4] = {0};
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "Div", "Div")
        .Input({"x1", data_types[0], shapes[0], datas[0]})
        .Input({"x2", data_types[1], shapes[1], datas[1]})
        .Output({"y", data_types[2], shapes[2], datas[2]});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect_out[4] = {5.0f, 5.0f, 6.0f, 5.0f};
    CompareResult<float>(output, expect_out, 4);
}

// ==================== coverage: type mismatch error ====================
TEST_F(TEST_REALDIV_UT, TestRealDiv_TypeMismatch) {
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    float input1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t input2[4] = {1, 2, 3, 4};
    float output[4] = {0};
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "RealDiv", "RealDiv")
        .Input({"x1", DT_FLOAT, shapes[0], datas[0]})
        .Input({"x2", DT_INT32, shapes[1], datas[1]})
        .Output({"y", DT_FLOAT, shapes[2], datas[2]});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ==================== coverage: broadcast with incompatible shapes ====================
TEST_F(TEST_REALDIV_UT, TestRealDiv_IncompatibleBroadcast) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{3}, {5}, {5}};
    float input1[3] = {1, 2, 3};
    float input2[5] = {1, 2, 3, 4, 5};
    float output[5] = {0};
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ==================== coverage: broadcast mid threshold (kParallelDataNumMid=16*1024) ====================
TEST_F(TEST_REALDIV_UT, TestRealDiv_BcastMidThreshold) {
    const int64_t rows = 128;
    const int64_t cols = 128;
    const int64_t out_num = rows * cols;
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{rows, cols}, {1, cols}, {rows, cols}};
    std::vector<float> input1(out_num, 8.0f);
    std::vector<float> input2(cols, 4.0f);
    std::vector<float> output(out_num, 0.0f);
    vector<void *> datas = {input1.data(), input2.data(), output.data()};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    std::vector<float> expect_out(out_num, 2.0f);
    CompareResult<float>(output.data(), expect_out.data(), out_num);
}

// ==================== coverage: NoBcast large X_ONE_ELEMENT parallel ====================
TEST_F(TEST_REALDIV_UT, TestRealDiv_ScalarDivVector_LargeParallel) {
    const int64_t num = 8192;
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1}, {num}, {num}};
    float input1[1] = {100.0f};
    std::vector<float> input2(num, 10.0f);
    std::vector<float> output(num, 0.0f);
    vector<void *> datas = {(void *)input1, input2.data(), output.data()};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    std::vector<float> expect_out(num, 10.0f);
    CompareResult<float>(output.data(), expect_out.data(), num);
}

// ==================== coverage: NoBcast large Y_ONE_ELEMENT parallel ====================
TEST_F(TEST_REALDIV_UT, TestRealDiv_VectorDivScalar_LargeParallel) {
    const int64_t num = 8192;
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{num}, {1}, {num}};
    std::vector<float> input1(num, 50.0f);
    float input2[1] = {5.0f};
    std::vector<float> output(num, 0.0f);
    vector<void *> datas = {input1.data(), (void *)input2, output.data()};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    std::vector<float> expect_out(num, 10.0f);
    CompareResult<float>(output.data(), expect_out.data(), num);
}