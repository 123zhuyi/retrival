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
#include <complex>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_Cross_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, dim_select, builder)        \
  do {                                                                         \
    (builder).Input({"x1", data_types[0], shapes[0], datas[0]})               \
            .Input({"x2", data_types[1], shapes[1], datas[1]})                \
            .Output({"y", data_types[2], shapes[2], datas[2]})                \
            .Attr("dim", dim_select);                                         \
  } while (0)

#define CREATE_NODEDEF_WITHOUT_DIM(shapes, data_types, datas, builder)        \
  do {                                                                         \
    (builder).Input({"x1", data_types[0], shapes[0], datas[0]})               \
            .Input({"x2", data_types[1], shapes[1], datas[1]})                \
            .Output({"y", data_types[2], shapes[2], datas[2]});               \
  } while (0)

template <typename T1, typename T2, typename T3>
void RunCrossKernel1(vector<DataType> data_types,
                     vector<vector<int64_t>>& shapes, int64_t dim_dim,
                     const T1 *input1_data,
                     const T2 *input2_data,
                     const T3 *output_exp_data) {

  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1* input1 = new T1[input1_size];

  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2* input2 = new T2[input2_size];

  for (uint64_t i = 0; i < input1_size; ++i) {
    input1[i] = input1_data[i];
  }
  for (uint64_t i = 0; i < input2_size; ++i) {
    input2[i] = input2_data[i];
  }

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3* output = new T3[output_size];
  vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder builder(node_def.get(), "Cross", "Cross");
  CREATE_NODEDEF(shapes, data_types, datas, dim_dim, builder);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);


  T3* output_exp = new T3[output_size];
  for (uint64_t i = 0; i < output_size; ++i) {
    output_exp[i] = output_exp_data[i];
  }

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_FLOAT64_SUCC) 
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
    const float input1_data[] = {2.7077589566656357f, 5.197479834591791f, 6.233047482133057f, 4.375215665760221f, 7.8406884533410945f, 9.12681929774893f};
    const float input2_data[] = {6.021063477291859f, 3.38675013593856f, 0.24726845735401404f, 6.8119374382102285f, 3.4433239030091043f, 1.725501009863517f};
    const float output_exp_data[] = {-19.82460158659752f, 36.860031166796105f, -22.123852991445368f, -17.897479202118383f, 54.621883016475586f, -38.34499453392479f};

    int64_t ddim = 2;
    RunCrossKernel1<float, float, float>(data_types, shapes, ddim, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_DOUBLE_SUCC1) 
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{3, 2, 3}, {3, 2, 3}, {3, 2, 3}};
    const double input1_data[] = {54.155179133312714, 103.94959669183581, 124.66094964266115, 87.50431331520443, 156.8137690668219, 182.53638595497864, 120.42126954583719, 67.7350027187712, 4.945369147080281, 136.23874876420456, 68.86647806018209, 34.51002019727034, 71.09907064812298, 150.49679028637476, 186.2817458852594, 16.09613730866255, 135.83370149633697, 46.157114460197036};
    const double input2_data[] = {63.17853420874613, 23.654768477113652, 32.582148794471834, 31.705177207030655, 123.49062954584826, 103.95472315282115, 156.86300697198396, 97.96767314601573, 169.99883941283485, 13.930085270036695, 108.03806347117562, 128.2126428175542, 71.74034709558342, 158.8784492903085, 91.97021211784579, 129.93378676074204, 75.98553209526817, 82.59681882796657};
    const double output_exp_data[] = {-2513.750339868926, -3982.1881656668374, -31212.8539548327, 17477.795965250574, -9442.35408485593, -3067.507743790804, 606.8237191051448, -12955.383995978937, -5395.634419320126, -10859.435901223087, 4858.611628918519, -10278.674744992291, 886.9049444955599, 8581.444305256731, 21031.086005960486, -3100.5311260564285, 8437.491205338347, 19815.99285804739};

    int64_t ddim = 0;
    RunCrossKernel1<double, double, double>(data_types, shapes, ddim, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_FLOAT16_SUCC1) 
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
    const Eigen::half input1_data[] = {static_cast<Eigen::half>(-792984.06f), static_cast<Eigen::half>(1474401.1f), static_cast<Eigen::half>(-884954.1f), static_cast<Eigen::half>(-715899.25f), static_cast<Eigen::half>(2184875.2f), static_cast<Eigen::half>(-1533799.5f)};
    const Eigen::half input2_data[] = {static_cast<Eigen::half>(1204.2126f), static_cast<Eigen::half>(677.35004f), static_cast<Eigen::half>(49.45369f), static_cast<Eigen::half>(1362.3875f), static_cast<Eigen::half>(688.6648f), static_cast<Eigen::half>(345.1002f)};
    const Eigen::half output_exp_data[] = {static_cast<Eigen::half>(-792984.06f), static_cast<Eigen::half>(1474401.1f), static_cast<Eigen::half>(-884954.1f), static_cast<Eigen::half>(-715899.25f), static_cast<Eigen::half>(2184875.2f), static_cast<Eigen::half>(-1533799.5f)};

    int64_t ddim = 2;
    RunCrossKernel1<Eigen::half, Eigen::half, Eigen::half>(data_types, shapes, ddim, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_Cross_UT, DEFAULT_DIM) 
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2, 3, 3}, {2, 3, 3}, {2, 3, 3}};
    const int16_t input1_data[] = {2, 5, 6, 4, 7, 9, 6, 3, 0, 6, 3, 1, 3, 7, 9, 0, 6, 2};
    const int16_t input2_data[] = {3, 1, 1, 1, 6, 5, 7, 4, 8, 0, 5, 6, 3, 7, 4, 6, 3, 4};
    const int16_t output_exp_data[] = {22, 10, 72, 4, -17, -48, -10, 23, 21, 18, -21, 28, -36, 21, 8, 18, -14, -50};

    int64_t ddim = -65530;
    RunCrossKernel1<int16_t, int16_t, int16_t>(data_types, shapes, ddim,  input1_data, input2_data, output_exp_data);
}

// exception instance
TEST_F(TEST_Cross_UT, INPUT_NULL_EXCEPTION) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
    int32_t output[6];
    vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
    int64_t ddim = 2;
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "Cross", "Cross");
    CREATE_NODEDEF(shapes, data_types, datas, ddim, builder);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Cross_UT, DIM_RANGE_EXCEPTION) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
    int32_t input1[6];
    int32_t input2[6];
    int32_t output[6];
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    int64_t ddim = 3;
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "Cross", "Cross");
    CREATE_NODEDEF(shapes, data_types, datas, ddim, builder);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Cross_UT, DIM_VALUE_EXCEPTION) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
    int32_t input1[6];
    int32_t input2[6];
    int32_t output[6];
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    int64_t ddim = 0;
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "Cross", "Cross");
    CREATE_NODEDEF(shapes, data_types, datas, ddim, builder);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Cross_UT, SHAPE_SIZE_EXCEPTION) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3}};
    int32_t input1[6];
    int32_t input2[24];
    int32_t output[6];
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    int64_t ddim = 2;
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "Cross", "Cross");
    CREATE_NODEDEF(shapes, data_types, datas, ddim, builder);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Cross_UT, SHAPE_VALUE_EXCEPTION) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 3, 3}, {1, 2, 3}};
    int32_t input1[6];
    int32_t input2[9];
    int32_t output[6];
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    int64_t ddim = 2;
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "Cross", "Cross");
    CREATE_NODEDEF(shapes, data_types, datas, ddim, builder);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Cross_UT, DTYPE_EXCEPTION) {
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
    int32_t input1[6];
    int32_t input2[6];
    int32_t output[6];
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
    int64_t ddim = 2;
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder builder(node_def.get(), "Cross", "Cross");
    CREATE_NODEDEF(shapes, data_types, datas, ddim, builder);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}