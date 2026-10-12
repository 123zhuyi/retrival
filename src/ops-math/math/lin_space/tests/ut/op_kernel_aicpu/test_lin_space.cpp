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
#include <Eigen/Core>
#include <iostream>

#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_LINSPACE_UT : public testing::Test {};

template <typename T>
bool CheckResult(T *output_data, T *output_exp, int64_t output_num) {
    for(int64_t i = 0; i < output_num; i++) {
       if(output_data[i] != output_exp[i]) {
        return false;
        }
    }
    return true;
}

TEST_F(TEST_LINSPACE_UT, SUCCESS_FLOAT_INT32) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {5}};
  float start_data = 2.0f;
  float stop_data = 10.0f;
  int32_t num_data = 5;
  float output_data[5] = {0};
  float output_exp[5] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")
      .Input({"start", data_types[0], shapes[0], datas[0]})
      .Input({"stop", data_types[1], shapes[1], datas[1]})
      .Input({"num", data_types[2], shapes[2], datas[2]})
      .Output({"output", data_types[3], shapes[3], datas[3]});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CheckResult<float>(output_data, output_exp, 5);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LINSPACE_UT, SUCCESS_DOUBLE_INT64) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {5}};
  double start_data = 2.0;
  double stop_data = 10.0;
  int64_t num_data = 5;
  double output_data[5] = {0};
  double output_exp[5] = {2.0, 4.0, 6.0, 8.0, 10.0};

  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_INT64, DT_DOUBLE};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")
      .Input({"start", data_types[0], shapes[0], datas[0]})
      .Input({"stop", data_types[1], shapes[1], datas[1]})
      .Input({"num", data_types[2], shapes[2], datas[2]})
      .Output({"output", data_types[3], shapes[3], datas[3]});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CheckResult<double>(output_data, output_exp, 5);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LINSPACE_UT, SUCCESS_NUM_1) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {1}};
  float start_data = 2.0f;
  float stop_data = 10.0f;
  int32_t num_data = 1;
  float output_data[1] = {0};
  float output_exp[1] = {2.0f};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")
      .Input({"start", data_types[0], shapes[0], datas[0]})
      .Input({"stop", data_types[1], shapes[1], datas[1]})
      .Input({"num", data_types[2], shapes[2], datas[2]})
      .Output({"output", data_types[3], shapes[3], datas[3]});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CheckResult<float>(output_data, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LINSPACE_UT, FAILED_NUM_NEGATIVE) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
  float start_data = 2.0f;
  float stop_data = 10.0f;
  int64_t num_data = -1;
  float output_data[1] = {0};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")
      .Input({"start", data_types[0], shapes[0], datas[0]})
      .Input({"stop", data_types[1], shapes[1], datas[1]})
      .Input({"num", data_types[2], shapes[2], datas[2]})
      .Output({"output", data_types[3], shapes[3], datas[3]});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LINSPACE_UT, FAILED_START_NOT_SCALAR) {
  vector<vector<int64_t>> shapes = {{2}, {}, {}, {}};
  float start_data[2] = {2.0f, 10.0f};
  float stop_data = 10.0f;
  int64_t num_data = 5;
  float output_data[1] = {0};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<void*> datas = {(void*)start_data, (void*)&stop_data, (void*)&num_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")
      .Input({"start", data_types[0], shapes[0], datas[0]})
      .Input({"stop", data_types[1], shapes[1], datas[1]})
      .Input({"num", data_types[2], shapes[2], datas[2]})
      .Output({"output", data_types[3], shapes[3], datas[3]});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LINSPACE_UT, FAILED_STOP_NOT_SCALAR) {
  vector<vector<int64_t>> shapes = {{}, {2}, {}, {}};
  float start_data = 2.0f;
  float stop_data[2] = {10.0f, 2.0f};
  int64_t num_data = 5;
  float output_data[1] = {0};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)stop_data, (void*)&num_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")
      .Input({"start", data_types[0], shapes[0], datas[0]})
      .Input({"stop", data_types[1], shapes[1], datas[1]})
      .Input({"num", data_types[2], shapes[2], datas[2]})
      .Output({"output", data_types[3], shapes[3], datas[3]});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LINSPACE_UT, FAILED_NUM_NOT_SCALAR) {
  vector<vector<int64_t>> shapes = {{}, {}, {2}, {}};
  float start_data = 2.0f;
  float stop_data = 10.0f;
  int64_t num_data[2] = {2, 5};
  float output_data[1] = {0};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)num_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")
      .Input({"start", data_types[0], shapes[0], datas[0]})
      .Input({"stop", data_types[1], shapes[1], datas[1]})
      .Input({"num", data_types[2], shapes[2], datas[2]})
      .Output({"output", data_types[3], shapes[3], datas[3]});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LINSPACE_UT, FAILED_START_STOP_TYPE_DIFF) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
  double start_data = 2.0;
  float stop_data = 10.0f;
  int64_t num_data = 5;
  float output_data[1] = {0};

  vector<DataType> data_types = {DT_DOUBLE, DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")
      .Input({"start", data_types[0], shapes[0], datas[0]})
      .Input({"stop", data_types[1], shapes[1], datas[1]})
      .Input({"num", data_types[2], shapes[2], datas[2]})
      .Output({"output", data_types[3], shapes[3], datas[3]});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LINSPACE_UT, FAILED_UNSUPPORTED_TYPE) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
  int64_t start_data = 2;
  int64_t stop_data = 10;
  int64_t num_data = 5;
  int64_t output_data[1] = {0};

  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")
      .Input({"start", data_types[0], shapes[0], datas[0]})
      .Input({"stop", data_types[1], shapes[1], datas[1]})
      .Input({"num", data_types[2], shapes[2], datas[2]})
      .Output({"output", data_types[3], shapes[3], datas[3]});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LINSPACE_UT, FAILED_NUM_WRONG_TYPE) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {5}};
  double start_data = 10.0;
  double stop_data = 2.0;
  int8_t num_data = 5;
  double output_data[5] = {0};

  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_INT8, DT_DOUBLE};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")
      .Input({"start", data_types[0], shapes[0], datas[0]})
      .Input({"stop", data_types[1], shapes[1], datas[1]})
      .Input({"num", data_types[2], shapes[2], datas[2]})
      .Output({"output", data_types[3], shapes[3], datas[3]});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}