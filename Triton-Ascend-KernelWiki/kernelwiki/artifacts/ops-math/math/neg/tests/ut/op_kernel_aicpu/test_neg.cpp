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

class TEST_NEG_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Neg", "Neg")                     \
      .Input({"x", data_types[0], shapes[0], datas[0]})           \
      .Output({"y", data_types[1], shapes[1], datas[1]})

#define ADD_CASE(base_type, aicpu_type)                                      \
  TEST_F(TEST_NEG_UT, TestNeg_##aicpu_type) {                                \
    vector<DataType> data_types = {aicpu_type, aicpu_type};                  \
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};                       \
    base_type input0[6] = {(base_type)1, (base_type)2, (base_type)3,         \
                           (base_type)4, (base_type)5, (base_type)6};        \
    base_type output[6] = {base_type(0)};                                    \
    vector<void *> datas = {(void *)input0, (void *)output};                 \
    CREATE_NODEDEF(shapes, data_types, datas);                               \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                            \
    base_type output_exp[6] = {(base_type)-1, (base_type)-2, (base_type)-3,  \
                               (base_type)-4, (base_type)-5, (base_type)-6}; \
    EXPECT_EQ(CompareResult<base_type>(output, output_exp, 6), true);        \
  }

TEST_F(TEST_NEG_UT, Failed) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  uint32_t input0[6] = {1, 2, 3, 4, 5, 6};
  uint32_t output[6] = {0};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

ADD_CASE(float, DT_FLOAT)
ADD_CASE(double, DT_DOUBLE)
ADD_CASE(Eigen::half, DT_FLOAT16)
ADD_CASE(int32_t, DT_INT32)
ADD_CASE(int64_t, DT_INT64)
