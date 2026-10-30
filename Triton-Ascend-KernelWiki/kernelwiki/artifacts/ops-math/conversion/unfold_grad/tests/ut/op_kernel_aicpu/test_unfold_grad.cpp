/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_unfold_grad.cpp
 * \brief
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "utils/aicpu_read_file.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_UNFOLD_GRAD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, dim, size, step)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "UnfoldGrad", "UnfoldGrad")       \
        .Input({"grad_out", data_types[0], shapes[0], datas[0]})           \
        .Input({"input_sizes", data_types[1], shapes[1], datas[1]})           \
        .Output({"grad_in", data_types[2], shapes[2], datas[2]})        \
        .Attr("dim", dim)                                                 \
        .Attr("size", size)                                           \
        .Attr("step", step)

TEST_F(TEST_UNFOLD_GRAD_UT, LAST_THIRD_DIM_COMPUTE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 2, 2, 1, 2}, {4}, {1, 3, 2, 1}};

  float input1[8];
  int64_t input2[4] = {1,3,2,1};
  float output[6];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, 1, 2, 1);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}