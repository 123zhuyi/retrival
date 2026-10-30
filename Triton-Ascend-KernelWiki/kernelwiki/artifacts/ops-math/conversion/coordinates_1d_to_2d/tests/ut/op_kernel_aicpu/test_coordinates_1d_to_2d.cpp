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

using namespace std;
using namespace aicpu;

class TEST_COORDINATES_1D_TO_2D_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                             \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();           \
  NodeDefBuilder node(node_def.get(), "Coordinates1DTo2D", "Coordinates1DTo2D"); \
  node.Input({"x", data_types[0], shapes[0], datas[0]})                       \
      .Input({"shape", data_types[1], shapes[1], datas[1]})                   \
      .Output({"row", data_types[2], shapes[2], datas[2]})                    \
      .Output({"col", data_types[3], shapes[3], datas[3]})                    \
      .Output({"n", data_types[4], shapes[4], datas[4]});

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, TestCoordinates1DTo2D_basic_int32) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1}, {4}, {1}, {1}, {1}};
    int32_t input_x[1] = {5};
    int32_t input_shape[4] = {1, 1, 1, 10};
    int32_t output_row[1] = {0};
    int32_t output_col[1] = {0};
    int32_t output_n[1] = {0};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output_row[0], 0);
    EXPECT_EQ(output_col[0], 5);
    EXPECT_EQ(output_n[0], 10);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, TestCoordinates1DTo2D_basic_int64) {
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{1}, {4}, {1}, {1}, {1}};
    int64_t input_x[1] = {25};
    int64_t input_shape[4] = {2, 2, 2, 5};
    int64_t output_row[1] = {0};
    int64_t output_col[1] = {0};
    int64_t output_n[1] = {0};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output_row[0], 5);
    EXPECT_EQ(output_col[0], 0);
    EXPECT_EQ(output_n[0], 5);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, TestCoordinates1DTo2D_basic_uint64) {
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{1}, {4}, {1}, {1}, {1}};
    uint64_t input_x[1] = {15};
    uint64_t input_shape[4] = {1, 1, 1, 10};
    uint64_t output_row[1] = {0};
    uint64_t output_col[1] = {0};
    uint64_t output_n[1] = {0};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output_row[0], 1);
    EXPECT_EQ(output_col[0], 5);
    EXPECT_EQ(output_n[0], 10);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, TestCoordinates1DTo2D_TYPE_MISMATCH) {
    vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1}, {4}, {1}, {1}, {1}};
    int32_t input_x[1] = {5};
    int64_t input_shape[4] = {1, 1, 1, 10};
    int32_t output_row[1] = {0};
    int32_t output_col[1] = {0};
    int32_t output_n[1] = {0};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, TestCoordinates1DTo2D_INVALID_SHAPE_SIZE) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1}, {3}, {1}, {1}, {1}};
    int32_t input_x[1] = {5};
    int32_t input_shape[3] = {1, 1, 10};
    int32_t output_row[1] = {0};
    int32_t output_col[1] = {0};
    int32_t output_n[1] = {0};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, TestCoordinates1DTo2D_zero_col) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1}, {4}, {1}, {1}, {1}};
    int32_t input_x[1] = {5};
    int32_t input_shape[4] = {1, 1, 1, 0};
    int32_t output_row[1] = {0};
    int32_t output_col[1] = {0};
    int32_t output_n[1] = {0};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, ThreadId_0_SUCCESS) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{}, {4}, {}, {}, {}};
    int32_t input_x[1] = {0};
    int32_t input_shape[4] = {1, 1, 4, 4};
    int32_t output_row[1] = {-1};
    int32_t output_col[1] = {-1};
    int32_t output_n[1] = {-1};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output_row[0], 0);
    EXPECT_EQ(output_col[0], 0);
    EXPECT_EQ(output_n[0], 4);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, ThreadId_5_SUCCESS) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{}, {4}, {}, {}, {}};
    int32_t input_x[1] = {5};
    int32_t input_shape[4] = {1, 1, 4, 4};
    int32_t output_row[1] = {-1};
    int32_t output_col[1] = {-1};
    int32_t output_n[1] = {-1};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output_row[0], 1);
    EXPECT_EQ(output_col[0], 1);
    EXPECT_EQ(output_n[0], 4);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, ThreadId_15_SUCCESS) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{}, {4}, {}, {}, {}};
    int32_t input_x[1] = {15};
    int32_t input_shape[4] = {1, 1, 4, 4};
    int32_t output_row[1] = {-1};
    int32_t output_col[1] = {-1};
    int32_t output_n[1] = {-1};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output_row[0], 3);
    EXPECT_EQ(output_col[0], 3);
    EXPECT_EQ(output_n[0], 4);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, ThreadId_0_int64_SUCCESS) {
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{}, {4}, {}, {}, {}};
    int64_t input_x[1] = {0};
    int64_t input_shape[4] = {1, 1, 4, 4};
    int64_t output_row[1] = {-1};
    int64_t output_col[1] = {-1};
    int64_t output_n[1] = {-1};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output_row[0], 0);
    EXPECT_EQ(output_col[0], 0);
    EXPECT_EQ(output_n[0], 4);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, ThreadId_0_uint64_SUCCESS) {
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{}, {4}, {}, {}, {}};
    uint64_t input_x[1] = {0};
    uint64_t input_shape[4] = {1, 1, 4, 4};
    uint64_t output_row[1] = {0};
    uint64_t output_col[1] = {0};
    uint64_t output_n[1] = {0};
    vector<void *> datas = {(void *)input_x, (void *)input_shape,
                            (void *)output_row, (void *)output_col, (void *)output_n};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output_row[0], 0);
    EXPECT_EQ(output_col[0], 0);
    EXPECT_EQ(output_n[0], 4);
}