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

class TEST_SPLITV_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, num_split)                \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();          \
  NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");                  \
  node.Input({"x", data_types[0], shapes[0], datas[0]})                 \
      .Input({"size_splits", data_types[1], shapes[1], datas[1]})           \
      .Input({"split_dim", data_types[2], shapes[2], datas[2]})             \
      .Attr("num_split", num_split);                                        \
  for(int i = 0; i < num_split; i++) {                                      \
    node.Output({"y", data_types[i + 3], shapes[i + 3], datas[i + 3]});     \
  }

#define ADD_CASE(case_name, aicpu_type, base_type, split_dim, num_split)                                         \
  TEST_F(TEST_SPLITV_UT, TestSplitV_##case_name##_##aicpu_type) {                                                \
    if(num_split == 1) {                                                                                         \
      vector<DataType> data_types = {aicpu_type, DT_INT64, DT_INT32, aicpu_type};                                \
      vector<vector<int64_t>> shapes = {{2, 2, 2}, {1}, {}, {2, 2, 2}};                                          \
      base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                              \
                            (base_type)5, (base_type)6, (base_type)7, (base_type)8};                             \
      int64_t size_split[1] = {2};                                                                               \
      base_type output[8] = {(base_type)0};                                                                      \
      vector<void *> datas = {(void *)input, (void *)size_split, (void *)&split_dim, (void *)output};            \
      CREATE_NODEDEF(shapes, data_types, datas, 1);                                                              \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                              \
      base_type expect_out[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                         \
                                 (base_type)5, (base_type)6, (base_type)7, (base_type)8};                        \
      EXPECT_EQ(CompareResult<base_type>(output, expect_out, 8), true);                                          \
    } else if(split_dim == 1){                                                                                   \
      vector<DataType> data_types = {aicpu_type, DT_INT64, DT_INT32, aicpu_type, aicpu_type};                    \
      vector<vector<int64_t>> shapes = {{2, 2, 2}, {2}, {}, {2, 1, 2},{2, 1, 2}};                                \
      base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                              \
                            (base_type)5, (base_type)6, (base_type)7, (base_type)8};                             \
      base_type output1[4] = {(base_type)0};                                                                     \
      base_type output2[4] = {(base_type)0};                                                                     \
      int64_t size_split[2] = {1, -1};                                                                           \
      vector<void *> datas = {(void *)input, (void *)size_split, (void *)&split_dim,                             \
                              (void *)output1, (void *)output2};                                                 \
      CREATE_NODEDEF(shapes, data_types, datas, 2);                                                              \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                              \
      base_type expect_out1[4] = {(base_type)1, (base_type)2, (base_type)5, (base_type)6};                       \
      base_type expect_out2[4] = {(base_type)3, (base_type)4, (base_type)7, (base_type)8};                       \
      EXPECT_EQ(CompareResult<base_type>(output1, expect_out1, 4), true);                                        \
      EXPECT_EQ(CompareResult<base_type>(output2, expect_out2, 4), true);                                        \
    } else if(split_dim == 0) {                                                                                  \
      vector<DataType> data_types = {aicpu_type, DT_INT64, DT_INT32, aicpu_type, aicpu_type};                    \
      vector<vector<int64_t>> shapes = {{2, 2, 2}, {2}, {}, {1, 2, 2},{1, 2, 2}};                                \
      base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                              \
                            (base_type)5, (base_type)6, (base_type)7, (base_type)8};                             \
      base_type output1[4] = {(base_type)0};                                                                     \
      base_type output2[4] = {(base_type)0};                                                                     \
      int64_t size_split[2] = {-1, 1};                                                                           \
      vector<void *> datas = {(void *)input, (void *)size_split, (void *)&split_dim,                             \
                              (void *)output1, (void *)output2};                                                 \
      CREATE_NODEDEF(shapes, data_types, datas, 2);                                                              \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                              \
      base_type expect_out1[4] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4};                       \
      base_type expect_out2[4] = {(base_type)5, (base_type)6, (base_type)7, (base_type)8};                       \
      EXPECT_EQ(CompareResult<base_type>(output1, expect_out1, 4), true);                                        \
      EXPECT_EQ(CompareResult<base_type>(output2, expect_out2, 4), true);                                        \
    }                                                                                                            \
  }

#define ADD_CASE_FAILED(case_name, aicpu_type, base_type, split_dim, num_split)                                  \
  TEST_F(TEST_SPLITV_UT, TestSplitV_##case_name##_##aicpu_type) {                                                \
      vector<DataType> data_types = {aicpu_type, DT_INT64, DT_INT32, aicpu_type, aicpu_type};                    \
      vector<vector<int64_t>> shapes = {{2, 2, 2}, {2}, {}, {2, 1, 2}, {2, 1, 2}};                               \
      base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                              \
                            (base_type)5, (base_type)6, (base_type)7, (base_type)8};                             \
      int64_t size_split[2] = {1, 1};                                                                            \
      base_type output1[4] = {(base_type)0};                                                                     \
      base_type output2[4] = {(base_type)0};                                                                     \
      vector<void *> datas = {(void *)input, (void *)size_split, (void *)&split_dim,                             \
                              (void *)output1, (void *)output2};                                                 \
      CREATE_NODEDEF(shapes, data_types, datas, num_split);                                                      \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                                                   \
  }

int64_t split_dim1 = 1;
int64_t split_dim0 = 0;
int64_t split_dim2 = 4;

ADD_CASE(two_split_with_dim_1, DT_FLOAT, float, split_dim1, 2)

ADD_CASE(two_split_with_dim_1, DT_DOUBLE, double, split_dim1, 2)

ADD_CASE(two_split_with_dim_1, DT_FLOAT16, Eigen::half, split_dim1, 2)

ADD_CASE(two_split_with_dim_0, DT_INT32, int32_t, split_dim0, 2)

ADD_CASE(two_split_with_dim_0, DT_INT16, int16_t, split_dim0, 2)

ADD_CASE(two_split_with_dim_0, DT_INT64, int64_t, split_dim0, 2)

ADD_CASE(two_split_with_dim_0, DT_INT8, int8_t, split_dim0, 2)

ADD_CASE(one_split_with_dim_1, DT_BOOL, bool, split_dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT8, uint8_t, split_dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT16, uint16_t, split_dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT32, uint32_t, split_dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT64, uint64_t, split_dim1, 1)

ADD_CASE_FAILED(split_num_not_equal_size_split_num, DT_INT64, int64_t, split_dim0, 0)

ADD_CASE_FAILED(split_num_not_equal_size_split_num, DT_INT32, int32_t, split_dim0, 1)

ADD_CASE_FAILED(split_num_not_equal_size_split_num, DT_INT16, int16_t, split_dim2, 2)

TEST_F(TEST_SPLITV_UT, TestSplitV_EMPTY_CASE1) {
  double input[1] = {0};
  double output1[1] = {0};
  double output2[1] = {0};
  int64_t size_split[2] = {1, 1};
  vector<int64_t> input_shape = {0, 2, 2};
  vector<int64_t> splits_shape = {2};
  vector<int64_t> out1_shape = {0, 1, 2};
  vector<int64_t> out2_shape = {0, 1, 2};
  int32_t split_dim = 1;

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
  node.Input({"x", DT_DOUBLE, input_shape, input})
      .Input({"size_splits", DT_INT64, splits_shape, size_split})
      .Input({"split_dim", DT_INT32, {}, &split_dim})
      .Attr("num_split", 2)
      .Output({"y1", DT_DOUBLE, out1_shape, output1})
      .Output({"y2", DT_DOUBLE, out2_shape, output2});
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPLITV_UT, TestSplitV_EMPTY_CASE2) {
  double input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  double output1[1] = {0};
  double output2[8] = {0};
  int64_t size_split[2] = {0, 2};
  vector<int64_t> input_shape = {2, 2, 2};
  vector<int64_t> splits_shape = {2};
  vector<int64_t> out1_shape = {2, 0, 2};
  vector<int64_t> out2_shape = {2, 2, 2};
  int32_t split_dim = 1;

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
  node.Input({"x", DT_DOUBLE, input_shape, input})
      .Input({"size_splits", DT_INT64, splits_shape, size_split})
      .Input({"split_dim", DT_INT32, {}, &split_dim})
      .Attr("num_split", 2)
      .Output({"y1", DT_DOUBLE, out1_shape, output1})
      .Output({"y2", DT_DOUBLE, out2_shape, output2});
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}