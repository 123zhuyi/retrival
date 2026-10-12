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

class TEST_SPLITD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, split_dim, num_split)     \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();          \
  NodeDefBuilder node(node_def.get(), "SplitD", "SplitD");                  \
  node.Input({"x", data_types[0], shapes[0], datas[0]})                 \
      .Attr("split_dim", split_dim)                                         \
      .Attr("num_split", num_split);                                        \
  for(int i = 0; i < num_split; i++) {                                      \
    node.Output({"y", data_types[i + 1], shapes[i + 1], datas[i + 1]});     \
  }

#define ADD_CASE(case_name, aicpu_type, base_type, split_dim, num_split)                                         \
  TEST_F(TEST_SPLITD_UT, TestSplitD_##case_name##_##aicpu_type) {                                                \
    if(num_split == 1) {                                                                                         \
      vector<DataType> data_types = {aicpu_type, aicpu_type};                                                    \
      vector<vector<int64_t>> shapes = {{2, 2, 2}, {2, 2, 2}};                                                   \
      base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                              \
                            (base_type)5, (base_type)6, (base_type)7, (base_type)8};                             \
      base_type output[8] = {(base_type)0};                                                                      \
      vector<void *> datas = {(void *)input, (void *)output};                                                    \
      CREATE_NODEDEF(shapes, data_types, datas, split_dim, 1);                                                   \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                              \
      base_type expect_out[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                         \
                                 (base_type)5, (base_type)6, (base_type)7, (base_type)8};                        \
      EXPECT_EQ(CompareResult<base_type>(output, expect_out, 8), true);                                          \
    } else if(split_dim == 1){                                                                                   \
      vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};                                        \
      vector<vector<int64_t>> shapes = {{2, 2, 2}, {2, 1, 2},{2, 1, 2}};                                         \
      base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                              \
                            (base_type)5, (base_type)6, (base_type)7, (base_type)8};                             \
      base_type output1[4] = {(base_type)0};                                                                     \
      base_type output2[4] = {(base_type)0};                                                                     \
      vector<void *> datas = {(void *)input, (void *)output1, (void *)output2};                                  \
      CREATE_NODEDEF(shapes, data_types, datas, split_dim, 2);                                                   \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                              \
      base_type expect_out1[4] = {(base_type)1, (base_type)2, (base_type)5, (base_type)6};                       \
      base_type expect_out2[4] = {(base_type)3, (base_type)4, (base_type)7, (base_type)8};                       \
      EXPECT_EQ(CompareResult<base_type>(output1, expect_out1, 4), true);                                        \
      EXPECT_EQ(CompareResult<base_type>(output2, expect_out2, 4), true);                                        \
    } else if(split_dim == 0) {                                                                                  \
      vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};                                        \
      vector<vector<int64_t>> shapes = {{2, 2, 2}, {1, 2, 2},{1, 2, 2}};                                         \
      base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                              \
                            (base_type)5, (base_type)6, (base_type)7, (base_type)8};                             \
      base_type output1[4] = {(base_type)0};                                                                     \
      base_type output2[4] = {(base_type)0};                                                                     \
      vector<void *> datas = {(void *)input, (void *)output1, (void *)output2};                                  \
      CREATE_NODEDEF(shapes, data_types, datas, split_dim, 2);                                                   \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                              \
      base_type expect_out1[4] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4};                       \
      base_type expect_out2[4] = {(base_type)5, (base_type)6, (base_type)7, (base_type)8};                       \
      EXPECT_EQ(CompareResult<base_type>(output1, expect_out1, 4), true);                                        \
      EXPECT_EQ(CompareResult<base_type>(output2, expect_out2, 4), true);                                        \
    }                                                                                                            \
  }

#define ADD_CASE_FAILED(case_name, aicpu_type, base_type, split_dim, num_split)                                  \
  TEST_F(TEST_SPLITD_UT, TestSplitD_##case_name##_##aicpu_type) {                                                \
      vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};                                        \
      vector<vector<int64_t>> shapes = {{2, 2, 2}, {2, 1, 2}, {2, 1, 2}};                                        \
      base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                              \
                            (base_type)5, (base_type)6, (base_type)7, (base_type)8};                             \
      base_type output1[4] = {(base_type)0};                                                                     \
      base_type output2[4] = {(base_type)0};                                                                     \
      vector<void *> datas = {(void *)input, (void *)output1, (void *)output2};                                  \
      CREATE_NODEDEF(shapes, data_types, datas, split_dim, num_split);                                           \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                                                   \
  }

#define ADD_CASE_FAILED_1(case_name, aicpu_type, base_type, split_dim, num_split)                                \
  TEST_F(TEST_SPLITD_UT, TestSplitD_##case_name##_##aicpu_type) {                                                \
      vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type, aicpu_type, aicpu_type};                \
      vector<vector<int64_t>> shapes = {{2, 2, 2}, {1, 2, 2}, {1, 2, 2}, {1, 2, 2}, {1, 2, 2}};                  \
      base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                              \
                            (base_type)5, (base_type)6, (base_type)7, (base_type)8};                             \
      base_type output1[4] = {(base_type)0};                                                                     \
      base_type output2[4] = {(base_type)0};                                                                     \
      base_type output3[4] = {(base_type)0};                                                                     \
      base_type output4[4] = {(base_type)0};                                                                     \
      vector<void *> datas = {(void *)input, (void *)output1, (void *)output2, (void *)output3, (void *)output4};\
      CREATE_NODEDEF(shapes, data_types, datas, split_dim, num_split);                                           \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                                                   \
  }

TEST_F(TEST_SPLITD_UT, TestSplitD_EMPTY_INPUT) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{0, 2, 2}, {0, 2, 2}};
    float input[1] = {(float)1};
    float output[1] = {(float)0};
    vector<void *> datas = {(void *)input, (void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, 0, 1);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                                                                 \
}

int64_t dim1 = 1;
int64_t dim0 = 0;
int64_t dim2 = 4;

ADD_CASE(two_split_with_dim_1, DT_FLOAT, float, dim1, 2)

ADD_CASE(two_split_with_dim_1, DT_DOUBLE, double, dim1, 2)

ADD_CASE(two_split_with_dim_1, DT_FLOAT16, Eigen::half, dim1, 2)

ADD_CASE(two_split_with_dim_0, DT_INT32, int32_t, dim0, 2)

ADD_CASE(two_split_with_dim_0, DT_INT16, int16_t, dim0, 2)

ADD_CASE(two_split_with_dim_0, DT_INT64, int64_t, dim0, 2)

ADD_CASE(two_split_with_dim_0, DT_INT8, int8_t, dim0, 2)

ADD_CASE(one_split_with_dim_1, DT_BOOL, bool, dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT8, uint8_t, dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT16, uint16_t, dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT32, uint32_t, dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT64, uint64_t, dim1, 1)

ADD_CASE_FAILED(split_num_not_equal_size_split_num, DT_INT64, int64_t, dim0, 0)

ADD_CASE_FAILED_1(split_num_not_equal_size_split_num, DT_INT32, int32_t, dim0, 4)

ADD_CASE_FAILED(split_num_not_equal_size_split_num, DT_INT16, int16_t, dim2, 2)

ADD_CASE_FAILED(split_num_not_equal_size_split_num, DT_COMPLEX64, std::complex<float>, dim0, 1)