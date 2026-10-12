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
#include <cstdint>
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

class TEST_CASE_CONDITION_UT : public testing::Test {};

#define CREATE_NODEDEF_WITH_ALGORITHM(shapes, dataTypes, datas, algorithm) \
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();        \
    NodeDefBuilder(nodeDef.get(), "CaseCondition", "CaseCondition")        \
        .Input({"x", dataTypes[0], shapes[0], datas[0]})                   \
        .Output({"y", dataTypes[1], shapes[1], datas[1]})                  \
        .Attr("algorithm", std::string(algorithm))

#define CREATE_NODEDEF(shapes, dataTypes, datas) CREATE_NODEDEF_WITH_ALGORITHM(shapes, dataTypes, datas, "LU")

TEST_F(TEST_CASE_CONDITION_UT, OUT_0_SUCCESS)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {}};
    int32_t input[3] = {3, 1, 2};
    int32_t output[1] = {-1};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int32_t outputExp[1] = {0};
    EXPECT_EQ(CompareResult(output, outputExp, 1), true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_1_SUCCESS)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {}};
    int32_t input[3] = {2, 2, 2};
    int32_t output[1] = {-1};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int32_t outputExp[1] = {1};
    EXPECT_EQ(CompareResult(output, outputExp, 1), true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_2_SUCCESS)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {}};
    int32_t input[3] = {3, 2, 2};
    int32_t output[1] = {-1};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int32_t outputExp[1] = {2};
    EXPECT_EQ(CompareResult(output, outputExp, 1), true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_3_SUCCESS)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {}};
    int32_t input[3] = {2, 3, 2};
    int32_t output[1] = {-1};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int32_t outputExp[1] = {3};
    EXPECT_EQ(CompareResult(output, outputExp, 1), true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_4_SUCCESS)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {}};
    int32_t input[3] = {2, 2, 1};
    int32_t output[1] = {-1};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int32_t outputExp[1] = {4};
    EXPECT_EQ(CompareResult(output, outputExp, 1), true);
}

TEST_F(TEST_CASE_CONDITION_UT, UINT64_SUCCESS)
{
    vector<DataType> dataTypes = {DT_UINT64, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {}};
    uint64_t input[3] = {2, 2, 3};
    int32_t output[1] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int32_t outputExp[1] = {5};
    EXPECT_EQ(CompareResult(output, outputExp, 1), true);
}

TEST_F(TEST_CASE_CONDITION_UT, INT64_SUCCESS)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {}};
    int64_t input[3] = {INT64_MAX, INT64_MAX, 1};
    int32_t output[1] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int32_t outputExp[1] = {4};
    EXPECT_EQ(CompareResult(output, outputExp, 1), true);
}

TEST_F(TEST_CASE_CONDITION_UT, UNSUPPORTED_TYPE_FAIL)
{
    vector<DataType> dataTypes = {DT_UINT32, DT_UINT32};
    vector<vector<int64_t>> shapes = {{3}, {}};
    uint32_t input[3] = {2, 2, 3};
    uint32_t output[1] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_CASE_CONDITION_UT, OUTPUT_TYPE_INVALID_FAIL)
{
    vector<DataType> dataTypes = {DT_INT32, DT_UINT32};
    vector<vector<int64_t>> shapes = {{3}, {}};
    int32_t input[3] = {2, 2, 1};
    uint32_t output[1] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_CASE_CONDITION_UT, INPUT_SHAPE_INVALID_FAIL)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {}};
    int32_t input[2] = {2, 2};
    int32_t output[1] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_CASE_CONDITION_UT, ALGORITHM_INVALID_FAIL)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {}};
    int32_t input[3] = {2, 2, 1};
    int32_t output[1] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF_WITH_ALGORITHM(shapes, dataTypes, datas, "QR");
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}
