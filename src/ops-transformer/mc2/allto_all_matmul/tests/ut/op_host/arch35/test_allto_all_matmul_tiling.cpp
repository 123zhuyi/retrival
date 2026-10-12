
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_allto_all_matmul_tiling.cpp
 * \brief hosttiling ut
 */
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace AlltoAllMatmulUT {

using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "AlltoAllMatmul";

struct AlltoAllMatmulTestParam {
    std::string caseName;
    // input
    // x1
    std::initializer_list<int64_t> x1Shape;
    ge::DataType x1Dtype;
    ge::Format x1Format;

    // x2
    std::initializer_list<int64_t> x2Shape;
    ge::DataType x2Dtype;
    ge::Format x2Format;

    // bias
    std::initializer_list<int64_t> biasShape;
    ge::DataType biasDtype;
    ge::Format biasFormat;

    // x1_scale
    std::initializer_list<int64_t> x1ScaleShape;
    ge::DataType x1ScaleDtype;
    ge::Format x1ScaleFormat;

    // x2_scale
    std::initializer_list<int64_t> x2ScaleShape;
    ge::DataType x2ScaleDtype;
    ge::Format x2ScaleFormat;

    // comm_scale
    std::initializer_list<int64_t> commScaleShape;
    ge::DataType commScaleDtype;
    ge::Format commScaleFormat;

    // x1_offset
    std::initializer_list<int64_t> x1OffsetShape;
    ge::DataType x1OffsetDtype;
    ge::Format x1OffsetFormat;

    // x2_offset
    std::initializer_list<int64_t> x2OffsetShape;
    ge::DataType x2OffsetDtype;
    ge::Format x2OffsetFormat;

    // output
    // y
    std::initializer_list<int64_t> yShape;
    ge::DataType youtputDtype;
    ge::Format youtputFormat;
    // all2allout
    std::initializer_list<int64_t> all2allOutShape;
    ge::DataType outputDtype;
    ge::Format outputFormat;

    // attrs
    std::string groupAttr;
    int64_t worldSizeAttr;
    int64_t alltoAllAxesAttr;
    int64_t yDtypeAttr;
    int64_t x1QuantModeAttr;
    int64_t x2QuantModeAttr;
    int64_t commQuantModeAttr;
    int64_t x1QuantDtypeAttr;
    int64_t commQuantDtypeAttr;
    bool transposex1Attr;
    bool transposex2Attr;
    int64_t groupSizeAttr;
    bool alltoalloutFlag;
    // soc version
    std::string socVersion;
    // expert result
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;
};

// ut/
// expectWorkspaces = 16 * 1024 * 1024
// tilingDataReservedLen = 43tilingDatamc2InitTilingmc2CcTiling
static AlltoAllMatmulTestParam testCases[] = {
    // legal
    {"alltoall_matmul_case_dtype_float16",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true, 
     "3510", 
     ge::GRAPH_SUCCESS, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_dtype_bf16_transposex2",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, true, 0, false,
    "3510",
    ge::GRAPH_SUCCESS, 16UL, "", {16822272}, 0},

    {"alltoall_matmul_case_normalshape_4p",
    {114172, 2304}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 4, 0, 0, 0, 0, 0, 0, 0, false, false, 0, false,
    "3510",
    ge::GRAPH_SUCCESS, 0UL, "", {1068986368}, 0},

    {"alltoall_matmul_case_bigshape_8p",
    {228344, 1152}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 8, 0, 0, 0, 0, 0, 0, 0, false, false, 0, false,
    "3510",
    ge::GRAPH_SUCCESS, 0UL, "", {1068986368}, 0},

    {"alltoall_matmul_case_bigshape_16p",
    {456688, 2304}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {36864, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 16, 0, 0, 0, 0, 0, 0, 0, false, false, 0, false,
    "3510",
    ge::GRAPH_SUCCESS, 0UL, "", {4225613824}, 0},

     // illegal
    {"alltoall_matmul_case_illegal_group_empty",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {6867328}, 0},

   {"alltoall_matmul_case_illegal_world_size_invalid",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 3, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {6867328}, 0},

   {"alltoall_matmul_case_illegal_x1_quant_mode_invalid",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 1, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {6867328}, 0},

  {"alltoall_matmul_case_illegal_x1_dtype_float",
    {88, 128}, ge::DT_FLOAT, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {6867328}, 0},

  {"alltoall_matmul_case_illegal_bias_dtype_mismatch",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {6867328}, 0},

  {"alltoall_matmul_case_illegal_x1_scale_not_null",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {1}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {6867328}, 0},

  {"alltoall_matmul_case_illegal_k_mismatch",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 64}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {6867328}, 0},

  {"alltoall_matmul_case_illegal_bias_shape_wrong",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {88}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {6867328}, 0},

  {"alltoall_matmul_case_illegal_y_shape_wrong",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {6867328}, 0},

  {"alltoall_matmul_case_illegal_transpose_x1_true",
    {128, 88}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {128, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, 0, true, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {6867328}, 0},

    {"alltoall_matmul_case_illegal_Empty_x1_m",
    {0, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_Empty_x1_k",
    {88, 0}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_Empty_x1_not_2d",
    {88}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_Empty_x1_format_not_nd",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_Empty_x2_empty_k",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {0, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_Empty_x2_empty_n",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 0}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_x2_not_2d",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256, 3}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_x2_dtype_invalid",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_x2_format_not_nd",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_y_format_not_nd",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_y_dtype_not_same_with_x",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_BF16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_alltoall_out_dtype_not_same_with_x",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_alltoall_out_format_not_nd",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_alltoall_out_shape_not_valid",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {1, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_bias_format_not_nd",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_group_over128",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345671", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_BS_not_divide_world_size",
    {85, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_k_too_large",
    {88, 32768}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {65536, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 65536}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"alltoall_matmul_case_illegal_m_too_large",
    {4294967295, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
    "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    // KC量化UT
    {"alltoall_matmul_kc_case_legal_nobias_float16_float8e4m3fn",
    {57086, 1536}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {3072, 9216}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 2, 0, 35, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_SUCCESS, 34UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_legal_nobias_float16_float8e5m2",
    {57086, 1536}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {3072, 9216}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 2, 0, 35, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_SUCCESS, 34UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_legal_nobias_bf16_float8e4m3fn",
    {57086, 1536}, ge::DT_BF16, ge::FORMAT_ND,
    {3072, 9216}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 2, 0, 35, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_SUCCESS, 34UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_legal_nobias_bf16_float8e5m2_x2trans",
    {57086, 1536}, ge::DT_BF16, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 2, 0, 35, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_SUCCESS, 50UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_illegal_bf16_float8e5m2_x1_true",
    {1536, 57086}, ge::DT_BF16, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 2, 0, 0, 0, true, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 33UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_illegal_nobias_bf16_float8e5m2_x2trans_x2scale_shape_error",
    {57086, 1536}, ge::DT_BF16, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {3072}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 2, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 49UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_illegal_nobias_bf16_float8e5m2_x2trans_x2scale_dtype_error",
    {57086, 1536}, ge::DT_BF16, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 2, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 49UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_illegal_nobias_bf16_float8e5m2_x2trans_x1quantmode_error",
    {57086, 1536}, ge::DT_BF16, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 3, 2, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 49UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_illegal_nobias_bf16_float8e5m2_x2trans_x2quantmode_error",
    {57086, 1536}, ge::DT_BF16, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 1, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 49UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_legal_nobias_bf16_float8e5m2_x2trans_x2scale_shape_error",
    {57086, 1536}, ge::DT_BF16, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 2, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 49UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_legal_nobias_bf16_float8e4m3fn_x1_shape_error",
    {57086, 1536, 1}, ge::DT_BF16, ge::FORMAT_ND,
    {3072, 9216}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 2, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 33UL, "", {279943680}, 0},

    {"alltoall_matmul_kc_case_legal_nobias_bf16_float8e4m3fn_x2_shape_error",
    {57086, 1536}, ge::DT_BF16, ge::FORMAT_ND,
    {3072, 9216, 1}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 7, 2, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 33UL, "", {279943680}, 0},

    // MX量化
    {"alltoall_matmul_mx_case_legal_nobias_float8e4m3fn_float8e4m3fn",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_SUCCESS, 51UL, "", {109941760}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_SUCCESS, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e5m2_float8e4m3fn",
    {57086, 1536}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_SUCCESS, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x1shape_error",
    {28543, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x2shape_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x1scale_shape_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {24, 57086, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x2scale_shape_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_x1scale_dtype_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x2scale_dtype_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x1quantmode_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 5, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x2quantmode_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 5, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_nobias_float8e4m3fn_float8e4m3fn_x2scale_shapedim_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {48, 9216}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {104461312}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x1scale_shapedim_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_bias_shape_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {3072}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x1_scale_null",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x2trans_alltoallout_dtype_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_bias_float8e4m3fn_float8e4m3fn_x2trans_alltoallout_shape_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 0}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109978624}, 0},

    {"alltoall_matmul_mx_case_legal_nobias_float8e4m3fn_float8e4m3fn_x2trans_error",
    {57086, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 48, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, false, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109941760}, 0},

    {"alltoall_matmul_mx_case_illegal_x1_k_not_divide_by_64",
    {57086, 96}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {9216, 192}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {57086, 2, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {9216, 3, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {28543, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {28543, 3072}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    "group", 2, 0, 0, 6, 6, 0, 0, 0, false, true, 0, true,
    "3510",
    ge::GRAPH_FAILED, 51UL, "", {109941760}, 0},
};

// setup & teardown
class AlltoAllMatmulArch35TilingTest : public testing::TestWithParam<AlltoAllMatmulTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulArch35TilingTest TearDown." << std::endl;
    }
};

struct AlltoAllMatmulCompileInfo {
} compileInfo;

// ut
static void TestOneParamCase(const AlltoAllMatmulTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    //
    //  Shape  tensor  shape  gert::StorageShape
    gert::StorageShape x1Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape x2Shape = {param.x2Shape, param.x2Shape};
    gert::StorageShape biasShape = {param.biasShape, param.biasShape};
    gert::StorageShape x1ScaleShape = {param.x1ScaleShape, param.x1ScaleShape};
    gert::StorageShape x2ScaleShape = {param.x2ScaleShape, param.x2ScaleShape};
    gert::StorageShape commScaleShape = {param.commScaleShape, param.commScaleShape};
    gert::StorageShape x1OffsetShape = {param.x1OffsetShape, param.x1OffsetShape};
    gert::StorageShape x2OffsetShape = {param.x2OffsetShape, param.x2OffsetShape};

    gert::StorageShape yShape = {param.yShape, param.yShape};
    gert::StorageShape all2allOutShape = {param.all2allOutShape, param.all2allOutShape};

    //  input tensor
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{x1Shape, param.x1Dtype, param.x1Format},
         {x2Shape, param.x2Dtype, param.x2Format},
         {biasShape, param.biasDtype, param.biasFormat},
         {x1ScaleShape, param.x1ScaleDtype, param.x1ScaleFormat},
         {x2ScaleShape, param.x2ScaleDtype, param.x2ScaleFormat},
         {commScaleShape, param.commScaleDtype, param.commScaleFormat},
         {x1OffsetShape, param.x1OffsetDtype, param.x1OffsetFormat},
         {x2OffsetShape, param.x2OffsetDtype, param.x2OffsetFormat}});

    //  output tensor
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.youtputDtype, param.youtputFormat}, {all2allOutShape, param.outputDtype, param.outputFormat}});

    //  attributes
    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.groupAttr)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.worldSizeAttr)},
         {"all2all_axes",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.alltoAllAxesAttr))},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.yDtypeAttr))},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantModeAttr)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2QuantModeAttr)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantModeAttr)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantDtypeAttr)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantDtypeAttr)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposex1Attr)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposex2Attr)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSizeAttr)},
         {"alltoallout_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.alltoalloutFlag)}});
    //
    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.socVersion);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                        param.expectWorkspaces, param.mc2TilingDataReservedLen);                                          
}

static void ThreadFunction(const AlltoAllMatmulTestParam *testCases, size_t caseNum, size_t threadIdx, size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        TestOneParamCase(testCases[idx]);
    }
}

static void TestExecMultiThread(const AlltoAllMatmulTestParam *testCases, size_t testCaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
    }
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(AlltoAllMatmulArch35TilingTest, GeneralCases)
{
    TestOneParamCase(GetParam());
}

TEST_F(AlltoAllMatmulArch35TilingTest, GeneralCasesMultiThread)
{
    TestExecMultiThread(testCases, sizeof(testCases) / sizeof(AlltoAllMatmulTestParam), 1);
}

INSTANTIATE_TEST_CASE_P(AlltoAllMatmulTilingUT, AlltoAllMatmulArch35TilingTest, testing::ValuesIn(testCases));

} // namespace
