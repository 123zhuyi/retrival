
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
     "Ascend910_93",
     ge::GRAPH_SUCCESS, 64UL, "", {16799744}, 0},

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
     "Ascend910_93",
     ge::GRAPH_SUCCESS, 66UL, "", {16822272}, 0},

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
     "Ascend910_93",
     ge::GRAPH_SUCCESS, 64UL, "", {1068986368}, 0},

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
     "Ascend910_93",
     ge::GRAPH_SUCCESS, 64UL, "", {1068986368}, 0},

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
     "Ascend910_93",
     ge::GRAPH_SUCCESS, 64UL, "", {4225613824}, 0},

     // illegal
    {"error-alltoall_matmul_x1_dtype_invalid",
     {88, 128}, ge::DT_INT32, ge::FORMAT_ND,  // x1数据类型非法
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
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_x2_dtype_invalid",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_INT32, ge::FORMAT_ND,  // x2数据类型非法
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_private_format_x1",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_NCHW, // x1数据格式非法
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
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_private_format_x2",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_NCHW, // x2数据格式非法
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_invalid_dim_x1",
     {88, 128, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, // x1维度不为2D
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
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_invalid_dim_x2",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, // x2维度不为2D
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_empty_tensor_x1",
     {88, 0}, ge::DT_FLOAT16, ge::FORMAT_ND, // x1为空tensor，第二维度为0
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
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_empty_tensor_x2_k",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {0, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // x2为空tensor，第一维度为0
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_empty_tensor_x2_N",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 0}, ge::DT_FLOAT16, ge::FORMAT_ND, // x2为空tensor，第二维度为0
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_BS_value_invalid_x1",
     {2147483648, 128}, ge::DT_FLOAT16, ge::FORMAT_ND, // x1BS值超出INT32_MAX
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
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_N_value_invalid_x2",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 2147483648}, ge::DT_FLOAT16, ge::FORMAT_ND, // x2N值超出INT32_MAX
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_outdtype_mismatch",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_INT32, ge::FORMAT_ND, // output数据类型非法
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_private_format_output",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_NCHW, // output数据格式非法
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_invalid_dim_output",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, // output维度不为2D
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_shape_output",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {43, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // output的shape不满足（BS/worldSize，H2的关系）
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_dtype_alltoallout",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_INT32, ge::FORMAT_ND, // all2allout数据类型非法
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_private_format_alltoallout",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_NCHW, // all2allout数据格式非法
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_shape_alltoallout",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 255}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout的shape非法
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_bias_dtype_invalid",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256}, ge::DT_BF16, ge::FORMAT_ND, // bias的数据类型非法
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_private_format_bias",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, // bias的数据格式非法
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_bias_dim_invalid",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // bias的维度不为1D
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_bias_shape_mismatch",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {255}, ge::DT_FLOAT16, ge::FORMAT_ND, // bias的shape非法
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_bias_dtype_mismatch",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256}, ge::DT_INT32, ge::FORMAT_ND, // bias的dtype非法
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_transpose_x1",
     {128, 88}, ge::DT_FLOAT16, ge::FORMAT_ND, // x1转置
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
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_transpose_x2_shape_mismatch",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {255, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // x2转置但shape非法
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_group_extra_long",
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
     "a_very_long_very_long_very_long_very_long_very_long_very_long_very_long_"
     "group_name_exceeding_128_characters_which_should_cause_an_error", // group长度超过128
     2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_group_empty",
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
     "", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true, // group为空
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_world_size_invalid",
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
     "group", 3, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true, // worldsize非法
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_x1_dtype_float",
     {88, 128}, ge::DT_FLOAT, ge::FORMAT_ND, // x1的dtype非法
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
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_bias_invalid1",
     {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
     {256, 256}, ge::DT_BF16, ge::FORMAT_ND,
     {256}, ge::DT_FLOAT16, ge::FORMAT_ND, // bias的dtype非法
     {}, ge::DT_BF16, ge::FORMAT_ND,
     {}, ge::DT_BF16, ge::FORMAT_ND,
     {}, ge::DT_BF16, ge::FORMAT_ND,
     {}, ge::DT_BF16, ge::FORMAT_ND,
     {}, ge::DT_BF16, ge::FORMAT_ND,
     {44, 256}, ge::DT_BF16, ge::FORMAT_ND,
     {44, 256}, ge::DT_BF16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_bias_invalid2",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {256}, ge::DT_FLOAT, ge::FORMAT_ND, // bias的dtype非法
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},

    {"error-alltoall_matmul_k_mismatch",
     {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {255, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // x2的shape不满足(H1*worldSize, H2)
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {44, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, // all2allout
     "group", 2, 0, 0, 0, 0, 0, 0, 0, false, false, 0, true,
     "Ascend910_93",
     ge::GRAPH_FAILED, 0UL, "", {16799744}, 0},
};

// setup & teardown
class AlltoAllMatmulA3TilingTest : public testing::TestWithParam<AlltoAllMatmulTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulA3TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulA3TilingTest TearDown." << std::endl;
    }
};

struct AlltoAllMatmulCompileInfo {
} compileInfoA3;

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
    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfoA3,
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

TEST_P(AlltoAllMatmulA3TilingTest, GeneralCases)
{
    TestOneParamCase(GetParam());
}

TEST_F(AlltoAllMatmulA3TilingTest, GeneralCasesMultiThread)
{
    TestExecMultiThread(testCases, sizeof(testCases) / sizeof(AlltoAllMatmulTestParam), 1);
}

INSTANTIATE_TEST_CASE_P(AlltoAllMatmulTilingUT, AlltoAllMatmulA3TilingTest, testing::ValuesIn(testCases));

} // namespace
