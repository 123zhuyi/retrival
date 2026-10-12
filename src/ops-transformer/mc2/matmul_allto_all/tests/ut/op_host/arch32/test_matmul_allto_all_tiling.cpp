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
 * \file test_matmul_allto_all_tiling.cpp
 * \brief tiling ut
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace MatmulAlltoAllUT {

using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "MatmulAlltoAll";

struct MatmulAlltoAllTestParam {
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

    // attrs
    std::string groupAttr;
    int64_t worldSizeAttr;
    int64_t alltoAllAxesAttr;
    int64_t yDtypeAttr;
    int64_t x1QuantModeAttr;
    int64_t x2QuantModeAttr;
    int64_t commQuantModeAttr;
    int64_t commQuantDtypeAttr;
    bool transposex1Attr;
    bool transposex2Attr;
    int64_t groupSizeAttr;
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
static MatmulAlltoAllTestParam g_testCases[] = {
    // legal
    {"matmul_alltoall_case_normal_dtype_float16",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {128, 256}, ge::DT_FLOAT16,ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128},ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_SUCCESS,
    0UL, "", {16867328}, 0},
    
    {"matmul_alltoall_case_bigshape_4p",
    {28543, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {3072, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {114172, 2304}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 4, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_SUCCESS, 
    0UL, "", {1068986368}, 0},

    {"matmul_alltoall_case_bigshape_8p",
    {28543, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {3072, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {228344, 1152}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 8, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_SUCCESS,
    0UL, "", {1068986368}, 0},

    {"matmul_alltoall_case_bigshape_16p",
    {28543, 3072}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {3072, 9216}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {456688, 576}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 16, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_SUCCESS,
    0UL, "", {1068986368}, 0},
    
    // illegal
    {"matmul_alltoall_case_illegal_group_empty",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_world_size_invalid",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 3, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_x1_quant_mode_invalid",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 1, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_x1_dtype_float",
    {88, 128}, ge::DT_FLOAT, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_bias_dtype_mismatch",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_x1_scale_not_null",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT, ge::FORMAT_ND,
    {1}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_k_mismatch",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 64}, ge::DT_BF16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_bias_shape_wrong",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {88}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_y_shape_wrong",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_x1_1d",
    {128}, ge::DT_BF16, ge::FORMAT_ND,
    {256, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_x2_3d",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {128, 256, 16}, ge::DT_BF16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {}, ge::DT_BF16, ge::FORMAT_ND,
    {176, 128}, ge::DT_BF16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_m_zero",
    {0, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,   
    {128, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {0, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16777216}, 0},

    {"matmul_alltoall_case_n_zero",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {128, 0}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {88, 0}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16777216}, 0},

    {"matmul_alltoall_case_k_zero",
    {88, 0}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {0, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16867328}, 0},

    {"matmul_alltoall_case_k_zero_transx2",
    {88, 0}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256, 0}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16867328}, 0},

    {"matmul_alltoall_case_n_zero_transx2",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {0, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {88, 0}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, true, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16777216}, 0},

    {"matmul_alltoall_case_illegal_k_too_large",
    {88, 65536}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    {65536, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_n_not_divisible",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {128, 257}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {257}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {88, 257}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_transpose_x1_true",
    {128, 88}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {128, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {256}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, true, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {6867328}, 0},

    {"matmul_alltoall_case_illegal_x1_k_empty",
    {88, 0}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {0, 256}, ge::DT_FLOAT16,ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128},ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16867328}, 0},

    {"matmul_alltoall_case_illegal_x1_format_not_nd",
    {88, 12}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,
    {12, 256}, ge::DT_FLOAT16,ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128},ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16867328}, 0},

    {"matmul_alltoall_case_illegal_x2_format_not_nd",
    {88, 12}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {12, 256}, ge::DT_FLOAT16,ge::FORMAT_FRACTAL_NZ,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128},ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16867328}, 0},

    {"matmul_alltoall_case_illegal_output_format_not_nd",
    {88, 12}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {12, 256}, ge::DT_FLOAT16,ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128},ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16867328}, 0},

    {"matmul_alltoall_case_illegal_bias_format_not_nd",
    {88, 12}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {12, 256}, ge::DT_FLOAT16,ge::FORMAT_ND,
    {256}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128},ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16867328}, 0},

    {"matmul_alltoall_case_illegal_groupName_len_over128",
    {88, 128}, ge::DT_BF16, ge::FORMAT_ND,
    {128, 256}, ge::DT_BF16,ge::FORMAT_ND,
    {}, ge::DT_FLOAT, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128},ge::DT_BF16, ge::FORMAT_ND,
    "012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345671", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_FAILED,
    0UL, "", {16867328}, 0},

};

// setup & teardown
class MatmulAlltoAllA3TilingTest : public testing::TestWithParam<MatmulAlltoAllTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAlltoAllA3TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAlltoAllA3TilingTest TearDown." << std::endl;
    }
};

// ut
static void TestOneParamCase(const MatmulAlltoAllTestParam &param)
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
        {{yShape, param.youtputDtype, param.youtputFormat}});

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
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantDtypeAttr)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposex1Attr)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposex2Attr)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSizeAttr)}});
    //
    struct MatmulAlltoAllCompileInfo {} compileInfoInstance; // 创建一个实例
    void* pCompileInfo = &compileInfoInstance;
    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, pCompileInfo,
                                              param.socVersion);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                        param.expectWorkspaces, param.mc2TilingDataReservedLen);                                          
}

static void ThreadFunction(const MatmulAlltoAllTestParam *testCases, size_t caseNum, size_t threadIdx, size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        TestOneParamCase(testCases[idx]);
    }
}

static void TestExecMultiThread(const MatmulAlltoAllTestParam *testCases, size_t testCaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
    }
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(MatmulAlltoAllA3TilingTest, GeneralCasesTest)
{
    TestOneParamCase(GetParam());
}

TEST_F(MatmulAlltoAllA3TilingTest, GeneralCasesMultiThread)
{
    TestExecMultiThread(g_testCases, sizeof(g_testCases) / sizeof(MatmulAlltoAllTestParam), 1);
}

INSTANTIATE_TEST_CASE_P(MatmulAlltoAllTilingUT, MatmulAlltoAllA3TilingTest, testing::ValuesIn(g_testCases));

} // namespace
