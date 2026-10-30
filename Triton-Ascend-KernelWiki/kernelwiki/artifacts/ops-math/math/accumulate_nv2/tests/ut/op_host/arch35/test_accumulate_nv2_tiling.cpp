/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace optiling {
struct AccumulateNV2CompileInfo {};
} // namespace optiling

class AccumulateNV2Tiling : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AccumulateNV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AccumulateNV2Tiling TearDown" << std::endl;
    }
};

// 3x float32 inputs, shape {8, 1024}, totalNum=8192 > 1024 => double buffer, key=210000000
TEST_F(AccumulateNV2Tiling, float32_double_buffer)
{
    optiling::AccumulateNV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "AccumulateNV2",
        {
            {{{8, 1024}, {8, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 1024}, {8, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 1024}, {8, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 1024}, {8, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 210000000UL, std::vector<size_t>{0});
}

// 2x float16 inputs, shape {16}, totalNum=16 <= 1024 => single buffer, key=210010000
TEST_F(AccumulateNV2Tiling, float16_single_buffer)
{
    optiling::AccumulateNV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "AccumulateNV2",
        {
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 210010000UL, std::vector<size_t>{0});
}

// 1x float32 input, shape {1}, totalNum=1 <= 1024 => single buffer, key=210010000
TEST_F(AccumulateNV2Tiling, single_input_scalar_like)
{
    optiling::AccumulateNV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "AccumulateNV2",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 210010000UL, std::vector<size_t>{0});
}

// 4x int32 inputs, shape {2048}, totalNum=2048 > 1024 => double buffer, key=210000000
TEST_F(AccumulateNV2Tiling, int32_double_buffer)
{
    optiling::AccumulateNV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "AccumulateNV2",
        {
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 210000000UL, std::vector<size_t>{0});
}

// 2x int8 inputs, shape {32}, totalNum=32 <= 1024 => single buffer, key=210010000
TEST_F(AccumulateNV2Tiling, int8_single_buffer)
{
    optiling::AccumulateNV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "AccumulateNV2",
        {
            {{{32}, {32}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{32}, {32}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{32}, {32}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 210010000UL, std::vector<size_t>{0});
}

// 2x uint8 inputs, shape {64}, totalNum=64 <= 1024 => single buffer, key=210010000
TEST_F(AccumulateNV2Tiling, uint8_single_buffer)
{
    optiling::AccumulateNV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "AccumulateNV2",
        {
            {{{64}, {64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{64}, {64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 210010000UL, std::vector<size_t>{0});
}

// 3x float32 inputs, shape {4, 8, 16}, 3D multi-dim, totalNum=512 <= 1024 => single buffer
TEST_F(AccumulateNV2Tiling, float32_3d_single_buffer)
{
    optiling::AccumulateNV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "AccumulateNV2",
        {
            {{{4, 8, 16}, {4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 8, 16}, {4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 8, 16}, {4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 8, 16}, {4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 210010000UL, std::vector<size_t>{0});
}

// scalar input (dimNum=0), hits EnsureNotScalar scalar→vec path
TEST_F(AccumulateNV2Tiling, scalar_input)
{
    optiling::AccumulateNV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "AccumulateNV2",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 210010000UL, std::vector<size_t>{0});
}

// zero-element tensor, shape {0}, hits totalNum<=0 early-exit path
TEST_F(AccumulateNV2Tiling, zero_element_tensor)
{
    optiling::AccumulateNV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "AccumulateNV2",
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 210010000UL, std::vector<size_t>{0});
}
