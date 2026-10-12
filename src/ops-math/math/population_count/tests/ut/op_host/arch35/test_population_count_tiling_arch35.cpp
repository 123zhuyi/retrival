/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

struct PopulationCountCompileInfo {};
PopulationCountCompileInfo g_compileInfo;

// dtype is driven by the op def; TilingKey only contains BUFFER_MODE.
constexpr uint64_t KEY_SINGLE = 0;
constexpr uint64_t KEY_DOUBLE = 1;

class PopulationCountTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PopulationCountTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "PopulationCountTilingTest TearDown" << std::endl; }
};

// INT16 small tensor — single buffer (totalNum <= 1024)
TEST_F(PopulationCountTilingTest, tiling_int16_small_single_buffer)
{
    gert::TilingContextPara tilingContextPara("PopulationCount", {{{{128}, {128}}, ge::DT_INT16, ge::FORMAT_ND}},
                                              {{{{128}, {128}}, ge::DT_UINT8, ge::FORMAT_ND}}, &g_compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, KEY_SINGLE, expectWorkspaces);
}

// UINT16 small tensor — single buffer
TEST_F(PopulationCountTilingTest, tiling_uint16_small_single_buffer)
{
    gert::TilingContextPara tilingContextPara("PopulationCount", {{{{256}, {256}}, ge::DT_UINT16, ge::FORMAT_ND}},
                                              {{{{256}, {256}}, ge::DT_UINT8, ge::FORMAT_ND}}, &g_compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, KEY_SINGLE, expectWorkspaces);
}

// INT16 large tensor — double buffer (totalNum > 1024)
TEST_F(PopulationCountTilingTest, tiling_int16_large_double_buffer)
{
    gert::TilingContextPara tilingContextPara("PopulationCount",
                                              {{{{8, 1024}, {8, 1024}}, ge::DT_INT16, ge::FORMAT_ND}},
                                              {{{{8, 1024}, {8, 1024}}, ge::DT_UINT8, ge::FORMAT_ND}}, &g_compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, KEY_DOUBLE, expectWorkspaces);
}

// UINT16 large tensor — double buffer
TEST_F(PopulationCountTilingTest, tiling_uint16_large_double_buffer)
{
    gert::TilingContextPara tilingContextPara(
        "PopulationCount", {{{{32, 64, 16}, {32, 64, 16}}, ge::DT_UINT16, ge::FORMAT_ND}},
        {{{{32, 64, 16}, {32, 64, 16}}, ge::DT_UINT8, ge::FORMAT_ND}}, &g_compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, KEY_DOUBLE, expectWorkspaces);
}

// Empty tensor — totalNum = 0
TEST_F(PopulationCountTilingTest, tiling_empty_tensor)
{
    gert::TilingContextPara tilingContextPara("PopulationCount", {{{{0}, {0}}, ge::DT_INT16, ge::FORMAT_ND}},
                                              {{{{0}, {0}}, ge::DT_UINT8, ge::FORMAT_ND}}, &g_compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, KEY_SINGLE, expectWorkspaces);
}

// Scalar input (0-dim) — treated as 1 element, single buffer
TEST_F(PopulationCountTilingTest, tiling_scalar_input)
{
    gert::TilingContextPara tilingContextPara("PopulationCount", {{{{}, {}}, ge::DT_UINT16, ge::FORMAT_ND}},
                                              {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}}, &g_compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, KEY_SINGLE, expectWorkspaces);
}

// Boundary: exactly at double buffer threshold (1024) — single buffer
TEST_F(PopulationCountTilingTest, tiling_at_threshold_boundary)
{
    gert::TilingContextPara tilingContextPara("PopulationCount", {{{{1024}, {1024}}, ge::DT_INT16, ge::FORMAT_ND}},
                                              {{{{1024}, {1024}}, ge::DT_UINT8, ge::FORMAT_ND}}, &g_compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, KEY_SINGLE, expectWorkspaces);
}

// Boundary: just above double buffer threshold (1025) — double buffer
TEST_F(PopulationCountTilingTest, tiling_above_threshold_boundary)
{
    gert::TilingContextPara tilingContextPara("PopulationCount", {{{{1025}, {1025}}, ge::DT_INT16, ge::FORMAT_ND}},
                                              {{{{1025}, {1025}}, ge::DT_UINT8, ge::FORMAT_ND}}, &g_compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, KEY_DOUBLE, expectWorkspaces);
}

// High-dimensional: 5D tensor (720 elements, single buffer)
TEST_F(PopulationCountTilingTest, tiling_5d_tensor)
{
    gert::TilingContextPara tilingContextPara(
        "PopulationCount", {{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_UINT16, ge::FORMAT_ND}},
        {{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_UINT8, ge::FORMAT_ND}}, &g_compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, KEY_SINGLE, expectWorkspaces);
}

// Very large tensor — multi-core splitting, double buffer
TEST_F(PopulationCountTilingTest, tiling_very_large_multicore)
{
    gert::TilingContextPara tilingContextPara(
        "PopulationCount", {{{{1024, 1024}, {1024, 1024}}, ge::DT_INT16, ge::FORMAT_ND}},
        {{{{1024, 1024}, {1024, 1024}}, ge::DT_UINT8, ge::FORMAT_ND}}, &g_compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, KEY_DOUBLE, expectWorkspaces);
}
