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
 * \file test_pad_tiling.cpp
 * \brief
 */

#include "../../../../op_host/arch35/pad_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
class PadTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PadTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PadTilingTest TearDown" << std::endl; }
};

TEST_F(PadTilingTest, Pad_AC_tiling_NDDMA_20000) // NDDMA_20000
{
    optiling::PadCompileInfo compileInfo;
    compileInfo.core_num = 64;
    compileInfo.ub_size = 245760; // 240 * 1024
    std::vector<int32_t> paddingsValue = {4, 1};
    gert::StorageShape xShape = {{55}, {55}};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{60}, {60}};
    gert::TilingContextPara tilingContextPara(
        "Pad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 20000;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadTilingTest, Pad_AC_tiling_NDDMA_30010) // NDDMA_30010  ----
{
    optiling::PadCompileInfo compileInfo;
    compileInfo.core_num = 64;
    compileInfo.ub_size = 245760; // 240 * 1024
    std::vector<int32_t> paddingsValue = {9, 17};
    gert::StorageShape xShape = {{1991203}, {1991203}};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};
    gert::StorageShape yShape = {{1991203}, {1991203}};
    gert::TilingContextPara tilingContextPara(
        "Pad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 30010;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadTilingTest, Pad_AC_tiling_NDDMA_30021) // NDDMA_30021
{
    optiling::PadCompileInfo compileInfo;
    compileInfo.core_num = 64;
    compileInfo.ub_size = 245760; // 240 * 1024
    std::vector<int32_t> paddingsValue = {20, 25, 10, 18};
    gert::StorageShape xShape = {{1239, 1025}, {1239, 1025}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape = {{1284, 1053}, {1284, 1053}};
    gert::TilingContextPara tilingContextPara(
        "Pad",
        {{xShape, ge::DT_BOOL, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_BOOL, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 30021;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadTilingTest, Pad_AC_tiling_NDDMA_30023) // NDDMA_30023   ---
{
    optiling::PadCompileInfo compileInfo;
    compileInfo.core_num = 64;
    compileInfo.ub_size = 245760; // 240 * 1024
    std::vector<int32_t> paddingsValue = {15, 22, 11, 1, 22, 5};
    gert::StorageShape xShape = {{109, 7078, 4}, {109, 7078, 4}};
    gert::StorageShape paddingsShape = {{3, 2}, {3, 2}};
    gert::StorageShape yShape = {{146, 7090, 31}, {146, 7090, 31}};
    gert::TilingContextPara tilingContextPara(
        "Pad",
        {{xShape, ge::DT_INT8, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_INT8, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 30023;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadTilingTest, Pad_AC_tiling_NDDMA_30031) // NDDMA_30031   ---
{
    optiling::PadCompileInfo compileInfo;
    compileInfo.core_num = 64;
    compileInfo.ub_size = 245760; // 240 * 1024
    std::vector<int32_t> paddingsValue = {5, 3, 4, 4, 4, 4, 5, 4};
    gert::StorageShape xShape = {{70, 73, 74, 69}, {70, 73, 74, 69}};
    gert::StorageShape paddingsShape = {{4, 2}, {4, 2}};
    gert::StorageShape yShape = {{78, 81, 82, 78}, {78, 81, 82, 78}};
    gert::TilingContextPara tilingContextPara(
        "Pad",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 30031;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(PadTilingTest, Pad_AC_tiling_NDDMA_30032) // NDDMA_30033   ---
{
    optiling::PadCompileInfo compileInfo;
    compileInfo.core_num = 64;
    compileInfo.ub_size = 245760; // 240 * 1024
    std::vector<int32_t> paddingsValue = {24, 14, 6, 17, 6, 7, 15, 17};
    gert::StorageShape xShape = {{4, 45, 42, 21}, {4, 45, 42, 21}};
    gert::StorageShape paddingsShape = {{4, 2}, {4, 2}};
    gert::StorageShape yShape = {{42, 68, 55, 53}, {42, 68, 55, 53}};
    gert::TilingContextPara tilingContextPara(
        "Pad",
        {{xShape, ge::DT_BOOL, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_BOOL, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 30033;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}
