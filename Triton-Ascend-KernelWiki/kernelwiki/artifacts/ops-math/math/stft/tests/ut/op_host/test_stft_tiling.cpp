/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/stft_tiling.h"

using namespace std;
using namespace ge;

class STFTTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "STFTTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "STFTTiling TearDown" << std::endl;
    }
};

TEST_F(STFTTiling, stft_tiling_001)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 30000}, {2, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {36422144};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_002)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 30000}, {2, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(-1);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_003)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 30000}, {2, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(-1);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_004)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 30000}, {2, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(-1);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_005)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 30000}, {2, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_006)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 30000}, {2, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_007)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 30000}, {2, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_008)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 30000}, {2, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_BF16, ge::FORMAT_ND}, {window_shape, ge::DT_BF16, ge::FORMAT_ND}},
        {{out_shape, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_009)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 3, 2}, {2, 3, 2}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_010)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 30000}, {2, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {36422144};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_011)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 30000}, {2, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};    
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_COMPLEX64, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {36422144};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_012)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{25, 2029797}, {25, 2029797}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "107374182400 1717988947797 54477365182624 137438966160 201 17179869248 111669149703 68719476760 30064771079 25769803782 137438953478 206158430209 38654705760 17179869202 4 1726576852993 1717986931084 111669150096 1717986931084 1099511627808 55834574880 4294967298 1 4294967296 510173395288064 32768 4294967297 4294967297 4294967309 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {9618565632};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_013)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{23, 2029637}, {23, 2029637}};
    gert::StorageShape window_shape = {{201, 401}, {201, 401}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "98784247808 1717988947637 54473070215328 137438966160 201 34359738432 223338299395 34359738418 55834574851 55834574861 68719476748 240518168577 4294967408 8589934594 2 1726576852993 1717986931083 223338299792 1717986931083 1099511627840 55834574880 4294967298 1 4294967296 738871813865472 65536 4294967297 4294967297 4294967309 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4382063104};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_014)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{40, 2029637}, {40, 2029637}};
    gert::StorageShape window_shape = {{201, 401}, {201, 401}};
    gert::StorageShape out_shape = {{2, 201, 188, 2}, {2, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "171798691840 1717988947637 54473070215328 137438966160 201 34359738432 223338299397 34359738418 55834574853 55834574861 68719476748 274877906945 4294967424 21474836482 5 1726576852993 1717986931083 223338299792 1717986931083 1099511627840 55834574880 4294967298 1 4294967296 738871813865472 65536 4294967297 4294967297 4294967309 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {7326243840};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_015)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 2029637}, {2, 2029637}};
    gert::StorageShape window_shape = {{200, 399}, {200, 399}};
    gert::StorageShape out_shape = {{2, 200, 187, 2}, {2, 200, 187, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(399);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(399);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "171798691840 1717988947637 54473070215328 137438966160 201 34359738432 223338299397 34359738418 55834574853 55834574861 68719476748 274877906945 4294967424 21474836482 5 1726576852993 1717986931083 223338299792 1717986931083 1099511627840 55834574880 4294967298 1 4294967296 738871813865472 65536 4294967297 4294967297 4294967309 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_016)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 2028677}, {2, 2028677}};
    gert::StorageShape window_shape = {{200, 399}, {200, 399}};
    gert::StorageShape out_shape = {{2, 200, 187, 2}, {2, 200, 187, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(37);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(37);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_017)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 2028517}, {2, 2028517}};
    gert::StorageShape window_shape = {{200, 399}, {200, 399}};
    gert::StorageShape out_shape = {{2, 200, 187, 2}, {2, 200, 187, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(10);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(10);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_018)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{30, 1861}, {30, 1861}};
    gert::StorageShape window_shape = {{768, 768}, {768, 768}};
    gert::StorageShape out_shape = {{30, 768, 34, 2}, {30, 768, 34, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(33);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(768);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(768);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_019)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{19, 30000}, {19, 30000}};
    gert::StorageShape window_shape = {{201, 400}, {201, 400}};
    gert::StorageShape out_shape = {{19, 201, 188, 2}, {19, 201, 188, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(160);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(400);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(400);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "81604378624 1717986948000 798863917216 137438953664 79164837200073 17179869248 111669149701 68719476760 30064771077 25769803782 137438953478 206158430209 38654705760 17179869202 4 1726576852993 1717986918586 111669150096 1717986918586 824633720864 42949673000 4294967300 1 4294967296 747667906887680 24576 4294967297 4294967297 8589934602 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117691904};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_020)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{30, 1861}, {30, 1861}};
    gert::StorageShape window_shape = {{768, 768}, {768, 768}};
    gert::StorageShape out_shape = {{30, 768, 34, 2}, {30, 768, 34, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(33);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(768);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(768);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_021)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{30, 1861}, {30, 1861}};
    gert::StorageShape window_shape = {{768, 768}, {768, 768}};
    gert::StorageShape out_shape = {{30, 768, 34, 2}, {30, 768, 34, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(33);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(768);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(768);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_022)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{1, 1}, {1, 1}};
    gert::StorageShape window_shape = {{1, 1}, {1, 1}};
    gert::StorageShape out_shape = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(1);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(1);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_023)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{1, 1}, {1, 1}};
    gert::StorageShape window_shape = {{1, 1}, {1, 1}};
    gert::StorageShape out_shape = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(1);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(1);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_024)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 524288, 65536, 65536, 131072, 0};
    gert::StorageShape input_shape = {{2, 1}, {2, 1}};
    gert::StorageShape window_shape = {{1, 1}, {1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1, 2}, {2, 1, 1, 2}};
    auto hop_length = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    auto win_length = Ops::Math::AnyValue::CreateFrom<int64_t>(1);
    auto normalized = Ops::Math::AnyValue::CreateFrom<bool>(true);
    auto oensided = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto return_complex = Ops::Math::AnyValue::CreateFrom<bool>(false);
    auto n_fft = Ops::Math::AnyValue::CreateFrom<int64_t>(1);

    gert::TilingContextPara tilingContextPara(
        "STFT", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {window_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("hop_length", hop_length),
         gert::TilingContextPara::OpAttr("win_length", win_length),
         gert::TilingContextPara::OpAttr("normalized", normalized),
         gert::TilingContextPara::OpAttr("oensided", oensided),
         gert::TilingContextPara::OpAttr("return_complex", return_complex),
         gert::TilingContextPara::OpAttr("n_fft", n_fft)},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "8589934592 1717986948000 798863917216 137438953664 79164837200073 8589934656 60129542145 137438953484 "
        "17179869185 12884901891 274877906947 274877906945 38654705792 4294967314 1 1726576852993 1717986918586 "
        "60129542544 1717986918586 824633720848 42949673000 4294967306 1 4294967296 1429365116108800 12288 4294967297 "
        "4294967297 21474836490 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}