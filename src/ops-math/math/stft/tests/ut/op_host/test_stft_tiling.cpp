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
    static void SetUpTestCase() { std::cout << "STFTTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "STFTTiling TearDown" << std::endl; }
};

TEST_F(STFTTiling, stft_tiling_001)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "8589934593 1717986948000 687194767776 798863917257 4417130515383975937 12884955136 8589969344 1 111669149704 "
        "17179869191 4294967344 0 274877906944 223338299393 1786706395184 223338299808 1717986918448 206158430272 "
        "17179869312 4294967300 1 4294967296 985162418487296 12288 4294967297 4294967297 17179869188 0 8589934594 1 0 "
        "0 0 0 0 0 0 0 223338299393 1786706395178 223338299808 1717986918442 206158430272 17179869312 4294967300 1 "
        "4294967296 985162418487296 12288 4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "214748364801 1786706395184 214748365216 1717986918448 206158430272 17179869312 4294967300 1 4294967296 "
        "985162418487296 12288 4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 214748364801 "
        "1786706395178 214748365216 1717986918442 206158430272 17179869312 4294967300 1 4294967296 985162418487296 "
        "12288 4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 30064771488 335007449094 137438953478 "
        "8589934610 798863917257 ";
    std::vector<size_t> expectWorkspaces = {35441152};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_006)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "8589934593 1717986948000 687194767776 798863917456 0 12884955136 8589969344 1 214748364808 17179869184 "
        "4294967344 0 274877906944 429496729601 1786706395184 429496730016 1717986918448 206158430320 8589934656 "
        "4294967303 1 4294967296 615726511554560 21504 4294967297 4294967297 30064771073 0 8589934594 1 0 0 0 0 0 0 0 "
        "0 429496729601 1786706395178 429496730016 1717986918442 206158430320 8589934656 4294967303 1 4294967296 "
        "615726511554560 21504 4294967297 4294967297 30064771073 0 8589934594 1 0 0 0 0 0 0 0 0 429496729601 "
        "1786706395184 429496730016 1717986918448 206158430320 8589934656 4294967303 1 4294967296 615726511554560 "
        "21504 4294967297 4294967297 30064771073 0 8589934594 1 0 0 0 0 0 0 0 0 429496729601 1786706395178 "
        "429496730016 1717986918442 206158430320 8589934656 4294967303 1 4294967296 615726511554560 21504 4294967297 "
        "4294967297 30064771073 0 8589934594 1 0 0 0 0 0 0 0 0 55834575264 335007449100 137438953478 8589934624 "
        "798863917456 ";
    std::vector<size_t> expectWorkspaces = {36695040};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_007)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "8589934593 1717986948000 687194767776 798863917257 0 12884955136 8589969344 1 111669149704 17179869191 "
        "4294967344 0 274877906944 223338299393 1786706395184 223338299808 1717986918448 206158430272 17179869312 "
        "4294967300 1 4294967296 985162418487296 12288 4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 "
        "0 223338299393 1786706395178 223338299808 1717986918442 206158430272 17179869312 4294967300 1 4294967296 "
        "985162418487296 12288 4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 214748364801 "
        "1786706395184 214748365216 1717986918448 206158430272 17179869312 4294967300 1 4294967296 985162418487296 "
        "12288 4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 214748364801 1786706395178 "
        "214748365216 1717986918442 206158430272 17179869312 4294967300 1 4294967296 985162418487296 12288 4294967297 "
        "4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 30064771488 335007449094 137438953478 8589934610 "
        "798863917257 ";
    std::vector<size_t> expectWorkspaces = {35441152};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_008)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    string expectTilingData = "107374182400 1717988947797 54477365182624 137438966160 201 17179869248 111669149703 "
                              "68719476760 30064771079 25769803782 137438953478 206158430209 38654705760 17179869202 4 "
                              "1726576852993 1717986931084 111669150096 1717986931084 1099511627808 55834574880 "
                              "4294967298 1 4294967296 510173395288064 32768 4294967297 4294967297 4294967309 0 "
                              "8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {9618565632};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_013)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    string expectTilingData = "98784247808 1717988947637 54473070215328 137438966160 201 34359738432 223338299395 "
                              "34359738418 55834574851 55834574861 68719476748 240518168577 4294967408 8589934594 2 "
                              "1726576852993 1717986931083 223338299792 1717986931083 1099511627840 55834574880 "
                              "4294967298 1 4294967296 738871813865472 65536 4294967297 4294967297 4294967309 0 "
                              "8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {4382063104};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_014)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    string expectTilingData = "171798691840 1717988947637 54473070215328 137438966160 201 34359738432 223338299397 "
                              "34359738418 55834574853 55834574861 68719476748 274877906945 4294967424 21474836482 5 "
                              "1726576852993 1717986931083 223338299792 1717986931083 1099511627840 55834574880 "
                              "4294967298 1 4294967296 738871813865472 65536 4294967297 4294967297 4294967309 0 "
                              "8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {7326243840};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_015)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "8589934593 1713693980342 687194767776 54473070215368 0 12885092096 8589944832 1 107374182408 17179869184 "
        "4294970472 0 274877906944 214748364801 1786706398312 214748365216 1713691954280 1374389534784 107374182416 "
        "4294967306 1 4294967296 1319413953331200 81920 4294967297 4294967297 21474836505 0 8589934594 1 0 0 0 0 0 0 0 "
        "0 214748364801 1786706398291 214748365216 1713691954259 1236950581312 107374182416 4294967306 1 4294967296 "
        "1231453023109120 73728 4294967297 4294967297 21474836505 0 8589934594 1 0 0 0 0 0 0 0 0 214748364801 "
        "1786706398312 214748365216 1713691954280 1374389534784 107374182416 4294967306 1 4294967296 1319413953331200 "
        "81920 4294967297 4294967297 21474836505 0 8589934594 1 0 0 0 0 0 0 0 0 214748364801 1786706398291 "
        "214748365216 1713691954259 1236950581312 107374182416 4294967306 1 4294967296 1231453023109120 73728 "
        "4294967297 4294967297 21474836505 0 8589934594 1 0 0 0 0 0 0 0 0 30064771488 335007449094 137438953478 "
        "8589934608 54473070215368 ";
    std::vector<size_t> expectWorkspaces = {117015040};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_016)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "8589934593 158915818592 687194767424 54460185313299 0 4295157504 8589944832 1 12884901896 17179869189 "
        "4294970472 0 274877906944 25769803777 274877910120 25769803840 158913793128 1099511627792 8589934624 "
        "4294967298 1 4294967296 299067162755072 16384 4294967297 4294967297 4294967298 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "25769803777 274877910096 25769803840 158913793104 1099511627792 8589934624 4294967298 1 4294967296 "
        "299067162755072 16384 4294967297 4294967297 4294967298 0 8589934594 1 0 0 0 0 0 0 0 0 17179869185 "
        "274877910120 17179869248 158913793128 1099511627792 8589934624 4294967298 1 4294967296 299067162755072 16384 "
        "4294967297 4294967297 4294967298 0 8589934594 1 0 0 0 0 0 0 0 0 17179869185 274877910096 17179869248 "
        "158913793104 1099511627792 8589934624 4294967298 1 4294967296 299067162755072 16384 4294967297 4294967297 "
        "4294967298 0 8589934594 1 0 0 0 0 0 0 0 0 4294967360 2194728288256 1 8589934630 54460185313299 ";
    std::vector<size_t> expectWorkspaces = {43911168};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_017)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "8589934593 42951701467 687194767392 54455890345990 0 4295157504 8589944832 1 8589934596 34359738370 "
        "4294968888 0 274877906944 17179869185 137438955064 17179869216 42949674552 1374389534736 4294967312 "
        "4294967301 5 4294967296 444202697621504 20480 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "17179869185 137438955007 17179869216 42949674495 1099511627792 4294967312 4294967302 6 4294967296 "
        "426610511577088 16384 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 8589934593 137438955064 "
        "8589934624 42949674552 1374389534736 4294967312 4294967301 5 4294967296 444202697621504 20480 4294967297 "
        "4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 8589934593 137438955007 8589934624 42949674495 "
        "1099511627792 4294967312 4294967302 6 4294967296 426610511577088 16384 4294967297 4294967297 4294967297 0 "
        "8589934594 1 0 0 0 0 0 0 0 0 4294967328 4389456576512 137438953472 8589934604 54455890345990 ";
    std::vector<size_t> expectWorkspaces = {38019584};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_018)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "128849018881 3298534884421 141733921536 146028888832 4401086909862903809 34359762080 34359778048 8589934596 "
        "412316860424 4294967296 34 0 274877906944 824633720833 3298534883362 824633721600 3298534883362 206158430400 "
        "17179869216 4294967320 1 4294967296 1055531162664960 36864 4294967297 4294967297 103079215106 0 8589934594 1 "
        "0 0 0 0 0 0 0 0 824633720833 3298534883362 824633721600 3298534883362 206158430400 17179869216 4294967320 1 "
        "4294967296 1055531162664960 36864 4294967297 4294967297 103079215106 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "824633720833 3298534883362 824633721600 3298534883362 206158430400 17179869216 4294967320 1 4294967296 "
        "1055531162664960 36864 4294967297 4294967297 103079215106 0 8589934594 1 0 0 0 0 0 0 0 0 824633720833 "
        "3298534883362 824633721600 3298534883362 206158430400 17179869216 4294967320 1 4294967296 1055531162664960 "
        "36864 4294967297 4294967297 103079215106 0 8589934594 1 0 0 0 0 0 0 0 0 103079215872 180388626456 12 "
        "128849018944 146028888832 ";
    std::vector<size_t> expectWorkspaces = {47673344};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_019)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    string expectTilingData = "81604378624 1717986948000 798863917216 137438953664 79164837200073 17179869248 "
                              "111669149701 68719476760 30064771077 25769803782 137438953478 206158430209 38654705760 "
                              "17179869202 4 1726576852993 1717986918586 111669150096 1717986918586 824633720864 "
                              "42949673000 4294967300 1 4294967296 747667906887680 24576 4294967297 4294967297 "
                              "8589934602 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117691904};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_020)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
        "128849018883 3298534884421 141733921536 146028888832 4401086909862903809 34359750240 34359800832 8589934596 "
        "412316860424 4294967296 34 0 274877906944 824633720833 3298534883362 824633721600 3298534883362 206158430400 "
        "51539607616 4294967308 1 4294967296 1583296743997440 36864 4294967297 4294967297 51539607564 0 8589934594 1 0 "
        "0 0 0 0 0 0 0 824633720833 3298534883362 824633721600 3298534883362 206158430400 51539607616 4294967308 1 "
        "4294967296 1583296743997440 36864 4294967297 4294967297 51539607564 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "824633720833 3298534883362 824633721600 3298534883362 206158430400 51539607616 4294967308 1 4294967296 "
        "1583296743997440 36864 4294967297 4294967297 51539607564 0 8589934594 1 0 0 0 0 0 0 0 0 824633720833 "
        "3298534883362 824633721600 3298534883362 206158430400 51539607616 4294967308 1 4294967296 1583296743997440 "
        "36864 4294967297 4294967297 51539607564 0 8589934594 1 0 0 0 0 0 0 0 0 103079215872 180388626456 12 "
        "128849018944 146028888832 ";
    std::vector<size_t> expectWorkspaces = {40613888};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_021)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "128849018881 3298534884421 141733921536 146028888832 4401086909862903809 34359762080 34359778048 8589934596 "
        "412316860424 4294967296 34 0 274877906944 824633720833 3298534883362 824633721600 3298534883362 206158430400 "
        "17179869216 4294967320 1 4294967296 1055531162664960 36864 4294967297 4294967297 103079215106 0 8589934594 1 "
        "0 0 0 0 0 0 0 0 824633720833 3298534883362 824633721600 3298534883362 206158430400 17179869216 4294967320 1 "
        "4294967296 1055531162664960 36864 4294967297 4294967297 103079215106 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "824633720833 3298534883362 824633721600 3298534883362 206158430400 17179869216 4294967320 1 4294967296 "
        "1055531162664960 36864 4294967297 4294967297 103079215106 0 8589934594 1 0 0 0 0 0 0 0 0 824633720833 "
        "3298534883362 824633721600 3298534883362 206158430400 17179869216 4294967320 1 4294967296 1055531162664960 "
        "36864 4294967297 4294967297 103079215106 0 8589934594 1 0 0 0 0 0 0 0 0 103079215872 180388626456 12 "
        "128849018944 146028888832 ";
    std::vector<size_t> expectWorkspaces = {47673344};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_022)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "4294967297 4294967296 8589934624 4294967297 4575657221408423937 190208 4294977536 1 4294967297 4294967296 1 0 "
        "8589934592 8589934593 137438953473 8589934624 4294967297 68719476752 4294967304 4294967297 1 4294967296 "
        "4398046511104 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 8589934593 137438953473 "
        "8589934624 4294967297 68719476752 4294967304 4294967297 1 4294967296 4398046511104 1024 4294967297 4294967297 "
        "4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 8589934593 137438953473 8589934624 4294967297 68719476752 "
        "4294967304 4294967297 1 4294967296 4398046511104 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 "
        "0 0 0 0 8589934593 137438953473 8589934624 4294967297 68719476752 4294967304 4294967297 1 4294967296 "
        "4398046511104 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 4294967328 4389456576512 "
        "137438953472 4294967298 4294967297 ";
    std::vector<size_t> expectWorkspaces = {33555968};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_023)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "4294967297 4294967296 8589934624 4294967297 4575657221408423937 190208 4294977536 1 4294967297 4294967296 1 0 "
        "8589934592 8589934593 137438953473 8589934624 4294967297 68719476752 4294967304 4294967297 1 4294967296 "
        "4398046511104 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 8589934593 137438953473 "
        "8589934624 4294967297 68719476752 4294967304 4294967297 1 4294967296 4398046511104 1024 4294967297 4294967297 "
        "4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 8589934593 137438953473 8589934624 4294967297 68719476752 "
        "4294967304 4294967297 1 4294967296 4398046511104 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 "
        "0 0 0 0 8589934593 137438953473 8589934624 4294967297 68719476752 4294967304 4294967297 1 4294967296 "
        "4398046511104 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 4294967328 4389456576512 "
        "137438953472 4294967298 4294967297 ";
    std::vector<size_t> expectWorkspaces = {33555968};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(STFTTiling, stft_tiling_024)
{
    optiling::STFTCompileInfo compileInfo = {20, 40, 20, 196608, 0};
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
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "8589934593 4294967296 8589934624 4294967297 4575657221408423937 190208 8589944832 1 4294967297 4294967296 1 0 "
        "8589934592 8589934593 137438953473 8589934624 4294967297 68719476752 4294967304 4294967297 1 4294967296 "
        "4398046511104 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 8589934593 137438953473 "
        "8589934624 4294967297 68719476752 4294967304 4294967297 1 4294967296 4398046511104 1024 4294967297 4294967297 "
        "4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 8589934593 137438953473 8589934624 4294967297 68719476752 "
        "4294967304 4294967297 1 4294967296 4398046511104 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 "
        "0 0 0 0 8589934593 137438953473 8589934624 4294967297 68719476752 4294967304 4294967297 1 4294967296 "
        "4398046511104 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 4294967328 4389456576512 "
        "137438953472 8589934594 4294967297 ";
    std::vector<size_t> expectWorkspaces = {33555968};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
