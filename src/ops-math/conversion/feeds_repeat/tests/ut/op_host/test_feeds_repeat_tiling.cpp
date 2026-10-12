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
#include "../../../op_host/feeds_repeat_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class FeedsRepeatTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "FeedsRepeat Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "FeedsRepeat Tiling TearDown" << std::endl;
    }
};

TEST_F(FeedsRepeatTiling, test_tiling_fp16_int32) {
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};
    
    gert::TilingContextPara tilingContextPara("FeedsRepeat",
                                              {{{{4, 5, 6, 7}, {4, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{4,}, {4,}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{15, 5, 6, 7}, {15, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
                                              &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "34359738372 210 224 64 16 0 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_fp32_int32) {
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};
    
    gert::TilingContextPara tilingContextPara("FeedsRepeat",
                                              {{{{4, 5, 6, 7}, {4, 5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{4,}, {4,}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{15, 5, 6, 7}, {15, 5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "34359738372 210 216 64 16 0 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_bf16_int32) {
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};
    
    gert::TilingContextPara tilingContextPara("FeedsRepeat",
                                              {{{{4, 50, 60, 7}, {4, 50, 60, 7}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{4,}, {4,}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{15, 50, 60, 7}, {15, 50, 60, 7}}, ge::DT_BF16, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
                                              &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "34359738372 21000 21008 64 16 0 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_fp32_int64) {
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};
    
    gert::TilingContextPara tilingContextPara("FeedsRepeat",
                                              {{{{48, 5, 6, 7}, {48, 5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{48,}, {48,}}, ge::DT_INT64, ge::FORMAT_ND}},
                                              {{{{100, 5, 6, 7}, {100, 5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "206158430256 210 216 64 1 16 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_fp16_int64) {
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};
    
    gert::TilingContextPara tilingContextPara("FeedsRepeat",
                                              {{{{50, 5, 6, 7}, {50, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{50,}, {50,}}, ge::DT_INT64, ge::FORMAT_ND}},
                                              {{{{50, 5, 6, 7}, {50, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "223338299442 210 224 64 1 14 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_bf16_int64) {
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};
    
    gert::TilingContextPara tilingContextPara("FeedsRepeat",
                                              {{{{100, 5, 6, 7}, {100, 5, 6, 7}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{100,}, {100,}}, ge::DT_INT64, ge::FORMAT_ND}},
                                              {{{{50, 5, 6, 7}, {50, 5, 6, 7}}, ge::DT_BF16, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
                                              &compileInfo);
    uint64_t expectTilingKey = 103;
    string expectTilingData = "429496729700 210 224 64 0 0 15 1 36 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_int32_int64) {
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};
    
    gert::TilingContextPara tilingContextPara("FeedsRepeat",
                                              {{{{48, 5, 6, 7}, {48, 5, 6, 7}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{48,}, {48,}}, ge::DT_INT64, ge::FORMAT_ND}},
                                              {{{{100, 5, 6, 7}, {100, 5, 6, 7}}, ge::DT_INT32, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
                                              &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "206158430256 0 0 64 1 16 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}