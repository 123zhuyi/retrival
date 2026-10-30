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
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/diag_part_tiling_arch35.h"

using namespace std;
using namespace ge;

class DiagPartTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DiagPartTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DiagPartTiling TearDown" << std::endl;
    }
};

TEST_F(DiagPartTiling, neg_test_tiling_001)
{
    struct DiagPartCompileInfo {
        int64_t coreNum = 64;
        int64_t ubSize = 253952;
    };

    DiagPartCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "DiagPart",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1000;
    string expectTilingData = "16 1 32 16 253952 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DiagPartTiling, neg_test_invalid_core_num)
{
    struct DiagPartCompileInfo {
        int64_t coreNum = 0;
        int64_t ubSize = 253952;
    };

    DiagPartCompileInfo compileInfo = {0, 253952};
    gert::TilingContextPara tilingContextPara(
        "DiagPart",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DiagPartTiling, neg_test_invalid_ub_size)
{
    struct DiagPartCompileInfo {
        int64_t coreNum = 64;
        int64_t ubSize = 0;
    };

    DiagPartCompileInfo compileInfo = {64, 0};
    gert::TilingContextPara tilingContextPara(
        "DiagPart",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DiagPartTiling, neg_test_x_dim_odd)
{
    struct DiagPartCompileInfo {
        int64_t coreNum = 64;
        int64_t ubSize = 253952;
    };

    DiagPartCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "DiagPart",
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DiagPartTiling, neg_test_x_front_back_half_mismatch)
{
    struct DiagPartCompileInfo {
        int64_t coreNum = 64;
        int64_t ubSize = 253952;
    };

    DiagPartCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "DiagPart",
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DiagPartTiling, neg_test_y_dim_mismatch)
{
    struct DiagPartCompileInfo {
        int64_t coreNum = 64;
        int64_t ubSize = 253952;
    };

    DiagPartCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "DiagPart",
        {
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(DiagPartTiling, neg_test_xy_front_dim_mismatch)
{
    struct DiagPartCompileInfo {
        int64_t coreNum = 64;
        int64_t ubSize = 253952;
    };

    DiagPartCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "DiagPart",
        {
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}