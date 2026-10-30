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
 * \file test_stateless_drop_out_gen_mask_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/stateless_drop_out_gen_mask_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class StatelessDropOutGenMaskTilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "StatelessDropOutGenMaskTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StatelessDropOutGenMaskTilingTest TearDown" << std::endl;
    }
};

TEST_F(StatelessDropOutGenMaskTilingTest, stateless_drop_out_gen_mask_test_01)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {32, 512};
    vector<float> probValue = {1.0};
    vector<int64_t> seedValue = {2};
    vector<int64_t> seed1Value = {0};
    vector<int64_t> offsetValue = {8};

    gert::TilingContextPara tilingContextPara(
        "StatelessDropOutGenMask",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, probValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seed1Value.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
    },
    {
        {{{ 2048 }, { 2048 } },ge::DT_UINT8, ge::FORMAT_ND},
    }, 
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 256 256 13056 34359738369 0 0 16384 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StatelessDropOutGenMaskTilingTest, stateless_drop_out_gen_mask_test_02)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int32_t> shapeValue = {8, 512, 10};
    vector<float> probValue = {0.6};
    vector<int32_t> seedValue = {24};
    vector<int32_t> seed1Value = {8};
    vector<int64_t> offsetValue = {36};

    gert::TilingContextPara tilingContextPara(
        "StatelessDropOutGenMask",
    {
        {{{3},{3}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, probValue.data()},
        {{{1},{1}}, ge::DT_INT32, ge::FORMAT_ND, true, seedValue.data()},
        {{{1},{1}}, ge::DT_INT32, ge::FORMAT_ND, true, seed1Value.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
    },
    {
        {{{ 5120 }, { 5120 } },ge::DT_UINT8, ge::FORMAT_ND},
    }, 
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "54 768 256 13056 17179869185 0 0 40960 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StatelessDropOutGenMaskTilingTest, stateless_drop_out_gen_mask_test_03)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int32_t> shapeValue = {8, 512, 2, 5};
    vector<float> probValue = {0.6};
    vector<int32_t> seedValue = {24};
    vector<int32_t> seed1Value = {8};
    vector<int64_t> offsetValue = {36};

    gert::TilingContextPara tilingContextPara(
        "StatelessDropOutGenMask",
    {
        {{{4},{4}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, probValue.data()},
        {{{1},{1}}, ge::DT_INT32, ge::FORMAT_ND, true, seedValue.data()},
        {{{1},{1}}, ge::DT_INT32, ge::FORMAT_ND, true, seed1Value.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
    },
    {
        {{{ 5120 }, { 5120 } },ge::DT_UINT8, ge::FORMAT_ND},
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "54 768 256 13056 17179869185 0 0 40960 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StatelessDropOutGenMaskTilingTest, stateless_drop_out_gen_mask_test_04)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int32_t> shapeValue = {8, 512, 2, 5};
    vector<float> probValue = {0.6};
    vector<int32_t> seedValue = {24};
    vector<int32_t> seed1Value = {8};
    vector<int64_t> offsetValue = {36};

    gert::TilingContextPara tilingContextPara(
        "StatelessDropOutGenMask",
    {
        {{{4},{4}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_BF16, ge::FORMAT_ND, true, probValue.data()},
        {{{1},{1}}, ge::DT_INT32, ge::FORMAT_ND, true, seedValue.data()},
        {{{1},{1}}, ge::DT_INT32, ge::FORMAT_ND, true, seed1Value.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
    },
    {
        {{{ 5120 }, { 5120 } },ge::DT_UINT8, ge::FORMAT_ND},
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "54 768 256 13056 17179869185 0 0 40960 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StatelessDropOutGenMaskTilingTest, stateless_drop_out_gen_mask_test_05)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {8, 512, 20, 5, 2};
    vector<float> probValue = {0.7};
    vector<int64_t> seedValue = {16};
    vector<int64_t> seed1Value = {8};
    vector<int64_t> offsetValue = {0, 36};

    gert::TilingContextPara tilingContextPara(
        "StatelessDropOutGenMask",
    {
        {{{5},{5}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_BF16, ge::FORMAT_ND, true, probValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seed1Value.data()},
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
    },
    {
        {{{ 102400 }, { 102400 } },ge::DT_UINT8, ge::FORMAT_ND},
    }, 
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 12800 12800 13056 34359738370 0 0 819200 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StatelessDropOutGenMaskTilingTest, stateless_drop_out_gen_mask_test_06)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {8, 256, 2, 5};
    vector<float> probValue = {0.6};
    vector<int64_t> seedValue = {24};
    vector<int64_t> seed1Value = {8};
    vector<int64_t> offsetValue = {0, 36};

    gert::TilingContextPara tilingContextPara(
        "StatelessDropOutGenMask",
    {
        {{{4},{4}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, probValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seed1Value.data()},
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
    },
    {
        {{{ 2560 }, { 2560 } },ge::DT_UINT8, ge::FORMAT_ND},
    }, 
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "40 512 512 13056 34359738370 0 0 20480 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StatelessDropOutGenMaskTilingTest, stateless_drop_out_gen_mask_test_07)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {8, 256, 3, 4};
    vector<float> probValue = {0.6};
    vector<int64_t> seedValue = {24};
    vector<int64_t> seed1Value = {8};
    vector<int64_t> offsetValue = {0, 36};

    gert::TilingContextPara tilingContextPara(
        "StatelessDropOutGenMask",
    {
        {{{4},{4}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, probValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seed1Value.data()},
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
    },
    {
        {{{ 3072 }, { 3072 } },ge::DT_UINT8, ge::FORMAT_ND},
    }, 
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "48 512 512 13056 34359738370 0 0 24576 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}