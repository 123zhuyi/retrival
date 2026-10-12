/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include <gtest/gtest.h>
#include <iostream>
#include "../../../../op_host/arch35/fills_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace std;

class FillsTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FillsTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FillsTilingTest TearDown" << std::endl;
  }
};

TEST_F(FillsTilingTest, fills_test_tiling_fp32_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Fills",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 7;
    string expectTilingData = "4096 4 32768 1024 4 1 1 1024 1024 32768 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(FillsTilingTest, fills_test_tiling_fp16_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Fills",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "4096 2 65536 2048 2 1 1 2048 2048 65536 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(FillsTilingTest, fills_test_tiling_int32_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Fills",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 9;
    string expectTilingData = "4096 4 32768 1024 4 1 1 1024 1024 32768 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(FillsTilingTest, fills_test_tiling_int64_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Fills",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 11;
    string expectTilingData = "4096 8 16384 512 8 1 1 512 512 16384 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(FillsTilingTest, fills_test_tiling_int8_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Fills",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 13;
    string expectTilingData = "4096 1 131072 4096 1 1 1 4096 4096 131072 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(FillsTilingTest, fills_test_tiling_uint8_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Fills",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 15;
    string expectTilingData = "4096 1 131072 4096 1 1 1 4096 4096 131072 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}

TEST_F(FillsTilingTest, fills_test_tiling_invalid_dtype)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Fills",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 7;
    string expectTilingData = "4096 4 32768 1024 4 1 1 1024 1024 32768 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_FAILED, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}
