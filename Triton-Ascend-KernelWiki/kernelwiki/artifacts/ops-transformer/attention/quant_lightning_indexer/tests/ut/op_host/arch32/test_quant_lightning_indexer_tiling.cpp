/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

class QuantLightningIndexerTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantLightningIndexerTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantLightningIndexerTiling TearDown" << std::endl;
    }
};

TEST_F(QuantLightningIndexerTiling, QuantLightningIndexer_910b_tiling_0)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 128, 64, 128}, {2, 128, 64, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2, 512, 1, 128}, {2, 512, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 1}, {2, 512, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 128, 1, 1024}, {2, 128, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    int64_t expectTilingKey = 197122;
    std::string expectTilingData =
        "2 549755813952 4398046511616 64 12884901888 ";
    std::vector<size_t> expectWorkspaces = {167903232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(QuantLightningIndexerTiling, QuantLightningIndexer_910b_tiling_1)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int64_t actualSeqQuery[] = {64, 128, 192};
    int64_t actualSeqKey[] = {128, 256, 384};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{192, 16, 128}, {192, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{384, 1, 128}, {384, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{192, 16}, {192, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 16}, {192, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{384, 1}, {384, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{192, 1, 512}, {192, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerTiling, QuantLightningIndexer_910b_tiling_2)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int64_t actualSeqKey[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 64, 64, 128}, {2, 64, 64, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{32, 16, 1, 128}, {32, 16, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2, 64, 64}, {2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 64}, {2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 16, 1}, {32, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 64, 1, 256}, {2, 64, 1, 256}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    int64_t expectTilingKey = 1090716162;
    std::string expectTilingData =
        "2 274877907008 1099511628288 68719476800 12884901920 ";
    std::vector<size_t> expectWorkspaces = {167903232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// input attr query_quant_mode only supported 0.
TEST_F(QuantLightningIndexerTiling, QuantLightningIndexer_910b_tiling_3)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 128, 64, 128}, {2, 128, 64, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2, 512, 1, 128}, {2, 512, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 1}, {2, 512, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 128, 1, 1024}, {2, 128, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// outside of PA, input attr layout_query and input attr layout_key must be the same,but now layout_key is TND, layout_query is BSND.
TEST_F(QuantLightningIndexerTiling, QuantLightningIndexer_910b_tiling_4)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int64_t actualSeqKey[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 64, 64, 128}, {2, 64, 64, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{32, 16, 1, 128}, {32, 16, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2, 64, 64}, {2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 64}, {2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 16, 1}, {32, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 64, 1, 256}, {2, 64, 1, 256}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// input attr sparse_count must > 0 and <= 2048, but now sparse_count is 0
TEST_F(QuantLightningIndexerTiling, QuantLightningIndexer_910b_tiling_5)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 128, 64, 128}, {2, 128, 64, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2, 512, 1, 128}, {2, 512, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 1}, {2, 512, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 128, 1, 0}, {2, 128, 1, 0}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
