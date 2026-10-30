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

class QuantLightningIndexerTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantLightningIndexerTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantLightningIndexerTilingArch35 TearDown" << std::endl;
    }
};

// 0
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_0)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 128, 64, 128}, {2, 128, 64, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 512, 1, 128}, {2, 512, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 512, 1}, {2, 512, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
        &compileInfo, "Ascend950", 64, 262144, 16384);

    int64_t expectTilingKey = 205860;
    std::string expectTilingData =
        "2 549755813952 4398046511616 64 12884901888 ";
    std::vector<size_t> expectWorkspaces = {167903232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 1
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_1)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int64_t actualSeqKey[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 64, 16, 128}, {2, 64, 16, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{32, 16, 1, 128}, {32, 16, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 16, 1}, {32, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 64, 1, 2048}, {2, 64, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// When query type is hifloat8, sparse_count must be 2048, but now sparse_count is 1024
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_2)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int64_t actualSeqKey[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 64, 16, 128}, {2, 64, 16, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{32, 16, 1, 128}, {32, 16, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 16, 1}, {32, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 64, 1, 1024}, {2, 64, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
