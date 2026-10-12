/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <iostream>
#include <gtest/gtest.h>
#include <string>

// 引入自动生成的 Tiling 头文件
#include "../../../op_host/block_sparse_attention_grad_tiling.h" 
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class BlockSparseAttentionGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "--- BlockSparseAttentionGradTiling UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "--- BlockSparseAttentionGradTiling UT TearDown ---" << std::endl;
    }
};

TEST_F(BlockSparseAttentionGradTilingTest, tiling_bnsd_case0)
{
    optiling::BlockSparseAttentionGradCompileInfo compileInfo;

    int64_t b = 1, n = 4, n_kv = 2, s = 128, s_kv = 128, d = 128;
    int64_t blockX = 64, blockY = 64;
    int64_t ceilQ = (s + blockX - 1) / blockX;
    int64_t ceilKv = (s_kv + blockY - 1) / blockY;

    int64_t blockShapeData[2] = {blockX, blockY};
    int64_t actualSeqData[1] = {s};
    int64_t actualSeqKvData[1] = {s_kv};

    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttentionGrad",
        {
            // --- Input Info ---
            {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, n, s}, {b, n, s}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{b, n, ceilQ, ceilKv}, {b, n, ceilQ, ceilKv}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND},
            
            // 将真实的内存地址传给框架，供 Tiling 内部解析
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, (void*)blockShapeData},
            {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, (void*)actualSeqData},
            {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, (void*)actualSeqKvData}
        },
        {
            // --- Output Info ---
            {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            // --- Attr Info ---

            {"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n_kv)},
            {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)}, 
            {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}
        },
        &compileInfo);

    uint64_t expectTilingKey = 1UL;
    
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}