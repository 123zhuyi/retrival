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
#include "../../../../op_kernel/arch35/matmul_reduce_scatter_v2_c_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace MatmulReduceScatterV2UT {

class MatmulReduceScatterV2Arch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulReduceScatterV2Arch35TilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MatmulReduceScatterV2Arch35TilingTest TearDown" << std::endl;
    }
};

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_e4m3fn_e4m3fn_fp32_rank8_reducescatterv2_david_ID000)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}

        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_e4m3fn_e5m2_fp32_rank8_reducescatterv2_david_ID001)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_e5m2_e5m2_fp32_rank8_reducescatterv2_david_ID002)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_hif8_hif8_fp32_rank8_reducescatterv2_david_ID003)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_e4m3fn_e4m3fn_fp16_rank8_reducescatterv2_david_ID004)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_e4m3fn_e5m2_fp16_rank8_reducescatterv2_david_ID005)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_e5m2_e5m2_fp16_rank8_reducescatterv2_david_ID006)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_hif8_hif8_fp16_rank8_reducescatterv2_david_ID007)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_e4m3fn_e4m3fn_bf16_rank8_reducescatterv2_david_ID008)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_e4m3fn_e5m2_bf16_rank8_reducescatterv2_david_ID009)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_e5m2_e5m2_bf16_rank8_reducescatterv2_david_ID010)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_hif8_hif8_bf16_rank8_reducescatterv2_david_ID011)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_e4m3fn_fp32_rank8_reducescatterv2_david_ID012)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_e5m2_e4m3fn_fp32_rank8_reducescatterv2_david_ID013)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_e5m2_e5m2_fp32_rank8_reducescatterv2_david_ID014)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_hif8_hif8_fp32_rank8_reducescatterv2_david_ID015)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_e4m3fn_e4m3fn_fp16_rank8_reducescatterv2_david_ID016)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_e5m2_e4m3fn_fp32_rank8_reducescatterv2_david_ID017)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_e5m2_e5m2_fp16_rank8_reducescatterv2_david_ID018)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120__hif8_hif8_fp16_rank8_reducescatterv2_david_ID019)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_e4m3fn_e4m3fn_bf16_rank8_reducescatterv2_david_ID020)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_e5m2_e4m3fn_bf16_rank8_reducescatterv2_david_ID021)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_e5m2_e5m2_bf16_rank8_reducescatterv2_david_ID022)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_hif8_hif8_bf16_rank8_reducescatterv2_david_ID023)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 40}, {32, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_fp16_fp16_rank8_reducescatterv2_david_ID024)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_bf16_bf16_rank8_reducescatterv2_david_ID025)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 5120}, {4096, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_1024_8192_fp16_fp16_rank8_reducescatterv2_david_ID026)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, 4096_640_5120_e5m2_e4m3fn_bf16_rank8_reducescatterv2_david_ID027)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Float16Test1)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Float16Test2)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Float16Test3)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{16384, 4096}, {16384, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 2752}, {4096, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16384, 2752}, {16384, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Float16Test4)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{16384, 4096}, {16384, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 2752}, {4096, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16384, 2752}, {16384, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Float16Test5)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 2752}, {4096, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 2752}, {4096, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 24
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Bfloat16)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, DoubleRing)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Test2pFullmesh)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 2}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Fp8e4m3Fp8e4m3YFp16)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 0ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Fp8e4m3Fp8e4m3YFp16Perblock)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 12}, {64, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12, 96}, {12, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Fpe4m3Fpe5m2YFloat32)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 64ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Fpe4m3Fpe5m2YFloat32Perblock)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 12}, {64, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12, 96}, {12, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Hif8Hif8YFloat32)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 64ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Hif8Hif8YFloat32Perblock)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 12}, {64, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12, 96}, {12, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Hif8Hif8YFloat32PerblockErrorDivid)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8191, 1536}, {8191, 1536}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 12}, {64, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12, 96}, {12, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8191, 12288}, {8191, 12288}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202623)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Hif8Hif8YFloat32PerblockSerial)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8184, 1536}, {8184, 1536}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 12}, {64, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12, 96}, {12, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8184, 12288}, {8184, 12288}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 65ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Fp8e4m3Fp8e4m3YFp16X1scaleerror)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, TestTilingCommBound)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{16384, 512}, {16384, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 15744}, {512, 15744}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16384, 15744}, {16384, 15744}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 40ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterV2Arch35TilingTest, Hif8Hif8YFloat32PerblockErrorScale1Shape)
{
    struct MatmulReduceScatterV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128, 12}, {128, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12, 96}, {12, 96}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aicpu")}
        },
        &compileInfo, "3510", 20
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // namespace