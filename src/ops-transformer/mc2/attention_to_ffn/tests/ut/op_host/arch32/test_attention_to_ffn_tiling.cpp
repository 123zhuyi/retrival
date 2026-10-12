/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_kernel/attention_to_ffn_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace AttentionToFFNUT {

class AttentionToFFNArch32TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AttentionToFFNArch32TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AttentionToFFNArch32TilingTest TearDown" << std::endl;
    }
};

TEST_F(AttentionToFFNArch32TilingTest, NoQuantSyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, NoQuantAsyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 2UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, QuantSyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 1UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, QuantAsyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 3UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, NoQuantSyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );
    uint64_t expectTilingKey = 4UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, NoQuantAsyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 6UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, QuantSyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 5UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, QuantAsyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 7UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, MicroBatchNumInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{2, 16, 7168}, {2, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, BsInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 513, 7168}, {1, 513, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, KInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 16}, {1, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, HInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 8196}, {1, 16, 8196}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, LayerNumInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 9, 4}, {2, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, MoeExpertNumInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1026, 4}, {1, 1026, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, FfnTokenInfoTableShapeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, FfnTokenDataShapeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttnTokenInfoTableShapeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, XDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168, 1}, {1, 16, 7168, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, SessionIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, MicroBatchIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, LayerIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ExpertIdsDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ExpertRankTableDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{9, 4}, {9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ScalesDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168, 1}, {1, 9, 7168, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ActiveMaskDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, XDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, SessionIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, MicroBatchIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, LayerIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ExpertIdsDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ExpertRankTableDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ScalesDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ActiveMaskDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_INT8, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, XFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, SessionIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, MicroBatchIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, LayerIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ExpertIdsFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ExpertRankTableDormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ScalesFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, ActiveMaskFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // AttentionToFFNUT