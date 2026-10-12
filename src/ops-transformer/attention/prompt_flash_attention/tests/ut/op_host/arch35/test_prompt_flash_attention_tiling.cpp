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
#include "../../../../op_host/prompt_flash_attention_tiling_compile_info.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
using namespace std;

class PromptFlashAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PromptFlashAttentionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PromptFlashAttentionTiling TearDown" << std::endl;
    }
};

// BNSD
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_0)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_qlist[] = {556, 732, 637, 573, 149, 158, 278, 1011, 623, 683, 680, 449, 538, 920, 396, 322, 268, 153, 452, 458, 821, 1001, 744};
    int64_t actual_seq_kvlist[] = {556, 732, 637, 573, 149, 158, 278, 1011, 623, 683, 680, 449, 538, 920, 396, 322, 268, 153, 452, 458, 821, 1001, 744};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{23, 40, 1024, 128}, {23, 40, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND},  // query input0
            {{{23, 40, 9088, 128}, {23, 40, 9088, 128}}, ge::DT_INT8, ge::FORMAT_ND},  // key input1
            {{{23, 40, 9088, 128}, {23, 40, 9088, 128}}, ge::DT_INT8, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // pse_shift input3
            {{{23, 1024, 9088}, {23, 1024, 9088}}, ge::DT_BOOL, ge::FORMAT_ND},    // atten_mask input4
            {{{23}, {23}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{23}, {23}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{40, 128}, {40, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{23, 40, 1024, 128}, {23, 40, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(488)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-127)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 4423156480;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// BSH
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_1)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t* actual_seq_qlist = nullptr;
    int64_t* actual_seq_kvlist = nullptr;
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // pse_shift input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},    // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266600705;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BSND
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_2)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_qlist[] = {2048, 2048, 1024, 1024, 2048, 2028, 2048, 1024};
    int64_t actual_seq_kvlist[] = {2048, 1024, 2048, 2048, 2048, 1024, 1024, 1024};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},  // query input0
            {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},  // key input1
            {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
            {{{8, 1, 2048, 2048}, {8, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},    // atten_mask input4
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{1, 1, 40, 128}, {1, 1, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // quant_scale2 input8
            {{{1, 1, 40, 128}, {1, 1, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_INT8, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1000)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266601217;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// TND
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_3)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    int64_t actual_seq_qlist[] = {409, 818, 1227, 1636, 2045, 2454, 2863, 3272, 3681, 4090, 4499, 4908, 5317, 5726, 6135, 6544, 6953, 7362, 7771, 8180, 8589,
                8998, 9407, 9816, 10225, 10634, 11043, 11452, 11861, 12270, 12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    int64_t actual_seq_kvlist[] = {409, 818, 1227, 1636, 2045, 2454, 2863, 3272, 3681, 4090, 4499, 4908, 5317, 5726, 6135, 6544, 6953, 7362, 7771, 8180, 8589,
                8998, 9407, 9816, 10225, 10634, 11043, 11452, 11861, 12270, 12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},  // query input0
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},  // key input1
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // pse_shift input3
            {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},    // atten_mask input4
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266601986;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BNSD_BSND
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_4)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_qlist[] = {37, 42, 2, 63, 6, 56, 40, 42};
    int64_t actual_seq_kvlist[] = {11, 11, 11, 11, 11, 11, 11, 11};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{8, 20, 65, 50}, {8, 20, 65, 50}}, ge::DT_BF16, ge::FORMAT_ND},  // query input0
            {{{8, 20, 65, 50}, {8, 20, 65, 50}}, ge::DT_BF16, ge::FORMAT_ND},  // key input1
            {{{8, 20, 65, 50}, {8, 20, 65, 50}}, ge::DT_BF16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{8, 65, 20, 50}, {8, 65, 20, 50}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.1443375672974065f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-9)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD_BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 266600960;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// k = 0/v = 0/out = 0
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_5)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
            {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
            {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 2277507072;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// bsngd 32k softcap
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_6)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132383488;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// DT_HIFLOAT8
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_7)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},  // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132383488;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// num_heads < 0 || num_key_value_heads < 0
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_8)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132383488;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// sparse_mode = 0
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_9)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_qlist[] = {2048};
    int64_t actual_seq_kvlist[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132383488;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

//  enableIFA = true
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_10)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 5, 1, 128}, {1, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // pse_shift input3
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{1, 5, 1, 128}, {1, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382976;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

//  enableIFA = true
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_950_tiling_11)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t actual_seq_qlist[] = {};
    int64_t actual_seq_kvlist[] = {};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{1, 5, 1, 128}, {1, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // pse_shift input3
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // atten_mask input4
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{1, 5, 1, 128}, {1, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);
    int64_t expectTilingKey = 132382976;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}
