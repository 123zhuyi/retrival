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
#include "../../../../../prompt_flash_attention/op_host/prompt_flash_attention_tiling_compile_info.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"


using namespace std;

class IncreFlashAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "IncreFlashAttentionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IncreFlashAttentionTiling TearDown" << std::endl;
    }
};

#if 1
TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_0)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2:v: 
         {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 132382977;
    // int64_t expectTilingKey = 4294967295;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "");
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_1)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    // int64_t expectTilingKey = 4294967295;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, "");
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_2)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2:v: 
         {{{5, 2, 237094, 21}, {5, 2, 237094, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{5, 1, 1, 237094}, {5, 1, 1, 237094}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{5, 20, 1, 21}, {5, 20, 1, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.218217f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 266600448;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_3)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{16, 128, 640}, {16, 128, 640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{16, 128, 640}, {16, 128, 640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1612975360;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_4)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,1,2048}, {1,1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1210322176;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_5)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,1,640}, {1,1,640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1,2048,640}, {1,2048,640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1,2048,640}, {1,2048,640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,1,2048}, {1,1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
        {{{1,1,640}, {1,1,640}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1210322177;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_6)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2,1,640}, {2,1,640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1,2048,640}, {1,2048,640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1,2048,640}, {1,2048,640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
        {{{1,1,640}, {1,1,640}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1210322177;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_7)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{16,5,128,128}, {16,5,128,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{16,5,128,128}, {16,5,128,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,1,2048}, {1,1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
         {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         
        },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1747193088;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_8)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{16,5,128,128,1}, {16,5,128,128,1}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{16,5,128,128,1}, {16,5,128,128,1}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
         {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         
        },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1747193088;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_9)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{16,5,128,128}, {16,5,128,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{16,5,128,128,1}, {16,5,128,128,1}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
         {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         
        },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1747193088;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}
#endif

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_10)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1:k: 
         {{{16, 128, 640}, {16, 128, 640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{16, 128, 640}, {16, 128, 640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_BF16, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1612975360;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_11)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16 , ge::FORMAT_ND},
         // 1:k: 
         {{{16, 128, 640}, {16, 128, 640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{16, 128, 640}, {16, 128, 640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_INT8, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1612975360;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_12)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 1:k: 
         {{{16, 128, 640}, {16, 128, 640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{16, 128, 640}, {16, 128, 640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1612975360;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_13)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{1,5,1,2048}, {1,5,1,2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    // int64_t expectTilingKey = 4294967295;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, "");
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_14)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1:k: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{1,5,1,2048}, {1,5,1,2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    // int64_t expectTilingKey = 4294967295;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, "");
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_15)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1:k: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         // 2:v: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_BF16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_16)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1:k: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND},
         // 2:v: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_BF16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_17)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1:k: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         // 2:v: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_18)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 1, 24, 128}, {1, 1, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 1, 24, 128}, {1, 1, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_19)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2364417;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
    // ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_20)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{11,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
    // ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_21)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{11,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
        // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_22)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048,2048}, {1,2048,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
        // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_23)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,1,1,2048}, {1,1,1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
        // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_24)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,1,1,1,2048}, {1,1,1,1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
        // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_25)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
        // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_26)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
        // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_27)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{11,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_INT8 , ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
        // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_28)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{16, 128, 640}, {16, 128, 640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{16, 128, 640}, {16, 128, 640}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1612975360;
        // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_29)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2:v: 
         {{{4, 2048, 128}, {4, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 3:pse_shift
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{4, 1, 1024}, {4, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1206124801;
    // int64_t expectTilingKey = 4294967295;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, "");
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_30)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1:k: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{1,5,1,2048}, {1,5,1,2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    // int64_t expectTilingKey = 4294967295;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, "");
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_31)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2, 2, 24, 128}, {2, 2, 24, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1:k: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{1,24,3,2048}, {1,24,3,2048}}, ge::DT_BF16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{2, 2, 24, 128}, {2, 2, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    // int64_t expectTilingKey = 4294967295;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, "");
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_32)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1, 127}, {1,5,1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1210322176;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_33)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1,5,2048,127}, {1,5,2048,127}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1210322176;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_34)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{16,5,128,128,16}, {16,5,128,128,16}}, ge::DT_INT4, ge::FORMAT_ND},
         // 2:v: 
         {{{16,5,128,128,16}, {16,5,128,128,16}}, ge::DT_INT4, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
         {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         
        },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1747193088;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_35)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{16,5,128,128,16}, {16,5,128,128,16}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{16,5,128,128,16}, {16,5,128,128,16}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{1,16}, {1,16}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
         {{{1,5,1,128}, {1,5,1,128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         
        },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1747193088;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_36)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{2, 736, 24, 128,1}, {2, 736, 24, 128,1}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    // int64_t expectTilingKey = 4294967295;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, "");
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_37)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         // 1:k: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         // 2:v: 
         {{{2, 736, 24, 128}, {2, 736, 24, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{2, 1, 24, 128}, {2, 1, 24, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 2362625;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
    // ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_38)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048,2048}, {1,2048,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1,1}, {1,1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
        // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_39)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048,2048}, {1,2048,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1,1,1,1,1}, {1,1,1,1,1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_40)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1, 736, 24, 128}, {1, 736, 24, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048,2048}, {1,2048,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{1,1,1,1,1}, {1,1,1,1,1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
        {{{1, 2, 24, 128}, {1, 2, 24, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580353;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_41)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,1,2048}, {1,1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
        {{{1,5,1, 128}, {1,5,1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 1210322176;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(IncreFlashAttentionTiling, IncreFlashAttention_950_tiling_42)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 65536, 65536, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
         // 0:q
        {{{1,5,2, 128}, {1,5,2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1:k: 
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 2:v: 
         {{{1,5,2048,128}, {1,5,2048,128}}, ge::DT_INT8, ge::FORMAT_ND},
         // 3:pse_shift
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4:atten_mask
         {{{1,2048}, {1,2048}}, ge::DT_BOOL, ge::FORMAT_ND},
         // 5:actual_seq_lengths
         {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},
         // 6:dequant_scale1
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 7:quant_scale1
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // 8:dequant_scale2
          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 9:quant_scale2
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 10:quant_offset2
          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 11:antiquant_scale
         {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 12:antiquant_offset
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 13:block_table
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // 14:kv_padding_size
         {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
         },
        {
        {{{1,5,2, 128}, {1,5,2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)}
        //  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
         },
                &compileInfo,"Ascend950",64,262144,16384);
    int64_t expectTilingKey = 136580352;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

