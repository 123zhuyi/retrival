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
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FusedInferAttentionScoreProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedInferAttentionScoreProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedInferAttentionScoreProto TearDown" << std::endl;
    }
};

TEST_F(FusedInferAttentionScoreProto, FusedInferAttentionScore_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore",
        // 输入Tensor
        {
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{4, 13, 10347}, {4, 13, 10347}}, ge::DT_INT8, ge::FORMAT_ND},            // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{4, 13, 16, 64}, {4, 13, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{4, 10347, 1, 64}, {4, 10347, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                       // 输出Tensor
         {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                              // softmax_lse
        {                                                                       // 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        // {},  // softmaxMax
        // {},  // softmaxSum
        {4, 13, 16, 512}, // attentionOut
        {0},              // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedInferAttentionScoreProto, FusedInferAttentionScore_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore",
        // 输入Tensor
        {
            {{{32, 1, 1024}, {32, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{32, 4096, 128}, {32, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{32, 4096, 128}, {32, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                              // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                            // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{1, 32, 4096}, {1, 32, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_antiquant_offset-input16
            {{{1, 32, 4096}, {1, 32, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{32, 1, 1024}, {32, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{32, 8, 1, 1}, {32, 8, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},  // softmax_lse
        {                                                                 // 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        // {},  // softmaxMax
        // {},  // softmaxSum
        {32, 1, 1024}, // attentionOut
        {32, 8, 1, 1}, // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedInferAttentionScoreProto, FusedInferAttentionScore_infershape_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore",
        // 输入Tensor
        {
            {{{4, 1, 8, 128}, {4, 1, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // query-input0
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{4, 8, 1, 2048}, {4, 8, 1, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // pse_shift-input3
            {{{4, 1, 1, 2048}, {4, 1, 1, 2048}}, ge::DT_INT8, ge::FORMAT_ND},        // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                     // query_padding_size-input13
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                   // kv_padding_size-input14
            {{{1, 1, 128}, {1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // key_antiquant_offset-input16
            {{{1, 1, 128}, {1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                     // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                     // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                     // kv_start_idx-input27
        },
        {                                                                   // 输出Tensor
         {{{4, 1, 8, 128}, {4, 1, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{4, 8, 1, 1}, {4, 8, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},      // softmax_lse
        {                                                                   // 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        // {},  // softmaxMax
        // {},  // softmaxSum
        {4, 1, 8, 128}, // attentionOut
        {4, 8, 1, 1},   // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedInferAttentionScoreProto, FusedInferAttentionScore_infershape_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore",
        // 输入Tensor
        {
            {{{8, 32, 3, 512}, {8, 32, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{8, 1, 256, 512}, {8, 1, 256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{8, 1, 256, 512}, {8, 1, 256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{8, 12}, {8, 12}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{10, 1, 128, 512}, {10, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{10, 1, 128, 512}, {10, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{8, 32, 3, 64}, {8, 32, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{8, 1, 256, 64}, {8, 1, 256, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{10, 1, 128, 64}, {10, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{8, 32, 3, 512}, {8, 32, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {                                                                     // 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         //  {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        // {},  // softmaxMax
        // {},  // softmaxSum
        {8, 32, 3, 512}, // attentionOut
        {0},             // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedInferAttentionScoreProto, FusedInferAttentionScore_infershape_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore",
        // 输入Tensor
        {
            {{{8, 32, 3, 512}, {8, 32, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // query-input0
            {{{8, 1, 256, 512}, {8, 1, 256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{8, 1, 256, 512}, {8, 1, 256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // pse_shift-input3
            {{{2048, 2048}, {}}, ge::DT_INT8, ge::FORMAT_ND},                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                              // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // antiquant_offset-input11
            {{{8, 12}, {8, 12}}, ge::DT_INT32, ge::FORMAT_ND},                     // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{10, 1, 128, 512}, {10, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{10, 1, 128, 512}, {10, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{8, 32, 3, 64}, {8, 32, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_rope_antiquant_scale-input23
            {{{8, 1, 256, 64}, {8, 1, 256, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},       // dequant_scale_query-input24
            {{{10, 1, 128, 64}, {10, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{8, 32, 3, 512}, {8, 32, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {                                                                     // 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         //  {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        // {},  // softmaxMax
        // {},  // softmaxSum
        {8, 32, 512}, // attentionOut
        {0},          // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedInferAttentionScoreProto, FusedInferAttentionScore_infershape_5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore",
        // 输入Tensor
        {
            {{{32, 1, 1024}, {32, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // query-input0
            {{{32, 4096, 128}, {32, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{32, 4096, 128}, {32, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                              // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                            // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // antiquant_offset-input11
            {{{8, 12}, {8, 12}}, ge::DT_INT32, ge::FORMAT_ND},                   // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_padding_size-input14
            {{{1, 32, 4096}, {1, 32, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_antiquant_offset-input16
            {{{1, 32, 4096}, {1, 32, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                           // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                             // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{32, 1, 1024}, {32, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{32, 8, 1, 1}, {32, 8, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},  // softmax_lse
        {                                                                 // 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        // {},  // softmaxMax
        // {},  // softmaxSum
        {32, 1, 1024}, // attentionOut
        {32, 8, 1, 1}, // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedInferAttentionScoreProto, FusedInferAttentionScore_infershape_6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore",
        // 输入Tensor
        {
            {{{3, 1, 16, 128}, {3, 1, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // query-input0
            {{{3, 55648, 8, 128}, {3, 55648, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{3, 55648, 8, 128}, {3, 55648, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},          // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{3, 1, 16, 64}, {3, 1, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // key_rope_antiquant_scale-input23
            {{{3, 55648, 8, 64}, {3, 55648, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{3, 16, 1, 128}, {3, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {                                                                     // 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687836487032f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(55648)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND_BNSD")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        // {},  // softmaxMax
        // {},  // softmaxSum
        {3, 16, 1, 128}, // attentionOut
        {0},             // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedInferAttentionScoreProto, FusedInferAttentionScore_infershape_7)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore",
        // 输入Tensor
        {
            {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{32, 1, 1020, 512}, {32, 1, 1020, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{32, 1, 1020, 512}, {32, 1, 1020, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{2048, 2048}, {2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{286, 1, 128, 512}, {286, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{286, 1, 128, 512}, {286, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{32, 1, 1020, 64}, {32, 1, 1020, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{286, 1, 128, 64}, {286, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {                                                                 // 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417382415922f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1020)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        // {},  // softmaxMax
        // {},  // softmaxSum
        {12, 64, 512}, // attentionOut
        {0},           // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedInferAttentionScoreProto, FusedInferAttentionScore_infershape_8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore",
        // 输入Tensor
        {
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_INT8, ge::FORMAT_ND},     // query-input0
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // key-input1
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // pse_shift-input3
            {{{4, 13, 10347}, {4, 13, 10347}}, ge::DT_INT8, ge::FORMAT_ND},         // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // key_rope-input22
            {{{4, 13, 16, 64}, {4, 13, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key_rope_antiquant_scale-input23
            {{{4, 10347, 1, 64}, {4, 10347, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // kv_start_idx-input27
        },
        {                                                                    // 输出Tensor
         {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                           // softmax_lse
        {                                                                    // 属性
         {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        // {},  // softmaxMax
        // {},  // softmaxSum
        {}, // attentionOut
        {}, // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(FusedInferAttentionScoreProto, FusedInferAttentionScore_infershape_9)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedInferAttentionScore",
        // 输入Tensor
        {
            {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // query-input0
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                // atten_mask-input4
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // actual_shared_prefix_len-空
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{64, 12, 512}, {64, 12, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {                                                                 // 属性
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND_NTD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        // {},  // softmaxMax
        // {},  // softmaxSum
        {64, 12, 512}, // attentionOut
        {0},           // softmaxOut
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}