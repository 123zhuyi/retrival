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
#include "mc2_tiling_case_executor.h"

using namespace std;

namespace MatmulReduceScatterUT {

class MatmulReduceScatterArch32TilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulReduceScatterArch32TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulReduceScatterArch32TilingTest TearDown" << std::endl;
    }
};

TEST_F(MatmulReduceScatterArch32TilingTest, Float16Test1)
{
    struct MatmulReduceScatterCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulReduceScatter",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterArch32TilingTest, Float16Test2)
{
    struct MatmulReduceScatterCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulReduceScatter",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterArch32TilingTest, Float16Test3)
{
    struct MatmulReduceScatterCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulReduceScatter",
        {
            {{{16384, 4096}, {16384, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 2752}, {4096, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{{{16384, 2752}, {16384, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterArch32TilingTest, Float16Test4)
{
    struct MatmulReduceScatterCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulReduceScatter",
        {
            {{{16384, 4096}, {16384, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 2752}, {4096, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{{{16384, 2752}, {16384, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterArch32TilingTest, Float16Test5)
{
    struct MatmulReduceScatterCompileInfo {} compileInfo;
    uint64_t coreNum = 24;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulReduceScatter",
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 2752}, {4096, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{{{4096, 2752}, {4096, 2752}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterArch32TilingTest, Float16Test6)
{
    struct MatmulReduceScatterCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulReduceScatter",
        {
            {{{8192, 512}, {8192, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 12288}, {512, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulReduceScatterArch32TilingTest, Bfloat16)
{
    struct MatmulReduceScatterCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulReduceScatter",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{{{8192, 12288}, {8192, 12288}}, ge::DT_BF16, ge::FORMAT_ND},},
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 7;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

} // namespace