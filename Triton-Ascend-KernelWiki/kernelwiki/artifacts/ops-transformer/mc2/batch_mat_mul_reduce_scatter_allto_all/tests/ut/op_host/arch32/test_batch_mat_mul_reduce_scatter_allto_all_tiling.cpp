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
#include <map>
#include <vector>
#include <string>
#include <float.h>
#include <array>
#include <gtest/gtest.h>
#include <opdev/platform.h>
#include <gmock/gmock.h>
#include "mc2_tiling_case_executor.h"

namespace BatchMatMulReduceScatterAlltoAllUT {

class BatchMatMulReduceScatterAlltoAllArch32TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BatchMatMulReduceScatterAlltoAllArch32TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BatchMatMulReduceScatterAlltoAllArch32TilingTest TearDown" << std::endl;
    }
};

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16Test1)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 17;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16Test1WeightTrans)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 21;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, M0)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 0}, {2, 1024, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 0, 128}, {2, 0, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16Shard)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 17;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16ShardWithBias)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 25;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16Shard0)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16ShardWithBiasTest1)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16ShardWithBiasTest2)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16ShardWithBiasTest3)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{1, 1024, 64}, {1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16ShardWithBiasTest4)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128, 1}, {2, 1, 128, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16ShardWithBiasTest5)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16ShardWithBiasTest6)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{1024, 64}, {1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16ShardWithBiasTest7)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 0}, {2, 1024, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Float16ShardWithBiasTest8)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 1, 128}, {3, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Fp16Shard0WithBias)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 8;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Fp16Shard0NonlocalETailFront)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{17, 3868, 637}, {17, 3868, 637}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{17, 637, 2366}, {17, 637, 2366}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17, 1, 1183}, {17, 1, 1183}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{68, 967, 1183}, {68, 967, 1183}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 8;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Bf16Shard0WithBias)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_BF16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 8;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Fp16Shard0WithBiasInvalidXshape)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1020, 64}, {2, 1020, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Fp16Shard1WithBiasInvalidXshape)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1020, 64}, {2, 1020, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(BatchMatMulReduceScatterAlltoAllArch32TilingTest, Fp16Shard0WithBiasInvalidH)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // namespace