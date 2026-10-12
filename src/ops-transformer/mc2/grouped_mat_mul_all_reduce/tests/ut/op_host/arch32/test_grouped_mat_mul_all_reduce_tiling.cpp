/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_mat_mul_all_reduce_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/op_tiling/grouped_mat_mul_all_reduce_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace GroupedMatMulAllReduceUT {
    
class GroupedMatMulAllReduceArch32TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatMulAllReduceArch32TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatMulAllReduceArch32TilingTest TearDown" << std::endl;
    }
};
struct GroupedMatMulAllReduceCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

TEST_F(GroupedMatMulAllReduceArch32TilingTest, Float16Test1)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {20, 196608};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 40960;
    std::string group("group");
    std::string reduceOp("sum");
    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        // input info
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        // output info
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        // attr
        {   
            {"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
            {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize
        );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t expectTilingKey = 0;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GroupedMatMulAllReduceArch32TilingTest, McutFloat16In910BTest1)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {20, 196608};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 40960;
    std::string group("group");
    std::string reduceOp("sum");
    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        // input info
        {
            {{{12290, 15360}, {12290, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{15360, 12288}, {15360, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        // output info
        {
            {{{12290, 12288}, {12290, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        // attr
        {   
            {"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
            {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t expectTilingKey = 0;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GroupedMatMulAllReduceArch32TilingTest, McutFloat16In910BTest2)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {20, 196608};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 40960;
    std::string group("group");
    std::string reduceOp("sum");
    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        // input info
        {
            {{{20, 2}, {20, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        // output info
        {
            {{{20, 2}, {20, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        // attr
        {   
            {"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
            {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    int64_t expectTilingKey = 0;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GroupedMatMulAllReduceArch32TilingTest, McutFloat16In910BTestwin2win)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {20, 196608};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 40960;
    std::string group("group");
    std::string reduceOp("sum");
    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        // input info
        {
            {{{12290, 15360}, {12290, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{15360, 12288}, {15360, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        // output info
        {
            {{{12290, 12288}, {12290, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        // attr
        {   
            {"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
            {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t expectTilingKey = 0;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GroupedMatMulAllReduceArch32TilingTest, Float16Test2)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {20, 196608};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 40960;
    std::string group("group");
    std::string reduceOp("sum");
    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        // input info
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        // output info
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        // attr
        {   
            {"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
            {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t expectTilingKey = 0;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GroupedMatMulAllReduceArch32TilingTest, Float16Test3)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {20, 196608};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 40960;
    std::string group("group");
    std::string reduceOp("sum");
    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        // input info
        {
            {{{128, 1536}, {128, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        // output info
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        // attr
        {   
            {"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
            {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t expectTilingKey = 0;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GroupedMatMulAllReduceArch32TilingTest, Float16Test4)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {20, 196608};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 40960;
    std::string group("group");
    std::string reduceOp("sum");
    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        // input info
        {
            {{{1024, 1536}, {1024, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        // output info
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        // attr
        {   
            {"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
            {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t expectTilingKey = 0;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GroupedMatMulAllReduceArch32TilingTest, Float16Test5)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {20, 196608};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 40960;
    std::string group("group");
    std::string reduceOp("sum");
    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        // input info
        {
            {{{256, 1536}, {256, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        // output info
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        // attr
        {   
            {"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
            {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t expectTilingKey = 0;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GroupedMatMulAllReduceArch32TilingTest, Float16Support3Dim)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {20, 196608};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 40960;
    std::string group("group");
    std::string reduceOp("sum");
    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        // input info
        {
            {{{1, 8192, 1536}, {1, 8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        // output info
        {
            {{{1, 8192, 12288}, {1, 8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        // attr
        {   
            {"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
            {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t expectTilingKey = 0;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GroupedMatMulAllReduceArch32TilingTest, Bfloat16)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {20, 196608};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 40960;
    std::string group("group");
    std::string reduceOp("sum");
    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        // input info
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        // output info
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        // attr
        {   
            {"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group)},
            {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduceOp)},
            {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t expectTilingKey = 0;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}
}
