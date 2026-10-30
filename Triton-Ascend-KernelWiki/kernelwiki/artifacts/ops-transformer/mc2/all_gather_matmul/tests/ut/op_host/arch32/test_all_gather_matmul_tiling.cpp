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
#include "../../../../op_kernel/all_gather_matmul_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace AllGatherMatmulUT {

class AllGatherMatmulArch32TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulArch32TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulArch32TilingTest TearDown" << std::endl;
    }
};

TEST_F(AllGatherMatmulArch32TilingTest, Float16Test1)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{512, 12288}, {512, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288, 3904}, {12288, 3904}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{4096, 3904}, {4096, 3904}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 12288}, {4096, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulArch32TilingTest, Float16Test2)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{16384, 1536}, {16384, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 4096}, {16384, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulArch32TilingTest, Float16Test3)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{327680, 15360}, {327680, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{15360, 10240}, {15360, 10240}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{2621440, 10240}, {2621440, 10240}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2621440, 15360}, {2621440, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulArch32TilingTest, Bfloat16)
{
    // tilingFunc simulate
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{16384, 1536}, {16384, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 4096}, {16384, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 7UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulArch32TilingTest, Float16TestL2cache)
{
    // tilingFunc simulate
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{8192, 5120}, {8192, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 12288}, {5120, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{65536, 12288}, {65536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{65536, 5120}, {65536, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 7UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulArch32TilingTest, N0)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{1024, 256}, {1024, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 0}, {256, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{8192, 0}, {8192, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 256}, {8192, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

} // AllGatherMatmulUT