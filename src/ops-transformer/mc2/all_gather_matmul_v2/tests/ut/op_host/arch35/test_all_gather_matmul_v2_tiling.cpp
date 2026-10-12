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
#include "../../../../../common/inc/kernel/mc2_tiling_struct.h"
#include "../../../../op_kernel/all_gather_matmul_aiv_mode_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace AllGatherMatmulV2UT {

class AllGatherMatmulV2Arch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulV2Arch35TilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulV2Arch35TilingTest TearDown" << std::endl;
    }
};

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_e4m3fn_fp32_rank8_david_ID000)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_e4m3fn_fp32_rank8_david_ID001)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_e4m3fn_fp32_rank8_david_ID002)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_e4m3fn_fp32_rank8_david_ID003)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_e4m3fn_fp32_rank8_david_ID004)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_e4m3fn_fp32_rank8_david_ID005)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_e4m3fn_fp32_rank8_david_ID006)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_e4m3fn_fp32_rank8_david_ID007)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_e4m3fn_fp32_rank8_david_ID008)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_e4m3fn_fp32_rank8_david_ID009)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_fp16_fp32_rank8_david_ID0010)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_bf16_fp32_rank8_david_ID0011)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4096, 1280}, {4096, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_5120_640_fp16_fp32_rank8_david_ID0012)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 5120}, {512, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 640}, {5120, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 640}, {4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 5120}, {4096, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_5120_640_fp16_fp32_rank8_david_ID0013)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{87, 518}, {87, 518}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{518, 1158}, {518, 1158}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{696, 1158}, {696, 1158}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{696, 518}, {696, 518}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_696_518_1185_e4m3_fp32_rank8_david_ID0014)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{87, 518}, {87, 518}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{518, 1158}, {518, 1158}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{696, 1158}, {696, 1158}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{696, 518}, {696, 518}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_696_518_1185_fp16_fp32_rank8_david_ID0015)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{87, 518}, {87, 518}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{518, 1158}, {518, 1158}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{696, 1158}, {696, 1158}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{696, 518}, {696, 518}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_696_518_1185_fp16_fp32_rank8_david_ID0016)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{87, 518}, {87, 518}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{518, 1158}, {518, 1158}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{696, 1158}, {696, 1158}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{696, 518}, {696, 518}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1952_4899_1173_e4m3_fp32_rank8_david_ID0017)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{244, 4899}, {244, 4899}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{4899, 1173}, {4899, 1158}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1952, 1173}, {1952, 1173}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1952, 4899}, {1952, 4899}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1952_4899_1173_e4m3_fp32_rank8_david_ID0018)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{244, 4899}, {244, 4899}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{4899, 1173}, {4899, 1158}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1952, 1173}, {1952, 1173}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1952, 4899}, {1952, 4899}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1952_4899_1173_e4m3_fp32_rank8_david_ID0019)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{244, 4899}, {244, 4899}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{4899, 1173}, {4899, 1158}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1952, 1173}, {1952, 1173}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1952, 4899}, {1952, 4899}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1952_4899_1173_e4m3_fp32_rank8_david_ID0020)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{244, 4899}, {244, 4899}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{4899, 1173}, {4899, 1158}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1952, 1173}, {1952, 1173}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1952, 4899}, {1952, 4899}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1184_6270_2662_e4m3_fp32_rank8_david_ID0021)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{148, 6270}, {148, 6270}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{6270, 2662}, {6270, 2662}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1184, 2662}, {1184, 2662}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1184, 6270}, {1184, 6270}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1776_4464_1591_bf16_fp32_rank8_david_ID0022)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{222, 4464}, {222, 4464}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4464, 1591}, {4464, 1591}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {{{1776, 1591}, {1776, 1591}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1776, 4464}, {1776, 4464}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1176_4472_1315_bf16_fp32_rank8_david_ID0023)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{147, 4472}, {147, 4472}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4472, 1315}, {4472, 1315}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1176, 1315}, {1176, 1315}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1176, 4472}, {1176, 4472}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1104_5482_1029_bf16_fp32_rank8_david_ID0024)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{138, 5482}, {138, 5482}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{5482, 1029}, {5482, 1029}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1104, 1029}, {1104, 1026}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1104, 5482}, {1104, 5482}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1104_4779_983_bf16_fp32_rank8_david_ID0025)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{138, 4779}, {138, 4779}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4779, 983}, {4779, 983}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1104, 983}, {1104, 983}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1104, 4779}, {1104, 4779}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1872_5251_1579_bf16_fp32_rank8_david_ID0026)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{234, 5251}, {234, 5251}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{5251, 1579}, {5251, 1579}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1872, 1579}, {1872, 1579}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1872, 5251}, {1872, 5251}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1720_4930_887_bf16_fp32_rank8_david_ID0027)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{215, 4930}, {215, 4930}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4930, 887}, {4930, 887}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1720, 887}, {1720, 887}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1720, 4930}, {1720, 4930}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1744_4904_2022_bf16_fp32_rank8_david_ID0028)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{218, 4904}, {218, 4904}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4904, 2022}, {4904, 2022}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1744, 2022}, {1744, 2022}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1744, 4904}, {1744, 4904}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1592_4124_1797_bf16_fp32_rank8_david_ID0029)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{199, 4124}, {199, 4124}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4124, 1797}, {4124, 1797}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1592, 1797}, {1592, 1797}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1592, 4124}, {1592, 4124}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1232_6049_1065_bf16_fp32_rank8_david_ID0030)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{154, 6049}, {154, 6049}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{6049, 1065}, {6049, 1065}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1232, 1065}, {1232, 1065}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1232, 6049}, {1232, 6049}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1520_6050_1463_bf16_fp32_rank8_david_ID0031)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{190, 6050}, {190, 6050}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{6050, 1463}, {6050, 1463}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1520, 1463}, {1520, 1463}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1520, 6050}, {1520, 6050}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1280_4212_2151_bf16_fp32_rank8_david_ID0032)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{160, 4212}, {160, 4212}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4212, 2151}, {4212, 2151}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1280, 2151}, {1280, 2151}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1280, 4212}, {1280, 4212}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_2184_4110_856_bf16_fp32_rank8_david_ID0033)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{273, 4110}, {273, 4110}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4110, 856}, {4110, 856}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2184, 856}, {2184, 856}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2184, 4110}, {2184, 4110}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1048_6808_1331_bf16_fp32_rank8_david_ID0034)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{131, 6808}, {131, 6808}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{6808, 1331}, {6808, 1331}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1048, 1331}, {1048, 1331}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1048, 6808}, {1048, 6808}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1288_4188_1822_bf16_fp32_rank8_david_ID0035)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{161, 4188}, {161, 4188}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4188, 1822}, {4188, 1822}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1288, 1822}, {1288, 1822}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1288, 4188}, {1288, 4188}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_2160_4154_1155_bf16_fp32_rank8_david_ID0036)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{270, 4154}, {270, 4154}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4154, 1155}, {4154, 1155}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2160, 1155}, {2160, 1155}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2160, 4154}, {2160, 4154}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1056_7241_784_bf16_fp32_rank8_david_ID0037)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{132, 7241}, {132, 7241}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{7241, 784}, {7241, 784}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1056, 784}, {1056, 784}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1056, 7241}, {1056, 7241}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1416_4499_1788_bf16_fp32_rank8_david_ID0039)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{177, 4499}, {177, 4499}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4499, 1788}, {4499, 1788}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1416, 4499}, {1416, 4499}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1416, 4499}, {1416, 4499}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_2296_4329_1927_bf16_fp32_rank8_david_ID0040)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{287, 4329}, {287, 4329}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4329, 1927}, {4329, 1927}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2296, 1927}, {2296, 1927}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2296, 4329}, {2296, 4329}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1240_4856_1443_bf16_fp32_rank8_david_ID0041)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{155, 4856}, {155, 4856}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4856, 1443}, {4856, 1443}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1240, 1443}, {1240, 1443}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1240, 4856}, {1240, 4856}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1240_4856_1443_bf16_fp32_rank8_david_ID0042)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{155, 4856}, {155, 4856}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4856, 1443}, {4856, 1443}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1240, 1443}, {1240, 1443}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1240, 4856}, {1240, 4856}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1442_4155_2097_bf16_fp32_rank8_david_ID0043)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{180, 4155}, {180, 4155}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4155, 2097}, {4155, 2097}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1440, 2097}, {1440, 2097}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1440, 4155}, {1440, 4155}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1936_4948_1415_bf16_fp32_rank8_david_ID0044)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{242, 4948}, {242, 4948}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4948, 1415}, {4948, 1415}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1936, 1415}, {1936, 1415}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1936, 4948}, {1936, 4948}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1672_4912_664_bf16_fp32_rank8_david_ID0045)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{209, 4912}, {209, 4912}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4912, 664}, {4912, 664}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1672, 664}, {1672, 664}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1672, 4912}, {1672, 4912}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1376_4814_1394_fp16_fp32_rank8_david_ID0047)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{172, 4814}, {172, 4814}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4814, 1394}, {4814, 1394}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1376, 1394}, {1376, 1394}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1376, 4814}, {1376, 4814}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1528_4227_798_fp16_fp32_rank8_david_ID0048)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{191, 4227}, {191, 4227}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4227, 798}, {4227, 798}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1528, 798}, {1528, 798}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1528, 4227}, {1528, 4227}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1056_4275_2219_fp16_fp32_rank8_david_ID0049)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{132, 4275}, {132, 4275}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4275, 2219}, {4275, 2219}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1056, 2219}, {1056, 2219}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1056, 4275}, {1056, 4275}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_1976_4724_1449_fp16_fp32_rank8_david_ID0050)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{247, 4724}, {247, 4724}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4724, 1449}, {4724, 1449}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1976, 1449}, {1976, 1449}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1976, 4724}, {1976, 4724}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    uint64_t expectTilingKey = 1ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_1001_6001_false_false_bf16_coreNum_is_0_950)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 0
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 2}
    };
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_1001_6001_false_false_bf16_950)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 2}
    };
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_1001_6001_false_false_hif8_hif8_y_float32_pertensor_950)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{8192, 1280}, {8192, 1280}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 1280}, {8192, 1280}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8192, 8192}, {8192, 8192}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 2}
    };
    uint64_t expectTilingKey = 8ULL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulV2Arch35TilingTest, all_gather_matmul_4096_8192_1280_false_true_hif8_hif8_y_float32_perblock_950)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            {{{512, 8192}, {512, 8192}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{8192, 1024}, {8192, 1024}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 8}, {64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(549764202624)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        },
        &compileInfo, "3510", 32
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", 8}
    };
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // namespace