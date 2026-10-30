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
 * \file test_acos_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/acos_tiling_data.h"
#include "../../../op_kernel/acos_tiling_key.h"

using namespace std;
using namespace ge;

class AcosTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AcosTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AcosTiling TearDown" << std::endl; }
};

TEST_F(AcosTiling, acos_tiling_001)
{
    struct AcosCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Acos",
        {
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo,
        40,          // Mock cube Core Num， vector core固定64
        196608 + 256 // Mock ubsize = 192k 供UT使用,获取的时候系统会自动减掉256
    );
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 0 64 0 1 64 64 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AcosTiling, acos_tiling_002)
{
    struct AcosCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Acos",
        {
            {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo,
        40,          // Mock cube Core Num， vector core固定64
        196608 + 256 // Mock ubsize = 192k 供UT使用,获取的时候系统会自动减掉256
    );
    uint64_t expectTilingKey = 0;
    string expectTilingData = "24 16 410 409 1 410 410 1 409 409 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AcosTiling, acos_tiling_003)
{
    struct AcosCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Acos",
        {
            {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo,
        40,          // Mock cube Core Num
        196608 + 256 // Mock ubsize = 192k 供UT使用,获取的时候系统会自动减掉256
    );
    uint64_t expectTilingKey = ge::DT_FLOAT;
    string expectTilingData = "1 39 52353 52352 11 4928 3073 11 4928 3072 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AcosTiling, acos_tiling_004)
{
    struct AcosCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Acos",
        {
            {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo,
        40,          // Mock cube Core Num
        196608 + 256 // Mock ubsize = 192k 供UT使用,获取的时候系统会自动减掉256
    );
    uint64_t expectTilingKey = ge::DT_FLOAT16;
    string expectTilingData = "1 39 52353 52352 9 6144 3201 9 6144 3200 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AcosTiling, acos_tiling_005)
{
    struct AcosCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Acos",
        {
            {{{1023, 2047}, {1023, 2047}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1023, 2047}, {1023, 2047}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo,
        40,          // Mock cube Core Num
        196608 + 256 // Mock ubsize = 192k 供UT使用,获取的时候系统会自动减掉256
    );
    uint64_t expectTilingKey = ge::DT_BF16;
    string expectTilingData = "1 39 52353 52352 9 6144 3201 9 6144 3200 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}