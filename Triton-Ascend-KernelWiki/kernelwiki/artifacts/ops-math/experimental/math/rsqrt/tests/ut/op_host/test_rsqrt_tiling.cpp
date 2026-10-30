/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_rsqrt_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/rsqrt_tiling_data.h"
#include "../../../op_kernel/rsqrt_tiling_key.h"

using namespace std;
using namespace ge;

class RsqrtTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RsqrtTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "RsqrtTiling TearDown" << std::endl; }
};

TEST_F(RsqrtTiling, rsqrt_tiling_001)
{
    struct RsqrtCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("Rsqrt",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo,
                                              40, // Mock cube Core Num， vector core固定64
                                              196608 + 256 // Mock ubsize = 192k 供UT使用,获取的时候系统会自动减掉256
    );
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 72 1 1 16296 64 72 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RsqrtTiling, rsqrt_tiling_002)
{
    struct RsqrtCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("Rsqrt",
                                              {
                                                  {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo,
                                              40, // Mock cube Core Num， vector core固定64
                                              196608 + 256 // Mock ubsize = 192k 供UT使用,获取的时候系统会自动减掉256
    );
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1024 1032 1 1 16296 1024 1032 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RsqrtTiling, rsqrt_tiling_003)
{
    struct RsqrtCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("Rsqrt",
                                              {
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo,
                                              40, // Mock cube Core Num
                                              196608 + 256 // Mock ubsize = 192k 供UT使用,获取的时候系统会自动减掉256
    );
    uint64_t expectTilingKey = 0;
    string expectTilingData = "52352 52360 7 7 8144 3488 3496 1 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RsqrtTiling, rsqrt_tiling_004)
{
    struct RsqrtCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("Rsqrt",
                                              {
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo,
                                              40, // Mock cube Core Num
                                              196608 + 256 // Mock ubsize = 192k 供UT使用,获取的时候系统会自动减掉256
    );
    uint64_t expectTilingKey = 0;
    string expectTilingData = "52352 52368 4 4 16288 3488 3504 1 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RsqrtTiling, rsqrt_tiling_005)
{
    struct RsqrtCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("Rsqrt",
                                              {
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo,
                                              40, // Mock cube Core Num
                                              196608 + 256 // Mock ubsize = 192k 供UT使用,获取的时候系统会自动减掉256
    );
    uint64_t expectTilingKey = 0;
    string expectTilingData = "52352 52368 7 7 8144 3488 3504 1 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// INT32 (ubDataNumber=2) 小数据场景：单核、单buffer，验证 INT32 tiling 路径
TEST_F(RsqrtTiling, rsqrt_tiling_006)
{
    struct RsqrtCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("Rsqrt",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 40, 196608 + 256);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 72 1 1 24448 64 72 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// UINT8 (ubDataNumber=4) 大数据场景：多核、双buffer，验证 UINT8 tiling 路径
TEST_F(RsqrtTiling, rsqrt_tiling_007)
{
    struct RsqrtCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("Rsqrt",
                                              {
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1023, 2047}, {1023, 2047}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 40, 196608 + 256);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "52352 52384 3 3 24448 3456 3488 1 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// INT8 (ubDataNumber=6) 中等数据场景：多核、单buffer，验证 INT8 tiling 路径
TEST_F(RsqrtTiling, rsqrt_tiling_008)
{
    struct RsqrtCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("Rsqrt",
                                              {
                                                  {{{8, 2048}, {8, 2048}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 2048}, {8, 2048}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 40, 196608 + 256);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1024 1056 1 1 32576 1024 1056 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// INT16 (ubDataNumber=2) 小数据场景：单核、单buffer，验证 INT16 tiling 路径
TEST_F(RsqrtTiling, rsqrt_tiling_009)
{
    struct RsqrtCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("Rsqrt",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 40, 196608 + 256);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 80 1 1 48896 64 80 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// BOOL (ubDataNumber=2) 小数据场景：单核、单buffer，验证 BOOL tiling 路径
TEST_F(RsqrtTiling, rsqrt_tiling_010)
{
    struct RsqrtCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("Rsqrt",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 40, 196608 + 256);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 96 1 1 97792 64 96 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
