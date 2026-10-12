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
 * \file test_add_v2_tiling.cpp
 * \brief add_v2 tiling UT for ascend950 (arch35)
 */

#include "math/add_v2/op_host/arch35/add_v2_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

// 空 Tensor 分支的 tiling key = 65550 = 0x1000E：
//   低 16 位 0x000E = 14，是 schMode 999 在 BRC_TEMP_SCH_MODE_KEY_DECL 取值表
//                     (1,2,101,102,103,104,109,201,202,301,302,303,304,305,999) 里的序号；
//   第 16 位 = userDef = 1。
// 常规通路是 userDef = 0，所以高位为 0，原有用例的 key 不受影响（仍为 8）。
static constexpr uint64_t ADD_V2_UT_EMPTY_TILING_KEY = 65550;

class AddV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddV2Tiling TearDown" << std::endl; }
};

TEST_F(AddV2Tiling, add_v2_tiling_fp32)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_fp16_broadcast)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_bf16)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_int64)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_complex64)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                  {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_invalid_dtype)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ── 空 Tensor ──────────────────────────────────────────────────────────────
// ATVOSS 的 BroadcastBaseTiling 在合轴后显式拒绝 0 元素，空 Tensor 必须走
// 自定义模板分支（schMode 999 + userDef 1），blockDim = 1，kernel 侧直接返回。
TEST_F(AddV2Tiling, add_v2_tiling_empty_1d)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = ADD_V2_UT_EMPTY_TILING_KEY;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AddV2Tiling, add_v2_tiling_empty_2d)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{0, 3}, {0, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = ADD_V2_UT_EMPTY_TILING_KEY;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 空张量 + 广播：x1 空、x2 是标量，输出仍为空
TEST_F(AddV2Tiling, add_v2_tiling_empty_broadcast)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 3}, {1, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = ADD_V2_UT_EMPTY_TILING_KEY;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 高维中间维为 0
TEST_F(AddV2Tiling, add_v2_tiling_empty_highrank)
{
    optiling::AddV2CompileInfoArch35 compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AddV2",
                                              {
                                                  {{{2, 0, 4}, {2, 0, 4}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{2, 0, 4}, {2, 0, 4}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 0, 4}, {2, 0, 4}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = ADD_V2_UT_EMPTY_TILING_KEY;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
