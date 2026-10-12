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
#include "tiling_case_executor.h"
#include "../../../op_kernel/not_equal_v2_tiling_data.h"

bool operator==(const NotEqualTilingData& lhs, const NotEqualTilingData& rhs) { return lhs.size == rhs.size; }

template <typename T>
void ExecuteTestCase(
    const gert::TilingContextPara& tilingContextPara, ge::graphStatus expectResult, uint64_t expectTilingKey,
    const T& expectTilingData, const std::vector<size_t>& expectWorkspaces)
{
    TilingInfo tiling_info;
    bool is_graph_success = ExecuteTiling(tilingContextPara, tiling_info);

    EXPECT_EQ(is_graph_success, expectResult == ge::GRAPH_SUCCESS);
    if (!is_graph_success)
        return;

    EXPECT_EQ(reinterpret_cast<std::vector<size_t>&>(tiling_info.workspaceSizes), expectWorkspaces);

    EXPECT_EQ(tiling_info.tilingKey, expectTilingKey);

    ASSERT_EQ(tiling_info.tilingDataSize, sizeof(T));

    EXPECT_EQ(*reinterpret_cast<T*>(tiling_info.tilingData.get()), expectTilingData);
}

class NotEqualTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NotEqualTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NotEqualTiling TearDown" << std::endl; }
};

constexpr size_t LIB_API_WORKSPACE_SIZE = 16 << 20;

TEST_F(NotEqualTiling, not_equal_0)
{
    struct NotEqualV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "NotEqualV2",
        {
            {{{19, 14, 21}, {19, 14, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{19, 14, 21}, {19, 14, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{19, 14, 21}, {19, 14, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    uint64_t expectTilingKey = 0;
    NotEqualTilingData expectTilingData{5586};
    std::vector expectWorkspaces = {LIB_API_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NotEqualTiling, not_equal_1)
{
    struct NotEqualV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "NotEqualV2",
        {
            {{{78, 23}, {78, 23}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{78, 23}, {78, 23}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{78, 23}, {78, 23}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    uint64_t expectTilingKey = 0;
    NotEqualTilingData expectTilingData{1794};
    std::vector expectWorkspaces = {LIB_API_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NotEqualTiling, not_equal_2)
{
    struct NotEqualV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "NotEqualV2",
        {
            {{{22, 84, 18, 5276}, {22, 84, 18, 5276}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{22, 84, 18, 5276}, {22, 84, 18, 5276}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{22, 84, 18, 5276}, {22, 84, 18, 5276}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    uint64_t expectTilingKey = 0;
    NotEqualTilingData expectTilingData{175500864};
    std::vector expectWorkspaces = {LIB_API_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NotEqualTiling, not_equal_3)
{
    struct NotEqualV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "NotEqualV2",
        {
            {{{97}, {97}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{97}, {97}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{97}, {97}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    uint64_t expectTilingKey = 0;
    NotEqualTilingData expectTilingData{97};
    std::vector expectWorkspaces = {LIB_API_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NotEqualTiling, not_equal_4)
{
    struct NotEqualV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "NotEqualV2",
        {
            {{{10, 115, 85}, {10, 115, 85}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{10, 115, 85}, {10, 115, 85}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{10, 115, 85}, {10, 115, 85}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    uint64_t expectTilingKey = 0;
    NotEqualTilingData expectTilingData{97750};
    std::vector expectWorkspaces = {LIB_API_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NotEqualTiling, not_equal_5)
{
    struct NotEqualV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "NotEqualV2",
        {
            {{{10, 41}, {10, 41}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{10, 41}, {10, 41}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{10, 41}, {10, 41}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    uint64_t expectTilingKey = 0;
    NotEqualTilingData expectTilingData{410};
    std::vector expectWorkspaces = {LIB_API_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NotEqualTiling, not_equal_6)
{
    struct NotEqualV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "NotEqualV2",
        {
            {{{3, 2503, 5049}, {3, 2503, 5049}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3, 2503, 5049}, {3, 2503, 5049}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{3, 2503, 5049}, {3, 2503, 5049}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    uint64_t expectTilingKey = 0;
    NotEqualTilingData expectTilingData{37912941};
    std::vector expectWorkspaces = {LIB_API_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
