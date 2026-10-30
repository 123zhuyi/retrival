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
#include <memory> // 必须：shared_ptr相关依赖
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/expandv_tiling_data.h"

using namespace std;
using namespace ge;

class ExpandvTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ExpandvTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ExpandvTiling TearDown" << std::endl; }
};

std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}};

TEST_F(ExpandvTiling, expandv_example_0)
{
    struct ExpandvCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Expandv",
        {
            {{{4, 1, 3}, {4, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 输入x1
        },
        {
            {{{4, 5, 3}, {4, 5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 输出y
        },
        {
            {"shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({4, 5, 3})},
        },
        &compileInfo); // 已修正原始笔误（删除多余的attrs,）
    uint64_t expectTilingKey = 0;
    std::string expectTilingData =
        "64 72 1 1 8184 64 72 0 3 3 4 1 3 0 0 0 0 0 0 0 4 5 3 0 0 0 0 0 0 0 3 3 1 0 0 0 0 0 0 0 15 3 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {1024 * 1024 * 16};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ExpandvTiling, expandv_example_1)
{
    struct ExpandvCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "Expandv",
        {
            {{{4, 1, 3}, {4, 1, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // input tensor1
        },
        {
            {{{4, 5, 3}, {4, 5, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // output tensor
        },
        {
            {"shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({4, 5, 3})},
        },
        &compileInfo);
    uint64_t expectTilingKey = 2;
    std::string expectTilingData =
        "64 80 1 1 16368 64 80 0 3 3 4 1 3 0 0 0 0 0 0 0 4 5 3 0 0 0 0 0 0 0 3 3 1 0 0 0 0 0 0 0 15 3 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {1024 * 1024 * 16};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}