/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/amp_update_scale_tiling.h"

using namespace std;
using namespace ge;
using optiling::AmpUpdateScaleCompileInfo;

class AmpUpdateScaleTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AmpUpdateScaleTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AmpUpdateScaleTiling TearDown" << std::endl;
    }
};

TEST_F(AmpUpdateScaleTilingTest, amp_update_scale_fp32_basic)
{
    AmpUpdateScaleCompileInfo compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AmpUpdateScale",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"growth_factor", ge::AnyValue::CreateFrom<ge::AnyValue::FLOAT>(2.0f)},
            {"backoff_factor", ge::AnyValue::CreateFrom<ge::AnyValue::FLOAT>(0.5f)},
            {"growth_interval", ge::AnyValue::CreateFrom<ge::AnyValue::INT>(5)},
        },
        &compileInfo,
        40,
        196608,
        4096);

    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AmpUpdateScaleTilingTest, amp_update_scale_fp16_basic)
{
    AmpUpdateScaleCompileInfo compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AmpUpdateScale",
        {
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"growth_factor", ge::AnyValue::CreateFrom<ge::AnyValue::FLOAT>(2.0f)},
            {"backoff_factor", ge::AnyValue::CreateFrom<ge::AnyValue::FLOAT>(0.5f)},
            {"growth_interval", ge::AnyValue::CreateFrom<ge::AnyValue::INT>(3)},
        },
        &compileInfo,
        40,
        196608,
        4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AmpUpdateScaleTilingTest, amp_update_scale_bf16_basic)
{
    AmpUpdateScaleCompileInfo compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AmpUpdateScale",
        {
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"growth_factor", ge::AnyValue::CreateFrom<ge::AnyValue::FLOAT>(2.0f)},
            {"backoff_factor", ge::AnyValue::CreateFrom<ge::AnyValue::FLOAT>(0.5f)},
            {"growth_interval", ge::AnyValue::CreateFrom<ge::AnyValue::INT>(10)},
        },
        &compileInfo,
        40,
        196608,
        4096);

    uint64_t expectTilingKey = 2;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}