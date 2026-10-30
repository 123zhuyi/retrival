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
 * \file test_power_tiling_arch35.cpp
 * \brief Power 算子 Tiling UT 测试
 */

#include "../../../../op_host/arch35/power_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class PowerTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PowerTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PowerTilingTest TearDown" << std::endl;
    }
};

TEST_F(PowerTilingTest, test_tiling_fp32_power_1_001)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("power", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("shift", Ops::Math::AnyValue::CreateFrom<float>(0.0f))
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "8192 70368744177672 1024 8 1 1 1024 1024 16384 1 1065353216 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(PowerTilingTest, test_tiling_fp32_power_2_002)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("power", Ops::Math::AnyValue::CreateFrom<float>(2.0f)),
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("shift", Ops::Math::AnyValue::CreateFrom<float>(0.0f))
        },
        &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "8192 56075093016584 1024 8 1 1 1024 1024 13056 1 1065353216 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(PowerTilingTest, test_tiling_fp32_power_3_003)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("power", Ops::Math::AnyValue::CreateFrom<float>(3.0f)),
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("shift", Ops::Math::AnyValue::CreateFrom<float>(0.0f))
        },
        &compileInfo);
    uint64_t expectTilingKey = 104;
    string expectTilingData = "8192 46729244180488 1024 8 1 1 1024 1024 10880 1 1065353216 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(PowerTilingTest, test_tiling_fp32_power_0_004)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("power", Ops::Math::AnyValue::CreateFrom<float>(0.0f)),
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("shift", Ops::Math::AnyValue::CreateFrom<float>(0.0f))
        },
        &compileInfo);
    uint64_t expectTilingKey = 98;
    string expectTilingData = "8192 140737488355336 1024 8 1 1 1024 1024 32768 1 1065353216 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(PowerTilingTest, test_tiling_fp16_power_1_005)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("power", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("shift", Ops::Math::AnyValue::CreateFrom<float>(0.0f))
        },
        &compileInfo);
    uint64_t expectTilingKey = 36;
    string expectTilingData = "8192 46729244180484 2048 4 1 1 2048 2048 10880 1 1065353216 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(PowerTilingTest, test_tiling_bf16_power_1_006)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("power", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("shift", Ops::Math::AnyValue::CreateFrom<float>(0.0f))
        },
        &compileInfo);
    uint64_t expectTilingKey = 68;
    string expectTilingData = "8192 46729244180484 2048 4 1 1 2048 2048 10880 1 1065353216 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(PowerTilingTest, test_tiling_fp32_generic_power_pos_007)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("power", Ops::Math::AnyValue::CreateFrom<float>(2.5f)),
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("shift", Ops::Math::AnyValue::CreateFrom<float>(0.0f))
        },
        &compileInfo);
    uint64_t expectTilingKey = 106;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PowerTilingTest, test_tiling_fp32_generic_power_neg_008)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("power", Ops::Math::AnyValue::CreateFrom<float>(-2.0f)),
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("shift", Ops::Math::AnyValue::CreateFrom<float>(0.0f))
        },
        &compileInfo);
    uint64_t expectTilingKey = 108;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PowerTilingTest, test_tiling_failed_dtype_input_output_diff_009)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(PowerTilingTest, test_tiling_failed_shape_input_output_diff_010)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(PowerTilingTest, test_tiling_failed_empty_tensor_011)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(PowerTilingTest, test_tiling_failed_unsupport_input_012)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(PowerTilingTest, test_tiling_failed_unsupport_output_013)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(PowerTilingTest, test_tiling_fp32_scale_shift_014)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("power", Ops::Math::AnyValue::CreateFrom<float>(2.0f)),
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(2.0f)),
            gert::TilingContextPara::OpAttr("shift", Ops::Math::AnyValue::CreateFrom<float>(1.0f))
        },
        &compileInfo);
    uint64_t expectTilingKey = 102;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PowerTilingTest, test_tiling_fp32_broadcast_scalar_015)
{
    optiling::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Power",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("power", Ops::Math::AnyValue::CreateFrom<float>(2.0f)),
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(0.0f)),
            gert::TilingContextPara::OpAttr("shift", Ops::Math::AnyValue::CreateFrom<float>(2.0f))
        },
        &compileInfo);
    uint64_t expectTilingKey = 98;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}