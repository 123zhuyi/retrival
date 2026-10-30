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
 * \file test_grouped_matmul_finalize_routing.cpp
 * \brief
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/grouped_matmul_finalize_routing_base_tiling.h"
#include "../../../op_host/grouped_matmul_finalize_routing_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace optiling
{
extern void DisablePatternCache();
extern void EnablePatternCache();
}  // namespace optiling

class GroupedMatmulFinalizeRoutingTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulFinalizeRoutingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulFinalizeRoutingTiling TearDown" << std::endl;
    }
};

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestW8A8NormalCase)
{

    optiling::GroupedMatmulFinalizeRoutingCompileInfo compileinfo = {20, 48, 196608, 524288, 33554432, 131072, 65536, 65536, 131072};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape{{e, k / 32, n / 16, 16, 32}, {e, k / 32, n / 16, 16, 32}};
    gert::StorageShape scaleShape = {{e, n}, {e, n}};
    gert::StorageShape pertoken_scaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};


    gert::TilingContextPara tilingContextPara("GroupedMatmulFinalizeRouting", 
        {
            {xShape, ge::DT_INT8, ge::FORMAT_ND},
            {wShape, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
            {scaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {pertoken_scaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
            {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 10000000000000000001UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestW4A8NormalCase)
{

    optiling::GroupedMatmulFinalizeRoutingCompileInfo compileinfo = {20, 48, 196608, 524288, 33554432, 131072, 65536, 65536, 131072};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape{{e, k / 32, n / 16, 16, 32}, {e, k / 32, n / 16, 16, 32}};
    gert::StorageShape scaleShape = {{e, n}, {e, n}};
    gert::StorageShape pertoken_scaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};


    gert::TilingContextPara tilingContextPara("GroupedMatmulFinalizeRouting", 
        {
            {xShape, ge::DT_INT8, ge::FORMAT_ND},
            {wShape, ge::DT_INT4, ge::FORMAT_ND},
            {scaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {pertoken_scaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
            {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 11000000000000000011UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestMXFP8NormalCaseTransposeWeightFalse)
{
    optiling::GroupedMatmulFinalizeRoutingCompileInfo compileinfo = {24,
                                                                     48,
                                                                     196608,
                                                                     524288,
                                                                     33554432,
                                                                     131072,
                                                                     65536,
                                                                     65536,
                                                                     131072,
                                                                     0,
                                                                     platform_ascendc::SocVersion::ASCEND950,
                                                                     false,
                                                                     true,
                                                                     NpuArch::DAV_3510};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape{{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape pertoken_scaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};

    gert::StorageShape yShape = {{m, n}, {m, n}};

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulFinalizeRouting",
        {{xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {wShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {pertoken_scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
         {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
         {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}},
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo);

    int64_t expectTilingKey = 0UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestMXFP8NormalCaseTransposeWeightTrue)
{
    optiling::GroupedMatmulFinalizeRoutingCompileInfo compileinfo = {24,
                                                                     48,
                                                                     196608,
                                                                     524288,
                                                                     33554432,
                                                                     131072,
                                                                     65536,
                                                                     65536,
                                                                     131072,
                                                                     0,
                                                                     platform_ascendc::SocVersion::ASCEND950,
                                                                     false,
                                                                     true,
                                                                     NpuArch::DAV_3510};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape{{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape pertoken_scaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};

    gert::StorageShape yShape = {{m, n}, {m, n}};

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulFinalizeRouting",
        {{xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {wShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {pertoken_scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
         {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
         {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}},
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo);

    int64_t expectTilingKey = 4UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestMXFP8IllegalCaseSharedOffset)
{
    optiling::GroupedMatmulFinalizeRoutingCompileInfo compileinfo = {20,
                                                                     48,
                                                                     196608,
                                                                     524288,
                                                                     33554432,
                                                                     131072,
                                                                     65536,
                                                                     65536,
                                                                     131072,
                                                                     0,
                                                                     platform_ascendc::SocVersion::ASCEND950,
                                                                     false,
                                                                     true,
                                                                     NpuArch::DAV_3510};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape{{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape pertoken_scaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};

    gert::StorageShape yShape = {{m, n}, {m, n}};

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulFinalizeRouting",
        {{xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {wShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {pertoken_scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
         {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
         {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}},
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1023)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestMXFP8IllegalCaseNullRowindex)
{
    optiling::GroupedMatmulFinalizeRoutingCompileInfo compileinfo = {20,
                                                                     48,
                                                                     196608,
                                                                     524288,
                                                                     33554432,
                                                                     131072,
                                                                     65536,
                                                                     65536,
                                                                     131072,
                                                                     0,
                                                                     platform_ascendc::SocVersion::ASCEND950,
                                                                     false,
                                                                     true,
                                                                     NpuArch::DAV_3510};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape{{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape pertoken_scaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{}, {}};

    gert::StorageShape yShape = {{m, n}, {m, n}};

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulFinalizeRouting",
        {{xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {wShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {pertoken_scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
         {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
         {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}},
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestMXFP8IllegalCaseWrongxDtype)
{
    optiling::GroupedMatmulFinalizeRoutingCompileInfo compileinfo = {20,
                                                                     48,
                                                                     196608,
                                                                     524288,
                                                                     33554432,
                                                                     131072,
                                                                     65536,
                                                                     65536,
                                                                     131072,
                                                                     0,
                                                                     platform_ascendc::SocVersion::ASCEND950,
                                                                     false,
                                                                     true,
                                                                     NpuArch::DAV_3510};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape{{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape pertoken_scaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};

    gert::StorageShape yShape = {{m, n}, {m, n}};

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulFinalizeRouting",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {wShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {pertoken_scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
         {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
         {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}},
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestMXFP8IllegalCaseWrongScaleDtype)
{
    optiling::GroupedMatmulFinalizeRoutingCompileInfo compileinfo = {20,
                                                                     48,
                                                                     196608,
                                                                     524288,
                                                                     33554432,
                                                                     131072,
                                                                     65536,
                                                                     65536,
                                                                     131072,
                                                                     0,
                                                                     platform_ascendc::SocVersion::ASCEND950,
                                                                     false,
                                                                     true,
                                                                     NpuArch::DAV_3510};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape{{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape pertoken_scaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};

    gert::StorageShape yShape = {{m, n}, {m, n}};

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulFinalizeRouting",
        {{xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {wShape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
         {scaleShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {pertoken_scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
         {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
         {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}},
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestCoreNumCheckFailNonZero)
{
    optiling::GroupedMatmulFinalizeRoutingCompileInfo compileinfo = {20, // aicNum
                                                                     50, // aivNum
                                                                     196608,
                                                                     524288,
                                                                     33554432,
                                                                     131072,
                                                                     65536,
                                                                     65536,
                                                                     131072,
                                                                     0,
                                                                     platform_ascendc::SocVersion::ASCEND950,
                                                                     false,
                                                                     true,
                                                                     NpuArch::DAV_3510};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape{{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape pertoken_scaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};

    gert::StorageShape yShape = {{m, n}, {m, n}};

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulFinalizeRouting",
        {{xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {wShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {pertoken_scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
         {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
         {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}},
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo);

    int64_t expectTilingKey = 4UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestCoreNumCheckFailZero)
{
    optiling::GroupedMatmulFinalizeRoutingCompileInfo compileinfo = {20, // aicNum
                                                                     0, // aivNum
                                                                     196608,
                                                                     524288,
                                                                     33554432,
                                                                     131072,
                                                                     65536,
                                                                     65536,
                                                                     131072,
                                                                     0,
                                                                     platform_ascendc::SocVersion::ASCEND950,
                                                                     false,
                                                                     true,
                                                                     NpuArch::DAV_3510};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape{{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape pertoken_scaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};

    gert::StorageShape yShape = {{m, n}, {m, n}};

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulFinalizeRouting",
        {{xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {wShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {pertoken_scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
         {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
         {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}},
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo);

    int64_t expectTilingKey = 4UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}