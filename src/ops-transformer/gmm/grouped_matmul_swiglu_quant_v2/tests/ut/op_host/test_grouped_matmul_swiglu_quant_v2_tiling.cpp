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

#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_v2_base_tiling.h"
#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_v2_fusion_tiling.h"
#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;

class GroupedMatmulSwigluQuantV2 : public testing::Test
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

TEST_F(GroupedMatmulSwigluQuantV2, test_w8a8_normal_case_1)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {192 * 1024 * 1024,
                        24, 48, 128, 256};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n / 32, k / 16, 16, 32}, {e, n / 32, k / 16, 16, 32}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_INT8, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 3;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_mxfp8_normal_case_1)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        24, 48, 128, 256, true};
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m, n / 64, 2}, {m, n / 64, 2}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 0;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_mxfp8_normal_case_2)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        24, 48, 128, 256, true};
    int m = 2048;
    int k = 2048;
    int n = 7168;
    int e = 4;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n, k}, {e, n, k}};
    gert::StorageShape wScaleShape = {{e, n, k / 64, 2}, {e, n, k / 64, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m, n / 64, 2}, {m, n / 64, 2}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 1;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_mxfp8_normal_case_3)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        24, 48, 128, 256, true};
    int m = 144;
    int k = 2048;
    int n = 7168;
    int e = 1;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m, n / 64, 2}, {m, n / 64, 2}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 0;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_mxfp8_normal_case_4)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        24, 48, 128, 256, true};
    int m = 288;
    int k = 7168;
    int n = 4096;
    int e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n, k}, {e, n, k}};
    gert::StorageShape wScaleShape = {{e, n, k / 64, 2}, {e, n, k / 64, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m, n / 64, 2}, {m, n / 64, 2}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 1;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_input_fp8_fp4_unsupport_case)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 288;
    int k = 7168;
    int n = 4096;
    int e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n, k}, {e, n, k}};
    gert::StorageShape wScaleShape = {{e, n, k / 64, 2}, {e, n, k / 64, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m, n / 64, 2}, {m, n / 64, 2}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = -1;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_input_fp4_y_fp4_N_less_than_4_unsupport_case)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 288;
    int k = 7168;
    int n = 3;
    int e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n, k}, {e, n, k}};
    gert::StorageShape wScaleShape = {{e, n, k / 64, 2}, {e, n, k / 64, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m, n / 64, 2}, {m, n / 64, 2}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = -1;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_input_int8_y_int8_pertoken_transpose_case)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 288;
    int k = 7168;
    int n = 3;
    int e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n, k}, {e, n, k}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e / 2}, {e / 2}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m}, {m}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_INT8, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_INT8, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_INT8, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = -1;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_input_int8_y_int8_pertoken_no_transpose_case)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 288;
    int k = 7168;
    int n = 3;
    int e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e / 2}, {e / 2}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m}, {m}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_INT8, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_INT8, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_INT8, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = -1;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_input_float8_y_float8_pertoken_transpose_case)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 288;
    int k = 7168;
    int n = 3;
    int e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n, k}, {e, n, k}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e / 2}, {e / 2}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m}, {m}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = -1;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_input_float8_y_float8_pertoken_no_transpose_case)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 288;
    int k = 7168;
    int n = 3;
    int e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e / 2}, {e / 2}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m}, {m}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = -1;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_input_hif8_y_hif8_pertoken_transpose_case)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 288;
    int k = 7168;
    int n = 3;
    int e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n, k}, {e, n, k}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e / 2}, {e / 2}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m}, {m}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = -1;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_input_hif8_y_hif8_pertoken_no_transpose_case)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 288;
    int k = 7168;
    int n = 3;
    int e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e / 2}, {e / 2}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m}, {m}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = -1;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_corenum_checkfail_nonzero)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        24, 50, 128, 256, true}; // aic: 24, aiv: 50
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m, n / 64, 2}, {m, n / 64, 2}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_corenum_checkfail_zero)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        24, 0, 128, 256, true}; // aic: 24, aiv: 0
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape outShape = {{m, n / 2}, {m, n / 2}};
    gert::StorageShape outScaleShape = {{m, n / 64, 2}, {m, n / 64, 2}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {outShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {outScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_w4a4_normal_case_1)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {192 * 1024 * 1024,
                        24, 48, 128, 256};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n / 64, k / 16, 16, 64}, {e, n / 64, k / 16, 16, 64}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_INT4, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT4, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 4;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_w4a4_wtrans_case_1)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {192 * 1024 * 1024,
                        24, 48, 128, 256};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k / 64, n / 16, 16, 64}, {e, k / 64, n / 16, 16, 64}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_INT4, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT4, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 5;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}