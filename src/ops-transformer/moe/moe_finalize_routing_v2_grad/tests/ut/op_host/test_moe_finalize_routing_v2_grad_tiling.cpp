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
#include "../../../op_host/moe_finalize_routing_v2_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class MoeFinalizeRoutingV2GradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2GradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2GradTiling TearDown" << std::endl;
    }
};

// ----------------------------------------------------------------------------------------------------------

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_10001_001)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            // output info
                                            {
                                                {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{5, 1}, {5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 10001; // tilngkey
    string expectTilingData = "5 0 5 5 0 5 0 1 8 5 65464 0 0 10001 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_10002_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape expanded_row_idx_shape = {{5}, {5}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape grad_scales_shape = {{5, 1}, {5, 1}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 10002; // tilngkey
    string expectTilingData = "5 0 5 5 0 5 0 1 262144 5 65464 4 288 10002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_20001_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 8}, {5, 8}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 8}, {15, 8}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 8}, {8, 8}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 8}, {15, 8}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 20001; // tilngkey
    string expectTilingData = "15 0 15 15 0 15 0 3 8 15 21808 0 0 20001 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_20002_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 262144}, {8, 262144}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData = "15 0 15 15 0 15 0 3 262144 15 21808 12 448 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_20002_002)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 262144}, {8, 262144}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expert_idx_shape, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData = "15 0 15 15 0 15 0 3 262144 15 21808 12 448 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_20002_003)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 262144}, {8, 262144}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 262144}, {15, 262144}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {expanded_x_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {scales_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                //{expert_idx_shape, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData = "15 0 15 15 0 15 0 3 262144 15 26176 10 384 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_10001_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 8}, {5, 8}};
    gert::StorageShape expanded_row_idx_shape = {{5}, {5}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{5, 8}, {5, 8}};
    gert::StorageShape grad_scales_shape = {{5, 1}, {5, 1}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 10001; // tilngkey
    string expectTilingData = "5 0 5 5 0 5 0 1 8 5 65464 0 0 10001 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_10002_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape expanded_row_idx_shape = {{5}, {5}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{5, 262144}, {5, 262144}};
    gert::StorageShape grad_scales_shape = {{5, 1}, {5, 1}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 10002; // tilngkey
    string expectTilingData = "5 0 5 5 0 5 0 1 262144 5 65464 4 288 10002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_20021_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 8}, {5, 8}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 8}, {15, 8}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 8}, {8, 8}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 8}, {15, 8}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 20001; // tilngkey
    string expectTilingData = "15 0 15 15 0 15 0 3 8 15 16360 0 0 20001 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_20022_001)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262000}, {5, 262000}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 262000}, {15, 262000}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 262000}, {8, 262000}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 262000}, {15, 262000}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData = "15 0 15 15 0 15 0 3 262000 15 16360 16 240 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_20022_002)
{
    // input shape
    gert::StorageShape grad_y_shape = {{5, 262000}, {5, 262000}};
    gert::StorageShape expanded_row_idx_shape = {{15}, {15}};
    gert::StorageShape expanded_x_shape = {{15, 262000}, {15, 262000}};
    gert::StorageShape scales_shape = {{5, 3}, {5, 3}};
    gert::StorageShape expert_idx_shape = {{5, 3}, {5, 3}};
    gert::StorageShape bias_shape = {{8, 262000}, {8, 262000}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{15, 262000}, {15, 262000}};
    gert::StorageShape grad_scales_shape = {{5, 3}, {5, 3}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(26)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 20022; // tilngkey
    string expectTilingData = "15 0 15 15 0 15 0 3 262000 15 16360 16 240 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_20022_003)
{
    // input shape
    gert::StorageShape grad_y_shape = {{18, 12849}, {18, 12849}};
    gert::StorageShape expanded_row_idx_shape = {{90}, {90}};
    gert::StorageShape expanded_x_shape = {{20, 58, 12849}, {20, 58, 12849}};
    gert::StorageShape scales_shape = {{18, 5}, {18, 5}};
    gert::StorageShape expert_idx_shape = {{18, 5}, {18, 5}};
    gert::StorageShape bias_shape = {{20, 12849}, {20, 12849}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{20, 12849}, {20, 12849}};
    gert::StorageShape grad_scales_shape = {{18, 5}, {18, 5}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(58)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData = "15 0 15 15 0 15 0 3 262000 15 16360 16 240 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradTiling, MoeFinalizeRoutingV2GradTiling_regbase_20022_004)
{
    // input shape
    gert::StorageShape grad_y_shape = {{18, 12849}, {18, 12849}};
    gert::StorageShape expanded_row_idx_shape = {{90}, {90}};
    gert::StorageShape expanded_x_shape = {{20, 58, 12849}, {20, 58, 12849}};
    gert::StorageShape scales_shape = {{18, 5}, {18, 5}};
    gert::StorageShape expert_idx_shape = {{18, 5}, {18, 5}};
    gert::StorageShape bias_shape = {{20, 12849}, {20, 12849}};

    // output shape
    gert::StorageShape grad_expanded_x_shape = {{20, 12849}, {20, 12849}};
    gert::StorageShape grad_scales_shape = {{18, 5}, {18, 5}};

    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {grad_y_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_row_idx_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {expert_idx_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {grad_expanded_x_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {grad_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            // attr
                                            {
                                                {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)},
                                                {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(58)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 20002; // tilngkey
    string expectTilingData = "15 0 15 15 0 15 0 3 262000 15 16360 16 240 20002 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
