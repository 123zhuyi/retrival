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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/attention_worker_combine_tiling.h"

using namespace std;
using namespace optiling;

class AttentionWorkerCombineTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AttentionWorkerCombineTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AttentionWorkerCombineTilingTest TearDown" << std::endl;
    }
};

TEST_F(AttentionWorkerCombineTilingTest, attention_worker_combine_tiling_test01)
{
    
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{32, 8}, {32, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};
    
    gert::StorageShape y_shape_out = {{32, 7168}, {32, 7168}};
    gert::StorageShape next_layer_id_shape_out = {{1}, {1}};

    AttentionWorkerCombineCompileInfo compileInfo = {64, 256*1024};
    gert::TilingContextPara tilingContextPara(
        "AttentionWorkerCombine",
        {
            // input
            {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
            {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            // output
            {y_shape_out, ge::DT_FLOAT16, ge::FORMAT_ND},
            {next_layer_id_shape_out, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7168)},
            {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}
        },
        &compileInfo);
    int64_t expectTilingKey = 10020;
    std::string expectTilingData = "32 32 8 7168 15 1 32 1 1 7168 0 1 0 1 2 0 4 ";
    std::vector<size_t> expectWorkspaces = {32};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AttentionWorkerCombineTilingTest, attention_worker_combine_tiling_test02) {
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{32, 8}, {32, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};

    gert::StorageShape y_shape_out = {{32, 20480}, {32, 20480}};
    gert::StorageShape next_layer_id_shape_out = {{1}, {1}};

    AttentionWorkerCombineCompileInfo compileInfo = {64, 256*1024};
    gert::TilingContextPara tilingContextPara(
        "AttentionWorkerCombine",
        {
            // input
            {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
            {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            // output
            {y_shape_out, ge::DT_FLOAT16, ge::FORMAT_ND},
            {next_layer_id_shape_out, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20480)},
            {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}
        },
        &compileInfo);
    int64_t expectTilingKey = 10010;
    std::string expectTilingData = "32 32 8 20480 15 1 32 1 1 15872 4608 1 0 2 1 0 8 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AttentionWorkerCombineTilingTest, attention_worker_combine_tiling_test03) {
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{32, 8}, {32, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};

    gert::StorageShape y_shape_out = {{32, 1024}, {32, 1024}};
    gert::StorageShape next_layer_id_shape_out = {{1}, {1}};

    AttentionWorkerCombineCompileInfo compileInfo = {64, 256*1024};
    gert::TilingContextPara tilingContextPara(
        "AttentionWorkerCombine",
        {
            // input
            {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
            {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            // output
            {y_shape_out, ge::DT_FLOAT16, ge::FORMAT_ND},
            {next_layer_id_shape_out, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}
        },
        &compileInfo);
    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "32 32 8 1024 15 1 32 1 1 1024 0 1 1 1 8 8 1 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AttentionWorkerCombineTilingTest, attention_worker_combine_tiling_test04) {
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{48, 8}, {48, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};

    gert::StorageShape y_shape_out = {{48, 7168}, {48, 7168}};
    gert::StorageShape next_layer_id_shape_out = {{1}, {1}};


    AttentionWorkerCombineCompileInfo compileInfo = {64, 256*1024};
    gert::TilingContextPara tilingContextPara(
        "AttentionWorkerCombine",
        {
            // input
            {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
            {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            // output
            {y_shape_out, ge::DT_FLOAT16, ge::FORMAT_ND},
            {next_layer_id_shape_out, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7168)},
            {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}
        },
        &compileInfo);
    int64_t expectTilingKey = 10021;
    std::string expectTilingData = "48 48 8 7168 15 1 48 1 1 7168 0 1 0 1 2 0 4 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AttentionWorkerCombineTilingTest, attention_worker_combine_tiling_test05) {
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{48, 8}, {48, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};

    gert::StorageShape y_shape_out = {{48, 20480}, {48, 20480}};
    gert::StorageShape next_layer_id_shape_out = {{1}, {1}};

    AttentionWorkerCombineCompileInfo compileInfo = {64, 256*1024};
    gert::TilingContextPara tilingContextPara(
        "AttentionWorkerCombine",
        {
            // input
            {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
            {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            // output
            {y_shape_out, ge::DT_FLOAT16, ge::FORMAT_ND},
            {next_layer_id_shape_out, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20480)},
            {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}
        },
        &compileInfo);
    int64_t expectTilingKey = 10011;
    std::string expectTilingData = "48 48 8 20480 15 1 48 1 1 15872 4608 1 0 2 1 0 8 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}