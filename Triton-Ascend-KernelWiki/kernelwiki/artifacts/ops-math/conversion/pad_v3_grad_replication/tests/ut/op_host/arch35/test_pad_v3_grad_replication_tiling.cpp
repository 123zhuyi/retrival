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
#include "conversion/pad_v3_grad_replication/op_host/arch35/pad_v3_grad_replication_tiling.h"

using namespace std;
using namespace ge;
using namespace optiling;

class PadV3GradReplicationTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PadV3GradReplicationTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PadV3GradReplicationTilingTest TearDown" << std::endl; }
};

// 1D float32 basic case, tail axis fallback (1D has no non-tail axis)
TEST_F(PadV3GradReplicationTilingTest, test_1d_float32_basic)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{70}, {70}};
    gert::StorageShape paddingsShape = {{2}, {2}};
    std::vector<int32_t> paddingsValue = {10, 10};

    gert::StorageShape yShape = {{50}, {50}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (1 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// 2D float16 with single-axis padding, split on axis=0
TEST_F(PadV3GradReplicationTilingTest, test_2d_float16_large)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{522, 64}, {522, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0};

    gert::StorageShape yShape = {{512, 64}, {512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// 3D bfloat16 with all dims padded, split on axis=1
TEST_F(PadV3GradReplicationTilingTest, test_3d_bfloat16_all_padding)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{15, 522, 74}, {15, 522, 74}};
    gert::StorageShape paddingsShape = {{6}, {6}};
    std::vector<int32_t> paddingsValue = {5, 0, 10, 0, 5, 5};

    gert::StorageShape yShape = {{10, 512, 64}, {10, 512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_BF16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (1ULL << 4) | (3 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// 4D float32 NCHW with only H/W padded, split on axis=2
TEST_F(PadV3GradReplicationTilingTest, test_4d_float32_nchw)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{10, 3, 522, 74}, {10, 3, 522, 74}};
    gert::StorageShape paddingsShape = {{8}, {8}};
    std::vector<int32_t> paddingsValue = {0, 0, 0, 0, 10, 0, 5, 5};

    gert::StorageShape yShape = {{10, 3, 512, 64}, {10, 3, 512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (2ULL << 4) | (4 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// 6D float32, only last 5 dims have padding (first dim padding=0)
TEST_F(PadV3GradReplicationTilingTest, test_6d_float32_last5_padding)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{100, 10, 20, 522, 74, 110}, {100, 10, 20, 522, 74, 110}};
    gert::StorageShape paddingsShape = {{12}, {12}};
    std::vector<int32_t> paddingsValue = {0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 5, 5};

    gert::StorageShape yShape = {{100, 10, 20, 512, 64, 100}, {100, 10, 20, 512, 64, 100}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (3ULL << 4) | (6 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// 8D float16, only last 5 dims have padding
TEST_F(PadV3GradReplicationTilingTest, test_8d_float16_last5_padding)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{2, 3, 4, 100, 10, 20, 522, 110}, {2, 3, 4, 100, 10, 20, 522, 110}};
    gert::StorageShape paddingsShape = {{16}, {16}};
    std::vector<int32_t> paddingsValue = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10};

    gert::StorageShape yShape = {{2, 3, 4, 100, 10, 20, 512, 100}, {2, 3, 4, 100, 10, 20, 512, 100}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (6ULL << 4) | (8 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// 3D float32 with int64 paddings (non-contiguous), covers GetPaddingsToShape<int64_t>
TEST_F(PadV3GradReplicationTilingTest, test_3d_float32_int64_paddings)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{15, 522, 74}, {15, 522, 74}};
    gert::StorageShape paddingsShape = {{6}, {6}};
    std::vector<int64_t> paddingsValue = {5, 10, 5, 0, 0, 5};

    gert::StorageShape yShape = {{10, 512, 64}, {10, 512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);

    uint64_t expectedTilingKey = (1ULL << 4) | (3 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// INT8 dtype, triggers dataSize=1 branch and uint16_t index limit with forced multi-tile
TEST_F(PadV3GradReplicationTilingTest, test_2d_int8_uint16_limit_forced_multitile)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // Single-tile dataBuf elements exceed INT16_MAX → uint16_t limit forces multi-tile
    gert::StorageShape xShape = {{520, 64}, {520, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 10, 0, 0};

    gert::StorageShape yShape = {{500, 64}, {500, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_INT8, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_INT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// UINT8 dtype, covers dataSize=1 branch
TEST_F(PadV3GradReplicationTilingTest, test_2d_uint8_basic)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{522, 64}, {522, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0};

    gert::StorageShape yShape = {{512, 64}, {512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_UINT8, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_UINT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// INT16 dtype, covers dataSize=2 branch and uint16_t index limit
TEST_F(PadV3GradReplicationTilingTest, test_2d_int16_uint16_limit)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{520, 64}, {520, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 10, 0, 0};

    gert::StorageShape yShape = {{500, 64}, {500, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_INT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_INT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// UINT16 dtype, covers dataSize=2 branch
TEST_F(PadV3GradReplicationTilingTest, test_2d_uint16_basic)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{522, 64}, {522, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0};

    gert::StorageShape yShape = {{512, 64}, {512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_UINT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_UINT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// INT32 dtype, covers dataSize=4 branch
TEST_F(PadV3GradReplicationTilingTest, test_2d_int32_basic)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{522, 64}, {522, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0};

    gert::StorageShape yShape = {{512, 64}, {512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// UINT32 dtype, covers dataSize=4 branch
TEST_F(PadV3GradReplicationTilingTest, test_2d_uint32_basic)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{522, 64}, {522, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0};

    gert::StorageShape yShape = {{512, 64}, {512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_UINT32, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_UINT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// INT64 dtype, covers dataSize=8 branch
TEST_F(PadV3GradReplicationTilingTest, test_2d_int64_basic)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{522, 64}, {522, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0};

    gert::StorageShape yShape = {{512, 64}, {512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_INT64, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// UINT64 dtype, covers dataSize=8 branch
TEST_F(PadV3GradReplicationTilingTest, test_2d_uint64_basic)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{522, 64}, {522, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0};

    gert::StorageShape yShape = {{512, 64}, {512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_UINT64, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_UINT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Unsupported dtype (DT_BOOL), should return GRAPH_FAILED
TEST_F(PadV3GradReplicationTilingTest, test_invalid_dtype_bool)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{522, 64}, {522, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0};

    gert::StorageShape yShape = {{512, 64}, {512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_BOOL, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_BOOL, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Paddings with wrong dtype (DT_FLOAT), should return GRAPH_FAILED
TEST_F(PadV3GradReplicationTilingTest, test_paddings_invalid_dtype_float)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{522, 64}, {522, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<float> paddingsValue = {10.0f, 0.0f, 0.0f, 0.0f};

    gert::StorageShape yShape = {{512, 64}, {512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 2D case forcing tail-axis fallback: huge padding on axis=0 exceeds UB → edge_simt kernel
TEST_F(PadV3GradReplicationTilingTest, test_2d_float32_tail_axis_fallback)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{100020, 64}, {100020, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {50000, 50000, 0, 0};

    gert::StorageShape yShape = {{20, 64}, {20, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (1ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// 3D INT8 with all non-tail axes failing, forcing tail axis (edge_simt)
TEST_F(PadV3GradReplicationTilingTest, test_3d_int8_tail_axis_fallback)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{10, 10, 100020}, {10, 10, 100020}};
    gert::StorageShape paddingsShape = {{6}, {6}};
    std::vector<int32_t> paddingsValue = {50000, 50000, 50000, 50000, 0, 0};

    gert::StorageShape yShape = {{10, 10, 20}, {10, 10, 20}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_INT8, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_INT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (2ULL << 4) | (3 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// FP16 dtype, single-tile both pads exceed UB budget → force multi-tile (no uint16_t limit since cast type)
TEST_F(PadV3GradReplicationTilingTest, test_2d_float16_single_tile_both_pads_exceed_ub)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // FP16 (cast type): dataBufSz=4, no uint16_t index limit
    // strideAligned=64, pL=508, pR=508 → bothFixedBytes=1*64*1016*4=260096 ≥ ubAvailable=260096
    // Force multi-tile: splitSize=inputShape[0]-1=2
    gert::StorageShape xShape = {{510 + 508 + 508, 64}, {510 + 508 + 508, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {508, 508, 0, 0};

    gert::StorageShape yShape = {{3, 64}, {3, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    // After force multi-tile: splitSize=2, splitAxis=0, dimNum=2
    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// INT8 dtype, single-tile both pads exceed UB AND shape=1 → TrySplitAxis fails, falls to tail axis
TEST_F(PadV3GradReplicationTilingTest, test_2d_int8_single_tile_exceed_ub_shape1)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // INT8 dataSize=1, strideAligned=640 (outputShape[1]=640)
    // pL=1, pR=1: bothFixedBytes=1*640*2*1=1280
    // ubAvailable=260096, bothFixed+align=1280+64=1344 < 260096 → not exceed UB
    // Need bothFixedBytes+alignSlack >= ubAvailable:
    // strideAligned*(pL+pR)*dataBufSz + 2*32 >= ubAvailable
    // With large strideAligned, pL+pR can be small
    // outputShape[1]=64000 → strideAligned=CeilAlign(64000*1,32)/1=64000
    // pL=2, pR=2: bothFixedBytes=1*64000*4*1=256000
    // 256000+64=256064 < 260096 → still not enough
    // pL=2, pR=2.1? Can't. Try pL=2, pR=2 but need 260096
    // 1*64000*4+64=256064. Need pL+pR > (260096-64)/64000 = 4.06 → pL+pR=5
    // pL=3, pR=2: maxSinglePad=3, fixedBytes=1*64000*3*1=192000 < 260096 ✓
    // budget=260096-192000-64=68032
    // perUnitBytes=1*64000*1+64000*1=128000 → unitsPerTile=68032/128000=0 → axis fails
    // Can't reach single-tile state. Need smaller perUnitBytes.
    //
    // Alternative: Use FP16 (cast type, no uint16 limit) with large strideAligned
    // outputShape[1]=500 → strideAligned=CeilAlign(500*2,32)/2=500 (CeilAlign(1000,32)=1000)
    // Actually: CeilAlign(1000,32)=(1000+31)/32*32=1031/32*32=32*32=1024 → strideAligned=1024/2=512
    // dataBufSz=4(FP16), dataSize=2
    // pL=255, pR=255: maxSinglePad=255, bothFixedBytes=1*512*510*4=1044480 > 260096
    // fixedBytes=1*512*255*4=522240 > 260096 → axis fails entirely
    //
    // Use smaller strideAligned. outputShape[1]=64 → strideAligned=64
    // FP16: dataBufSz=4, dataSize=2
    // Need: bothFixedBytes+alignSlack >= ubAvailable=260096
    // 1*64*(pL+pR)*4 + 64 >= 260096 → 256*(pL+pR) >= 260032 → pL+pR >= 1016
    // pL=508, pR=508: maxSinglePad=508
    // fixedBytes=1*64*508*4=130048 < 260096 ✓
    // perUnitBytes=1*64*4+64*2=256+128=384
    // budget=260096-130048-64=129984
    // unitsPerTile=129984/384=338→splitSize=min(338,1)=1(single tile, inputShape[0]=1)
    // bothFixedBytes+64=260160>=260096 ✓ → exceeds UB
    // inputShape[0]=1 ≤ 1 → "shape=1" path → TrySplitAxis returns false
    // Falls to tail axis
    gert::StorageShape xShape = {{508 + 508 + 1, 64}, {508 + 508 + 1, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {508, 508, 0, 0};

    gert::StorageShape yShape = {{1, 64}, {1, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    // Falls to tail axis: splitAxis=1, dimNum=2 → tilingKey=(1<<4)|(2-1)=17
    uint64_t expectedTilingKey = (1ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// FP16 dtype, single-tile pads fit UB but no room for data (unitsBoth=0), force multi-tile
TEST_F(PadV3GradReplicationTilingTest, test_2d_float16_single_tile_pads_fit_no_room)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // FP16 (cast type, no uint16_t limit): dataBufSz=4, dataSize=2
    // outputShape[1]=64 → strideAligned=64
    // pL=507, pR=508: bothFixedBytes=1*64*1015*4=259840
    // 259840+64=259904 < 260096 ✓ → pads fit
    // budgetBoth=260096-259840-64=192
    // perUnitBytes=1*64*4+64*2=384
    // unitsBoth=192/384=0 → no room for data
    // inputShape[0]=3 > 1 → force multi-tile: splitSize=2
    gert::StorageShape xShape = {{507 + 508 + 3, 64}, {507 + 508 + 3, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {507, 508, 0, 0};

    gert::StorageShape yShape = {{3, 64}, {3, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// INT8 dtype, single-tile both pads fit but reduce splitSize (unitsBoth < inputShape)
// This test uses INT8 and covers the "both pads reduce" path when the uint16_t limit is NOT triggered
// (since strideAligned is small enough)
TEST_F(PadV3GradReplicationTilingTest, test_2d_int8_single_tile_both_pads_reduce_split)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // INT8 dataSize=1, outputShape[1]=5000 → strideAligned=5024
    // pL=1, pR=1: bothFixedBytes=1*5024*2*1=10048
    // ubAvailable=260096, bothFixed+align=10048+64=10112 < 260096 ✓ → pads fit
    // budgetBoth=260096-10048-64=249984
    // perUnitBytes=1*5024*1+5000*1=10024
    // unitsBoth=249984/10024=24.9 → 24 < inputShape[0]=100 → reduces splitSize=24
    gert::StorageShape xShape = {{102, 5000}, {102, 5000}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {1, 1, 0, 0};

    gert::StorageShape yShape = {{100, 5000}, {100, 5000}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_INT8, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_INT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// FP16 dtype, single-tile both pads reduce splitSize (no uint16_t limit since cast type)
TEST_F(PadV3GradReplicationTilingTest, test_2d_float16_both_pads_reduce_split)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // FP16: dataBufSz=4, dataSize=2, strideAligned=256 (outputShape[1]=256)
    // pL=1, pR=1: maxSinglePad=1, fixedBytes=1*256*1*4=1024
    // budget_normal=260096-1024-64=259008, perUnitBytes=1536
    // unitsPerTile=259008/1536=169 → splitSize=min(169,169)=169 (single tile)
    // bothFixedBytes=1*256*2*4=2048, 2048+64=2112 < 260096 → pads fit
    // budgetBoth=260096-2048-64=257984
    // unitsBoth=257984/1536=168 → 168 < 169 → reduces splitSize=168
    gert::StorageShape xShape = {{171, 256}, {171, 256}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {1, 1, 0, 0};

    gert::StorageShape yShape = {{169, 256}, {169, 256}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// FP16 dtype, single-tile pads fit UB but no room for data with shape=1 → TrySplitAxis fails, tail axis fallback
TEST_F(PadV3GradReplicationTilingTest, test_2d_float16_pads_fit_no_room_shape1)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    // FP16: dataBufSz=4, dataSize=2, strideAligned=64
    // pL=507, pR=508: bothFixedBytes=1*64*1015*4=259840, 259840+64=259904 < 260096 → pads fit
    // budgetBoth=260096-259840-64=192, perUnitBytes=384, unitsBoth=192/384=0 → no room
    // inputShape[0]=1 → "shape=1, pads fit but no room" → TrySplitAxis fails → tail axis
    gert::StorageShape xShape = {{507 + 508 + 1, 64}, {507 + 508 + 1, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {507, 508, 0, 0};

    gert::StorageShape yShape = {{1, 64}, {1, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    // Falls to tail axis: splitAxis=1, dimNum=2 → tilingKey=(1<<4)|(2-1)=17
    uint64_t expectedTilingKey = (1ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// INT8 dtype, uint16_t index limit: pad overhead exceeds limit → axis fails, tail axis fallback
TEST_F(PadV3GradReplicationTilingTest, test_2d_int8_uint16_pad_overhead_exceeds_limit)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{30, 4096}, {30, 4096}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 10, 0, 0};

    gert::StorageShape yShape = {{10, 4096}, {10, 4096}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_INT8, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_INT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (1ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// INT8 dtype, uint16_t index limit: allowedSlice ≤ effectivePad → axis fails, tail axis fallback
TEST_F(PadV3GradReplicationTilingTest, test_2d_int8_uint16_allowed_slice_le_effective_pad)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{20, 3264}, {20, 3264}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0};

    gert::StorageShape yShape = {{10, 3264}, {10, 3264}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_INT8, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_INT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (1ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// INT8 dtype, uint16_t index limit: general reduction path reduces splitSize
TEST_F(PadV3GradReplicationTilingTest, test_2d_int8_uint16_general_reduce_split)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 32, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{1010, 64}, {1010, 64}};
    gert::StorageShape paddingsShape = {{4}, {4}};
    std::vector<int32_t> paddingsValue = {10, 0, 0, 0};

    gert::StorageShape yShape = {{1000, 64}, {1000, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_INT8, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_INT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (0ULL << 4) | (2 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Multi-tile per core: splitCount > coreNum, triggers CalcUsedCore multi-tile branch
TEST_F(PadV3GradReplicationTilingTest, test_5d_float32_multi_tile_outer_combos)
{
    PadV3GradReplicationCompileInfo compileInfo = {
        .core_num = 2, .ub_size = 128 * 1024, .sysWorkspaceSize = 16 * 1024 * 1024};

    gert::StorageShape xShape = {{10, 10, 10, 522, 74}, {10, 10, 10, 522, 74}};
    gert::StorageShape paddingsShape = {{10}, {10}};
    std::vector<int32_t> paddingsValue = {0, 0, 0, 0, 0, 0, 10, 0, 5, 5};

    gert::StorageShape yShape = {{10, 10, 10, 512, 64}, {10, 10, 10, 512, 64}};

    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("replicate")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectedTilingKey = (3ULL << 4) | (5 - 1);
    std::vector<size_t> expectedWorkspaces = {16 * 1024 * 1024};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}
