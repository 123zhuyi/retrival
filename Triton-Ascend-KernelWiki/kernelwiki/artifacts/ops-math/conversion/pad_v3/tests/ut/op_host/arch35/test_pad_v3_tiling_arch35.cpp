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
#include "../../../../op_host/arch35/pad_v3_tiling_arch35.h"

using namespace std;
using namespace ge;
using namespace optiling;

class PadV3TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PadV3TilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PadV3TilingTest TearDown" << std::endl; }
};

// Test scenario: pad_v3 constant mode with INT32 paddings and paddings_contiguous=true, expect SIMT tiling
TEST_F(PadV3TilingTest, pad_v3_tiling_test_001)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{2}, {2}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[2] = {3, 4};
    int64_t constantValue = 62;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, true, &pad_value},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 20000;
    string
        expectTilingData = "1 0 0 24 3 4 0 0 0 0 0 60 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 36 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test scenario: pad_v3 reflect mode with INT64 paddings and small shape, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_reflect_simt_int64)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{4, 5, 6}, {4, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 symmetric mode with small shape, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_symmetric_simt)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{4, 5, 6}, {4, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 edge mode with small shape, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_edge_simt)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{4, 5, 6}, {4, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 circular mode with small shape, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_circular_simt)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{4, 5, 6}, {4, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with INT64 paddings dtype and paddings_contiguous=true, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_int64_contiguous)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 2, 3, 4, 5, 6};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{3, 7, 15}, {3, 7, 15}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with INT32 paddings and paddings_contiguous=false, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_int32_non_contiguous)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 3, 5, 2, 4, 6};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{5, 10, 15}, {5, 10, 15}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with FP16 dtype and larger shape, expect non-SIMT tiling branch to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_fp16_large_shape)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{256, 256, 256}, {256, 256, 256}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{258, 258, 258}, {258, 258, 258}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with all-negative paddings, expect slice tiling branch to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_all_negative_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10, 10}, {10, 10, 10}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {-2, -2, -2, -2, -2, -2};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{6, 6, 6}, {6, 6, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 reflect mode with all-negative paddings, expect slice tiling branch to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_reflect_all_negative_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10, 10}, {10, 10, 10}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {-2, -2, -2, -2, -2, -2};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{6, 6, 6}, {6, 6, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 edge mode with all-negative paddings, expect slice tiling branch to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_edge_all_negative_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10, 10}, {10, 10, 10}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {-2, -2, -2, -2, -2, -2};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{6, 6, 6}, {6, 6, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 circular mode with all-negative paddings, expect slice tiling branch to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_circular_all_negative_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10, 10}, {10, 10, 10}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {-2, -2, -2, -2, -2, -2};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{6, 6, 6}, {6, 6, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with mixed positive/negative paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_mixed_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10, 10}, {10, 10, 10}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, -2, 0, 1, -2, 0};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{11, 8, 10}, {11, 8, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 reflect mode with mixed positive/negative paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_reflect_mixed_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10, 10}, {10, 10, 10}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, -2, 0, 1, -2, 0};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{11, 8, 10}, {11, 8, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 edge mode with mixed paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_edge_mixed_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10, 10}, {10, 10, 10}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, -2, 0, 1, -2, 0};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{11, 8, 10}, {11, 8, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 circular mode with mixed paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_circular_mixed_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10, 10}, {10, 10, 10}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, -2, 0, 1, -2, 0};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{11, 8, 10}, {11, 8, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with empty tensor (0-dim), expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_empty_tensor)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{0, 3, 4}, {0, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{1, 5, 6}, {1, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 reflect mode with reflect validation failure (padFront > inShape), expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_reflect_invalid_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {3, 1, 1, 3, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{5, 5, 6}, {5, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 symmetric mode with symmetric validation failure (padFront > inShape), expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_symmetric_invalid_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {3, 1, 1, 3, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{5, 5, 6}, {5, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 circular mode with circular validation failure (padFront > inShape), expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_circular_invalid_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {3, 1, 1, 3, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{5, 5, 6}, {5, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 edge mode with inShape=0 and nonzero pad, expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_edge_zero_inshape_nonzero_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{0, 3, 4}, {0, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 0, 0, 1, 0, 0};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 constant mode with unsupported paddings dtype (FLOAT), expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_invalid_paddings_dtype)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    float pad_value_arr[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float constantValue = 0.0f;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {padShape, ge::DT_FLOAT, ge::FORMAT_ND, true, pad_value_arr},
         {constantShape, ge::DT_FLOAT, ge::FORMAT_ND, true, &constantValue}},
        {{{{3, 7, 15}, {3, 7, 15}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 constant mode with invalid mode value, expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_invalid_mode)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{3, 5, 6}, {3, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("invalid_mode")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 constant mode with >8 dims input, expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_too_many_dims)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}};
    gert::StorageShape padShape = {{18}, {18}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[18] = {0};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 constant mode with negative dim in input shape, expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_negative_input_dim)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, -1, 4}, {2, -1, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{3, 0, 6}, {3, 0, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 constant mode with FP8 dtype and negative padding, expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_fp8_negative_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {-1, 1, 1, -1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_HIFLOAT8, ge::FORMAT_ND, true, &constantValue}},
        {{{{1, 5, 6}, {1, 5, 6}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 constant mode with FP4 dtype, odd last dim, expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_fp4_odd_last_dim)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 5}, {2, 3, 5}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {2, 2, 2, 2, 2, 2};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, true, &constantValue}},
        {{{{4, 5, 9}, {4, 5, 9}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 constant mode with FP4 dtype, odd last-dim padding, expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_fp4_odd_padding)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {2, 2, 1, 2, 2, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, true, &constantValue}},
        {{{{4, 5, 6}, {4, 5, 6}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 constant mode with mismatched constant_values dtype, expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_dtype_mismatch_constant_values)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    float constantValue = 0.0f;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT, ge::FORMAT_ND, true, &constantValue}},
        {{{{3, 5, 6}, {3, 5, 6}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 constant mode with output shape producing negative result, expect GRAPH_FAILED
TEST_F(PadV3TilingTest, pad_v3_tiling_negative_output_shape)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {-5, -5, -5, -6, -6, -6};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{-9, -8, -7}, {-9, -8, -7}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test scenario: pad_v3 reflect mode with 6D input (dim>5 triggers SIMT), expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_reflect_6d_simt)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7}};
    gert::StorageShape padShape = {{12}, {12}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{4, 5, 6, 7, 8, 9}, {4, 5, 6, 7, 8, 9}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with INT8 dtype and INT64 paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_int8_int64_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10, 10}, {10, 10, 10}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_INT8, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_INT8, ge::FORMAT_ND, true, &constantValue}},
        {{{{12, 12, 12}, {12, 12, 12}}, ge::DT_INT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with BF16 dtype and INT64 paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_bf16_int64_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10, 10}, {10, 10, 10}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {1, 1, 1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_BF16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_BF16, ge::FORMAT_ND, true, &constantValue}},
        {{{{12, 12, 12}, {12, 12, 12}}, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with all-1 dims and zero padding, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_all_ones_no_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {0, 0, 0, 0, 0, 0};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{1, 1, 1}, {1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with some 1-dims that collapse with adjacent no-pad dims, expect tiling to
// succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_partial_collapse)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{1, 3, 4}, {1, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {0, 1, 1, 0, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{1, 5, 6}, {1, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 edge mode with 2D input and positive paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_edge_2d_positive_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10}, {10, 10}};
    gert::StorageShape padShape = {{4}, {4}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[4] = {2, 2, 2, 2};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{14, 14}, {14, 14}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 reflect mode with 2D input and valid paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_reflect_2d_valid_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10}, {10, 10}};
    gert::StorageShape padShape = {{4}, {4}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[4] = {2, 2, 2, 2};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{14, 14}, {14, 14}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 symmetric mode with 2D input and valid paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_symmetric_2d_valid_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10}, {10, 10}};
    gert::StorageShape padShape = {{4}, {4}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[4] = {2, 2, 2, 2};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{14, 14}, {14, 14}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("symmetric")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 circular mode with 2D input and valid paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_circular_2d_valid_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10, 10}, {10, 10}};
    gert::StorageShape padShape = {{4}, {4}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[4] = {2, 2, 2, 2};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{14, 14}, {14, 14}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with small 2D shape, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_simt_small)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape padShape = {{4}, {4}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[4] = {1, 1, 1, 1};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{4, 5}, {4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 edge mode with empty tensor and all-zero paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_edge_empty_all_zero_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{0, 3, 4}, {0, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {0, 0, 0, 0, 0, 0};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{0, 3, 4}, {0, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 circular mode with empty tensor and all-zero paddings, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_circular_empty_all_zero_pad)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{0, 3, 4}, {0, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[6] = {0, 0, 0, 0, 0, 0};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{0, 3, 4}, {0, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("circular")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test scenario: pad_v3 constant mode with 1D input and padding, expect tiling to succeed
TEST_F(PadV3TilingTest, pad_v3_tiling_constant_1d_input)
{
    PadV3CompileInfo compileInfo = {0, 0, 0, 0, 0, false, "constant", 0, 0, 0, false, "ascend950"};

    gert::StorageShape xShape = {{10}, {10}};
    gert::StorageShape padShape = {{2}, {2}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t pad_value[2] = {1, 2};
    int64_t constantValue = 0;

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value},
         {constantShape, ge::DT_FLOAT16, ge::FORMAT_ND, true, &constantValue}},
        {{{{13}, {13}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}
