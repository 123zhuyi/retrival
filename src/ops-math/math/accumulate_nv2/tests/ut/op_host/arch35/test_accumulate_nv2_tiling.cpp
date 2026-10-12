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
#include "../../../../op_kernel/arch35/accumulate_nv2_tiling_data.h"

namespace optiling {
struct AccumulateNV2CompileInfo {};
} // namespace optiling

namespace {
constexpr uint64_t KEY_SINGLE = 0;
constexpr uint64_t KEY_DOUBLE = 1;
optiling::AccumulateNV2CompileInfo g_compileInfo;

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape storageShape;
    auto& originShape = storageShape.MutableOriginShape();
    auto& runtimeShape = storageShape.MutableStorageShape();
    originShape.SetDimNum(dimensions.size());
    runtimeShape.SetDimNum(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        originShape.SetDim(i, dimensions[i]);
        runtimeShape.SetDim(i, dimensions[i]);
    }
    return storageShape;
}

static gert::TilingContextPara MakeTilingContext(const std::vector<std::vector<int64_t>>& inputShapes,
                                                 const std::vector<int64_t>& outputShape, ge::DataType dataType,
                                                 int64_t attrN = -1, bool withAttrN = true)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    for (const auto& shape : inputShapes) {
        inputs.emplace_back(MakeStorageShape(shape), dataType, ge::FORMAT_ND);
    }
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {MakeStorageShape(outputShape), dataType, ge::FORMAT_ND},
    };
    int64_t inputNum = static_cast<int64_t>(inputShapes.size());
    if (attrN < 0) {
        attrN = inputNum;
    }
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    if (withAttrN) {
        attrs.emplace_back("N", Ops::Math::AnyValue::CreateFrom<int64_t>(attrN));
    }
    return gert::TilingContextPara("AccumulateNV2", inputs, outputs, attrs, {static_cast<uint32_t>(inputNum)}, {1},
                                   &g_compileInfo);
}
} // namespace

class AccumulateNV2Tiling : public ::testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AccumulateNV2Tiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AccumulateNV2Tiling TearDown" << std::endl; }
};

TEST_F(AccumulateNV2Tiling, float32_double_buffer)
{
    auto context = MakeTilingContext({{8, 1024}, {8, 1024}, {8, 1024}}, {8, 1024}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, KEY_DOUBLE, std::vector<size_t>{0});
}

TEST_F(AccumulateNV2Tiling, float16_single_buffer)
{
    auto context = MakeTilingContext({{16}, {16}}, {16}, ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, KEY_SINGLE, std::vector<size_t>{0});
}

TEST_F(AccumulateNV2Tiling, single_input_scalar)
{
    auto context = MakeTilingContext({{}}, {}, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_EQ(info.tilingKey, KEY_SINGLE);
    const auto* data = reinterpret_cast<const AccumulateNV2TilingData*>(info.tilingData.get());
    EXPECT_EQ(data->totalNum, 1);
    EXPECT_EQ(data->rank, 1);
    EXPECT_EQ(data->needBroadcast, 0);
}

TEST_F(AccumulateNV2Tiling, int32_double_buffer)
{
    auto context = MakeTilingContext({{2048}, {2048}, {2048}, {2048}}, {2048}, ge::DT_INT32);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, KEY_DOUBLE, std::vector<size_t>{0});
}

TEST_F(AccumulateNV2Tiling, int8_single_buffer)
{
    auto context = MakeTilingContext({{32}, {32}}, {32}, ge::DT_INT8);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, KEY_SINGLE, std::vector<size_t>{0});
}

TEST_F(AccumulateNV2Tiling, uint8_single_buffer)
{
    auto context = MakeTilingContext({{64}, {64}}, {64}, ge::DT_UINT8);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, KEY_SINGLE, std::vector<size_t>{0});
}

TEST_F(AccumulateNV2Tiling, broadcast_metadata)
{
    auto context = MakeTilingContext({{2, 1}, {1, 3}}, {2, 3}, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_EQ(info.tilingKey, KEY_SINGLE);
    ASSERT_EQ(info.tilingDataSize, sizeof(AccumulateNV2TilingData));
    const auto* data = reinterpret_cast<const AccumulateNV2TilingData*>(info.tilingData.get());
    EXPECT_EQ(data->totalNum, 6);
    EXPECT_EQ(data->inputNum, 2);
    EXPECT_EQ(data->rank, 2);
    EXPECT_EQ(data->needBroadcast, 1);
    EXPECT_EQ(data->outputShape[0], 2);
    EXPECT_EQ(data->outputShape[1], 3);
    EXPECT_EQ(data->inputStrides[0][0], 1);
    EXPECT_EQ(data->inputStrides[0][1], 0);
    EXPECT_EQ(data->inputStrides[1][0], 0);
    EXPECT_EQ(data->inputStrides[1][1], 1);
}

TEST_F(AccumulateNV2Tiling, zero_element_broadcast)
{
    // The first input is non-empty, while broadcasting with the second input
    // produces an empty output.
    auto context = MakeTilingContext({{1, 2, 1, 2}, {0, 1, 3, 2}}, {0, 2, 3, 2}, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const AccumulateNV2TilingData*>(info.tilingData.get());
    EXPECT_EQ(data->totalNum, 0);
    EXPECT_EQ(info.blockNum, 1);
}

TEST_F(AccumulateNV2Tiling, missing_n_uses_dynamic_input_number)
{
    auto context = MakeTilingContext({{16}, {16}}, {16}, ge::DT_FLOAT, -1, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const AccumulateNV2TilingData*>(info.tilingData.get());
    EXPECT_EQ(data->inputNum, 2);
}

TEST_F(AccumulateNV2Tiling, attr_n_must_match_input_number)
{
    auto context = MakeTilingContext({{16}, {16}}, {16}, ge::DT_FLOAT, 3);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(AccumulateNV2Tiling, incompatible_input_shapes_fail)
{
    auto context = MakeTilingContext({{2, 3}, {4, 3}}, {4, 3}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(AccumulateNV2Tiling, output_must_equal_broadcast_shape)
{
    auto context = MakeTilingContext({{1, 3}, {1, 3}}, {2, 3}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(AccumulateNV2Tiling, input_number_limit)
{
    std::vector<std::vector<int64_t>> inputShapes(ACCUMULATE_NV2_MAX_INPUT_NUM + 1, {1});
    auto context = MakeTilingContext(inputShapes, {1}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
