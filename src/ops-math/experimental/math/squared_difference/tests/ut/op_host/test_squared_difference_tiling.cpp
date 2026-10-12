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
#include "../../../op_kernel/squared_difference_tiling_data.h"
#include "../../../op_kernel/squared_difference_tiling_key.h"

namespace SquaredDifferenceUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "SquaredDifference";

struct SquaredDifferenceTestParam {
    std::string caseName;
    std::initializer_list<int64_t> x1Shape;
    ge::DataType x1Dtype;
    ge::Format x1Format;
    std::initializer_list<int64_t> x2Shape;
    ge::DataType x2Dtype;
    ge::Format x2Format;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

static SquaredDifferenceTestParam testCases[] = {
    {"same_shape_bf16",
     {1, 19},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {1, 19},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {1, 19},
     ge::DT_BF16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     SD_KEY_BF16_ONEDIM,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096},
    {"same_shape_fp32",
     {4, 16},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {4, 16},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {4, 16},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     SD_KEY_FP32_ONEDIM,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096},
    {"scalar_broadcast_fp16",
     {1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {19},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {19},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     SD_KEY_FP16_ONEDIM,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096},
    {"multi_axis_broadcast_fp32",
     {2, 1, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1, 4, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {2, 4, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     SD_KEY_FP32_BRC,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096},
    {"same_shape_int64",
     {2, 8},
     ge::DT_INT64,
     ge::FORMAT_ND,
     {2, 8},
     ge::DT_INT64,
     ge::FORMAT_ND,
     {2, 8},
     ge::DT_INT64,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     SD_KEY_INT64_ONEDIM,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096},
    {"not_broadcastable",
     {2, 2},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {2, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {2, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_FAILED,
     0UL,
     EMPTY_EXPECT_TILING_DATA,
     {},
     64,
     262144,
     4096},
};

class SquaredDifferenceTilingTest : public testing::TestWithParam<SquaredDifferenceTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "SquaredDifferenceTilingTest SetUp." << std::endl; }
    static void TearDownTestCase() { std::cout << "SquaredDifferenceTilingTest TearDown." << std::endl; }
};

struct SquaredDifferenceCompileInfo {
} compileInfo;

static void TestOneParamCase(const SquaredDifferenceTestParam& param)
{
    gert::StorageShape x1Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape x2Shape = {param.x2Shape, param.x2Shape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{x1Shape, param.x1Dtype, param.x1Format}, {x2Shape, param.x2Dtype, param.x2Format}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.maxAIVNum, param.ubSize, param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces);
}

TEST_P(SquaredDifferenceTilingTest, tiling_test)
{
    const SquaredDifferenceTestParam& param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(SquaredDifferenceTilingTests, SquaredDifferenceTilingTest, testing::ValuesIn(testCases));

TEST(SquaredDifferenceTilingTest, TilingDataContainsBroadcastPlan)
{
    gert::StorageShape x1Shape = {{2, 1, 3}, {2, 1, 3}};
    gert::StorageShape x2Shape = {{1, 4, 3}, {1, 4, 3}};
    gert::StorageShape yShape = {{2, 4, 3}, {2, 4, 3}};
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {x1Shape, ge::DT_FLOAT, ge::FORMAT_ND},
        {x2Shape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    gert::TilingContextPara contextPara("SquaredDifference", inputs, outputs, &compileInfo, 64, 262144, 4096);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(contextPara, info));
    ASSERT_EQ(info.tilingKey, SD_KEY_FP32_BRC);
    ASSERT_GE(info.tilingDataSize, sizeof(SquaredDifferenceTilingData));
    const auto* tiling = reinterpret_cast<const SquaredDifferenceTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->mode, SD_MODE_BRC);
    EXPECT_EQ(tiling->dtypeKey, SD_DT_FP32);
    EXPECT_EQ(tiling->shapeLen, 3);
    EXPECT_EQ(tiling->outDims[0], 2);
    EXPECT_EQ(tiling->outDims[1], 4);
    EXPECT_EQ(tiling->outDims[2], 3);
    EXPECT_EQ(tiling->totalLength, 24);
    EXPECT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], 0U);
}

} // namespace SquaredDifferenceUT
