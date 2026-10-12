/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_kernel/softshrink_grad_tiling_data.h"
#include "../../../op_kernel/softshrink_grad_tiling_key.h"

namespace SoftShrinkGradUT {

using namespace ge;
using namespace gert;

static const std::string OP_NAME = "SoftshrinkGrad";

struct SoftShrinkGradTestParam {
    std::string caseName;
    std::initializer_list<int64_t> gradShape;
    std::initializer_list<int64_t> xShape;
    std::initializer_list<int64_t> outputShape;
    ge::DataType gradDtype;
    ge::DataType xDtype;
    ge::DataType outputDtype;
    float lambd;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::vector<int64_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    int64_t expectTotalNum;
    int64_t expectBlockFactor;
    int32_t expectUbFactor;
    uint32_t expectBlockNum;
};

static SoftShrinkGradTestParam testCases[] = {
    {"fp16_small_single_buffer",
     {3, 5},
     {3, 5},
     {3, 5},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     0.5f,
     ge::GRAPH_SUCCESS,
     GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_0),
     {0},
     64,
     262144,
     15,
     15,
     256,
     1},
    {"fp32_small_single_buffer",
     {4096},
     {4096},
     {4096},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     0.0f,
     ge::GRAPH_SUCCESS,
     GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_2),
     {0},
     64,
     262144,
     4096,
     4096,
     4096,
     1},
    {"bf16_unaligned_single_buffer",
     {257},
     {257},
     {257},
     ge::DT_BF16,
     ge::DT_BF16,
     ge::DT_BF16,
     1.0f,
     ge::GRAPH_SUCCESS,
     GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_4),
     {0},
     64,
     262144,
     257,
     257,
     512,
     1},
    {"fp16_large_double_buffer",
     {4194304},
     {4194304},
     {4194304},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     0.5f,
     ge::GRAPH_SUCCESS,
     GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_1),
     {0},
     48,
     262144,
     4194304,
     87552,
     11520,
     48},
    {"scalar_fp32",
     {},
     {},
     {},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     0.5f,
     ge::GRAPH_SUCCESS,
     GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_2),
     {0},
     64,
     262144,
     1,
     1,
     128,
     1},
    {"invalid_shape",
     {2, 3},
     {2, 4},
     {2, 3},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     0.5f,
     ge::GRAPH_FAILED,
     0,
     {},
     64,
     262144,
     0,
     0,
     0,
     0},
    {"invalid_dtype_mismatch",
     {2, 3},
     {2, 3},
     {2, 3},
     ge::DT_FLOAT16,
     ge::DT_FLOAT,
     ge::DT_FLOAT16,
     0.5f,
     ge::GRAPH_FAILED,
     0,
     {},
     64,
     262144,
     0,
     0,
     0,
     0},
    {"invalid_negative_lambda",
     {2, 3},
     {2, 3},
     {2, 3},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     -0.5f,
     ge::GRAPH_FAILED,
     0,
     {},
     64,
     262144,
     0,
     0,
     0,
     0},
    {"invalid_ub_size",
     {65},
     {65},
     {65},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     0.5f,
     ge::GRAPH_FAILED,
     0,
     {},
     64,
     8192,
     0,
     0,
     0,
     0},
};

class SoftShrinkGradTilingTest : public testing::TestWithParam<SoftShrinkGradTestParam> {};

struct SoftShrinkGradCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildContextParam(const SoftShrinkGradTestParam& param)
{
    gert::StorageShape gradShape = {param.gradShape, param.gradShape};
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape outputShape = {param.outputShape, param.outputShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc(
        {{gradShape, param.gradDtype, ge::FORMAT_ND}, {xShape, param.xDtype, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc(
        {{outputShape, param.outputDtype, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    attrs.emplace_back("lambd", Ops::NN::AnyValue::CreateFrom<float>(param.lambd));
    return gert::TilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo, param.maxAIVNum,
                                   param.ubSize, 4096);
}

TEST_P(SoftShrinkGradTilingTest, Tiling)
{
    const SoftShrinkGradTestParam& param = GetParam();
    auto tilingContextPara = BuildContextParam(param);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);

    if (param.status == ge::GRAPH_FAILED) {
        EXPECT_FALSE(ok) << param.caseName;
        return;
    }

    ASSERT_TRUE(ok) << param.caseName;
    EXPECT_EQ(static_cast<uint64_t>(tilingInfo.tilingKey), param.expectTilingKey);
    EXPECT_EQ(tilingInfo.workspaceSizes, param.expectWorkspaces);
    EXPECT_EQ(tilingInfo.blockNum, static_cast<size_t>(param.expectBlockNum));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(SoftShrinkGradTilingData));

    auto* tilingData = reinterpret_cast<SoftShrinkGradTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalNum, param.expectTotalNum);
    EXPECT_EQ(tilingData->blockFactor, param.expectBlockFactor);
    EXPECT_EQ(tilingData->ubFactor, param.expectUbFactor);
    EXPECT_FLOAT_EQ(tilingData->lambd, param.lambd);
}

INSTANTIATE_TEST_SUITE_P(SoftShrinkGradTilingTests, SoftShrinkGradTilingTest, testing::ValuesIn(testCases));

} // namespace SoftShrinkGradUT
