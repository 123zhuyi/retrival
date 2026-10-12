/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_kernel/im2col_tiling_data.h"

extern "C" __global__ __aicore__ void im2col_fp16_contiguous(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" __global__ __aicore__ void im2col_fp32_channel_template(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                                   GM_ADDR tiling);

namespace {
constexpr size_t BLOCK_BYTES = 32;
constexpr uint32_t SINGLE_BLOCK_COUNT = 1U;
constexpr uint64_t TEST_CORE_NUM = 40;
constexpr uint64_t TEST_UB_SIZE = 262144;
constexpr uint16_t FP16_TEST_VALUE_BASE = 0x3C00U;
constexpr uint32_t FP32_TEST_VALUE_BASE = 0x3F000000U;
struct Im2colCompileInfo {};

size_t AlignUp(size_t value, size_t alignment)
{
    if (alignment == 0U) {
        return 0U;
    }
    return (value + alignment - 1U) / alignment * alignment;
}

gert::TilingContextPara BuildIdentityCase()
{
    const gert::StorageShape inputShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    const gert::StorageShape outputShape = {{1, 1, 8}, {1, 1, 8}};
    const std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {inputShape, ge::DT_FLOAT16, ge::FORMAT_NCHW},
    };
    const std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {outputShape, ge::DT_FLOAT16, ge::FORMAT_ND},
    };
    const std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
        {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
        {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
        {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
        {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
    };
    static Im2colCompileInfo compileInfo;
    return gert::TilingContextPara("Im2col", inputs, outputs, attrs, &compileInfo, TEST_CORE_NUM, TEST_UB_SIZE,
                                   sizeof(Im2colTilingData));
}

gert::TilingContextPara BuildChannelTemplateCase()
{
    const gert::StorageShape inputShape = {{1, 2, 16, 16}, {1, 2, 16, 16}};
    const gert::StorageShape outputShape = {{1, 16, 5}, {1, 16, 5}};
    const std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {inputShape, ge::DT_FLOAT, ge::FORMAT_NCHW},
    };
    const std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {outputShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    const std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
        {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({16, 2})},
        {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
        {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
        {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
    };
    static Im2colCompileInfo compileInfo;
    return gert::TilingContextPara("Im2col", inputs, outputs, attrs, &compileInfo, TEST_CORE_NUM, TEST_UB_SIZE,
                                   sizeof(Im2colTilingData));
}
} // namespace

class Im2colKernelTest : public testing::Test {};

TEST_F(Im2colKernelTest, fp16_identity_is_binary_equal)
{
    auto context = BuildIdentityCase();
    ::TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_EQ(tilingInfo.blockNum, SINGLE_BLOCK_COUNT);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(Im2colTilingHeader));

    const auto* hostTiling = reinterpret_cast<const Im2colTilingData*>(tilingInfo.tilingData.get());
    ASSERT_EQ(hostTiling->fastChannel, IM2COL_TILING_FLAG_ENABLED);
    ASSERT_EQ(hostTiling->channelIdentity, IM2COL_TILING_FLAG_ENABLED);

    constexpr size_t elementCount = 8;
    constexpr size_t dataBytes = elementCount * sizeof(uint16_t);
    const size_t allocationBytes = AlignUp(dataBytes, BLOCK_BYTES);
    auto* input = static_cast<uint8_t*>(AscendC::GmAlloc(allocationBytes));
    auto* output = static_cast<uint8_t*>(AscendC::GmAlloc(allocationBytes));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(BLOCK_BYTES));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(tilingInfo.tilingDataSize));
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    auto* inputValues = reinterpret_cast<uint16_t*>(input);
    for (size_t i = 0; i < elementCount; ++i) {
        inputValues[i] = static_cast<uint16_t>(FP16_TEST_VALUE_BASE + i);
    }
    std::memset(output, 0, allocationBytes);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(im2col_fp16_contiguous, tilingInfo.blockNum, input, output, workspace, tiling);

    EXPECT_EQ(std::memcmp(input, output, dataBytes), 0);

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(Im2colKernelTest, fp32_int16_template_is_binary_equal)
{
    auto context = BuildChannelTemplateCase();
    ::TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_EQ(tilingInfo.blockNum, SINGLE_BLOCK_COUNT);

    const auto* hostTiling = reinterpret_cast<const Im2colTilingData*>(tilingInfo.tilingData.get());
    ASSERT_EQ(hostTiling->fastChannel, IM2COL_TILING_FLAG_ENABLED);
    ASSERT_EQ(hostTiling->channelIndexTemplateValid, IM2COL_CHANNEL_INDEX_TEMPLATE_INT16);
    ASSERT_EQ(hostTiling->channelIndexTemplateElements * sizeof(uint16_t) % BLOCK_BYTES, 0U);

    constexpr size_t channelCount = 2U;
    constexpr size_t kernelWidth = 8U;
    constexpr size_t outputWidth = 5U;
    constexpr size_t inputHeight = 16U;
    constexpr size_t inputWidth = 16U;
    constexpr size_t strideWidth = 2U;
    constexpr size_t inputElements = channelCount * inputHeight * inputWidth;
    constexpr size_t outputElements = channelCount * kernelWidth * outputWidth;
    constexpr size_t inputBytes = inputElements * sizeof(uint32_t);
    constexpr size_t outputBytes = outputElements * sizeof(uint32_t);
    auto* input = static_cast<uint8_t*>(AscendC::GmAlloc(inputBytes));
    auto* output = static_cast<uint8_t*>(AscendC::GmAlloc(outputBytes));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(BLOCK_BYTES));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(tilingInfo.tilingDataSize));
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    auto* inputValues = reinterpret_cast<uint32_t*>(input);
    auto* outputValues = reinterpret_cast<uint32_t*>(output);
    for (size_t i = 0; i < inputElements; ++i) {
        inputValues[i] = FP32_TEST_VALUE_BASE + static_cast<uint32_t>(i);
    }
    std::memset(output, 0, outputBytes);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(im2col_fp32_channel_template, tilingInfo.blockNum, input, output, workspace, tiling);

    for (size_t channel = 0; channel < channelCount; ++channel) {
        for (size_t kernelColumn = 0; kernelColumn < kernelWidth; ++kernelColumn) {
            for (size_t outputColumn = 0; outputColumn < outputWidth; ++outputColumn) {
                const size_t outputIndex = (channel * kernelWidth + kernelColumn) * outputWidth + outputColumn;
                const size_t inputIndex = channel * inputHeight * inputWidth + outputColumn * strideWidth +
                                          kernelColumn;
                EXPECT_EQ(outputValues[outputIndex], inputValues[inputIndex]);
            }
        }
    }

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
