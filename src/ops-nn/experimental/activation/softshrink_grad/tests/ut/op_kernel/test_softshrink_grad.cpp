/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "softshrink_grad_tiling.h"
#include "../../../op_kernel/softshrink_grad.cpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

namespace {

struct BFloat16Element {
    uint16_t value;
};

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16U) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exponent >= 31) {
        uint16_t payload = mantissa == 0 ? 0U : 0x0200U;
        return static_cast<uint16_t>(sign | 0x7c00U | payload);
    }
    uint32_t rounded = mantissa + 0x00001000U;
    if ((rounded & 0x00800000U) != 0U) {
        rounded = 0;
        ++exponent;
        if (exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7c00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) | (rounded >> 13U));
}

float HalfToFloat(uint16_t value)
{
    uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1fU;
    uint32_t mantissa = value & 0x03ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        bits = sign;
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t roundingBias = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16U);
}

float BFloat16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename Element>
void Fill(uint8_t* gm, const std::vector<float>& values)
{
    if constexpr (std::is_same_v<Element, float>) {
        std::memcpy(gm, values.data(), values.size() * sizeof(float));
    } else if constexpr (std::is_same_v<Element, uint16_t>) {
        for (size_t i = 0; i < values.size(); ++i) {
            uint16_t converted = FloatToHalf(values[i]);
            std::memcpy(gm + i * sizeof(converted), &converted, sizeof(converted));
        }
    } else {
        for (size_t i = 0; i < values.size(); ++i) {
            uint16_t converted = FloatToBFloat16(values[i]);
            std::memcpy(gm + i * sizeof(converted), &converted, sizeof(converted));
        }
    }
}

template <typename Element>
std::vector<float> Read(uint8_t* gm, size_t size)
{
    std::vector<float> result(size);
    if constexpr (std::is_same_v<Element, float>) {
        std::memcpy(result.data(), gm, size * sizeof(float));
    } else {
        for (size_t i = 0; i < size; ++i) {
            uint16_t converted = 0;
            std::memcpy(&converted, gm + i * sizeof(converted), sizeof(converted));
            if constexpr (std::is_same_v<Element, uint16_t>) {
                result[i] = HalfToFloat(converted);
            } else {
                result[i] = BFloat16ToFloat(converted);
            }
        }
    }
    return result;
}

std::vector<float> Golden(const std::vector<float>& grad, const std::vector<float>& self, float lambd)
{
    std::vector<float> result(grad.size());
    for (size_t i = 0; i < grad.size(); ++i) {
        result[i] = std::fabs(self[i]) <= lambd ? 0.0f : grad[i];
    }
    return result;
}

template <typename Element>
float Epsilon()
{
    if constexpr (std::is_same_v<Element, float>) {
        return 1e-6f;
    }
    if constexpr (std::is_same_v<Element, uint16_t>) {
        return 1e-3f;
    }
    return 1e-2f;
}

template <typename Element, uint32_t TilingKey>
void RunCase(const std::vector<float>& gradHost, const std::vector<float>& selfHost, float lambd, uint32_t numBlocks,
             int64_t ubFactor)
{
    ASSERT_EQ(gradHost.size(), selfHost.size());
    const size_t size = gradHost.size();
    const size_t dataBytes = std::max<size_t>(size * sizeof(Element), 32U);
    const size_t tilingDataSize = sizeof(SoftShrinkGradTilingData);

    uint8_t* grad = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(dataBytes));
    uint8_t* self = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(dataBytes));
    uint8_t* output = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(dataBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));
    if (grad == nullptr || self == nullptr || output == nullptr || workspace == nullptr || tiling == nullptr) {
        if (grad != nullptr) {
            AscendC::GmFree(grad);
        }
        if (self != nullptr) {
            AscendC::GmFree(self);
        }
        if (output != nullptr) {
            AscendC::GmFree(output);
        }
        if (workspace != nullptr) {
            AscendC::GmFree(workspace);
        }
        if (tiling != nullptr) {
            AscendC::GmFree(tiling);
        }
        FAIL() << "GmAlloc failed";
    }

    std::memset(output, 0, dataBytes);
    Fill<Element>(grad, gradHost);
    Fill<Element>(self, selfHost);

    auto* tilingData = reinterpret_cast<SoftShrinkGradTilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(size);
    tilingData->blockFactor = static_cast<int64_t>((size + numBlocks - 1U) / numBlocks);
    tilingData->ubFactor = static_cast<int32_t>(ubFactor);
    tilingData->lambd = lambd;

    ICPU_SET_TILING_KEY(TilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((softshrink_grad<TilingKey>), numBlocks, grad, self, output, workspace, tiling);

    std::vector<float> actual = Read<Element>(output, size);
    std::vector<float> expected = Golden(gradHost, selfHost, lambd);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(actual[i])) << "index=" << i;
        } else if (std::isinf(expected[i])) {
            EXPECT_TRUE(std::isinf(actual[i])) << "index=" << i;
            EXPECT_EQ(std::signbit(actual[i]), std::signbit(expected[i])) << "index=" << i;
        } else {
            EXPECT_NEAR(actual[i], expected[i], Epsilon<Element>()) << "index=" << i;
        }
    }

    AscendC::GmFree(grad);
    AscendC::GmFree(self);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST(SoftShrinkGradKernelTest, Float16BoundarySingleBuffer)
{
    std::vector<float> grad = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
    std::vector<float> self = {-1.0f, -0.5f, -0.25f, -0.0f, 0.0f, 0.5f, 0.75f, 2.0f};
    RunCase<uint16_t, SOFTSHRINKGRAD_TPL_SCH_MODE_0>(grad, self, 0.5f, 1, 256);
}

TEST(SoftShrinkGradKernelTest, Float16MultiBlockDoubleBuffer)
{
    std::vector<float> grad(2048);
    std::vector<float> self(2048);
    for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
        self[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;
    }
    RunCase<uint16_t, SOFTSHRINKGRAD_TPL_SCH_MODE_1>(grad, self, 0.5f, 4, 256);
}

TEST(SoftShrinkGradKernelTest, Float32SpecialValuesSingleBuffer)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> grad = {1.0f, -2.0f, nan, inf, -inf, 6.0f, 7.0f};
    std::vector<float> self = {nan, inf, -inf, -0.5f, 0.5f, 0.0f, 0.75f};
    RunCase<float, SOFTSHRINKGRAD_TPL_SCH_MODE_2>(grad, self, 0.5f, 1, 128);
}

TEST(SoftShrinkGradKernelTest, Float32UnalignedDoubleBuffer)
{
    std::vector<float> grad(2051);
    std::vector<float> self(2051);
    for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.0625f;
        self[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.125f;
    }
    RunCase<float, SOFTSHRINKGRAD_TPL_SCH_MODE_3>(grad, self, 0.75f, 1, 256);
}

TEST(SoftShrinkGradKernelTest, BFloat16BoundarySingleBuffer)
{
    std::vector<float> grad = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    std::vector<float> self = {-1.0f, -0.5f, -0.25f, 0.0f, 0.5f, 1.0f};
    RunCase<BFloat16Element, SOFTSHRINKGRAD_TPL_SCH_MODE_4>(grad, self, 0.5f, 1, 256);
}

TEST(SoftShrinkGradKernelTest, BFloat16UnalignedDoubleBuffer)
{
    std::vector<float> grad(1025);
    std::vector<float> self(1025);
    for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.125f;
        self[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.125f;
    }
    RunCase<BFloat16Element, SOFTSHRINKGRAD_TPL_SCH_MODE_5>(grad, self, 1.0f, 1, 256);
}

} // namespace
