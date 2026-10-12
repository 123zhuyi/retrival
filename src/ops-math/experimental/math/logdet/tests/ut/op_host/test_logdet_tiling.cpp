/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_kernel/logdet_tiling_data.h"

namespace LogdetUT {
using namespace ge;
using namespace gert;

namespace {

constexpr uint64_t kSystemWorkspaceBytes = 16ULL * 1024ULL * 1024ULL;
constexpr uint32_t kBlockedBlockSize = 64U;
constexpr uint32_t kBlockedMaxN = 4095U;
constexpr float kExpectedEps = 1.0e-38f;
constexpr uint64_t kCoreNum = 32ULL;
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint64_t kTilingDataMaxSize = 4096ULL;

uint64_t Align8(uint64_t value)
{
    return ((value + 7ULL) / 8ULL) * 8ULL;
}

uint64_t ComputeExpectedWorkspace(uint32_t n, bool blocked)
{
    if (!blocked) {
        return kSystemWorkspaceBytes;
    }
    const uint64_t blockedBytes = static_cast<uint64_t>(n) * Align8(n) * sizeof(float);
    return std::max(kSystemWorkspaceBytes, blockedBytes);
}

struct LogdetTilingExpectation {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    std::initializer_list<int64_t> yShape;
    graphStatus expectStatus;
    uint32_t expectMatSizeN;
    uint64_t expectMatrixNumCount;
    uint32_t expectBlockSize;
    uint32_t expectTilingBlockNum;
    size_t expectBlockDim;
    uint64_t expectWorkspaceBytes;
};

static LogdetTilingExpectation g_testCases[] = {
    {
        "logdet_small_single_3x3",
        {3, 3}, {},
        GRAPH_SUCCESS,
        3U, 1ULL, 3U, 1U, 1U, kSystemWorkspaceBytes
    },
    {
        "logdet_small_batch_10x3x3",
        {10, 3, 3}, {10},
        GRAPH_SUCCESS,
        3U, 10ULL, 3U, 1U, 10U, kSystemWorkspaceBytes
    },
    {
        "logdet_small_batch_5x3x3",
        {5, 3, 3}, {5},
        GRAPH_SUCCESS,
        3U, 5ULL, 3U, 1U, 5U, kSystemWorkspaceBytes
    },
    {
        "logdet_small_batch_32x3x3",
        {32, 3, 3}, {32},
        GRAPH_SUCCESS,
        3U, 32ULL, 3U, 1U, 32U, kSystemWorkspaceBytes
    },
    {
        "logdet_small_batch_100x3x3",
        {100, 3, 3}, {100},
        GRAPH_SUCCESS,
        3U, 100ULL, 3U, 1U, 32U, kSystemWorkspaceBytes
    },
    {
        "logdet_small_boundary_156x156",
        {156, 156}, {},
        GRAPH_SUCCESS,
        156U, 1ULL, 156U, 1U, 1U, kSystemWorkspaceBytes
    },
    {
        "logdet_large_boundary_209x209",
        {209, 209}, {},
        GRAPH_SUCCESS,
        209U, 1ULL, kBlockedBlockSize, 1U, 1U, ComputeExpectedWorkspace(209U, true)
    },
    {
        "logdet_large_single_256x256",
        {256, 256}, {},
        GRAPH_SUCCESS,
        256U, 1ULL, kBlockedBlockSize, 1U, 1U, ComputeExpectedWorkspace(256U, true)
    },
    {
        "logdet_large_batch_4x256x256",
        {4, 256, 256}, {4},
        GRAPH_SUCCESS,
        256U, 4ULL, kBlockedBlockSize, 1U, 1U, ComputeExpectedWorkspace(256U, true)
    },
    {
        "logdet_large_batch_2x512x512",
        {2, 512, 512}, {2},
        GRAPH_SUCCESS,
        512U, 2ULL, kBlockedBlockSize, 1U, 1U, ComputeExpectedWorkspace(512U, true)
    },
    {
        "logdet_large_max_4095x4095",
        {4095, 4095}, {},
        GRAPH_SUCCESS,
        4095U, 1ULL, kBlockedBlockSize, 1U, 1U, ComputeExpectedWorkspace(4095U, true)
    },
    {
        "logdet_exceeds_supported_max_4096x4096",
        {4096, 4096}, {},
        GRAPH_FAILED,
        0U, 0ULL, 0U, 0U, 0U, 0ULL
    },
    {
        "logdet_empty_tensor",
        {0, 3, 3}, {0},
        GRAPH_FAILED,
        0U, 0ULL, 0U, 0U, 0U, 0ULL
    },
    {
        "logdet_small_1x1",
        {1, 1}, {},
        GRAPH_SUCCESS,
        1U, 1ULL, 1U, 1U, 1U, kSystemWorkspaceBytes
    },
};

struct LogdetCompileInfo {
} g_compileInfo;

LogdetTilingData DecodeTilingData(const TilingInfo& info)
{
    LogdetTilingData tilingData{};
    if (info.tilingDataSize < sizeof(LogdetTilingData)) {
        ADD_FAILURE() << "tilingDataSize(" << info.tilingDataSize
                      << ") is smaller than LogdetTilingData(" << sizeof(LogdetTilingData) << ")";
        return tilingData;
    }
    std::memcpy(&tilingData, info.tilingData.get(), sizeof(LogdetTilingData));
    return tilingData;
}

void RunOneCase(const LogdetTilingExpectation& param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    StorageShape xShape = {param.xShape, param.xShape};
    StorageShape yShape = {param.yShape, param.yShape};
    std::vector<TilingContextPara::TensorDescription> inputTensorDesc(
        {{xShape, DT_FLOAT, FORMAT_ND}});
    std::vector<TilingContextPara::TensorDescription> outputTensorDesc(
        {{yShape, DT_FLOAT, FORMAT_ND}});
    std::vector<TilingContextPara::OpAttr> attrs;

    TilingContextPara tilingContextPara(
        "Logdet",
        inputTensorDesc,
        outputTensorDesc,
        attrs,
        &g_compileInfo,
        kCoreNum,
        kUbSize,
        kTilingDataMaxSize);

    TilingInfo info;
    const bool ret = ExecuteTiling(tilingContextPara, info);
    EXPECT_EQ(ret, param.expectStatus == GRAPH_SUCCESS);

    if (param.expectStatus != GRAPH_SUCCESS) {
        return;
    }

    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(static_cast<uint64_t>(info.workspaceSizes[0]), param.expectWorkspaceBytes);
    EXPECT_EQ(info.blockNum, param.expectBlockDim);

    const LogdetTilingData tilingData = DecodeTilingData(info);
    EXPECT_EQ(tilingData.matSizeN, param.expectMatSizeN);
    EXPECT_EQ(tilingData.matrixNumCount, param.expectMatrixNumCount);
    EXPECT_EQ(tilingData.blockSize, param.expectBlockSize);
    EXPECT_EQ(tilingData.blockNum, param.expectTilingBlockNum);
    EXPECT_FLOAT_EQ(tilingData.epsSingular, kExpectedEps);

    const bool blocked = tilingData.blockSize == kBlockedBlockSize && tilingData.matSizeN != tilingData.blockSize;
    if (blocked) {
        EXPECT_LE(tilingData.matSizeN, kBlockedMaxN);
        EXPECT_EQ(info.blockNum, 1U);
    } else {
        EXPECT_EQ(info.blockNum, std::min<size_t>(param.expectMatrixNumCount, kCoreNum));
    }
}

class LogdetTilingTest : public testing::TestWithParam<LogdetTilingExpectation> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LogdetTilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LogdetTilingTest TearDown." << std::endl;
    }
};

}  // namespace

TEST_P(LogdetTilingTest, tiling_matches_current_host_behavior)
{
    RunOneCase(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    LogdetTilingCases,
    LogdetTilingTest,
    testing::ValuesIn(g_testCases));

}  // namespace LogdetUT
