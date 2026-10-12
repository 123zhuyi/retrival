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
#include <limits>

#include "op_common/log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/bernoulli_tiling_key.h"
#include "../op_kernel/bernoulli_tiling_data.h"

namespace optiling {

namespace {
constexpr uint64_t kElementsPerBlock = 256;
constexpr size_t kWorkspaceBytes = 512U;
constexpr uint64_t kUbReserveBytes = 8U * 1024U;
constexpr uint64_t kFallbackUbBytes = 192U * 1024U;
constexpr uint64_t kMaxRandomTileElements = 32U * 1024U;
constexpr uint64_t kMaxConstantTileElements = 32U * 1024U;
constexpr uint64_t kMaxWideConstantTileElements = 4U * 1024U;
constexpr uint64_t kBufferNum = 2;
constexpr uint64_t kBitsPerByte = 8U;
constexpr uint64_t kMinRandomTileElements = kElementsPerBlock;
constexpr uint64_t kVectorCyclesPerBlock = 72U;
constexpr uint64_t kTileSetupCycles = 280U;
constexpr uint64_t kStageSetupCycles = 320U;
constexpr uint64_t kSyncAllCycles = 2600U;
constexpr uint64_t kActiveCoreCycles = 96U;

struct RandomTilingConfig {
    uint64_t blockNum = 1;
    uint64_t blockElements = kElementsPerBlock;
    uint64_t stageElements = kElementsPerBlock;
    uint64_t tileElements = kElementsPerBlock;
    uint64_t estimatedCycles = std::numeric_limits<uint64_t>::max();
};

struct BernoulliCompileInfo {};

bool GetStorageBytes(ge::DataType dtype, uint64_t& storageBytes)
{
    switch (dtype) {
        case ge::DT_UINT8:
        case ge::DT_INT8:
        case ge::DT_BOOL:
            storageBytes = 1;
            return true;
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
            storageBytes = 2;
            return true;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            storageBytes = 4;
            return true;
        case ge::DT_INT64:
        case ge::DT_DOUBLE:
            storageBytes = 8;
            return true;
        default:
            return false;
    }
}

uint64_t GetRandomScratchBytesPerElement(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_UINT8:
        case ge::DT_INT8:
        case ge::DT_BOOL:
            return 2;
        case ge::DT_INT64:
        case ge::DT_DOUBLE:
            return 4;
        default:
            return 0;
    }
}

uint64_t AlignDownToBlock(uint64_t elements) { return (elements / kElementsPerBlock) * kElementsPerBlock; }

uint64_t AlignUpToBlock(uint64_t elements)
{
    return ((elements + kElementsPerBlock - 1U) / kElementsPerBlock) * kElementsPerBlock;
}

uint64_t CeilDiv(uint64_t dividend, uint64_t divisor)
{
    return dividend / divisor + (dividend % divisor != 0U ? 1U : 0U);
}

bool IsRandomMode(uint32_t mode) { return mode == BERNOULLI_MODE_RANDOM_ALIASED; }

uint64_t GetRandomVectorWeight(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_UINT8:
        case ge::DT_INT8:
        case ge::DT_BOOL:
            return 3U; // Select plus half-to-byte conversion.
        case ge::DT_INT64:
        case ge::DT_DOUBLE:
            return 4U; // Select plus the wide conversion/bit-pattern step.
        default:
            return 2U; // Duplicate and Select.
    }
}

bool EvaluateRandomCandidate(uint64_t total, uint64_t ubSize, ge::DataType dtype, uint64_t blockNum,
                             RandomTilingConfig& best)
{
    uint64_t storageBytes = 0;
    if (!GetStorageBytes(dtype, storageBytes) || ubSize <= kUbReserveBytes || blockNum == 0U) {
        return false;
    }

    const uint64_t availableUb = ubSize - kUbReserveBytes;
    const uint64_t computeBytesPerElement = kBufferNum * storageBytes + GetRandomScratchBytesPerElement(dtype);
    if (computeBytesPerElement == 0) {
        return false;
    }

    const uint64_t blockElements = AlignUpToBlock(CeilDiv(total, blockNum));
    const uint64_t maxTileElements = std::min(
        blockElements, std::min(kMaxRandomTileElements, AlignDownToBlock(availableUb / computeBytesPerElement)));
    if (maxTileElements < kMinRandomTileElements) {
        return false;
    }

    // The input mask aliases output storage. Every active core must finish
    // reading its mask stage before any core writes output, so choose cores,
    // mask stage, and vector tile together instead of using a size threshold.
    for (uint64_t tileElements = kMinRandomTileElements; tileElements <= maxTileElements;) {
        const uint64_t computeBytes = tileElements * computeBytesPerElement;
        if (computeBytes < availableUb) {
            uint64_t stageElements = AlignDownToBlock((availableUb - computeBytes) * kBitsPerByte);
            stageElements = std::min(stageElements, blockElements);
            if (stageElements >= kMinRandomTileElements) {
                const uint64_t tilesPerCore = CeilDiv(blockElements, tileElements);
                const uint64_t stageCount = CeilDiv(total, blockNum * stageElements);
                const uint64_t vectorBlocks = CeilDiv(blockElements, kElementsPerBlock);
                const uint64_t syncCost = blockNum > 1U ? stageCount * kSyncAllCycles : 0U;
                const uint64_t score = vectorBlocks * GetRandomVectorWeight(dtype) * kVectorCyclesPerBlock +
                                       tilesPerCore * kTileSetupCycles + stageCount * kStageSetupCycles + syncCost +
                                       blockNum * kActiveCoreCycles;
                if (score < best.estimatedCycles) {
                    best = {blockNum, blockElements, stageElements, tileElements, score};
                }
            }
        }

        if (tileElements == maxTileElements) {
            break;
        }
        const uint64_t nextTile = tileElements * 2U;
        tileElements = nextTile >= maxTileElements ? maxTileElements : nextTile;
    }
    return best.estimatedCycles != std::numeric_limits<uint64_t>::max();
}

bool GetRandomTiling(uint64_t total, uint64_t coreNum, uint64_t ubSize, ge::DataType dtype, RandomTilingConfig& result)
{
    const uint64_t maxBlockNum = std::min(coreNum, std::max<uint64_t>(1U, CeilDiv(total, kElementsPerBlock)));
    RandomTilingConfig best;
    for (uint64_t blockNum = 1U; blockNum <= maxBlockNum; ++blockNum) {
        EvaluateRandomCandidate(total, ubSize, dtype, blockNum, best);
    }
    if (best.estimatedCycles == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    result = best;
    return true;
}

uint64_t GetConstantTileElements(uint64_t ubSize, ge::DataType dtype, uint64_t blockElements)
{
    uint64_t storageBytes = 0;
    if (!GetStorageBytes(dtype, storageBytes) || ubSize <= kUbReserveBytes) {
        return 0;
    }

    const uint64_t bytesPerBlock = kBufferNum * storageBytes * kElementsPerBlock;
    const uint64_t availableUb = ubSize - kUbReserveBytes;
    uint64_t tileElements = (availableUb / bytesPerBlock) * kElementsPerBlock;
    uint64_t maxTileElements = kMaxConstantTileElements;
    if (dtype == ge::DT_INT64 || dtype == ge::DT_DOUBLE) {
        // The constant wide-type path uses a 128-bit mask repeat count.
        maxTileElements = kMaxWideConstantTileElements;
    }
    tileElements = std::min(tileElements, maxTileElements);
    tileElements = std::min(tileElements, std::max(blockElements, kElementsPerBlock));
    return AlignDownToBlock(tileElements);
}

ge::graphStatus BernoulliTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto inputShape = context->GetInputShape(0);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    const auto& shape = inputShape->GetStorageShape();
    constexpr uint64_t kMaxElements = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    uint64_t total = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            OP_LOGE(context, "Bernoulli input shape contains a negative dimension.");
            return ge::GRAPH_FAILED;
        }
        const uint64_t unsignedDim = static_cast<uint64_t>(dim);
        if (total != 0 && unsignedDim > kMaxElements / total) {
            OP_LOGE(context, "Bernoulli input element count exceeds int64_t range.");
            return ge::GRAPH_FAILED;
        }
        total *= unsignedDim;
    }
    auto* tiling = context->GetTilingData<BernoulliTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (memset_s(tiling, sizeof(BernoulliTilingData), 0, sizeof(BernoulliTilingData)) != EOK) {
        OP_LOGE(context, "failed to clear tiling data");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* modePtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, modePtr);
    if (*modePtr < BERNOULLI_MODE_RANDOM_ALIASED || *modePtr > BERNOULLI_MODE_ONE) {
        OP_LOGE(context, "Bernoulli mode must be 0, 1, or 2, but got %ld.", *modePtr);
        return ge::GRAPH_FAILED;
    }
    const uint32_t mode = static_cast<uint32_t>(*modePtr);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    uint64_t storageBytes = 0;
    if (!GetStorageBytes(dtype, storageBytes)) {
        OP_LOGE(context, "Bernoulli got unsupported dtype %d.", static_cast<int32_t>(dtype));
        return ge::GRAPH_FAILED;
    }

    uint64_t coreNum = 1;
    uint64_t ubSize = kFallbackUbBytes;
    if (context->GetPlatformInfo() != nullptr) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        coreNum = static_cast<uint64_t>(std::max<int64_t>(platform.GetCoreNumAiv(), 1));
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }

    uint64_t blockNum = std::min(coreNum, std::max<uint64_t>(1U, CeilDiv(total, kElementsPerBlock)));
    if (IsRandomMode(mode)) {
        RandomTilingConfig randomTiling;
        if (!GetRandomTiling(total, coreNum, ubSize, dtype, randomTiling)) {
            OP_LOGE(context, "Bernoulli cannot fit one random stage in %lu bytes of UB.", ubSize);
            return ge::GRAPH_FAILED;
        }
        blockNum = randomTiling.blockNum;
        tiling->blockElements = randomTiling.blockElements;
        tiling->stageElements = randomTiling.stageElements;
        tiling->tileElements = randomTiling.tileElements;
    } else {
        const uint64_t blockElements = AlignUpToBlock(CeilDiv(total, blockNum));
        tiling->blockElements = blockElements;
        tiling->tileElements = GetConstantTileElements(ubSize, dtype, blockElements);
        if (tiling->tileElements == 0) {
            OP_LOGE(context, "Bernoulli cannot fit one constant tile in %lu bytes of UB.", ubSize);
            return ge::GRAPH_FAILED;
        }
    }
    tiling->totalElements = total;
    tiling->mode = mode;
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = kWorkspaceBytes;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
    context->SetTilingKey(GET_TPL_TILING_KEY(BERNOULLI_TPL_SCH_MODE_0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BernoulliTilingParse([[maybe_unused]] gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }

} // namespace

IMPL_OP_OPTILING(Bernoulli).Tiling(BernoulliTilingFunc).TilingParse<BernoulliCompileInfo>(BernoulliTilingParse);

} // namespace optiling
