/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILING_FUNCTION_DEF_H
#define TILING_FUNCTION_DEF_H

#include <iostream>
#include <string>
#include <cstdint>
#include <vector>
#include <cmath>
#include "non_finite_check_tiling.h"
#include "exe_graph/runtime/shape.h"
#include "graph/types.h"
#include "graph/ge_error_codes.h"

namespace NonFiniteCheckTest {
constexpr int32_t MAX_LOG_COUNT = 800;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_REPEAT = 256;

constexpr uint8_t DTYPE_SIZE_FLOAT = 4;
constexpr uint8_t NUM_TWO = 2;
constexpr uint32_t COEFFICIENT_1 = 128;

class NonFiniteCheckTiling {
public:
    NonFiniteCheckTiling() {
        compileInfo.totalCoreNum = 48;
        compileInfo.ubSizePlatForm = 196608;
    };

    void Init(const std::vector<std::vector<uint64_t>>& shapeInfos, const ge::DataType dtypeIn = ge::DT_FLOAT);
    uint32_t RunBigKernelTiling();

public:
    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b) const {
        T1 bTemp(b);
        return bTemp == 0 ? a : (a + bTemp - 1) / bTemp;
    };

    void InitTilingDataItems();
    bool CalcNeedCoreNum();
    void AssignDataToEachCore();
    bool DivideUbMemory();
    uint32_t GetReduceRetValSize(uint32_t srcDataSize, uint32_t dtypeSize) const;
    uint64_t GetTilingKeyVal() const;
    void FillTilingData(NonFiniteCheckTilingData* tilingData);

private:
    NonFiniteCheckTilingData tilingData;
    NonFiniteCheckCompileInfo compileInfo;

    uint32_t maxProcCount = 0;
    uint32_t tempValUbSize = 0;
    int64_t tensorDataCountAlignedList[MAX_TENSOR_COUNT] = {0};
    int64_t* tensorDataCountList = nullptr;
    uint16_t* tensorStartList = nullptr;
    uint16_t* tensorEndList = nullptr;
    int64_t* tensorStartOffsetList = nullptr;
    int64_t* tensorEndOffsetList = nullptr;
    int64_t totalDataCountAligned = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    int32_t dataTypeSize = 0;
    int32_t elementsPerBlock = 0;
    int32_t totalTensorCount = 0;
    uint32_t needCoreNum = 0;
};

void NonFiniteCheckTiling::Init(const std::vector<std::vector<uint64_t>>& shapeInfos,
                                const ge::DataType dtypeIn /*= ge::DT_FLOAT*/) {
    InitTilingDataItems();
    dataType = dtypeIn;
    if (dataType == ge::DT_FLOAT) {
        dataTypeSize = 4;
    } else {
        dataTypeSize = 2;
    }
    elementsPerBlock = BYTE_BLOCK / dataTypeSize;

    for (auto shape : shapeInfos) {
        uint64_t elementCount = 1;
        for (auto i : shape) {
            elementCount *= i;
        }
        tensorDataCountList[totalTensorCount] = elementCount;
        tensorDataCountAlignedList[totalTensorCount] = CeilDiv(elementCount, elementsPerBlock) * elementsPerBlock;
        totalDataCountAligned += tensorDataCountAlignedList[totalTensorCount];
        totalTensorCount++;
    }

    // printf("dataType:%d, totalTensorCount:%d, totalDataCountAligned:%ld.", static_cast<int32_t>(dataType),
    //        totalTensorCount, totalDataCountAligned);
}

uint32_t NonFiniteCheckTiling::RunBigKernelTiling() {
    CalcNeedCoreNum();
    AssignDataToEachCore();
    DivideUbMemory();
    return needCoreNum;
}

void NonFiniteCheckTiling::InitTilingDataItems() {
    tensorDataCountList = tilingData.tensorDataCountList;
    tensorStartList = tilingData.tensorStartList;
    tensorEndList = tilingData.tensorEndList;
    tensorStartOffsetList = tilingData.tensorStartOffsetList;
    tensorEndOffsetList = tilingData.tensorEndOffsetList;
}

bool NonFiniteCheckTiling::CalcNeedCoreNum() {
    needCoreNum = uint32_t(totalDataCountAligned / elementsPerBlock);
    if (needCoreNum > uint32_t(compileInfo.totalCoreNum)) {
        needCoreNum = compileInfo.totalCoreNum;
    }
    if (needCoreNum == 0) {
        return false;
    } else {
        return true;
    }
}

void NonFiniteCheckTiling::AssignDataToEachCore() {
    int64_t blockCount = totalDataCountAligned / elementsPerBlock;
    // Divisible, representing the amount of data each core needs to process.
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = blockCount % needCoreNum;  // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPos = 0;
    tensorStartList[coreIndex] = 0;
    tensorStartOffsetList[coreIndex] = 0;
    for (uint16_t i = 0; i < totalTensorCount; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount != 0 && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountAlignedList[i] - cursorPos;
        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPos = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList[coreIndex] = i;
        cursorPos = cursorPos + curCmpCount - dataCount;
        tensorEndOffsetList[coreIndex] = cursorPos - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPos < tensorDataCountAlignedList[i]) {
            tensorStartList[coreIndex] = i;
            tensorStartOffsetList[coreIndex] = cursorPos;
            --i;  // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList[coreIndex] = i + 1;
            tensorStartOffsetList[coreIndex] = 0;
            cursorPos = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manually set the offset of the last core. */
    if (dataCount != 0) {
        tensorEndList[coreIndex] = totalTensorCount - 1;
        tensorEndOffsetList[coreIndex] = tensorDataCountAlignedList[totalTensorCount - 1] - 1;
    }
}

bool NonFiniteCheckTiling::DivideUbMemory() {
    // A 32-byte alignment is performed on the UB available space.
    uint32_t canUseUbSize = uint32_t(compileInfo.ubSizePlatForm / BYTE_BLOCK * BYTE_BLOCK);
    uint32_t dtypeSizeTemp = dataTypeSize;
    if (dataType == ge::DT_BF16) {
        dtypeSizeTemp = DTYPE_SIZE_FLOAT;
    }
    uint32_t predictSGUbSize =
        uint32_t((canUseUbSize - BYTE_BLOCK) * COEFFICIENT_1 * 1.0 / (BYTE_REPEAT + dtypeSizeTemp));
    uint32_t maxDataUbSize = predictSGUbSize / BYTE_BLOCK * BYTE_BLOCK;
    maxProcCount = maxDataUbSize / dtypeSizeTemp;
    tempValUbSize = GetReduceRetValSize(maxDataUbSize, dtypeSizeTemp);
    if ((NUM_TWO * maxDataUbSize + tempValUbSize) > compileInfo.ubSizePlatForm) {
        return false;
    } else {
        return true;
    }
}

uint32_t NonFiniteCheckTiling::GetReduceRetValSize(uint32_t srcDataSize, uint32_t dtypeSize) const {
    /* Calculate the space size of the intermediate variable workLocal and
        the result variable dstLocal of ReduceMax and ReduceMin. */
    uint8_t perBlockCount = BYTE_BLOCK / dtypeSize;
    uint32_t iter1OutputCount = uint32_t(std::ceil(NUM_TWO * 1.0 * srcDataSize / BYTE_REPEAT));
    uint32_t iter1AlignEnd = CeilDiv(iter1OutputCount, perBlockCount) * perBlockCount;
    return iter1AlignEnd * dtypeSize;
}

uint64_t NonFiniteCheckTiling::GetTilingKeyVal() const {
    switch (dataType) {
        case ge::DT_FLOAT:
            return static_cast<uint64_t>(NonFiniteCheckTilingKey::KEY_FLOAT);
        case ge::DT_FLOAT16:
            return static_cast<uint64_t>(NonFiniteCheckTilingKey::KEY_FLOAT16);
        case ge::DT_BF16:
            return static_cast<uint64_t>(NonFiniteCheckTilingKey::KEY_BF16);
        default:
            return 0;
    }
}

void NonFiniteCheckTiling::FillTilingData(NonFiniteCheckTilingData* tilingDataPtr) {
    std::ostringstream tempOSS;
    tempOSS << "maxProcCount: " << maxProcCount << ", tempValUbSize: " << tempValUbSize << "." << std::endl
            << "tensorDataCountList: ";
    for (uint32_t i = 0; i < uint32_t(totalTensorCount); i++) {
        tempOSS << "[" << i << "]: " << tensorDataCountList[i] << ", ";
    }
    tempOSS << std::endl << "Per core info: " << std::endl;
    for (uint32_t i = 0; i < needCoreNum; i++) {
        tempOSS << "index:[" << i << "], tensorStartList: " << tensorStartList[i]
                << ", tensorEndList:" << tensorEndList[i] << ", tensorStartOffsetList: " << tensorStartOffsetList[i]
                << ", tensorEndOffsetList: " << tensorEndOffsetList[i] << "." << std::endl;
    }
    std::string logDataStr = tempOSS.str();
    int32_t index = 0;
    do {
        // printf("%s", logDataStr.substr(index, MAX_LOG_COUNT).c_str());
        index += MAX_LOG_COUNT;
    } while (index < int32_t(logDataStr.length()));

    tilingData.maxProcCount = maxProcCount;
    tilingData.tempValUbSize = tempValUbSize;
    memcpy(tilingDataPtr, &tilingData, sizeof(NonFiniteCheckTilingData));
}

}  // namespace NonFiniteCheckTest

#endif  // TILING_FUNCTION_DEF_H