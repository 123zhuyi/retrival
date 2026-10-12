/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_tiling_noncontiguous.cpp
 * \brief atvoss broadcast template tiling
 */
#include "op_common/atvoss/broadcast/broadcast_tiling_noncontiguous.h"

namespace Ops {
namespace Base {
namespace {
constexpr uint64_t MIN_PER_CORE_ELEMS = 512;
}

struct LastTransposeSplitResult {
    uint64_t fusedProduct;
    uint64_t dimProductBeforeSplit;
    uint64_t curProduct;
};

struct ReverseDeriveResult {
    uint64_t maxElemNum;
    uint64_t fusedProduct;
    ubSplitInfo ubInfo;
};

LastTransposeSplitResult GetBlockSplitFactorLastTranspose(const BroadcastTilingData& broadcastTilingData,
                                                          ubSplitInfo& ubInfo, uint64_t maxElemNum,
                                                          const BroadcastComputeParams& computeParams,
                                                          uint64_t ubFormerHint = 0)
{
    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t minDtypeBlockAlignSize = BLOCK_LENGTH * BROADCAST_BITS_NUM / minDtypeBits;

    // 做ub切分
    uint64_t curProduct = 1;
    uint64_t ubSplitAxes = 0;
    bool flag = true;
    int32_t shapeLen = broadcastTilingData.dims.back().size();
    int64_t lastAxis = broadcastTilingData.dims.back()[shapeLen - 1];
    int64_t axisBeforelastAxis = broadcastTilingData.dims.back()[shapeLen - 2];

    int64_t ubOuter = 1;
    int64_t ubTail = 1;
    int64_t ubOuterLastAxis = 1;
    int64_t ubTailLastAxis = 1;

    int64_t axisBeforelastAxisSplitFactor = 128 / minDtypeBlockAlignSize;
    int64_t ubFormerCap = axisBeforelastAxisSplitFactor;
    if (ubFormerHint > 0 && ubFormerHint < static_cast<uint64_t>(axisBeforelastAxisSplitFactor)) {
        ubFormerCap = static_cast<int64_t>(ubFormerHint);
    }
    // -2轴切分至少128B, last轴转置至少有两维，所以不用考虑一维场景
    int64_t ubFormer = axisBeforelastAxis > ubFormerCap ? ubFormerCap : axisBeforelastAxis;
    int64_t ubFormerLastAxis = maxElemNum / ubFormer;

    if (lastAxis > ubFormerLastAxis) {
        // 如果-2轴切分完之后，-1轴还能切分则直接切-1轴，-2轴切分直接应用
        ubSplitAxes = shapeLen - 2;
        ubOuterLastAxis = (lastAxis + ubFormerLastAxis - 1) / ubFormerLastAxis;
        ubTailLastAxis = lastAxis - (ubOuterLastAxis - 1) * ubFormerLastAxis;

        ubOuter = (axisBeforelastAxis + ubFormer - 1) / ubFormer;
        ubTail = axisBeforelastAxis - (ubOuter - 1) * ubFormer;
        curProduct = 0;
    } else {
        // 如果 -1轴不够切分，则-2轴往前切分，尽可能多的切分-2轴，甚至到-3轴。
        ubOuterLastAxis = 1;
        ubFormerLastAxis = lastAxis;
        ubTailLastAxis = lastAxis;

        // curProduct需要将补维能力加入，确保后续ub切分的结果是带pad的。
        for (int64_t i = shapeLen - 1; i >= 0; i--) {
            // 遇到倒数第三轴的时候，需要对前两维补pad。
            if (i == shapeLen - 3) {
                curProduct = (curProduct + minDtypeBlockAlignSize - 1) / minDtypeBlockAlignSize *
                             minDtypeBlockAlignSize;
            }
            curProduct *= broadcastTilingData.dims.back()[i];
            if (curProduct > maxElemNum) {
                curProduct = curProduct / broadcastTilingData.dims.back()[i];
                ubSplitAxes = i;
                flag = false;
                break;
            }
        }

        if (flag) {
            curProduct = curProduct / broadcastTilingData.dims.back()[0];
        }

        // ubFormer 不可能在最后一根轴
        ubFormer = maxElemNum / curProduct;
        ubOuter = (broadcastTilingData.dims.back()[ubSplitAxes] + ubFormer - 1) / ubFormer;
        ubTail = broadcastTilingData.dims.back()[ubSplitAxes] - (ubOuter - 1) * ubFormer;
    }

    // 计算ub外轴乘积
    uint64_t dimProductBeforeSplit = 1;
    for (uint64_t i = 0; i < ubSplitAxes; i++) {
        dimProductBeforeSplit *= broadcastTilingData.dims.back()[i];
    }
    uint64_t fusedProduct = ubOuter * dimProductBeforeSplit;

    ubInfo.ubSplitAxis = ubSplitAxes;
    ubInfo.ubFormer = ubFormer;
    ubInfo.ubOuter = ubOuter;
    ubInfo.ubTail = ubTail;
    ubInfo.ubFormerLastAxis = ubFormerLastAxis;
    ubInfo.ubOuterLastAxis = ubOuterLastAxis;
    ubInfo.ubTailLastAxis = ubTailLastAxis;

    return {fusedProduct, dimProductBeforeSplit, curProduct};
}

ReverseDeriveResult ReverseDeriveLastTransposeMaxElemNum(const BroadcastTilingData& broadcastTilingData,
                                                         uint64_t targetFusedProduct, uint64_t maxUbElems,
                                                         const BroadcastComputeParams& computeParams)
{
    if (targetFusedProduct == 0) {
        return {maxUbElems, 0, ubSplitInfo{}};
    }
    uint64_t totalElems = 1;
    for (uint64_t k = 0; k < broadcastTilingData.dims.back().size(); k++) {
        totalElems *= static_cast<uint64_t>(broadcastTilingData.dims.back()[k]);
    }

    uint64_t initMaxElem = (totalElems + targetFusedProduct - 1) / targetFusedProduct;
    initMaxElem = (initMaxElem + static_cast<uint64_t>(CACHE_LINE) - 1) / static_cast<uint64_t>(CACHE_LINE) *
                  static_cast<uint64_t>(CACHE_LINE);
    if (initMaxElem < MIN_PER_CORE_ELEMS) {
        initMaxElem = MIN_PER_CORE_ELEMS;
    }
    if (initMaxElem > maxUbElems) {
        initMaxElem = maxUbElems;
    }

    ubSplitInfo ubInfo;
    auto split = GetBlockSplitFactorLastTranspose(broadcastTilingData, ubInfo, initMaxElem, computeParams);

    if (split.curProduct == 0) {
        int64_t shapeLen = broadcastTilingData.dims.back().size();
        int64_t lastAxis = broadcastTilingData.dims.back()[shapeLen - 1];

        int64_t minDtypeBits = computeParams.minDtypeBits;
        int64_t minDtypeBlockAlignSize = BLOCK_LENGTH * BROADCAST_BITS_NUM / minDtypeBits;
        int64_t splitFactor = 128 / minDtypeBlockAlignSize;
        int64_t ubFormerCap = broadcastTilingData.dims.back()[shapeLen - 2];
        if (ubFormerCap > splitFactor) {
            ubFormerCap = splitFactor;
        }

        uint64_t optHint = initMaxElem / static_cast<uint64_t>(lastAxis) + 1;
        if (optHint > static_cast<uint64_t>(ubFormerCap)) {
            optHint = static_cast<uint64_t>(ubFormerCap);
        }

        ubSplitInfo optUbInfo;
        auto optSplit = GetBlockSplitFactorLastTranspose(broadcastTilingData, optUbInfo, initMaxElem, computeParams,
                                                         optHint);

        return {initMaxElem, optSplit.fusedProduct, optUbInfo};
    }

    if (split.fusedProduct > targetFusedProduct) {
        uint64_t maxUbOuter = targetFusedProduct / split.dimProductBeforeSplit;
        if (maxUbOuter == 0) {
            maxUbOuter = 1;
        }
        uint64_t dimSplit = static_cast<uint64_t>(broadcastTilingData.dims.back()[ubInfo.ubSplitAxis]);
        uint64_t minUbFormer = (dimSplit + maxUbOuter - 1) / maxUbOuter;
        if (minUbFormer < 1) {
            minUbFormer = 1;
        }
        uint64_t optMaxElem = minUbFormer * split.curProduct;
        optMaxElem = (optMaxElem + static_cast<uint64_t>(CACHE_LINE) - 1) / static_cast<uint64_t>(CACHE_LINE) *
                     static_cast<uint64_t>(CACHE_LINE);
        if (optMaxElem < MIN_PER_CORE_ELEMS) {
            optMaxElem = MIN_PER_CORE_ELEMS;
        }
        if (optMaxElem > maxUbElems) {
            optMaxElem = maxUbElems;
        }
        split = GetBlockSplitFactorLastTranspose(broadcastTilingData, ubInfo, optMaxElem, computeParams);
        return {optMaxElem, split.fusedProduct, ubInfo};
    }

    return {initMaxElem, split.fusedProduct, ubInfo};
}

ge::graphStatus DoBrodcastTilingLastTranspose(const BroadcastTilingParams& broadcastTilingParams,
                                              BroadcastTilingData& broadcastTilingData)
{
    uint64_t computeKey = BroadcastGetComputeKey();
    auto iter = broadcastTilingParams.computeMap.find(computeKey);
    BroadcastComputeParams computeParams;
    if (iter != broadcastTilingParams.computeMap.end()) {
        computeParams = iter->second;
    } else {
        OP_LOGE("BroadcastTiling", "can not find computeKey: %lu", computeKey);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(broadcastTilingParams.ubSize < computeParams.extraSize[0],
                OP_LOGE("BroadcastTiling", "ubSize is smaller than extra size."), return ge::GRAPH_FAILED);

    // 获取最大存活空间大小
    uint64_t maxElemNum = BroadcastGetMaxElemNum(broadcastTilingParams.ubSize, computeParams);
    OP_LOGI("Broadcast", "Broadcast DoBrodcastTiling. origin maxElemNum: %lu ubSize: %ld", maxElemNum,
            broadcastTilingParams.ubSize);
    OP_CHECK_IF(broadcastTilingParams.ubSize <= 0, OP_LOGE("BroadcastTiling", "ubSize can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(broadcastTilingParams.coreNum <= 0, OP_LOGE("BroadcastTiling", "coreNum can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxElemNum == 0, OP_LOGE("BroadcastTiling", "maxElemNum can not be 0"), return ge::GRAPH_FAILED);

    ubSplitInfo ubInfo;
    if (broadcastTilingData.dims.back().size() < 2) {
        OP_LOGE("BroadcastTiling", "The last transpose shape at least 2 dims.");
        return ge::GRAPH_FAILED;
    }
    auto initSplit = GetBlockSplitFactorLastTranspose(broadcastTilingData, ubInfo, maxElemNum, computeParams);
    uint64_t fusedProduct = initSplit.fusedProduct;
    uint64_t blockFormer = (fusedProduct + broadcastTilingParams.coreNum - 1) / broadcastTilingParams.coreNum;
    uint64_t blockNum = (fusedProduct + blockFormer - 1) / blockFormer;

    // 非连续场景默认多核优先：当核数未用满时，反推maxElemNum使blockNum尽可能接近coreNum
    if (blockNum < static_cast<uint64_t>(broadcastTilingParams.coreNum)) {
        uint64_t coreNum = static_cast<uint64_t>(broadcastTilingParams.coreNum);
        uint64_t originFusedProduct = fusedProduct;
        uint64_t originMaxElemNum = maxElemNum;

        uint64_t blockFormerLowerBound = (fusedProduct + coreNum - 1) / coreNum;
        uint64_t target = blockFormerLowerBound * coreNum;

        auto result = ReverseDeriveLastTransposeMaxElemNum(broadcastTilingData, target, originMaxElemNum,
                                                           computeParams);
        if (result.fusedProduct > originFusedProduct) {
            OP_LOGI("Broadcast",
                    "Broadcast DoBrodcastTiling. reverseDerive applied: originFusedProduct: %lu "
                    "newFusedProduct: %lu originMaxElemNum: %lu newMaxElemNum: %lu",
                    originFusedProduct, result.fusedProduct, originMaxElemNum, result.maxElemNum);
            maxElemNum = result.maxElemNum;
            fusedProduct = result.fusedProduct;
            ubInfo = result.ubInfo;
            blockFormer = (fusedProduct + coreNum - 1) / coreNum;
            blockNum = (fusedProduct + blockFormer - 1) / blockFormer;
        }
    }

    uint64_t blockTail = fusedProduct - (blockNum - 1) * blockFormer;
    uint64_t dimProductBeforeUbInner = fusedProduct;
    OP_LOGI("Broadcast",
            "Broadcast DoBrodcastTiling. maxElemNum: %lu fusedProduct: %lu ubFormer: %ld "
            "blockFormer: %lu blockNum: %lu",
            maxElemNum, fusedProduct, ubInfo.ubFormer, blockFormer, blockNum);

    broadcastTilingData.ubSplitAxis = ubInfo.ubSplitAxis;
    broadcastTilingData.ubFormer = ubInfo.ubFormer;
    broadcastTilingData.ubOuter = ubInfo.ubOuter;
    broadcastTilingData.ubTail = ubInfo.ubTail;
    broadcastTilingData.ubFormerLastAxis = ubInfo.ubFormerLastAxis;
    broadcastTilingData.ubOuterLastAxis = ubInfo.ubOuterLastAxis;
    broadcastTilingData.ubTailLastAxis = ubInfo.ubTailLastAxis;

    broadcastTilingData.blockFormer = blockFormer;
    broadcastTilingData.blockNum = blockNum;
    broadcastTilingData.blockTail = blockTail;
    broadcastTilingData.dimProductBeforeUbInner = dimProductBeforeUbInner;
    broadcastTilingData.elemNum = maxElemNum;

    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t minDtypeBlockAlignSize = BLOCK_LENGTH * BROADCAST_BITS_NUM / minDtypeBits;
    broadcastTilingData.minDtypeBlockAlignSize = minDtypeBlockAlignSize;

    uint64_t scheduleKey = BroadcastGetScheduleKey(broadcastTilingData.shapeLen - broadcastTilingData.ubSplitAxis);
    broadcastTilingData.innerKey = computeKey * BROADCAST_COMPUTE_KEY_OFFSET + scheduleKey;
    return ge::GRAPH_SUCCESS;
}

struct NLastSplitResult {
    uint64_t fusedProduct;
    uint64_t dimProductBeforeSplit;
    uint64_t curProduct;
};

NLastSplitResult GetBlockSplitFactorNLastTranspose(const BroadcastTilingData& broadcastTilingData, ubSplitInfo& ubInfo,
                                                   uint64_t maxElemNum, const BroadcastComputeParams& computeParams,
                                                   bool isUbBroadcast)
{
    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t minDtypeBlockAlignSize = BLOCK_LENGTH * BROADCAST_BITS_NUM / minDtypeBits;

    // 做ub切分
    uint64_t curProduct = 1;
    uint64_t ubSplitAxes = 0;
    bool flag = true;

    int64_t boundaryNum = 2;
    if (isUbBroadcast) {
        boundaryNum = 1;
    }

    for (int64_t i = broadcastTilingData.dims.back().size() - 1; i >= 0; i--) {
        curProduct *= broadcastTilingData.dims.back()[i];
        if (i == static_cast<int64_t>(broadcastTilingData.dims.back().size() - boundaryNum)) {
            curProduct = ((curProduct + minDtypeBlockAlignSize - 1) / minDtypeBlockAlignSize) * minDtypeBlockAlignSize;
        }
        if (curProduct > maxElemNum) {
            curProduct = curProduct / broadcastTilingData.dims.back()[i];
            ubSplitAxes = i;
            flag = false;
            break;
        }
    }

    // 全部能放下，则去掉第一个维度
    if (flag) {
        curProduct = curProduct / broadcastTilingData.dims.back()[0];
    }

    uint32_t ubFormer = 0; // 表示当前切分轴的切分因子
    if (broadcastTilingData.dims.back().size() == 1) {
        ubFormer = maxElemNum;
    } else {
        ubFormer = maxElemNum / curProduct;
    }

    uint64_t ubOuter = (broadcastTilingData.dims.back()[ubSplitAxes] + ubFormer - 1) / ubFormer;
    uint64_t ubTail = broadcastTilingData.dims.back()[ubSplitAxes] - (ubOuter - 1) * ubFormer;

    // 计算ub外轴乘积
    uint64_t dimProductBeforeSplit = 1;
    for (uint64_t i = 0; i < ubSplitAxes; i++) {
        dimProductBeforeSplit *= broadcastTilingData.dims.back()[i];
    }
    uint64_t fusedProduct = ubOuter * dimProductBeforeSplit;

    ubInfo.ubFormer = ubFormer;
    ubInfo.ubSplitAxis = ubSplitAxes;
    ubInfo.ubOuter = ubOuter;
    ubInfo.ubTail = ubTail;

    return {fusedProduct, dimProductBeforeSplit, curProduct};
}

ReverseDeriveResult ReverseDeriveNLastTransposeMaxElemNum(const BroadcastTilingData& broadcastTilingData,
                                                          uint64_t targetFusedProduct, uint64_t maxUbElems,
                                                          const BroadcastComputeParams& computeParams,
                                                          bool isUbBroadcast)
{
    if (targetFusedProduct == 0) {
        return {maxUbElems, 0, ubSplitInfo{}};
    }
    uint64_t totalElems = 1;
    for (uint64_t k = 0; k < broadcastTilingData.dims.back().size(); k++) {
        totalElems *= static_cast<uint64_t>(broadcastTilingData.dims.back()[k]);
    }

    uint64_t initMaxElem = (totalElems + targetFusedProduct - 1) / targetFusedProduct;
    initMaxElem = (initMaxElem + static_cast<uint64_t>(CACHE_LINE) - 1) / static_cast<uint64_t>(CACHE_LINE) *
                  static_cast<uint64_t>(CACHE_LINE);
    if (initMaxElem < MIN_PER_CORE_ELEMS) {
        initMaxElem = MIN_PER_CORE_ELEMS;
    }
    if (initMaxElem > maxUbElems) {
        initMaxElem = maxUbElems;
    }

    ubSplitInfo ubInfo;
    auto split = GetBlockSplitFactorNLastTranspose(broadcastTilingData, ubInfo, initMaxElem, computeParams,
                                                   isUbBroadcast);

    if (split.fusedProduct > targetFusedProduct) {
        uint64_t maxUbOuter = targetFusedProduct / split.dimProductBeforeSplit;
        if (maxUbOuter == 0) {
            maxUbOuter = 1;
        }
        uint64_t dimSplit = static_cast<uint64_t>(broadcastTilingData.dims.back()[ubInfo.ubSplitAxis]);
        uint64_t minUbFormer = (dimSplit + maxUbOuter - 1) / maxUbOuter;
        if (minUbFormer < 1) {
            minUbFormer = 1;
        }
        uint64_t optMaxElem = minUbFormer * split.curProduct;
        optMaxElem = (optMaxElem + static_cast<uint64_t>(CACHE_LINE) - 1) / static_cast<uint64_t>(CACHE_LINE) *
                     static_cast<uint64_t>(CACHE_LINE);
        if (optMaxElem < MIN_PER_CORE_ELEMS) {
            optMaxElem = MIN_PER_CORE_ELEMS;
        }
        if (optMaxElem > maxUbElems) {
            optMaxElem = maxUbElems;
        }
        split = GetBlockSplitFactorNLastTranspose(broadcastTilingData, ubInfo, optMaxElem, computeParams,
                                                  isUbBroadcast);
        return {optMaxElem, split.fusedProduct, ubInfo};
    }

    return {initMaxElem, split.fusedProduct, ubInfo};
}

ge::graphStatus DoBrodcastTilingNLastTranspose(const BroadcastTilingParams& broadcastTilingParams,
                                               BroadcastTilingData& broadcastTilingData, bool isUbBroadcast)
{
    uint64_t computeKey = BroadcastGetComputeKey();
    auto iter = broadcastTilingParams.computeMap.find(computeKey);
    BroadcastComputeParams computeParams;
    if (iter != broadcastTilingParams.computeMap.end()) {
        computeParams = iter->second;
    } else {
        OP_LOGE("BroadcastTiling", "can not find computeKey: %lu", computeKey);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(broadcastTilingParams.ubSize < computeParams.extraSize[0],
                OP_LOGE("BroadcastTiling", "ubSize is smaller than extra size."), return ge::GRAPH_FAILED);

    // 获取最大存活空间大小
    uint64_t maxElemNum = BroadcastGetMaxElemNum(broadcastTilingParams.ubSize, computeParams);
    OP_LOGI("Broadcast", "Broadcast DoBrodcastTiling. origin maxElemNum: %lu ubSize: %ld", maxElemNum,
            broadcastTilingParams.ubSize);
    OP_CHECK_IF(broadcastTilingParams.ubSize <= 0, OP_LOGE("BroadcastTiling", "ubSize can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(broadcastTilingParams.coreNum <= 0, OP_LOGE("BroadcastTiling", "coreNum can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxElemNum == 0, OP_LOGE("BroadcastTiling", "maxElemNum can not be 0"), return ge::GRAPH_FAILED);

    ubSplitInfo ubInfo;
    auto split = GetBlockSplitFactorNLastTranspose(broadcastTilingData, ubInfo, maxElemNum, computeParams,
                                                   isUbBroadcast);
    uint64_t fusedProduct = split.fusedProduct;
    uint64_t blockFormer = (fusedProduct + broadcastTilingParams.coreNum - 1) / broadcastTilingParams.coreNum;
    uint64_t blockNum = (fusedProduct + blockFormer - 1) / blockFormer;

    // 非连续场景默认多核优先：当核数未用满时，反推maxElemNum使blockNum尽可能接近coreNum
    if (blockNum < static_cast<uint64_t>(broadcastTilingParams.coreNum)) {
        uint64_t coreNum = static_cast<uint64_t>(broadcastTilingParams.coreNum);
        uint64_t originFusedProduct = fusedProduct;
        uint64_t originMaxElemNum = maxElemNum;

        uint64_t blockFormerLowerBound = (fusedProduct + coreNum - 1) / coreNum;
        uint64_t target = blockFormerLowerBound * coreNum;
        OP_LOGI("Broadcast",
                "Broadcast DoBrodcastTiling. reverseDerive: originFusedProduct: %lu target: %lu "
                "blockFormerLowerBound: %lu",
                fusedProduct, target, blockFormerLowerBound);

        auto result = ReverseDeriveNLastTransposeMaxElemNum(broadcastTilingData, target, originMaxElemNum,
                                                            computeParams, isUbBroadcast);
        if (result.fusedProduct > originFusedProduct) {
            OP_LOGI("Broadcast",
                    "Broadcast DoBrodcastTiling. reverseDerive applied: originFusedProduct: %lu "
                    "newFusedProduct: %lu originMaxElemNum: %lu newMaxElemNum: %lu",
                    originFusedProduct, result.fusedProduct, originMaxElemNum, result.maxElemNum);
            maxElemNum = result.maxElemNum;
            fusedProduct = result.fusedProduct;
            ubInfo = result.ubInfo;
            blockFormer = (fusedProduct + coreNum - 1) / coreNum;
            blockNum = (fusedProduct + blockFormer - 1) / blockFormer;
        }
    }

    uint64_t blockTail = fusedProduct - (blockNum - 1) * blockFormer;
    uint64_t dimProductBeforeUbInner = fusedProduct;
    OP_LOGI("Broadcast",
            "Broadcast DoBrodcastTiling. maxElemNum: %lu fusedProduct: %lu ubFormer: %ld "
            "blockFormer: %lu blockNum: %lu",
            maxElemNum, fusedProduct, ubInfo.ubFormer, blockFormer, blockNum);

    broadcastTilingData.ubSplitAxis = ubInfo.ubSplitAxis;
    broadcastTilingData.ubFormer = ubInfo.ubFormer;
    broadcastTilingData.ubOuter = ubInfo.ubOuter;
    broadcastTilingData.ubTail = ubInfo.ubTail;

    broadcastTilingData.blockFormer = blockFormer;
    broadcastTilingData.blockNum = blockNum;
    broadcastTilingData.blockTail = blockTail;
    broadcastTilingData.dimProductBeforeUbInner = dimProductBeforeUbInner;
    broadcastTilingData.elemNum = maxElemNum;
    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t minDtypeBlockAlignSize = BLOCK_LENGTH * BROADCAST_BITS_NUM / minDtypeBits;
    broadcastTilingData.minDtypeBlockAlignSize = minDtypeBlockAlignSize;
    uint64_t scheduleKey = BroadcastGetScheduleKey(broadcastTilingData.shapeLen - broadcastTilingData.ubSplitAxis);
    broadcastTilingData.innerKey = computeKey * BROADCAST_COMPUTE_KEY_OFFSET + scheduleKey;
    return ge::GRAPH_SUCCESS;
}

NLastSplitResult GetBlockSplitFactorNonContiguousBase(const BroadcastTilingData& broadcastTilingData,
                                                      ubSplitInfo& ubInfo, uint64_t maxElemNum)
{
    uint64_t curProduct = 1;
    uint64_t ubSplitAxes = 0;
    bool flag = true;
    for (int64_t i = broadcastTilingData.dims.back().size() - 1; i >= 0; i--) {
        curProduct *= broadcastTilingData.dims.back()[i];
        if (curProduct > maxElemNum) {
            curProduct = curProduct / broadcastTilingData.dims.back()[i];
            ubSplitAxes = i;
            flag = false;
            break;
        }
    }
    if (flag) {
        curProduct = curProduct / broadcastTilingData.dims.back()[0];
    }

    uint32_t ubFormer = 0;
    if (broadcastTilingData.dims.back().size() == 1) {
        ubFormer = maxElemNum;
    } else {
        ubFormer = maxElemNum / curProduct;
    }

    uint64_t ubOuter = (broadcastTilingData.dims.back()[ubSplitAxes] + ubFormer - 1) / ubFormer;
    uint64_t ubTail = broadcastTilingData.dims.back()[ubSplitAxes] - (ubOuter - 1) * ubFormer;

    uint64_t dimProductBeforeSplit = 1;
    for (uint64_t i = 0; i < ubSplitAxes; i++) {
        dimProductBeforeSplit *= broadcastTilingData.dims.back()[i];
    }
    uint64_t fusedProduct = ubOuter * dimProductBeforeSplit;

    ubInfo.ubFormer = ubFormer;
    ubInfo.ubSplitAxis = ubSplitAxes;
    ubInfo.ubOuter = ubOuter;
    ubInfo.ubTail = ubTail;

    return {fusedProduct, dimProductBeforeSplit, curProduct};
}

ReverseDeriveResult ReverseDeriveNonContiguousBaseMaxElemNum(const BroadcastTilingData& broadcastTilingData,
                                                             uint64_t targetFusedProduct, uint64_t maxUbElems)
{
    if (targetFusedProduct == 0) {
        return {maxUbElems, 0, ubSplitInfo{}};
    }
    uint64_t totalElems = 1;
    for (uint64_t k = 0; k < broadcastTilingData.dims.back().size(); k++) {
        totalElems *= static_cast<uint64_t>(broadcastTilingData.dims.back()[k]);
    }

    uint64_t initMaxElem = (totalElems + targetFusedProduct - 1) / targetFusedProduct;
    initMaxElem = (initMaxElem + static_cast<uint64_t>(CACHE_LINE) - 1) / static_cast<uint64_t>(CACHE_LINE) *
                  static_cast<uint64_t>(CACHE_LINE);
    if (initMaxElem < MIN_PER_CORE_ELEMS) {
        initMaxElem = MIN_PER_CORE_ELEMS;
    }
    if (initMaxElem > maxUbElems) {
        initMaxElem = maxUbElems;
    }

    ubSplitInfo ubInfo;
    auto split = GetBlockSplitFactorNonContiguousBase(broadcastTilingData, ubInfo, initMaxElem);

    if (split.fusedProduct > targetFusedProduct) {
        uint64_t maxUbOuter = targetFusedProduct / split.dimProductBeforeSplit;
        if (maxUbOuter == 0) {
            maxUbOuter = 1;
        }
        uint64_t dimSplit = static_cast<uint64_t>(broadcastTilingData.dims.back()[ubInfo.ubSplitAxis]);
        uint64_t minUbFormer = (dimSplit + maxUbOuter - 1) / maxUbOuter;
        if (minUbFormer < 1) {
            minUbFormer = 1;
        }
        uint64_t optMaxElem = minUbFormer * split.curProduct;
        optMaxElem = (optMaxElem + static_cast<uint64_t>(CACHE_LINE) - 1) / static_cast<uint64_t>(CACHE_LINE) *
                     static_cast<uint64_t>(CACHE_LINE);
        if (optMaxElem < MIN_PER_CORE_ELEMS) {
            optMaxElem = MIN_PER_CORE_ELEMS;
        }
        if (optMaxElem > maxUbElems) {
            optMaxElem = maxUbElems;
        }
        split = GetBlockSplitFactorNonContiguousBase(broadcastTilingData, ubInfo, optMaxElem);
        return {optMaxElem, split.fusedProduct, ubInfo};
    }

    return {initMaxElem, split.fusedProduct, ubInfo};
}

ge::graphStatus DoBrodcastTilingNonContiguousBase(const BroadcastTilingParams& broadcastTilingParams,
                                                  BroadcastTilingData& broadcastTilingData)
{
    uint64_t computeKey = BroadcastGetComputeKey();
    auto iter = broadcastTilingParams.computeMap.find(computeKey);
    BroadcastComputeParams computeParams;
    if (iter != broadcastTilingParams.computeMap.end()) {
        computeParams = iter->second;
    } else {
        OP_LOGE("BroadcastTiling", "can not find computeKey");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(broadcastTilingParams.ubSize < computeParams.extraSize[0],
                OP_LOGE("BroadcastTiling", "ubSize is smaller than extra size."), return ge::GRAPH_FAILED);

    uint64_t maxElemNum = BroadcastGetMaxElemNum(broadcastTilingParams.ubSize, computeParams);
    OP_LOGI("Broadcast", "Broadcast DoBrodcastTiling. origin maxElemNum: %lu ubSize: %ld", maxElemNum,
            broadcastTilingParams.ubSize);
    OP_CHECK_IF(broadcastTilingParams.ubSize <= 0, OP_LOGE("BroadcastTiling", "ubSize can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(broadcastTilingParams.coreNum <= 0, OP_LOGE("BroadcastTiling", "coreNum can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxElemNum == 0, OP_LOGE("BroadcastTiling", "maxElemNum can not be 0"), return ge::GRAPH_FAILED);

    ubSplitInfo ubInfo;
    auto split = GetBlockSplitFactorNonContiguousBase(broadcastTilingData, ubInfo, maxElemNum);
    uint64_t fusedProduct = split.fusedProduct;
    uint64_t blockFormer = (fusedProduct + broadcastTilingParams.coreNum - 1) / broadcastTilingParams.coreNum;
    uint64_t blockNum = (fusedProduct + blockFormer - 1) / blockFormer;

    // 非连续场景默认多核优先：当核数未用满时，反推maxElemNum使blockNum尽可能接近coreNum
    if (blockNum < static_cast<uint64_t>(broadcastTilingParams.coreNum)) {
        uint64_t coreNum = static_cast<uint64_t>(broadcastTilingParams.coreNum);
        uint64_t originFusedProduct = fusedProduct;
        uint64_t originMaxElemNum = maxElemNum;

        uint64_t blockFormerLowerBound = (fusedProduct + coreNum - 1) / coreNum;
        uint64_t target = blockFormerLowerBound * coreNum;
        OP_LOGI("Broadcast",
                "Broadcast DoBrodcastTiling. reverseDerive: originFusedProduct: %lu target: %lu "
                "blockFormerLowerBound: %lu",
                fusedProduct, target, blockFormerLowerBound);

        auto result = ReverseDeriveNonContiguousBaseMaxElemNum(broadcastTilingData, target, originMaxElemNum);
        if (result.fusedProduct > originFusedProduct) {
            OP_LOGI("Broadcast",
                    "Broadcast DoBrodcastTiling. reverseDerive applied: originFusedProduct: %lu "
                    "newFusedProduct: %lu originMaxElemNum: %lu newMaxElemNum: %lu",
                    originFusedProduct, result.fusedProduct, originMaxElemNum, result.maxElemNum);
            maxElemNum = result.maxElemNum;
            fusedProduct = result.fusedProduct;
            ubInfo = result.ubInfo;
            blockFormer = (fusedProduct + coreNum - 1) / coreNum;
            blockNum = (fusedProduct + blockFormer - 1) / blockFormer;
        }
    }

    uint64_t blockTail = fusedProduct - (blockNum - 1) * blockFormer;
    uint64_t dimProductBeforeUbInner = fusedProduct;
    OP_LOGI("Broadcast",
            "Broadcast DoBrodcastTiling. maxElemNum: %lu fusedProduct: %lu ubFormer: %ld "
            "blockFormer: %lu blockNum: %lu",
            maxElemNum, fusedProduct, ubInfo.ubFormer, blockFormer, blockNum);

    broadcastTilingData.ubSplitAxis = ubInfo.ubSplitAxis;
    broadcastTilingData.ubFormer = ubInfo.ubFormer;
    broadcastTilingData.ubOuter = ubInfo.ubOuter;
    broadcastTilingData.ubTail = ubInfo.ubTail;

    broadcastTilingData.blockFormer = blockFormer;
    broadcastTilingData.blockNum = blockNum;
    broadcastTilingData.blockTail = blockTail;
    broadcastTilingData.dimProductBeforeUbInner = dimProductBeforeUbInner;
    broadcastTilingData.elemNum = maxElemNum;

    uint64_t scheduleKey = BroadcastGetScheduleKey(broadcastTilingData.shapeLen - broadcastTilingData.ubSplitAxis);
    broadcastTilingData.innerKey = computeKey * BROADCAST_COMPUTE_KEY_OFFSET + scheduleKey;
    return ge::GRAPH_SUCCESS;
}

} // namespace Base
} // namespace Ops
