/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_gmm.h
 * \brief
 */
#ifndef __ALLTO_ALLV_GMM_H__
#define __ALLTO_ALLV_GMM_H__

#include "allto_allv_gmm_utils.h"

namespace ALLTO_ALLV_GMM {
constexpr uint32_t thresholdBlockNum = 8;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
constexpr uint32_t thresholdDimM = 1;
#else
constexpr uint32_t thresholdDimM = 5;
#endif
static constexpr uint32_t MAX_EP_RANK_SIZE = 128U;
static constexpr uint32_t MAX_EXPERT_PER_EP = 32U;

struct MNConfig {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t mIdx;
    uint32_t nIdx;
    uint32_t numBlocksM;
    uint32_t numBlocksN;
    uint32_t singleM;
    uint32_t singleN;
    uint64_t xBaseOffset = 0;
    uint64_t yBaseOffset = 0;
    uint64_t wBaseOffset = 0;
};

/** @brief GroupMatmul operator Class
 */
template <typename ComputeType>
class GMMProcess
{
private:
    ComputeType& computeOp; // inernal computation operator
    uint32_t blockIdx;
    uint32_t coreNum;
    uint32_t baseM;
    uint32_t baseN;

    __aicore__ inline void MnBlockIdxCompute(
        MNConfig& mnConfig, const uint32_t curBlock, const uint32_t count, const uint32_t thresholdM_dimN);

public:
    __aicore__ inline GMMProcess(ComputeType& computeOp_) : computeOp(computeOp_)
    {}

    __aicore__ inline void Init(uint32_t curBaseM, uint32_t curBaseN, uint32_t curCoreNum);
    __aicore__ inline void Process(
        uint16_t rankId, uint32_t mmH, uint32_t mmN, uint64_t* inOffset, uint64_t* outOffset, uint32_t* tokenNum,
        uint16_t computerNum, uint32_t wIndex);
};

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Init(uint32_t curBaseM, uint32_t curBaseN, uint32_t curCoreNum)
{
    blockIdx = GetBlockIdx();
    baseM = curBaseM;
    baseN = curBaseN;
    coreNum = curCoreNum;
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::MnBlockIdxCompute(
    MNConfig& mnConfig, const uint32_t curBlock, const uint32_t count, const uint32_t thresholdM_dimN)
{
    if (mnConfig.numBlocksM <= thresholdDimM || thresholdDimM == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.numBlocksN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.numBlocksN;
    } else {
        uint32_t relativeBlock = curBlock - count;
        uint32_t curThresholdM = relativeBlock >= AlignDown(mnConfig.numBlocksM * mnConfig.numBlocksN, thresholdM_dimN) ?
                                     mnConfig.numBlocksM % thresholdBlockNum :
                                     thresholdBlockNum;
        uint32_t curThresholdM_thresholdN = curThresholdM * thresholdBlockNum;
        uint32_t curThresholdN =
            relativeBlock % thresholdM_dimN >= AlignDown(curThresholdM * mnConfig.numBlocksN, curThresholdM_thresholdN) ?
                mnConfig.numBlocksN % thresholdBlockNum :
                thresholdBlockNum;

        uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
        mnConfig.mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * thresholdBlockNum;
        mnConfig.nIdx = (localRelativeBlock + localRelativeBlock / LeastCommonMultiple(curThresholdM, curThresholdN)) %
                            curThresholdN +
                        relativeBlock % thresholdM_dimN / curThresholdM_thresholdN * thresholdBlockNum;
    }
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Process(
    uint16_t rankId, uint32_t mmH, uint32_t mmN, uint64_t* inOffset, uint64_t* outOffset, uint32_t* tokenNum,
    uint16_t computerNum, uint32_t wIndex)
{
    // aicore function
    MNConfig mnConfig;
    mnConfig.baseM = baseM;
    mnConfig.baseN = baseN;

    // eIndex 顺序
    mnConfig.k = mmH;
    mnConfig.n = mmN;

    mnConfig.wBaseOffset = mnConfig.k * mnConfig.n;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;

    for (uint16_t computerIdx = 0U, count = 0U; computerIdx < computerNum; ++computerIdx) {
        mnConfig.m = tokenNum[computerIdx];
        mnConfig.xBaseOffset = inOffset[computerIdx];
        mnConfig.yBaseOffset = outOffset[computerIdx];
        if (mnConfig.m == 0U || mnConfig.k == 0U || mnConfig.n == 0U) {
            return;
        }

        mnConfig.numBlocksM = Ceil(mnConfig.m, mnConfig.singleM); // 向上取整
        mnConfig.numBlocksN = Ceil(mnConfig.n, mnConfig.singleN); // 向上取整

        uint32_t curCount = count + mnConfig.numBlocksM * mnConfig.numBlocksN;
        uint32_t curBlock = blockIdx >= count ? blockIdx : blockIdx + coreNum;
        uint32_t thresholdM_dimN = thresholdBlockNum * mnConfig.numBlocksN;

        while (curBlock < curCount) {
            MnBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
            computeOp.MMCompute(mnConfig, wIndex, rankId);
            curBlock += coreNum;
        }
        count = curCount % coreNum;
    }
}

/** @brief intenal computation class
 */
template <class mmType, bool sync = false>
class GMMCompute
{
public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using CT = typename mmType::CT::T;

    __aicore__ inline GMMCompute(typename mmType::MT& mm_) : mm(mm_)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y);
    __aicore__ inline void MMCompute(MNConfig& mnConfig, uint32_t wIndex, uint16_t rankId);

private:
    typename mmType::MT& mm; // matmul operator
    GM_ADDR xTensorPtr;
    GM_ADDR weightTensorPtr;
    GM_ADDR yTensorPtr;
    GlobalTensor<AT> xGm;
    GlobalTensor<BT> weightGm;
    GlobalTensor<CT> yGm;
};

template <class mmType, bool sync>
__aicore__ inline void GMMCompute<mmType, sync>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y)
{
    xTensorPtr = x;
    weightTensorPtr = weight;
    yTensorPtr = y;
}

template <class mmType, bool sync>
__aicore__ inline void GMMCompute<mmType, sync>::MMCompute(MNConfig& mnConfig, uint32_t wIndex, uint16_t rankId)
{
    uint32_t curSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.numBlocksN - 1) {
        curSingleN = mnConfig.n - mnConfig.nIdx * curSingleN;
    }
    uint32_t curSingleM = mnConfig.singleM;
    if (mnConfig.mIdx >= mnConfig.numBlocksM - 1) {
        curSingleM = mnConfig.m - mnConfig.mIdx * curSingleM;
    }

    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    uint64_t wOffset = mnConfig.nIdx * mnConfig.singleN;
    if (mmType::BT::isTrans) {
        wOffset = mnConfig.nIdx * mnConfig.k * mnConfig.singleN;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + mnConfig.nIdx * mnConfig.singleN;

    // Init global buffer
    xGm.SetGlobalBuffer((__gm__ AT*)xTensorPtr + mnConfig.xBaseOffset);
    weightGm.SetGlobalBuffer((__gm__ BT*)weightTensorPtr + wIndex * mnConfig.wBaseOffset);
    yGm.SetGlobalBuffer((__gm__ CT*)yTensorPtr + mnConfig.yBaseOffset);

    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGm[xOffset]);
    mm.SetTensorB(weightGm[wOffset], mmType::BT::isTrans);
    mm.template IterateAll<sync>(yGm[outOffset], false);
}
} // namespace ALLTO_ALLV_GMM

#endif // __ALLTO_ALLV_GMM_H__