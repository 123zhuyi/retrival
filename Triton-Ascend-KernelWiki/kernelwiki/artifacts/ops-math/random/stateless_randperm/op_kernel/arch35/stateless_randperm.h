/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_randperm.h
 * \brief
 */

#ifndef STATELESS_RANDPERM_H
#define STATELESS_RANDPERM_H

#include <cmath>
#include <limits.h>
#include "op_kernel/math_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "stateless_randperm_sort.h"
#include "stateless_randperm_random.h"
#include "../stateless_randperm_struct.h"
#include "simt_api/asc_simt.h"

namespace StatelessRandperm {
using namespace AscendC;

template <typename N, typename T, typename Y>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH) inline void FindAndFisherYares(
    __gm__ T* y1WorkSpace_, __gm__ volatile N* indexWorkSpace_, __gm__ volatile Y* outGm_,
    uint32_t repeatTimes, uint64_t randomBits, uint32_t factor, uint32_t coreNum, int64_t offset,
    uint32_t key0, uint32_t key1, int64_t n)
{
    if (blockIdx.x >= coreNum) return;
    uint32_t counter[ALG_COUNTER_SIZE] = {0};
    uint32_t key[ALG_KEY_SIZE] = {key0, key1};
    uint32_t results;
    uint32_t last[ALG_COUNTER_SIZE];
    uint32_t state = 0;
    T mask = static_cast<T>((1UL << randomBits) - 1);
    for (uint16_t i = 0; i < repeatTimes; i++){
        uint64_t blockIdxVal = blockIdx.x * factor + i;
        uint64_t tid = blockIdxVal * blockDim.x + threadIdx.x;
        
        // find the beginning of islands
        if (tid >= n - 1) continue; // out of range
        if ((y1WorkSpace_[tid] & mask) != (y1WorkSpace_[tid + 1] & mask)) continue;
        if (tid != 0 && ((y1WorkSpace_[tid] & mask) == (y1WorkSpace_[tid - 1] & mask))) continue;

        // find the size of islands
        int islandSize = 0;
        do { islandSize++; }
        while ((tid + islandSize < n) && ((y1WorkSpace_[tid + islandSize] & mask) == (y1WorkSpace_[tid] & mask)));

        // do random permutation inside each island.
        uint64_t dataOffset = tid;
        RandInit(state, tid, offset, key, counter, last);
        for (int j = islandSize - 1; j > 0; j--) {
            Rand1(state, key, counter, results, last); 
            T r = results % (j + 1);
            if (j != r) {
                N tmp = indexWorkSpace_[dataOffset + j];
                indexWorkSpace_[dataOffset + j] = indexWorkSpace_[dataOffset + r];
                indexWorkSpace_[dataOffset + r] = tmp;
            }
        }
    }
}

template <typename Y, typename N>
__simt_vf__ __aicore__ LAUNCH_BOUND(DATACOPY_THREAD_LAUNCH) inline void CopyData(
    __gm__ volatile Y* outGm_, __gm__ N* indexWorkSpace_, int64_t n,
    uint32_t factor, uint32_t repeatTimes)
{
    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint64_t blockIdxVal = blockIdx.x * factor + i;
        uint64_t tid = blockIdxVal * blockDim.x + threadIdx.x;
        if (tid > n - 1) return; // out of range
        outGm_[tid] = static_cast<Y>(indexWorkSpace_[tid]);
    }   
}

template <typename R, typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(PHILOX_THREAD_LAUNCH) inline void Philox(
__gm__ volatile R* randWorkSpace, T n, int32_t randomBits,
uint32_t factor, uint32_t repeatTimes, uint32_t coreNum, T offset,
uint32_t key0, uint32_t key1)
{
    if (blockIdx.x >= coreNum) return;
    uint32_t counter[ALG_COUNTER_SIZE] = {0};
    uint32_t key[ALG_KEY_SIZE] = {key0, key1};
    uint32_t results[ALG_COUNTER_SIZE];
    uint32_t last[ALG_COUNTER_SIZE];
    T converted[CONVERT_COUNTER_SIZE];
    T mask = static_cast<T>((1UL << randomBits) - 1);
    uint32_t state = 0;
    T dimX = (n + PHILOX_USED_THREAD - 1) / PHILOX_USED_THREAD;
    dimX = min(dimX, static_cast<T>(MAX_DIM_X)); 
    T roundedSize = ((n - 1) / (PHILOX_USED_THREAD * dimX * UNROLL_FACTOR) + 1) * 
                            (PHILOX_USED_THREAD * dimX * UNROLL_FACTOR);
    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t blockIdxVal = blockIdx.x * factor + i;
        uint32_t idx = blockIdxVal * blockDim.x + threadIdx.x;
        RandInit(state, idx, offset, key, counter, last);
        T linearStep = PHILOX_USED_THREAD * dimX * UNROLL_FACTOR;
        for(T linearIndex = idx; linearIndex < roundedSize; linearIndex += linearStep) {
            Rand4(state, key, counter, results, last);
            ConvertToResult<T>(converted, results); 
            for (T ii = 0; ii < 2; ii++) {
                T li = linearIndex + PHILOX_USED_THREAD * dimX * ii;
                if (li < n) {
                    randWorkSpace[li] = static_cast<R>(converted[ii] & mask);
                }
            }
        }
    }
}

template <typename Tn, typename Tr, typename Ty, uint64_t schId, uint64_t isInt32, uint64_t isDescend>
class StatelessRandperm{

public:
    __aicore__ inline StatelessRandperm(TPipe* pipe, StatelessRandpermTilingData* __restrict tiling) : pipe_(pipe), tilingData_(tiling)
    {};
    __aicore__ inline void Init(GM_ADDR n, GM_ADDR seed, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace)
    {
        InitParams();

        auto blockIdxVal = GetBlockIdx();

        outGm_.SetGlobalBuffer((__gm__ Ty *)(y));

        usrWorkspace = AscendC::GetUserWorkspace(workspace);

        uint64_t wkOffsetBits = 0;
        nRangeWorkSpace_.SetGlobalBuffer((__gm__ Tn *)(usrWorkspace + wkOffsetBits), n_);

        wkOffsetBits = wkOffsetBits + static_cast<uint64_t>(n_ * sizeof(Tn));
        indexWorkSpace_.SetGlobalBuffer((__gm__ Tn *)(usrWorkspace + wkOffsetBits), n_);

        wkOffsetBits = wkOffsetBits + static_cast<uint64_t>(n_ * sizeof(Tn));
        randWorkSpace_.SetGlobalBuffer((__gm__ Tr *)(usrWorkspace + wkOffsetBits), n_);

        wkOffsetBits = wkOffsetBits + static_cast<uint64_t>(n_ * sizeof(Tr));
        y1WorkSpace_.SetGlobalBuffer((__gm__ Tr *)(usrWorkspace + wkOffsetBits), n_);
    }

    __aicore__ inline void InitParams()
    {
        randomBits_ = tilingData_->randomBits;
        realCoreNum_ = GetBlockNum();
        factor_ = tilingData_->islandFactor;              // 每个AiCore的线程循环次数
        factorTail_ = tilingData_->islandFactorTail;      // 最后一个AiCore的线程循环次数
        castFactor_ = tilingData_->castFactor;
        castFactorTail_ = tilingData_->castFactorTail;
        randomWkSizeByte_ = tilingData_->randomWkSizeByte;
        sortTilingData_ = &(tilingData_->sortTilingData);
        subNTileCount_ = tilingData_->subNTileCount;

        n_ = tilingData_->n;
        offset_ = tilingData_->philoxOffset;

        uint32_t dimX = static_cast<uint32_t>((n_ + PHILOX_USED_THREAD - 1) / PHILOX_USED_THREAD);
        uint32_t cudaThreadBlockCount = dimX > MAX_DIM_X ? MAX_DIM_X : dimX;
        philoxFactor_ = Ops::Base::CeilDiv(static_cast<uint32_t>(cudaThreadBlockCount), static_cast<uint32_t>(realCoreNum_));
        philoxNeedCoreNum_ = Ops::Base::CeilDiv(static_cast<uint32_t>(cudaThreadBlockCount), static_cast<uint32_t>(philoxFactor_));
        philoxFactorTail_ = cudaThreadBlockCount - philoxFactor_ * (philoxNeedCoreNum_ - 1);
        if (GetBlockIdx() == philoxNeedCoreNum_-1) {
            philoxRepeatTimes = philoxFactorTail_;
        } else {
            philoxRepeatTimes = philoxFactor_;
        }

        uint32_t cudaFisherThreadBlockCount = static_cast<uint32_t>((n_ + 511) / 512);
        fisherNeedCoreNum_ = Ops::Base::CeilDiv(cudaFisherThreadBlockCount, factor_);
        if (GetBlockIdx() == fisherNeedCoreNum_ - 1) {
            repeatTimes = factorTail_;
        } else {
            repeatTimes = factor_;
        }

        for (uint32_t i = 0; i < ALG_KEY_SIZE; i++) {
            key_[i] = tilingData_->philoxKey[i];
        }
        for (uint32_t i = 0; i < SUB_N_TILE_COUNT; i++) {
            subNTile_[i] = tilingData_->subNTile[i];
        }
    }

    __aicore__ inline void Process()
    {
        randomProcess(n_, offset_);

        // Sort
        pipe_->Reset();
        Sort<Tr, Tn, schId, isInt32, isDescend>((GM_ADDR)(randWorkSpace_.GetPhyAddr()), (GM_ADDR)(y1WorkSpace_.GetPhyAddr()), (GM_ADDR)(indexWorkSpace_.GetPhyAddr()), 
                                        usrWorkspace + randomWkSizeByte_, reinterpret_cast<SortRegBaseTilingData*>(sortTilingData_), pipe_);

        SyncAll();
        asc_vf_call<FindAndFisherYares<Tn, Tr, Ty>>(dim3{USED_THREAD}, 
            (__gm__ Tr*)(y1WorkSpace_.GetPhyAddr()), (__gm__ Tn*)(indexWorkSpace_.GetPhyAddr()),
            (__gm__ Ty*)(outGm_.GetPhyAddr()), repeatTimes, randomBits_, factor_, fisherNeedCoreNum_, counterOffset_,
            key_[0], key_[1], n_);

        SyncAll();
        if (GetBlockIdx() == realCoreNum_ -1) {
            repeatTimes = castFactorTail_;
        } else {
            repeatTimes = castFactor_;
        }
        asc_vf_call<CopyData<Ty, Tn>>(dim3{DATACOPY_THREAD_LAUNCH},
                        (__gm__ Ty*)(outGm_.GetPhyAddr()), (__gm__ Tn*)(indexWorkSpace_.GetPhyAddr()), n_,
                        castFactor_, repeatTimes);
        SyncAll();
    }


private:

    __aicore__ inline bool canUse32bitIndexing(int64_t curN)
    {
        int64_t maxVal = std::numeric_limits<int32_t>::max();
        if (curN > maxVal) {
            return false;
        }
        int64_t maxOffset = 1 + (curN - 1) * sizeof(Tr);
        if (maxOffset > maxVal) {
            return false;
        }
        return true;
    }

    __aicore__ inline uint64_t calcCounterOffset(int64_t currentN)
    {
        uint32_t dimX = static_cast<uint32_t>((currentN + PHILOX_USED_THREAD - 1) / PHILOX_USED_THREAD);
        uint32_t cudaThreadBlockCount = dimX > MAX_DIM_X ? MAX_DIM_X : dimX;
        uint64_t counterOffset = (currentN / (PHILOX_USED_THREAD * cudaThreadBlockCount * UNROLL_FACTOR) + 1) * RESULT_ELEMENT_CNT;
        return counterOffset;
    }

    __aicore__ inline void randomProcess(int64_t curN, uint64_t curOffset)
    {
        uint64_t counterOffsetList[SUB_N_TILE_COUNT] = {curOffset};

        if (subNTileCount_ > 1) {
            counterOffsetList[0] = calcCounterOffset(curN) + curOffset;
        }

        uint64_t currentLow;
        uint64_t currentCounterOffset;
        for (int i = 0; i < subNTileCount_; i++) {
            int64_t currentN = subNTile_[i];
            if (i == 0) {
                currentLow = 0;
                currentCounterOffset = counterOffsetList[i];
            } else {
                currentLow = currentLow + subNTile_[i - 1];
                counterOffsetList[i] = counterOffsetList[i - 1] + calcCounterOffset(currentN);
                currentCounterOffset = counterOffsetList[i];
            }

            // 计算needCore
            uint32_t dimX = static_cast<uint32_t>((currentN + PHILOX_USED_THREAD - 1) / PHILOX_USED_THREAD);
            uint32_t cudaThreadBlockCount = dimX > MAX_DIM_X ? MAX_DIM_X : dimX;
            philoxFactor_ = Ops::Base::CeilDiv(static_cast<uint32_t>(cudaThreadBlockCount), static_cast<uint32_t>(realCoreNum_));
            philoxNeedCoreNum_ = Ops::Base::CeilDiv(static_cast<uint32_t>(cudaThreadBlockCount), static_cast<uint32_t>(philoxFactor_));
            philoxFactorTail_ = cudaThreadBlockCount - philoxFactor_ * (philoxNeedCoreNum_ - 1);
            if (GetBlockIdx() == philoxNeedCoreNum_-1) {
                philoxRepeatTimes = philoxFactorTail_;
            } else {
                philoxRepeatTimes = philoxFactor_;
            }              

            if (randomBits_ <= 32) {
                asc_vf_call<Philox<Tr, int32_t>>(dim3(PHILOX_USED_THREAD),
                    (__gm__ Tr*)(randWorkSpace_[currentLow].GetPhyAddr()), static_cast<int32_t>(currentN), randomBits_,
                    philoxFactor_, philoxRepeatTimes, philoxNeedCoreNum_, currentCounterOffset, 
                    key_[0], key_[1]);
            } else {
                asc_vf_call<Philox<Tr, int64_t>>(dim3(PHILOX_USED_THREAD),
                    (__gm__ Tr*)(randWorkSpace_[currentLow].GetPhyAddr()), currentN, randomBits_,
                    philoxFactor_, philoxRepeatTimes, philoxNeedCoreNum_, currentCounterOffset, 
                    key_[0], key_[1]);
            }

            SyncAll();
        }

        counterOffset_ = counterOffsetList[subNTileCount_ - 1] + calcCounterOffset(subNTile_[subNTileCount_ - 1]);
    }

private:
    GlobalTensor<Tn> nRangeWorkSpace_;
    GlobalTensor<Tr> randWorkSpace_;
    GlobalTensor<Tr> y1WorkSpace_;
    GlobalTensor<Tn> indexWorkSpace_;
    GlobalTensor<Ty> outGm_;

private:
    TPipe* pipe_;

private:
    StatelessRandpermTilingData* tilingData_;
    int32_t randomBits_;
    uint32_t realCoreNum_;
    uint32_t philoxNeedCoreNum_;
    uint32_t fisherNeedCoreNum_;
    uint32_t philoxWkOffset_ = 0;
    uint32_t key_[ALG_KEY_SIZE] = {0};
    int16_t subNTileCount_;
    int32_t subNTile_[SUB_N_TILE_COUNT];
    uint32_t factor_;
    uint32_t philoxFactor_;
    uint32_t philoxFactorTail_;
    uint32_t factorTail_;
    uint32_t castFactor_;
    uint32_t castFactorTail_;
    uint32_t repeatTimes;
    uint32_t philoxRepeatTimes;
    uint64_t randomWkSizeByte_;
    struct SortRegBaseTilingDataForRandperm *sortTilingData_;

    int64_t n_; // n
    int64_t seed_; // seed
    int64_t offset_; // offset
    int64_t counterOffset_ = 0;

    GM_ADDR usrWorkspace;
}; // class StatelessRandperm
} // namespace StatelessRandperm

#endif // STATELESS_RANDPERM_H
