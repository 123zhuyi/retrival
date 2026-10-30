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
 * \file coalesce_sparse_simt.h
 * \brief
 */
#ifndef COALESCE_SPARSE_SIMT_H
#define COALESCE_SPARSE_SIMT_H

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"

using namespace AscendC;
constexpr int64_t THREAD_NUM = 1024;

template <typename uIdxType, typename idxType, typename dataType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SetIndices(__gm__ uIdxType* uniqueIndices,
                                                                       __gm__ idxType* indices,
                                                                       __gm__ idxType* newIndices, uint32_t taskLen,
                                                                       uint32_t m)
{
    for (uint64_t index = threadIdx.x; index < static_cast<uint64_t>(taskLen) * m; index += blockDim.x) {
        uint64_t outputIdx = *(uniqueIndices + (index / m));
        __gm__ idxType* indices_base = (__gm__ idxType*)indices + index;
        __gm__ idxType* newIndices_base = (__gm__ idxType*)newIndices + outputIdx * m + index % m;
        asc_stcg(newIndices_base, asc_ldcg(indices_base));
    }
}

template <typename uIdxType, typename idxType, typename dataType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ValueAtomicAdd(__gm__ uIdxType* uniqueIndices,
                                                                           __gm__ dataType* values,
                                                                           __gm__ dataType* newValue, uint32_t taskLen,
                                                                           uint32_t valueSize)
{
    for (uint64_t index = threadIdx.x; index < static_cast<uint64_t>(taskLen) * valueSize; index += blockDim.x) {
        uint64_t outputIdx = *(uniqueIndices + (index / valueSize));
        __gm__ dataType* values_base = (__gm__ dataType*)values + index;
        __gm__ dataType* newValue_base = (__gm__ dataType*)newValue + outputIdx * valueSize + index % valueSize;
        asc_atomic_add(newValue_base, asc_ldcg(values_base));
    }
}

template <typename uIdxType, typename idxType, typename dataType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SetIndicesDeterministic(__gm__ uIdxType* uniqueIndices,
                                                                                    __gm__ idxType* indices,
                                                                                    __gm__ idxType* newIndices,
                                                                                    uint32_t taskLen, uint32_t m)
{
    for (uint32_t index = threadIdx.x; index < taskLen; index += blockDim.x) {
        uint32_t outputIdx = *(uniqueIndices + index);
        __gm__ idxType* indices_base = (__gm__ idxType*)indices + index * m;
        __gm__ idxType* newIndices_base = (__gm__ idxType*)newIndices + outputIdx * m;
        for (uint32_t m_iter = 0; m_iter < m; m_iter++) {
            asc_stcg(newIndices_base, asc_ldcg(indices_base));
            indices_base++;
            newIndices_base++;
        }
    }
}

template <typename uIdxType, typename idxType, typename dataType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ValueAtomicAddDeterministic(__gm__ uIdxType* uniqueIndices,
                                                                                        __gm__ dataType* values,
                                                                                        __gm__ dataType* newValue,
                                                                                        uint32_t taskLen,
                                                                                        uint32_t valueSize)
{
    for (uint32_t index = 0; index < taskLen; index += 1) {
        uint32_t outputIdx = *(uniqueIndices + index);
        __gm__ dataType* values_base = (__gm__ dataType*)values + index * valueSize;
        __gm__ dataType* newValue_base = (__gm__ dataType*)newValue + outputIdx * valueSize;
        for (uint32_t v_iter = threadIdx.x; v_iter < valueSize; v_iter += blockDim.x) {
            asc_atomic_add(newValue_base + v_iter, asc_ldcg(values_base + v_iter));
        }
    }
}

template <typename uIdxType, typename idxType, typename dataType>
class KernelCoalesceSparseSimt {
public:
    __aicore__ inline KernelCoalesceSparseSimt() = default;
    __aicore__ inline void Init(GM_ADDR uniqueIndices, GM_ADDR indices, GM_ADDR values, GM_ADDR newIndices,
                                GM_ADDR newValue, const CoalesceSparseTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTilingValue(const CoalesceSparseTilingData* __restrict tilingData);

private:
    __gm__ uIdxType* uniqueIndicesAddr_ = nullptr;
    __gm__ idxType* indicesAddr_ = nullptr;
    __gm__ dataType* valuesAddr_ = nullptr;
    __gm__ idxType* newIndicesAddr_ = nullptr;
    __gm__ dataType* newValueAddr_ = nullptr;

    uint64_t usedCoreNum_{0};
    uint64_t m_{0};
    uint64_t valueSize_{0};
    uint64_t taskNum_{0};
    uint64_t taskTail_{0};
    uint64_t taskLen_{0};
    uint64_t deterministicFlag_{0};
    const CoalesceSparseTilingData* __restrict tilingDevice{nullptr};
};

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparseSimt<uIdxType, idxType, dataType>::Init(
    GM_ADDR uniqueIndices, GM_ADDR indices, GM_ADDR values, GM_ADDR newIndices, GM_ADDR newValue,
    const CoalesceSparseTilingData* __restrict tilingData)
{
    InitTilingValue(tilingData);
    uint64_t coreId = GetBlockIdx();
    uint64_t beginOffset = coreId * taskNum_;
    uint64_t indicesBeginOffset = beginOffset * m_;
    uint64_t valueBeginOffset = beginOffset * valueSize_;

    if (coreId < usedCoreNum_ - 1) {
        taskLen_ = taskNum_;
    } else if (coreId == usedCoreNum_ - 1) {
        taskLen_ = taskTail_;
    }

    uniqueIndicesAddr_ = (__gm__ uIdxType*)uniqueIndices + beginOffset;
    indicesAddr_ = (__gm__ idxType*)indices + indicesBeginOffset;
    valuesAddr_ = (__gm__ dataType*)values + valueBeginOffset;
    newIndicesAddr_ = (__gm__ idxType*)newIndices;
    newValueAddr_ = (__gm__ dataType*)newValue;
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparseSimt<uIdxType, idxType, dataType>::InitTilingValue(
    const CoalesceSparseTilingData* __restrict tilingData)
{
    // Get tilingData
    this->tilingDevice = tilingData;
    usedCoreNum_ = tilingDevice->usedCoreNum;
    m_ = tilingDevice->m;
    valueSize_ = tilingDevice->valueSize;
    taskNum_ = tilingDevice->taskNum;
    taskTail_ = tilingDevice->taskTail;
    deterministicFlag_ = tilingDevice->deterministicFlag;
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparseSimt<uIdxType, idxType, dataType>::Process()
{
    if (taskLen_ == 0) {
        return;
    }

    if (deterministicFlag_ == 1) {
        asc_vf_call<SetIndicesDeterministic<uIdxType, idxType, dataType>>(dim3{THREAD_NUM, 1, 1}, uniqueIndicesAddr_,
                                                                          indicesAddr_, newIndicesAddr_, taskLen_, m_);
        asc_vf_call<ValueAtomicAddDeterministic<uIdxType, idxType, dataType>>(
            dim3{THREAD_NUM, 1, 1}, uniqueIndicesAddr_, valuesAddr_, newValueAddr_, taskLen_, valueSize_);
    } else {
        asc_vf_call<SetIndices<uIdxType, idxType, dataType>>(dim3{THREAD_NUM, 1, 1}, uniqueIndicesAddr_, indicesAddr_,
                                                             newIndicesAddr_, taskLen_, m_);
        asc_vf_call<ValueAtomicAdd<uIdxType, idxType, dataType>>(dim3{THREAD_NUM, 1, 1}, uniqueIndicesAddr_,
                                                                 valuesAddr_, newValueAddr_, taskLen_, valueSize_);
    }
}

#endif // COALESCE_SPARSE_SIMT
