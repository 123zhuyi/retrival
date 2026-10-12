/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_SIMT_CHUNK_PRE_H
#define SPLIT_V_SIMT_CHUNK_PRE_H
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "simt_api/asc_simt.h"

#ifdef __DAV_FPGA__
constexpr int32_t CHUNK_PRE_THREAD_DIM = 512;
#else
constexpr int32_t CHUNK_PRE_THREAD_DIM = 2048;
#endif

constexpr int32_t CHUNK_PRE_THREADS_PER_LOGIC_BLOCK = 128;
constexpr int32_t CHUNK_PRE_LOGIC_BLOCK_PER_CORE = 16;
constexpr int32_t CHUNK_PRE_UB_FIELD_PER_BLOCK = 7;
constexpr int32_t CHUNK_PRE_UB_TOTAL_U64 = CHUNK_PRE_LOGIC_BLOCK_PER_CORE * CHUNK_PRE_UB_FIELD_PER_BLOCK;

namespace SplitV {
using namespace AscendC;

// vf: 与 chunk_ub.h CopyChunkUB 完全一致 — 只读 UB, 每个 block 7 个字段
// bStart, bEnd, strideDiff, magic, shift, inAddr, outAddr
template <typename T>
__simt_vf__ LAUNCH_BOUND(CHUNK_PRE_THREAD_DIM) __aicore__ void CopyChunkPre(__ubuf__ uint64_t* e)
{
    __ubuf__ uint64_t* b = e + (threadIdx.x >> 7) * CHUNK_PRE_UB_FIELD_PER_BLOCK;
    int64_t bStart = (int64_t)b[0];
    int64_t bEnd = (int64_t)b[1];
    int64_t strideDiff = (int64_t)b[2];
    uint64_t magic = b[3];
    uint64_t shift = b[4];
    __gm__ T* inBase = reinterpret_cast<__gm__ T*>(b[5]);
    __gm__ volatile T* outBase = reinterpret_cast<__gm__ volatile T*>(b[6]);

    if (bStart >= bEnd) {
        return;
    }

    for (int64_t idx = bStart + (threadIdx.x & (CHUNK_PRE_THREADS_PER_LOGIC_BLOCK - 1)); idx < bEnd;
         idx += CHUNK_PRE_THREADS_PER_LOGIC_BLOCK) {
        int64_t row = (int64_t)Simt::UintDiv<uint64_t>((uint64_t)idx, magic, shift);
        outBase[idx] = inBase[idx + row * strideDiff];
    }
}

template <typename T>
class SplitVSIMTChunkPre {
public:
    __aicore__ inline SplitVSIMTChunkPre(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const SplitVSIMTChunkPreTilingData* tilingData);
    __aicore__ inline void Process();

private:
    const SplitVSIMTChunkPreTilingData* tilingData_ = nullptr;
    GlobalTensor<T> xGm_;
    ListTensorDesc outputList_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> ubBuf_;
    int32_t blockIdx_ = 0;
    int32_t realCoreNum_ = 0;
    int32_t splitNum_ = 0;
};

template <typename T>
__aicore__ inline void SplitVSIMTChunkPre<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                   const SplitVSIMTChunkPreTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    outputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));
    realCoreNum_ = tilingData_->realCoreNum;
    splitNum_ = tilingData_->splitNum;
    pipe_.InitBuffer(ubBuf_, CHUNK_PRE_UB_TOTAL_U64 * sizeof(uint64_t));
}

// 标量侧: 遍历 16 个 block, 从 tilingData_ 读预存字段, 算 inAddr/outAddr, 全部写 UB
// 与 chunk_ub.h Process 完全一致, 区别只是 q/r/magic/shift 从 tilingData_ 预存读而非现场算
template <typename T>
__aicore__ inline void SplitVSIMTChunkPre<T>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }

    int32_t gbeg = blockIdx_ * CHUNK_PRE_LOGIC_BLOCK_PER_CORE;
    LocalTensor<uint64_t> ubTensor = ubBuf_.Get<uint64_t>();

    __gm__ T* xPhy = (__gm__ T*)xGm_.GetPhyAddr();

    int32_t oi = 0;
    for (int32_t k = 0; k < CHUNK_PRE_LOGIC_BLOCK_PER_CORE; k++) {
        int32_t gb = gbeg + k;

        while (oi < splitNum_ && static_cast<int32_t>(tilingData_->blockPrefix[oi]) <= gb) {
            oi++;
        }

        if (oi >= splitNum_) {
            for (int32_t f = 0; f < CHUNK_PRE_UB_FIELD_PER_BLOCK; f++) {
                ubTensor.SetValue(k * CHUNK_PRE_UB_FIELD_PER_BLOCK + f, (uint64_t)0);
            }
            continue;
        }

        uint32_t base0 = (oi == 0) ? 0 : tilingData_->blockPrefix[oi - 1];
        uint32_t blkCnt = tilingData_->blockPrefix[oi] - base0;
        if (blkCnt == 0) {
            for (int32_t f = 0; f < CHUNK_PRE_UB_FIELD_PER_BLOCK; f++) {
                ubTensor.SetValue(k * CHUNK_PRE_UB_FIELD_PER_BLOCK + f, (uint64_t)0);
            }
            continue;
        }

        int64_t j = (int64_t)gb - (int64_t)base0;
        int64_t q = tilingData_->qPerBlk[oi];
        int64_t r = tilingData_->rPerBlk[oi];
        int64_t bStart = j * q + (j < r ? j : r);
        int64_t bEnd = bStart + q + (j < r ? 1 : 0);

        uint64_t magic = tilingData_->divMagic[oi];
        uint64_t shift = static_cast<uint64_t>(tilingData_->divShift[oi]);
        int64_t sd = tilingData_->strideDiff[oi];
        int32_t colOff = static_cast<int32_t>(tilingData_->colOffset[oi]);

        uint64_t inAddr = reinterpret_cast<uint64_t>(xPhy) +
                          static_cast<uint64_t>(colOff) * static_cast<uint64_t>(sizeof(T));
        __gm__ T* outPtr = outputList_.GetDataPtr<T>(static_cast<uint32_t>(oi));
        uint64_t outAddr = reinterpret_cast<uint64_t>(outPtr);

        ubTensor.SetValue(k * CHUNK_PRE_UB_FIELD_PER_BLOCK + 0, (uint64_t)bStart);
        ubTensor.SetValue(k * CHUNK_PRE_UB_FIELD_PER_BLOCK + 1, (uint64_t)bEnd);
        ubTensor.SetValue(k * CHUNK_PRE_UB_FIELD_PER_BLOCK + 2, (uint64_t)sd);
        ubTensor.SetValue(k * CHUNK_PRE_UB_FIELD_PER_BLOCK + 3, magic);
        ubTensor.SetValue(k * CHUNK_PRE_UB_FIELD_PER_BLOCK + 4, shift);
        ubTensor.SetValue(k * CHUNK_PRE_UB_FIELD_PER_BLOCK + 5, inAddr);
        ubTensor.SetValue(k * CHUNK_PRE_UB_FIELD_PER_BLOCK + 6, outAddr);
    }

    __ubuf__ uint64_t* eUb = (__ubuf__ uint64_t*)ubTensor.GetPhyAddr();
    asc_vf_call<CopyChunkPre<T>>(dim3(CHUNK_PRE_THREAD_DIM), eUb);
}

} // namespace SplitV
#endif // SPLIT_V_SIMT_CHUNK_PRE_H
